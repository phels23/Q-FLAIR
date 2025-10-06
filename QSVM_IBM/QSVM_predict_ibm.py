import warnings

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Statevector
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import argparse
import os.path
from collections import OrderedDict
import pickle

from QSVM_ibm import evaluateCircuit_QSVM_IBM, TA_loss

# Evaluate target alignment for K_train and K_test
def target_alignment(K, Y, Y2=None):
    K = np.asarray(K)
    # Y must be shape (n_samples,) with values -1 or 1
    Y = np.asarray(Y).reshape(-1, 1)
    if Y2 is None:
        Y2 = Y
    else:
        Y2 = np.asarray(Y2).reshape(-1, 1)
    Y_outer = Y @ Y2.T
    K_flat = K.flatten()
    Y_flat = Y_outer.flatten()
    return - np.dot(K_flat, Y_flat) / (np.linalg.norm(K_flat) * np.linalg.norm(Y_flat))

def remove_idle_qwires(circ):
	dag = circuit_to_dag(circ)

	idle_wires = list(dag.idle_wires())
	dag.remove_qubits(*idle_wires)

	return dag_to_circuit(dag)

def read_data():
	'''
	read in the data files and the setup parameters
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--parameter", type=str)
	args = parser.parse_args()
	param_file = args.parameter
	allData = [[],[],[],[],[],[]] #X_test,X_train,X_val,Y_test,Y_train,Y_val
	seed=0
	repetition=0
	layers=0
	kernelFile = 'a'
	gateFile = 'a'
	costFile = 'a'
	train_start = 0
	train_length=False
	accOut = 'a'
	nTest = 0
	param_open = open(param_file,'r')
	for line in param_open:
		line_split = line.split()
		if line_split[0].strip() == 'Xtest': allData[0] = np.genfromtxt(line_split[1].strip(),delimiter=',')
		elif line_split[0].strip() == 'Xtrain': allData[1] = np.genfromtxt(line_split[1].strip(),delimiter=',')
		elif line_split[0].strip() == 'Xval': allData[2] = np.genfromtxt(line_split[1].strip(),delimiter=',')
		elif line_split[0].strip() == 'Ytest': allData[3] = np.genfromtxt(line_split[1].strip(),delimiter=',')
		elif line_split[0].strip() == 'Ytrain': allData[4] = np.genfromtxt(line_split[1].strip(),delimiter=',')
		elif line_split[0].strip() == 'Yval': allData[5] = np.genfromtxt(line_split[1].strip(),delimiter=',')
		elif line_split[0].strip() == 'seed': seed = float(line_split[1].strip())
		elif line_split[0].strip() == 'repetition': repetition = int(line_split[1].strip())
		elif line_split[0].strip() == 'layers': layers = int(line_split[1].strip())
		elif line_split[0].strip() == 'kernelFile': kernelFile = line_split[1].strip()
		elif line_split[0].strip() == 'gateFile': gateFile = line_split[1].strip()
		elif line_split[0].strip() == 'costFile': costFile = line_split[1].strip()
		elif line_split[0].strip() == 'train_start': train_start = int(line_split[1].strip())
		elif line_split[0].strip() == 'train_length': train_length = int(line_split[1].strip())
		elif line_split[0].strip() == 'accOut': accOut = line_split[1].strip()
		elif line_split[0].strip() == 'nTest': nTest = int(line_split[1].strip())
		elif line_split[0].strip() == 'train_batch': train_batch = int(line_split[1].strip())
		elif line_split[0].strip() == 'nNystroem': nNystroem = int(line_split[1].strip())
		elif line_split[0].strip() == 'shots': shots = int(line_split[1].strip())
	if train_length==False:
		train_length=len(allData[1])
		train_start=0
	elif train_start+train_length>len(allData[1]):
		train_length = len(allData[1])-train_start
		print('new length for training dataset:',train_length)
	allData[0] = allData[0][:nTest]
	allData[3] = allData[3][:nTest]
	scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi),clip=True)			#normalize the data to the range 0 to 2PI
	allData[1] =scaler.fit_transform(allData[1])
	allData[0]=scaler.transform(allData[0])
	allData[2] =scaler.transform(allData[2])

	# cut train data:
	allData[1] = allData[1][train_start:train_start+train_length]
	allData[4] = allData[4][train_start:train_start+train_length]

	param_open.close()
	return [allData,seed,repetition,layers,kernelFile,gateFile,costFile,accOut,train_batch,shots,nNystroem]


def read_circuit(gateFile):
	gate=[]
	angle=[]
	feature=[]

	all_data=open(gateFile,'r')
	for line in all_data:
		gateL=[]
		entries=line.split('\t')
		angle.append(float(entries[-2]))
		feature.append(int(entries[-1]))
		for q in range(len(entries)-2):
			gateL.append(entries[q])
		gate.append(gateL)
	all_data.close()
	return [gate,angle,feature]


def initializeMainCirc():
	qreg_qc = QuantumRegister(n_qubit, 'qc')
	qc = QuantumCircuit(qreg_qc)	#initialize quantum circuit
	return qc

def circ_convert(seq,m,p):
	'''
	Construct the quantum circuit based on the given sequence of gates and parameters.

	Args:
		seq: Sequence of gates (number of features x m).
		m: Number of layers.
		p: Parameter values of the Z-rotation gates (number of features x m).

	Returns:
		Quantum circuit.
	'''
	Qubits = n_qubit
	Descriptor=np.reshape(list(seq),(Qubits,m))
	Rotation=np.reshape(list(p),(Qubits,m))
	qreg_q = QuantumRegister(Qubits, 'q')
	circuit = QuantumCircuit(qreg_q)
	for i in range(len(Descriptor[0])):
		for j in range(len(Descriptor)):
			descript = Descriptor[j][i].split('_')
			if len(descript)==2:
				desc,ID = descript
				if ID == 'x':
					ID = 1
				ID = int(ID)
				if desc == 'U1':
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'U2':
					circuit.rx(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Rz':
					circuit.rz(Rotation[j][i],qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'H':
					circuit.h(qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'X':
					circuit.cx(qreg_q[j-1], qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Xn':
					circuit.cx(qreg_q[j-2], qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Xm':
					circuit.cx(qreg_q[j-3], qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Z':
					circuit.cz(qreg_q[j-1], qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Zn':
					circuit.cz(qreg_q[j-2], qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Zm':
					circuit.cz(qreg_q[j-3], qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				#additional gates
				elif desc == 'T':
					circuit.p(np.pi*0.25, qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Rx':
					circuit.rx(Rotation[j][i],qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Rxx':
					circuit.rxx(Rotation[j][i]*PARAM[ID],qreg_q[j-1],qreg_q[j])
					#circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Ry':
					circuit.ry(Rotation[j][i],qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Ryy':
					circuit.ryy(Rotation[j][i]*PARAM[ID],qreg_q[j-1],qreg_q[j])
					#circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Rzx':
					circuit.rzx(Rotation[j][i],qreg_q[j-1],qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Rzz':
					circuit.rzz(Rotation[j][i]*PARAM[ID],qreg_q[j-1],qreg_q[j])
					#circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'S':
					circuit.s(qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Sdg':
					circuit.sdg(qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'swap':
					circuit.swap(qreg_q[j-1],qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Sx':
					circuit.sx(qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Sxdg':
					circuit.sxdg(qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Y':
					circuit.y(qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'ZH':
					circuit.cz(qreg_q[j-1], qreg_q[j])
					circuit.h(qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'ZHn':
					circuit.cz(qreg_q[j-1], qreg_q[j])
					circuit.h(qreg_q[j-1])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j-1])
	return(circuit)



def evaluation(y_pred,Y_real):#,P_pred):
    '''
    Calculating statistical criteria for the proposed model.

    Args:
        y_pred: Predicted labels.
        Y_real: Real labels in the dataset.
        P_pred: Predicted probability of correct label.

    Returns:
        acc0:  True Positive Rate (TPR)
        acc1: True Negative Rate (TNR)
        Low_acc: Min(TPR,TNR)
        BIC: Bayesian Information Criteria.
        f1: F1-score
    '''
    count0=0
    count1=0
    t=0
    r=0
    Yr=Y_real
    for j in range(len(y_pred)):
        if Yr[j]==-1:
            t+=1
            if y_pred[j]==Yr[j]:
                count0+=1
        else:
            r+=1
            if y_pred[j]==Yr[j]:
                count1+=1
    acc0=count0/t
    acc1=count1/r
    Low_acc = min(acc0,acc1)
    BIC=0
    f1 = 0
    return (acc0,acc1,Low_acc,BIC,f1)




def predict_qc(X_train,X_dataset):
	inv_n_data=1/len(X_dataset)
	qc=initializeMainCirc()
	y_sign_gates=[]
	y_prob_pos_gates=[]
	Kernel_gate = []
	X_train_orig = np.asarray(X_train)
	X_dataset_orig = np.asarray(X_dataset)
	for i in tqdm(range(0,len(gate)), desc='Iterating over gates'):
		if X_train_orig.ndim == 3:  # different batches per iteration
			X_train = X_train_orig[i]
		if X_dataset_orig.ndim == 3:  # different batches per iteration
			X_dataset = X_dataset_orig[i]
		Kernel = np.zeros((len(X_dataset),len(X_train)))
		y_sign=[]
		y_prob_pos=0
		for j in range(len(gate[i])):
			if gate[i][j]!='111111':
				gTyp,fID = gate[i][j].split('_')
				gate[i][j] = f'{gTyp}_{feature[i]}'
		angle_list = [angle[i]]*n_qubit
		newGate = circ_convert(gate[i],1,angle_list)
		qc.compose(newGate,qubits=list(range(0,n_qubit)),inplace=True)
		qc_red = remove_idle_qwires(qc)
		print(qc_red,flush=True)

		# parallelize the "outer loop" that iterates over the data points (for dp in range(len(X_dataset)))
		def compute_kernel_for_dp(dp, X_dataset, X_train, n_feature, PARAM, qc):
			y_pred=0
			qc_test = qc.assign_parameters({PARAM[nf]: X_dataset[dp][nf]
											for nf in range(n_feature) if PARAM[nf] in qc.parameters}, inplace = False)

			Kernel_row = []
			for tr in range(len(X_train)):
				qc_train = qc.assign_parameters({PARAM[nf]: X_train[tr][nf]
												for nf in range(n_feature) if PARAM[nf] in qc.parameters}, inplace=False)


				qc_kernel = qc_test & qc_train.inverse()


				k_ij = Statevector(qc_kernel).probabilities()[0]
				Kernel_row.append(k_ij)
			return Kernel_row

		Kernel = np.array(
			Parallel(n_jobs=min(32, len(X_dataset)))(
				delayed(compute_kernel_for_dp)(dp, X_dataset, X_train, n_feature, PARAM, qc_red)
				for dp in range(len(X_dataset))
			)
		)

		Kernel_gate.append(Kernel)

	return Kernel_gate

def writeKernel(kernel,outname):
	outfile=open(outname,'w')
	for i_gate in range(len(kernel)):
		outfile.write(f'GATE,{i_gate+1}\n')
		for i_line in range(len(kernel[i_gate])):
			for i_column in range(len(kernel[i_gate][i_line])-1):
				outfile.write(f'{kernel[i_gate][i_line][i_column]},')
			outfile.write(f'{kernel[i_gate][i_line][-1]}\n')
	outfile.close()

def readKernel(inname):
	infile=open(inname,'r')
	kernel = []
	for line in infile:
		line_entries = line.split(',')
		if len(line_entries)>0:
			if line_entries[0] == 'GATE': kernel.append([])
			else:
				kernel[-1].append([float(i) for i in line_entries])
	infile.close()
	return kernel


def sv_evaluate_pub(pub):
	qc, _, ps = pub
	evs = []
	for p in ps:
		qc_p = qc.copy()
		for k, v in p.items():
			try:
				qc_p.assign_parameters({k: v}, inplace=True)
			except:
				pass
		ev = Statevector(qc_p).probabilities()[0]
		evs.append(ev)
	return evs


test_train,seed,repetition,layer,kernelFile,gateFile,costFile,accOut,train_batch,shots,nNystroem = read_data()
gate, angle, feature = read_circuit(gateFile)
normalized_Xtest,normalized_Xtrain,normalized_Xval,Y_test,Y_train,Y_val=test_train
accOut = os.path.splitext(accOut)[0]

#Modify basic data
for yt in range(len(Y_train)):
	if Y_train[yt] == 0:
		Y_train[yt] = -1
for yt in range(len(Y_test)):
	if Y_test[yt] == 0:
		Y_test[yt] = -1
for yt in range(len(Y_val)):
	if Y_val[yt] == 0:
		Y_val[yt] = -1

# Load mini batches:
normalized_Xtrain_mini = []
Y_train_mini = []
for i in range(len(gate)):
	train_mini_i = np.load(os.path.join(os.path.dirname(accOut), 'train_minibatches', f'{i:03d}.npz'))
	normalized_Xtrain_mini.append(train_mini_i['normalized_Xtrain'])
	Y_train_mini.append(train_mini_i['Y_train'])


print('read in finished',flush=True)
n_feature = len(normalized_Xtrain[0])
n_qubit=30
PARAM = ParameterVector('x', n_feature)

if os.path.exists(accOut+'K_train.txt'):
	print('reading K_train',flush=True)
	K_train = readKernel(accOut+'K_train.txt')
else:
	print('calculate K_train',flush=True)
	K_train = predict_qc(normalized_Xtrain,normalized_Xtrain)
if os.path.exists(accOut+'K_test.txt'):
	print('reading K_test',flush=True)
	K_test = readKernel(accOut+'K_test.txt')
else:
	print('calculate K_test',flush=True)
	K_test = predict_qc(normalized_Xtrain,normalized_Xtest)
K_train = np.asarray(K_train)
K_test = np.asarray(K_test)


y_predicted = []
y_predicted_train = []
losses = []
losses_train = []
for i in range(len(K_train)):
	model = SVC(kernel='precomputed', class_weight='balanced')
	model = GridSearchCV(model,
						 param_grid={'C': list(np.logspace(start=-4, stop=4, num=19, base=10.0))},
						 scoring='balanced_accuracy', refit=True, cv=20, n_jobs=-1)
	model.fit(X=K_train[i], y=Y_train)
	predictions = model.predict(X=K_test[i])
	predictions_train = model.predict(X=K_train[i])

	y_predicted.append(predictions)
	y_predicted_train.append(predictions_train)

	losses_train.append(target_alignment(K_train[i], Y_train))
	losses.append(target_alignment(K_test[i], Y_test, Y_train))

if os.path.exists(accOut+'K_train.txt'): print('no new K_train written')
else: writeKernel(K_train,accOut+'K_train.txt')
if os.path.exists(accOut+'K_test.txt'): print('no new K_test written')
else: writeKernel(K_test,accOut+'K_test.txt')
del K_train, K_test  # avoids accidental reuse of the kernel data


# Load the precomputed kernels from the IBM backend
if os.path.exists(accOut+'K_test_ibm.txt'):
	print('reading K_test_ibm',flush=True)
	K_test_ibm = readKernel(accOut+'K_test_ibm.txt')
else:
	print('calculate K_test_ibm',flush=True)
	K_test_ibm = []
	for i in range(len(gate)):  # number of iterations performed
		K_test_ibm_i = np.load(os.path.join(os.path.dirname(accOut),
											'ibm_results', f'test_{i:03d}_kernel.npz'))['kernel']
		K_test_ibm.append(K_test_ibm_i)
	writeKernel(K_test_ibm, accOut+'K_test_ibm.txt')
if os.path.exists(accOut+'K_train_ibm.txt'):
	print('reading K_train_ibm',flush=True)
	K_train_ibm = readKernel(accOut+'K_train_ibm.txt')
else:
	print('calculate K_train_ibm',flush=True)
	K_train_ibm = []
	for i in range(len(gate)):  # number of iterations performed
		K_train_ibm_i = np.load(os.path.join(os.path.dirname(accOut),
											 'ibm_results', f'{i:03d}_kernel.npz'))['kernel']
		K_train_ibm.append(K_train_ibm_i)
	writeKernel(K_train_ibm, accOut+'K_train_ibm.txt')
K_train_ibm = np.asarray(K_train_ibm)
K_test_ibm = np.asarray(K_test_ibm)


y_predicted_ibm = []
y_predicted_train_ibm = []
losses_ibm = []
losses_train_ibm = []
for i in range(len(K_train_ibm)):
	model = SVC(kernel='precomputed', class_weight='balanced')
	model = GridSearchCV(model,
						 param_grid={'C': list(np.logspace(start=-4, stop=4, num=19, base=10.0))},
						 scoring='balanced_accuracy', refit=True, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42),
						 n_jobs=-1)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", UserWarning)
		model.fit(X=K_train_ibm[i], y=Y_train_mini[i])
	predictions = model.predict(X=K_test_ibm[i])
	predictions_train = model.predict(X=K_train_ibm[i])

	y_predicted_ibm.append(predictions)
	y_predicted_train_ibm.append(predictions_train)

	losses_train_ibm.append(target_alignment(K_train_ibm[i], Y_train_mini[i]))
	losses_ibm.append(target_alignment(K_test_ibm[i], Y_test, Y_train_mini[i]))

del K_train_ibm, K_test_ibm  # avoids accidental reuse of the kernel data


# Exact simulation of the kernel but for mini batches
if os.path.exists(accOut+'K_test_mini.txt'):
	print('reading K_test_mini',flush=True)
	K_test_mini = readKernel(accOut+'K_test_mini.txt')
else:
	print('calculate K_test_mini',flush=True)
	K_test_mini = predict_qc(normalized_Xtrain_mini,normalized_Xtest)
	writeKernel(K_test_mini, accOut+'K_test_mini.txt')
if os.path.exists(accOut+'K_train_mini.txt'):
	print('reading K_train_mini',flush=True)
	K_train_mini = readKernel(accOut+'K_train_mini.txt')
else:
	print('calculate K_train_mini',flush=True)
	K_train_mini = predict_qc(normalized_Xtrain_mini,normalized_Xtrain_mini)
	writeKernel(K_train_mini, accOut+'K_train_mini.txt')
K_train_mini = np.asarray(K_train_mini)
K_test_mini = np.asarray(K_test_mini)

y_predicted_mini = []
y_predicted_train_mini = []
losses_mini = []
losses_train_mini = []
for i in range(len(K_train_mini)):
	model = SVC(kernel='precomputed', class_weight='balanced')
	model = GridSearchCV(model,
						 param_grid={'C': list(np.logspace(start=-4, stop=4, num=19, base=10.0))},
						 scoring='balanced_accuracy', refit=True, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42),
						 n_jobs=-1)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", UserWarning)
		model.fit(X=K_train_mini[i], y=Y_train_mini[i])
	predictions = model.predict(X=K_test_mini[i])
	predictions_train = model.predict(X=K_train_mini[i])

	y_predicted_mini.append(predictions)
	y_predicted_train_mini.append(predictions_train)

	losses_train_mini.append(target_alignment(K_train_mini[i], Y_train_mini[i]))
	losses_mini.append(target_alignment(K_test_mini[i], Y_test, Y_train_mini[i]))

del K_train_mini, K_test_mini  # avoids accidental reuse of the kernel data



# We load the evs from the IBM backend (with and without error mitigation contained in .npz) and determine the error mitigation effect relative to the exact simulation
# load npz:
rmses = []
opt_param_diff = []
opt_cost_diff = []
for i in tqdm(range(len(gate)), desc="Reconstruction quality evaluation"):
	evs_ibm_backend = np.load(os.path.join(os.path.dirname(accOut), 'ibm_results', f'{i:03d}_evs.npz'))
	evs_pool = [v for k, v in sorted(evs_ibm_backend.items()) if k.startswith('evs')]  # mitigated
	raw_evs_pool = [v for k, v in sorted(evs_ibm_backend.items()) if k.startswith('raw')]  # raw (unmitigated)
	# For exact simulation, load pubs_orig.pickle:
	with open(os.path.join(os.path.dirname(accOut), 'ibm_results', f'{i:03d}_pubs_orig.pickle'), 'rb') as f:
		pubs_orig = pickle.load(f)
	# Evaluate the circuits with exact simulation (to obtain sv_evs):
	sv_evs_pool = []
	for pub in pubs_orig:
		sv_evs = sv_evaluate_pub(pub)
		sv_evs_pool.append(sv_evs)
	assert len(sv_evs_pool) == len(evs_pool) == len(raw_evs_pool) == len(pubs_orig)
	# For lists, add tuple where first element is w.r.t the mitigated evs, second element is w.r.t. the raw evs
	rmses.append([])
	opt_param_diff.append([])
	opt_cost_diff.append([])
	num_samples = int(1e2)
	for evs, raw_evs, sv_evs in zip(evs_pool, raw_evs_pool, sv_evs_pool):  # iterate over the pool

		(AA, BB, CC, XX, YY) = \
			evaluateCircuit_QSVM_IBM(Y_train_mini[i], normalized_Xtrain_mini[i],
									 0, None, None, 1, None, None,
									 i, len(normalized_Xtrain_mini[i]), evs, return_reconstruction_only=True)
		(AA_raw, BB_raw, CC_raw, XX_raw, YY_raw) = \
			evaluateCircuit_QSVM_IBM(Y_train_mini[i], normalized_Xtrain_mini[i],
									 0, None, None, 1, None, None,
									 i, len(normalized_Xtrain_mini[i]), raw_evs, return_reconstruction_only=True)
		(AA_sv, BB_sv, CC_sv, XX_sv, YY_sv) = \
			evaluateCircuit_QSVM_IBM(Y_train_mini[i], normalized_Xtrain_mini[i],
									 0, None, None, 1, None, None,
									 i, len(normalized_Xtrain_mini[i]), sv_evs, return_reconstruction_only=True)

		# Sample reconstructions
		alpha = np.linspace(-np.pi, np.pi, num_samples)
		alpha = alpha.reshape(-1, 1)
		kernel_mat_flat = np.clip(AA.flatten() * np.cos(alpha - BB.flatten()) + CC.flatten(), 0.0, 1.0)
		costs = TA_loss(kernel_mat_flat, YY.flatten())
		opt_idx = np.argmin(costs)
		opt_param, opt_cost = alpha[opt_idx, 0], costs[opt_idx]
		kernel_mat_flat_raw = np.clip(AA_raw.flatten() * np.cos(alpha - BB_raw.flatten()) + CC_raw.flatten(), 0.0, 1.0)
		costs_raw = TA_loss(kernel_mat_flat_raw, YY_raw.flatten())
		opt_idx_raw = np.argmin(costs_raw)
		opt_param_raw, opt_cost_raw = alpha[opt_idx_raw, 0], costs_raw[opt_idx_raw]
		kernel_mat_flat_sv = np.clip(AA_sv.flatten() * np.cos(alpha - BB_sv.flatten()) + CC_sv.flatten(), 0.0, 1.0)
		costs_sv = TA_loss(kernel_mat_flat_sv, YY_sv.flatten())
		opt_idx_sv = np.argmin(costs_sv)
		opt_param_sv, opt_cost_sv = alpha[opt_idx_sv, 0], costs_sv[opt_idx_sv]

		# Calculate the RMSE for the mitigated and raw evs w.r.t. the sv_evs
		rmse = np.sqrt(np.mean((kernel_mat_flat - kernel_mat_flat_sv) ** 2, axis=0))
		rmse_raw = np.sqrt(np.mean((kernel_mat_flat_raw - kernel_mat_flat_sv) ** 2, axis=0))
		rmses[-1].extend(zip(rmse, rmse_raw))
		# Calculate the absolute difference in optimal parameters and costs
		opt_param_diff[-1].append((np.abs(opt_param - opt_param_sv), np.abs(opt_param_raw - opt_param_sv)))
		opt_cost_diff[-1].append((np.abs(opt_cost - opt_cost_sv), np.abs(opt_cost_raw - opt_cost_sv)))


metrics_keys = ['Acc0', 'Acc1', 'AccLow', 'Avg-Acc', 'loss']
keys = metrics_keys.copy()
keys.extend([k + '_tr' for k in metrics_keys])
keys.extend([k + '_ibm' for k in metrics_keys])
keys.extend([k + '_ibm_tr' for k in metrics_keys])
keys.extend([k + '_mini' for k in metrics_keys])
keys.extend([k + '_mini_tr' for k in metrics_keys])
keys.extend(['RMSE_mean', 'RMSE_std', 'RMSE_mean_raw', 'RMSE_std_raw'])
keys.extend(['OptParamDiff_mean', 'OptParamDiff_std', 'OptParamDiff_mean_raw', 'OptParamDiff_std_raw'])
keys.extend(['OptCostDiff_mean', 'OptCostDiff_std', 'OptCostDiff_mean_raw', 'OptCostDiff_std_raw'])
outfile=open(accOut + '.txt','w')
outfile.write('\t'.join(keys) + '\n')  # write header with keys
for i in range(len(y_predicted)):
	# Compute all metrics
	acc0, acc1, Low, BIC, F1 = evaluation(y_predicted[i], Y_test)
	acc0_tr, acc1_tr, Low_tr, BIC_tr, F1_tr = evaluation(y_predicted_train[i], Y_train)
	acc0_ibm, acc1_ibm, Low_ibm, BIC_ibm, F1_ibm = evaluation(y_predicted_ibm[i], Y_test)
	acc0_ibm_tr, acc1_ibm_tr, Low_ibm_tr, BIC_ibm_tr, F1_ibm_tr = evaluation(y_predicted_train_ibm[i], Y_train_mini[i])
	acc0_mini, acc1_mini, Low_mini, BIC_mini, F1_mini = evaluation(y_predicted_mini[i], Y_test)
	acc0_mini_tr, acc1_mini_tr, Low_mini_tr, BIC_mini_tr, F1_mini_tr = evaluation(y_predicted_train_mini[i], Y_train_mini[i])

	# Build ordered result
	result = OrderedDict([
		('Acc0', acc0), ('Acc1', acc1), ('AccLow', Low), ('Avg-Acc', (acc0 + acc1) / 2), ('loss', losses[i]),
		('Acc0_tr', acc0_tr), ('Acc1_tr', acc1_tr), ('AccLow_tr', Low_tr), ('Avg-Acc_tr', (acc0_tr + acc1_tr) / 2), ('loss_tr', losses_train[i]),
		('Acc0_ibm', acc0_ibm), ('Acc1_ibm', acc1_ibm), ('AccLow_ibm', Low_ibm), ('Avg-Acc_ibm', (acc0_ibm + acc1_ibm) / 2), ('loss_ibm', losses_ibm[i]),
		('Acc0_ibm_tr', acc0_ibm_tr), ('Acc1_ibm_tr', acc1_ibm_tr), ('AccLow_ibm_tr', Low_ibm_tr), ('Avg-Acc_ibm_tr', (acc0_ibm_tr + acc1_ibm_tr) / 2), ('loss_ibm_tr', losses_train_ibm[i]),
		('Acc0_mini', acc0_mini), ('Acc1_mini', acc1_mini), ('AccLow_mini', Low_mini), ('Avg-Acc_mini', (acc0_mini + acc1_mini) / 2), ('loss_mini', losses_mini[i]),
		('Acc0_mini_tr', acc0_mini_tr), ('Acc1_mini_tr', acc1_mini_tr), ('AccLow_mini_tr', Low_mini_tr), ('Avg-Acc_mini_tr', (acc0_mini_tr + acc1_mini_tr) / 2), ('loss_mini_tr', losses_train_mini[i]),
		('RMSE_mean', np.mean([r[0] for r in rmses[i]])), ('RMSE_std', np.std([r[0] for r in rmses[i]])),
		('RMSE_mean_raw', np.mean([r[1] for r in rmses[i]])), ('RMSE_std_raw', np.std([r[1] for r in rmses[i]])),
		('OptParamDiff_mean', np.mean([d[0] for d in opt_param_diff[i]])), ('OptParamDiff_std', np.std([d[0] for d in opt_param_diff[i]])),
		('OptParamDiff_mean_raw', np.mean([d[1] for d in opt_param_diff[i]])), ('OptParamDiff_std_raw', np.std([d[1] for d in opt_param_diff[i]])),
		('OptCostDiff_mean', np.mean([c[0] for c in opt_cost_diff[i]])), ('OptCostDiff_std', np.std([c[0] for c in opt_cost_diff[i]])),
		('OptCostDiff_mean_raw', np.mean([c[1] for c in opt_cost_diff[i]])), ('OptCostDiff_std_raw', np.std([c[1] for c in opt_cost_diff[i]]))
	])

	# Write to file and print
	outfile.write('\t'.join(str(result[k]) for k in keys) + '\n')
	for k, v in result.items():
		print(f"{k}: {v}")
	print()  # Blank line between results
outfile.close()
