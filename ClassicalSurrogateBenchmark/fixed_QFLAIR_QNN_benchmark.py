from sklearn.svm import SVC
from sklearn import preprocessing
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Operator
from qiskit.quantum_info import Pauli
from scipy.optimize import minimize
import numpy as np
import argparse
from sklearn.metrics import log_loss
from qiskit import transpile
from qiskit_aer import AerSimulator
import itertools
from collections import defaultdict
import torch
from tqdm import tqdm


def read_data():
	'''
	read in the data files and the setup parameters
	Returns:
		Data of starting files.
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--parameter", type=str)
	parser.add_argument("-m", "--measure", action='store_true')
	parser.add_argument("-in", "--initialize", action='store_true')
	parser.add_argument("-ra", "--randomangle", action='store_true')
	args = parser.parse_args()

	
	param_file = args.parameter
	allData = [[],[],[],[],[],[]]
	seed=0
	repetition=0
	layers=0
	kernelFile = 'a'
	gateFile = 'a'
	costFile = 'a'
	n_jobs = 1
	train_start = 0
	train_length=False
	accOut = 'a'
	nTest = False
	nGate = False
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
		elif line_split[0].strip() == 'gateList': gateList = line_split[1].split(',')
		elif line_split[0].strip() == 'kernelFile': kernelFile = line_split[1].strip()
		elif line_split[0].strip() == 'gateFile': gateFile = line_split[1].strip()
		elif line_split[0].strip() == 'costFile': costFile = line_split[1].strip()
		elif line_split[0].strip() == 'parallel': n_jobs = int(line_split[1].strip())
		elif line_split[0].strip() == 'train_start': train_start = int(line_split[1].strip())
		elif line_split[0].strip() == 'train_length': train_length = int(line_split[1].strip())
		elif line_split[0].strip() == 'accOut': accOut = line_split[1].strip()
		elif line_split[0].strip() == 'nTest': nTest = int(line_split[1].strip())
		elif line_split[0].strip() == 'nGate': nGate = int(line_split[1].strip())
	if train_length==False:
		train_length=len(allData[1])
		train_start=0
	elif train_start+train_length>len(allData[1]):
		train_length = len(allData[1])-train_start
		print('new length for training dataset:',train_length)
	allData[1] = allData[1][train_start:train_start+train_length]
	allData[4] = allData[4][train_start:train_start+train_length]
	if nTest == False:
		nTest = len(allData[0])
	elif nTest>len(allData[1]):
		nTest = len(allData[1])
	allData[0] = allData[0][:nTest]
	allData[3] = allData[3][:nTest]
	scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi),clip=True)
	allData[1] =scaler.fit_transform(allData[1])
	allData[0]=scaler.transform(allData[0])
	allData[2] =scaler.transform(allData[2])
	param_open.close()
	
	if args.measure or args.initialize or args.randomangle:
		return [allData,seed,repetition,layers,kernelFile,gateFile,costFile,n_jobs,gateList,accOut,
			args.measure,args.initialize,args.randomangle,nGate]
	else:
		return [allData,seed,repetition,layers,kernelFile,gateFile,costFile,n_jobs,gateList,accOut,nGate]

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
	qc = QuantumCircuit(qreg_qc)
	return qc

def circ_convert(seq,m,p):
	'''
	Construct the quantum circuit based on the given sequence of gates and parameters.
	Args:
		seq: Sequence of gates (number of features x m).
		m: Number of layers.
		p: Parameter values of the Z-rotation gates (number of features x m).
		n_feature: Number of features.
		PARAM: List of parameters.
	Returns:
		Quantum circuit.
	'''
	Qubits = n_qubit
	print('descript',seq,Qubits)	
	Descriptor=np.reshape(list(seq),(Qubits,1))
	Rotation=np.reshape(list(p),(Qubits,1))
	qreg_q = QuantumRegister(Qubits, 'q')
	circuit = QuantumCircuit(qreg_q)
	l_descr=len(Descriptor)
	l_descr0=len(Descriptor[0])
	for i in range(l_descr0):
		for j in range(l_descr):
			descript = Descriptor[j][i].split('_')
			if len(descript)==2:
				desc,ID = descript
				if ID == 'x' or ID == 'y':
					ID = 1
				ID = int(ID)
				if desc == 'U1':
					circuit.rz(ANGLE[m]*PARAM[ID],qreg_q[j])
				elif desc == 'U2':
					circuit.rx(ANGLE[m]*PARAM[ID],qreg_q[j])
				elif desc == 'U3':
					circuit.ry(ANGLE[m]*PARAM[ID],qreg_q[j])
				elif desc == 'P':
					circuit.p(ANGLE[m]*PARAM[ID],qreg_q[j])
				elif desc == 'CP':
					circuit.cp(ANGLE[m]*PARAM[ID],qreg_q[j-1],qreg_q[j])
				elif desc == 'H':
					circuit.h(qreg_q[j])
				elif desc == 'X':
					circuit.cx(qreg_q[j-1], qreg_q[j])
					circuit.rz(ANGLE[m]*PARAM[ID],qreg_q[j])
				elif desc == 'Xn':
					circuit.cx(qreg_q[j-2], qreg_q[j])
					circuit.rz(ANGLE[m]*PARAM[ID],qreg_q[j])
				elif desc == 'Xm':
					circuit.cx(qreg_q[j-3], qreg_q[j])
					circuit.rz(ANGLE[m]*PARAM[ID],qreg_q[j])
				elif desc == 'Rz':
					circuit.rz(ANGLE[m],qreg_q[j])
				elif desc == 'Z':
					circuit.cz(qreg_q[j-1], qreg_q[j])
					circuit.rz(ANGLE[m]*PARAM[ID],qreg_q[j])
				elif desc == 'Zn':
					circuit.cz(qreg_q[j-2], qreg_q[j])
					circuit.rz(ANGLE[m]*PARAM[ID],qreg_q[j])
				elif desc == 'Zm':
					circuit.cz(qreg_q[j-3], qreg_q[j])
					circuit.rz(ANGLE[m]*PARAM[ID],qreg_q[j])
				elif desc == 'T':
					circuit.p(np.pi*0.25, qreg_q[j])
				elif desc == 'Rx':
					circuit.rx(ANGLE[m],qreg_q[j])
				elif desc == 'Rxx':
					circuit.rxx(ANGLE[m]*PARAM[ID],qreg_q[j-1],qreg_q[j])
				elif desc == 'Ry':
					circuit.ry(ANGLE[m],qreg_q[j])
				elif desc == 'Ryy':
					circuit.ryy(ANGLE[m]*PARAM[ID],qreg_q[j-1],qreg_q[j])
				elif desc == 'Rzx':
					circuit.rzx(ANGLE[m]*PARAM[ID],qreg_q[j-1],qreg_q[j])
				elif desc == 'Rzz':
					circuit.rzz(ANGLE[m]*PARAM[ID],qreg_q[j-1],qreg_q[j])
				elif desc == 'S':
					circuit.s(qreg_q[j])
				elif desc == 'Sdg':
					circuit.sdg(qreg_q[j])
				elif desc == 'swap':
					circuit.swap(qreg_q[j-1],qreg_q[j])
				elif desc == 'Sx':
					circuit.sx(qreg_q[j])
				elif desc == 'Sxdg':
					circuit.sxdg(qreg_q[j])
				elif desc == 'Y':
					circuit.y(qreg_q[j])
					circuit.rz(ANGLE[m]*PARAM[ID],qreg_q[j])
				elif desc == 'ZH':
					circuit.cz(qreg_q[j-1], qreg_q[j])
					circuit.h(qreg_q[j])
					circuit.rz(ANGLE[m]*PARAM[ID],qreg_q[j])
				elif desc == 'ZHn':
					circuit.cz(qreg_q[j-1], qreg_q[j])
					circuit.h(qreg_q[j-1])
					circuit.rz(ANGLE[m]*PARAM[ID],qreg_q[j-1])
	return(circuit)


def rebuild_qc():
	qc=initializeMainCirc()
	for i in range(0,len(gate)):
		print('gate',i,'added')
		for j in range(len(gate[i])):
			if gate[i][j]!='111111':
				gTyp,fID = gate[i][j].split('_')
				gate[i][j] = f'{gTyp}_{feature[i]}'
		angle_list = [angle[i]]*n_qubit
		newGate = circ_convert(gate[i],i,angle_list)
		qc.compose(newGate,qubits=list(range(0,n_qubit)),inplace=True)
	print(qc)
	return qc




def Kernel_qm(gate,angle,feature,qc):
	'''
	Calculate the kernel matrix for all training values, with variable thetas
	'''
	kernel= [[] for i in range(norm_Xtrain_l)]
	for i in range(norm_Xtrain_l):
		kernel.append([])
		qc_i = qc.copy()
		for nf in range(n_feature):
			try:
				qc_i.assign_parameters({PARAM[nf]: normalized_Xtrain[i][nf]}, inplace = True)
			except: pass
		kernel[i]=qc_i
	return kernel


def calcCost(theta):
    cost = 0
    expect_collect = []
    n_Y = len(Y_train)
    n_Y_inv = 1 / n_Y
    for i in range(n_Y):
        datai = np.array(normalized_Xtrain[i])
        qc_eval = kernel[i].copy()

        for lg in range(len(gate)):
            try:
                qc_eval.assign_parameters({ANGLE[lg]: theta[lg]}, inplace=True)
            except:
                pass
        st = Statevector(qc_eval)
        expect = st.probabilities()[0]
        expect_collect.append(expect)

    cost = log_loss(Y_train, expect_collect)
    return cost


test_train,seed,repetition,layer,kernelFile,gateFile,costFile,n_jobs,gateList,accOut,nGate = read_data()
gate, angle, feature = read_circuit(gateFile)
if nGate!=False and len(gate)>nGate:
	gate = gate[:nGate]
	angle = angle[:nGate]
	feature = feature[:nGate]
print(gate,flush=True)
print(angle,flush=True)
print(feature,flush=True)
normalized_Xtest,normalized_Xtrain,normalized_Xval,Y_test,Y_train,Y_val=test_train
n_feature = len(normalized_Xtrain[0])
norm_Xtrain_l = len(normalized_Xtrain)
norm_Xval_l=len(normalized_Xval)



#Modify basic data
for yt in range(len(Y_train)):
	if Y_train[yt] == -1:
		Y_train[yt] = 0
for yt in range(len(Y_test)):
	if Y_test[yt] == -1:
		Y_test[yt] = 0
for yt in range(len(Y_val)):
	if Y_val[yt] == -1:
		Y_val[yt] = 0

print('read in finished',flush=True)
n_feature = len(normalized_Xtrain[0])
n_qubit = 5
PARAM = ParameterVector('x', n_feature)
ANGLE = ParameterVector('t', len(gate))
opZ = SparsePauliOp("Z"*n_qubit)
qc = rebuild_qc()
kernel=Kernel_qm(gate,angle,feature,qc)
initCost = calcCost(angle)


# Collect frequencies per feature:
eig_vals = defaultdict(list)
# angle holds the current frequencies (more precisely +- eig vals)
# feature holds the corresponding feature index
# qc is fully parameterized circuit
for a, f in zip(angle, feature):
    eig_vals[f].extend([a, -a])  # eigenvalues come in pairs (only two distinct eigenvalues per gate generator)

selected_features = list(sorted(eig_vals.keys()))
# print number of distinct frequencies per feature
for f in eig_vals:
    distinct_freqs = set(np.round(eig_vals[f], decimals=6))  # rounding to avoid floating point issues
    print(f'Feature {f} has {len(distinct_freqs)} distinct frequencies: {distinct_freqs}', flush=True)
# compute in list comprehension the number of Fourier components per feature, which is half the number of distinct frequencies + 1 (for zero frequency)
np.prod([(len(set(np.round(eig_vals[f], decimals=6))) // 2 + 1) for f in eig_vals])
# compute unique frequencies across all features as the unique differences in eigenvalues
unique_freqs = defaultdict(set)
for f in eig_vals:
    eigs = sorted(set(np.round(eig_vals[f], decimals=6)))
    for i in range(len(eigs)):
        for j in range(len(eigs)):
            freq = eigs[j] - eigs[i]
            # round
            # fixme maybe this rounding technique in combination with set is not ideal. BUT I THINK IT DOES NOT CHANGE ANYTHING FOR THE NUMBER OF FOURIER COEFFS
            freq = np.round(freq, decimals=6)
            unique_freqs[f].add(freq)
print("Number of Fourier coefficients estimate:", np.prod(list(map(len, unique_freqs.values()))))

#raise SystemExit("Safeguard for too large number of Fourier coeffs - remove to run full code")

# Improved torch version: create Omega_torch directly
freq_vectors = [torch.tensor(sorted(freqs), dtype=torch.float64) for _, freqs in sorted(unique_freqs.items())]
Omega_torch = torch.cartesian_prod(*freq_vectors)

# Original (slow) Python native implementation:
# # sort by feature index / key
# unique_freqs = dict(sorted(unique_freqs.items()))
# abs_unique_freqs = {feat: set(sorted(abs(freq) for freq in freqs)) for feat, freqs in unique_freqs.items()}
# # create Fourier coeffs dict (numpy arrays) initialized to zero
# fourier_coeffs = {feat: np.zeros((len(freqs),), dtype=np.complex128) for feat, freqs in unique_freqs.items()}
#
# n_selected_features = len(selected_features)
# omega = [[f * e_i for f in sorted(freqs)] for e_i, (_, freqs) in zip(np.eye(n_selected_features), sorted(unique_freqs.items())) ]
# Omega = [sum(pair) for pair in itertools.product(*omega)]
# print(f"Number of total Fourier features: {len(Omega)}", flush=True)
#
# # assert that Omega matches Omega_torch
# Omega_np = np.array(Omega)
# assert np.allclose(Omega_np, Omega_torch.numpy()), "Omega matrices do not match!"
#
# # switch to torch tensors
# Omega_torch = torch.tensor(np.asarray(Omega), dtype=torch.float64).T
Omega_torch = Omega_torch.T  # shape (n_selected_features, n_fourier_features)
print(f"Number of Fourier coefficients confirmed: {Omega_torch.numel()}", flush=True)
X_train_torch = torch.tensor(normalized_Xtrain[:, selected_features], dtype=torch.float64)
Y_train_torch = torch.tensor(Y_train, dtype=torch.float64).view(-1, 1)

X_val_torch = torch.tensor(normalized_Xval[:, selected_features], dtype=torch.float64)
Y_val_torch = torch.tensor(Y_val, dtype=torch.float64).view(-1, 1)


class FourierLinearModel(torch.nn.Module):
    def __init__(self, omega_matrix, out_features=1):
        super().__init__()
        self.register_buffer('Omega', omega_matrix)
        n_fourier_feats = 2 * omega_matrix.shape[1]

        # Linear layer without bias (as bias is included in Fourier features via zero frequency component)
        self.linear = torch.nn.Linear(n_fourier_feats, out_features, bias=False, dtype=torch.float64)

        # Initialize weights
        with torch.no_grad():
            self.linear.weight.zero_()

    def forward(self, x):
        # 1. Project: X @ Omega.T
        projections = x @ self.Omega

        # 2. Compute Features: cos(proj), sin(proj)
        features = torch.cat([torch.cos(projections), torch.sin(projections)], dim=-1)

        # 3. Linear Prediction
        return self.linear(features)


classical_surrogate_model = FourierLinearModel(Omega_torch)

# TODO use log loss, which is more appropriate for classification and should maintain convexity!
loss_fn = torch.nn.MSELoss()
# use stochastic gradient descent to learn fourier coeffs
learning_rate = 1e-7
n_epochs = 1000
optimizer = torch.optim.SGD(classical_surrogate_model.parameters(), lr=learning_rate)
# use minibatches
batch_size = 64
# use dataset loader for batching
train_dataset = torch.utils.data.TensorDataset(X_train_torch, Y_train_torch)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(X_val_torch, Y_val_torch)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# after each epoch, evaluate on validation set and print loss + accuracy
for epoch in tqdm(range(n_epochs)):
    classical_surrogate_model.train()  # switch to train mode
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        Y_pred = classical_surrogate_model(X_batch)
        loss = loss_fn(Y_pred, Y_batch)
        loss.backward()
        batch_acc = ((Y_pred >= 0.5).float() == Y_batch).float().mean()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        classical_surrogate_model.eval()  # switch to eval mode
        with torch.no_grad():
            # use batch loader for evaluation too:
            Y_val_pred = torch.cat([classical_surrogate_model(X_val_batch) for X_val_batch, _ in val_loader], dim=0)
            val_loss = loss_fn(Y_val_pred, Y_val_torch)
            val_acc = ((Y_val_pred >= 0.5).float() == Y_val_torch).float().mean()
            Y_train_pred, Y_train_true = map(torch.cat, zip(*[(classical_surrogate_model(X_b), Y_b) for X_b, Y_b in train_loader]))
            train_loss = loss_fn(Y_train_pred, Y_train_true)
            train_acc = ((Y_train_pred >= 0.5).float() == Y_train_true).float().mean()
        print(f'\nEpoch {epoch + 1}/{n_epochs}, Training Loss: {train_loss.item()}, Accuracy: {train_acc.item():.4f}\n'
              f'Epoch {epoch + 1}/{n_epochs}, Validation Loss: {val_loss.item()}, Accuracy: {val_acc.item():.4f}\n',
              flush=True)



#def fourier_model(theta, x):
    # result = 0
    # for feat in selected_features:
    #     freqs = sorted(unique_freqs[feat])
    #     coeffs = fourier_coeffs[feat]
    #     sum_feat = 0
    #     for idx, freq in enumerate(freqs):
    #         sum_feat += coeffs[idx] * np.exp(-1j * freq * x[feat])
    #     result += sum_feat
    # return np.real(result)



# Optional: add frequencies that are close
# TODO sanity check by fitting surrogate from learned QFLAIR circuit

#
# print('initCost',initCost,flush=True)
# bounds=[]
# for i in range(len(angle)):
# 	bounds.append((-1,1))
# optimRes = minimize(calcCost, x0=angle, method='L-BFGS-B',bounds=bounds)
# print(optimRes,flush=True)
# OutCost=open(accOut,'w')
# OutCost.write(f'initial\t{initCost}\n')
# OutCost.write(f"optimized\t{optimRes['fun']}\n")
# OutCost.write(f"newAngles\t{optimRes['x']}")
# OutCost.close()
#










