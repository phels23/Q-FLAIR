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
	qc = QuantumCircuit(qreg_qc)	#initialize quantum circuit
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
	n_Y_inv = 1/n_Y
	for i in range(n_Y):
		datai = np.array(normalized_Xtrain[i])
		qc_eval = kernel[i].copy()
		
		for lg in range(len(gate)):
			try:		
				qc_eval.assign_parameters({ANGLE[lg]: theta[lg]}, inplace = True)
			except:
				pass
		st = Statevector(qc_eval)
		expect = st.probabilities()[0]
		expect_collect.append(expect)
	
	cost = log_loss(Y_train,expect_collect)		
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
opZ = SparsePauliOp("Z"*n_qubit)#+"I"*(n_feature-1))


qc = rebuild_qc()
kernel=Kernel_qm(gate,angle,feature,qc)

initCost = calcCost(angle)
print('initCost',initCost,flush=True)
bounds=[]
for i in range(len(angle)):
	bounds.append((-1,1))


optimRes = minimize(calcCost, x0=angle, method='COBYLA',bounds=bounds)

print(optimRes,flush=True)

OutCost=open(accOut+'_cobyla','w')
OutCost.write(f'initial\t{initCost}\n')
OutCost.write(f"optimized\t{optimRes['fun']}\n")
OutCost.write(f"newAngles\t{optimRes['x']}")
OutCost.close()











