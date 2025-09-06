from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector



import numpy as np
import argparse
from sklearn.metrics import log_loss
import pandas as pd
import os.path


from qiskit import transpile
from qiskit_aer import AerSimulator

import pylab as plt

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
	if train_length==False:
		train_length=len(allData[1])
		train_start=0
	elif train_start+train_length>len(allData[1]):
		train_length = len(allData[1])-train_start
		print('new length for training dataset:',train_length)
	allData[1] = allData[1][train_start:train_start+train_length]
	allData[4] = allData[4][train_start:train_start+train_length]
	allData[0] = allData[0][:nTest]
	allData[3] = allData[3][:nTest]
	scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi),clip=True)
	allData[1] =scaler.fit_transform(allData[1])
	allData[0]=scaler.transform(allData[0])
	allData[2] =scaler.transform(allData[2])
	param_open.close()
	return [allData,seed,repetition,layers,kernelFile,gateFile,costFile,accOut]
	
'''def read_kernel(kernelFile):
	#
	#reads the kernel which has been produced for the training data set
	#
	allKernel=[]
	countKernel=-1
	allData=open(kernelFile,'r')
	for line in allData:
		entries=line.split()
		if entries[0]=='Kernel':
			allKernel.append([])
			countKernel+=1
		else:
			kernelLine=[]
			for en in range(len(entries)):
				kernelLine.append(float(entries[en]))
			allKernel[countKernel].append(kernelLine)
	allData.close()
	return allKernel'''

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
				elif desc == 'T':
					circuit.p(np.pi*0.25, qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Rx':
					circuit.rx(Rotation[j][i],qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Rxx':
					circuit.rxx(Rotation[j][i]*PARAM[ID],qreg_q[j-1],qreg_q[j])
				elif desc == 'Ry':
					circuit.ry(Rotation[j][i],qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Ryy':
					circuit.ryy(Rotation[j][i]*PARAM[ID],qreg_q[j-1],qreg_q[j])
				elif desc == 'Rzx':
					circuit.rzx(Rotation[j][i],qreg_q[j-1],qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Rzz':
					circuit.rzz(Rotation[j][i]*PARAM[ID],qreg_q[j-1],qreg_q[j])
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
    print('acc0,acc1',acc0,acc1)
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
	for i in range(0,len(gate)):
		Kernel = np.zeros((len(X_dataset),len(X_train)))
		print('gate',i,flush=True)
		y_sign=[]
		y_prob_pos=0
		for j in range(len(gate[i])):
			if gate[i][j]!='111111':
				gTyp,fID = gate[i][j].split('_')
				gate[i][j] = f'{gTyp}_{feature[i]}'
		angle_list = [angle[i]]*n_qubit
		newGate = circ_convert(gate[i],1,angle_list)
		qc.compose(newGate,qubits=list(range(0,n_qubit)),inplace=True)
		print(qc,flush=True)
		for dp in range(len(X_dataset)):
			y_pred=0
			qc_test = qc.copy()
			for nf in range(n_feature):
				try: qc_test.assign_parameters({PARAM[nf]: X_dataset[dp][nf]}, inplace = True)
				except: pass
			psi_j = Statevector(qc_test)
			for tr in range(len(X_train)):
				qc_train = qc.copy()
				for nf in range(n_feature):
					try: qc_train.assign_parameters({PARAM[nf]: X_train[tr][nf]}, inplace = True)
					except: pass
				psi_i = Statevector(qc_train)
				k_ij = abs(psi_j.inner(psi_i))**2
				Kernel[dp][tr] = k_ij

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
	return np.array(kernel)
			



test_train,seed,repetition,layer,kernelFile,gateFile,costFile,accOut = read_data()
gate, angle, feature = read_circuit(gateFile)
normalized_Xtest,normalized_Xtrain,normalized_Xval,Y_test,Y_train,Y_val=test_train
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

print('read in finished',flush=True)
n_feature = len(normalized_Xtrain[0])
n_qubit=5#n_feature
PARAM = ParameterVector('x', n_feature)

if os.path.exists(accOut+'K_train'):
	print('reading K_train',flush=True)
	K_train = readKernel(accOut+'K_train')
else:
	print('calculate K_train',flush=True)
	K_train = predict_qc(normalized_Xtrain,normalized_Xtrain)
if os.path.exists(accOut+'K_train'):
	print('reading K_test',flush=True)
	K_test = readKernel(accOut+'K_train')
else:
	print('calculate K_test',flush=True)
	K_test = predict_qc(normalized_Xtrain,normalized_Xtest)



y_predicted = []
for i in range(len(K_train)):
	print(f'train SVM for {i+1} gates')
	model = SVC(kernel='precomputed', class_weight='balanced')
	model = GridSearchCV(model,param_grid={'C':list(np.logspace(start=-4,
		stop=4,num=19,base=10.0))}, scoring='balanced_accuracy',refit=True,cv=20,n_jobs=1)
	model.fit(X=K_train[i], y=Y_train)
	predictions = model.predict(X=K_test[i])
	y_predicted.append(predictions)


if os.path.exists(accOut+'K_train'): print('no new K_train written')
else: writeKernel(K_train,accOut+'K_train')
if os.path.exists(accOut+'K_test'): print('no new K_test written')
else: writeKernel(K_test,accOut+'K_test')

outfile=open(accOut,'w')
outfile.write('Acc0\tAcc1\tAccLow\tBIC\tF1\n')
for i in range(len(y_predicted)):
	acc0_t,acc1_t,Low_t,BIC_t,F1 = evaluation(y_predicted[i],Y_test)#,P_t[i])
	result={'Acc-1':acc0_t,'Acc1':acc1_t,'Acc_low':Low_t,'BIC_t':BIC_t}
	outfile.write(f'{acc0_t}\t{acc1_t}\t{Low_t}\t{BIC_t}\t{F1}\n')
	print(result)
outfile.close()
















