from sklearn.svm import SVC
from sklearn import preprocessing
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator
from qiskit.quantum_info import Pauli
from qiskit.quantum_info import SparsePauliOp





import numpy as np
import argparse
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve

import pandas as pd

from qiskit import transpile
from qiskit_aer import AerSimulator

import pylab as plt

from QNN_circuit import circ_convertSinge
from QNN_in_out import read_data


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

def read_test_offset(accOut):
	train = False	#is true if it is the test data file and false otherwise
	offset = 0
	if 'test' in accOut:
		train = True
		testOut = accOut.replace('test','train')

		offset = np.genfromtxt(testOut,usecols=3,skip_header=1)
	return [train,offset]


def initializeMainCirc():
	qreg_qc = QuantumRegister(n_qubit, 'qc')
	qc = QuantumCircuit(qreg_qc)	#initialize quantum circuit
	return qc





def calculate_bic(n, LL, num_params):
    '''
    Calculating Bayesian Information Criteria (BIC) for the proposed model.

    Args:
        n: Number of training points .
        num_params: Number of parameters in the model.
        LL: Log_liklihood.

    Returns:
        BIC value of the proposed model.
    '''
    bic = (-2) * LL + num_params * np.log(n)
    return bic


def evaluation(y_pred,Y_real,P_pred):
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





def predict_qc(X_train,Y_train,X_dataset,train,offset):
	inv_n_data=1/len(X_dataset)
	qc=initializeMainCirc()

	y_sign_gates=[]
	y_prob_pos_gates=[]
	expect_collect = []
	gate_cutoff = min([150,len(gate)])
	for i in range(0,gate_cutoff):
		print('gate',i)
		y_sign=[]
		y_prob_pos=0
		for j in range(len(gate[i])):
			if gate[i][j]!='111111':
				gTyp,fID = gate[i][j].split('_')
				gate[i][j] = f'{gTyp}_{feature[i]}'
		angle_list = [angle[i]]*n_qubit
		newGate = circ_convertSinge(gate[i],1,angle_list,n_qubit,PARAM)
		qc.compose(newGate,qubits=list(range(0,n_qubit)),inplace=True)
		expect_collect.append([])
		for dp in range(len(X_dataset)):
			y_pred=0
			qc_test = qc.copy()
			for nf in range(n_feature):
				try: 
					qc_test.assign_parameters({PARAM[nf]: X_dataset[dp][nf]}, inplace = True)
				except: pass
			
			st = Statevector(qc_test)#.data
			expect = -1+2*st.probabilities()[0]
			expect_collect[-1].append(expect)

			expect_sign = np.sign(expect)
			if expect_sign == 0: expect_sign = 1
			y_sign.append(expect_sign)
			if y_sign[dp]>0: y_prob_pos+=inv_n_data

		y_sign_gates.append(y_sign)
		y_prob_pos_gates.append(y_prob_pos)
	return [y_sign_gates,y_prob_pos_gates,expect_collect]



test_train,seed,repetition,layer,kernelFile,gateFile,costFile,n_jobs,gateList,accOut = read_data()

train,offset = read_test_offset(accOut)

gate, angle, feature = read_circuit(gateFile)
normalized_Xtest,normalized_Xtrain,normalized_Xval,Y_test,Y_train,Y_val=test_train
#Modify basic data
for i in range(len(normalized_Xtest)):
    normalized_Xtest[i][-1] = np.pi
for i in range(len(normalized_Xtrain)):
    normalized_Xtrain[i][-1] = np.pi


for yt in range(len(Y_train)):
	if Y_train[yt] == 0:
		Y_train[yt] = -1
for yt in range(len(Y_test)):
	if Y_test[yt] == 0:
		Y_test[yt] = -1
for yt in range(len(Y_val)):
	if Y_val[yt] == 0:
		Y_val[yt] = -1

print('read in finished')
n_feature = len(normalized_Xtrain[0])
n_qubit = n_feature	#5
PARAM = ParameterVector('x', n_feature)
opZ = SparsePauliOp("Z"*n_qubit)

y_predicted,P_t,expect_predict = predict_qc(normalized_Xtrain,Y_train,normalized_Xtest,train,offset)
for i in range(len(P_t)): P_t[i]=[P_t[i]]*len(y_predicted[i])
print('test calculated')


outfile=open(accOut,'w')
outfile.write('Acc0\tAcc1\tAccAver\tthreshold\n')
for i in range(len(y_predicted)):
	acc0_t,acc1_t,Low_t,BIC_t,F1 = evaluation(y_predicted[i],Y_test,P_t[i])
	if train:
		tpr = np.array([acc1_t])
		tnr = np.array([acc0_t])
		thresholds = np.array([offset[i]])
		accs = (tpr+tnr)*0.5
		maxID = 0
	else:
		fpr,tpr,thresholds = roc_curve(np.array(Y_test),np.array(expect_predict[i]))
		tnr = 1-fpr
		accs = (tpr+tnr)*0.5
		maxID = np.argmax(accs)
		
	# Re-evaluation with new threshold predictions
	y_predicted_threshold = np.where(expect_predict[i] >=thresholds[maxID],1,-1)
	acc0_t_roc,acc1_t_roc,Low_t_roc,BIC_t_roc,F1_roc = evaluation(y_predicted_threshold,Y_test,P_t[i])
	averAcc = (acc0_t_roc+acc1_t_roc)*0.5

	result={'Acc-1':acc0_t_roc,'Acc1':acc1_t_roc,'threshold':thresholds[maxID]}
	outfile.write(f'{acc0_t_roc}\t{acc1_t_roc}\t{averAcc}\t{thresholds[maxID]}\n')
	print(result)
outfile.close()
















