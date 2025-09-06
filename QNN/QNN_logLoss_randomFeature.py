import argparse
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import log_loss
from qiskit.utils import QuantumInstance, algorithm_globals
import scipy as sc
import random as rnd
from qiskit.circuit import Parameter, ParameterVector
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_ibm_provider import IBMProvider
from qiskit import Aer, transpile
from qiskit import BasicAer
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Operator
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Pauli


import os.path
import time
from joblib import Parallel, delayed
import sympy as sp


from QNN_in_out import read_data, write_kernel, write_gates, write_cost
from QNN_circuit import gate_pool, circ_convert, circ_convertSinge, initializeMainCirc
from QNN_cost import determine_sine_curve, eq_sin, kernel, cost, find_cost_angle, cost_sym, find_cost_angle_sym
from QNN_circuitConstruction import firstKernelTest, layerParallel, layerBuild

featureID=0


def find_cost_exp_QNN(qc,Y,normalized_Xtrain,feat_select,gate,n_feature,PARAM,opZ):
	'''
	Cost function calculation for the QNN. The predicted value is determined directly from the expectation value
	Args:
		qc: previous circuit
		Y: labels of the dataset
		normalized_Xtrain: features of the dataset
		feat_select: the feature that is to be tested
		gate: the gate that is to be tested
		n_feature: number of features
		PARAM: list of parameters	
	Returns:
		opt_param: the best angle theta
		opt_y: the best cost function value
	'''
	def cost_exp(theta):
		cost_collect = np.zeros(len(theta))
		for th in range(len(theta)):
			cost = 0
			n_Y = len(Y)
			n_Y_inv = 1/n_Y
			agreement = 0
			expect_collect = []
			for i in range(n_Y):
				qc_eval = qc.copy()
				datai = np.array(normalized_Xtrain[i])
				for nf in range(n_feature):
					try:
						qc_eval.assign_parameters({PARAM[nf]: datai[nf]}, inplace = True)
					except: pass	
				qc_new = circ_convertSinge(gate,1,[theta[th]]*n_qubit,n_qubit,PARAM)
				try:
					qc_new.assign_parameters({PARAM[1]: datai[feat_select]}, inplace = True)
				except: pass
				qc_eval.compose(qc_new,qubits=list(range(0,n_qubit)),inplace=True)
				st = Statevector(qc_eval)
				expect = st.probabilities()[0]
				expect_collect.append(expect)
				
			cost_collect[th] = log_loss(Y,expect_collect)
			
		return cost_collect
	
	num_xs = int(num_samples_classical)
	if num_xs%2==0:
		num_xs+=1
	xs = np.linspace(-1, 1, num_xs)
	ys = cost_exp(xs)

	opt_param = xs[np.argmin(ys)]
	opt_y = np.min(ys)

	return [opt_param, opt_y]

def constructCircuit_QNN(Y,normalized_Xtrain,mf,matrix_feature,theta_test_matrix,n_feature,PARAM,opZ,count,norm_Xtrain_l,featureID):
	'''
	
	'''
	def layerParallel(ID1,mf,AA,BB,CC):
		'''
		Determine a,b,c for all but the first gate added to the circuit in parallel
		Args:
			ID1: data point iteration 1
			ID2: data point iteration 2
			mf: gate iteration parameter
			AA: amplitude parameter matrix
			BB: shift parameter matrix
			CC:	offset parameter matrix
		Returns:
			Parameters of the cosine functions.
		'''
		datai = np.array(normalized_Xtrain[ID1])
						
		expect_test = [0,0,0]
		for kt in range(3):	#iterate over the three test angles len(kernel_test)
			'''
			Calculate the kernel matrix entries explicitly for theta=0,PI/2,-PI/2
			'''
			qc_old_i = qc.copy()
			for nf in range(n_feature):
				try:
					qc_old_i.assign_parameters({PARAM[nf]: datai[nf]}, inplace = True)
				except: pass
			qc_new_test = circ_convertSinge(matrix_feature[mf],1,theta_test_matrix[kt],n_qubit,PARAM)
			qc_new_test_i = qc_new_test
			try:
				qc_new_test_i.assign_parameters({PARAM[1]: 1}, inplace = True)
			except: pass
						
			qc_unite_test_i = qc_old_i.compose(qc_new_test_i,qubits=list(range(0,n_qubit)),inplace=False)
		
			psi_i = Statevector(qc_unite_test_i)
			expect_test[kt] = psi_i.probabilities()[0]	
		'''
		Determine from the three explicit kernels the values for a, b, c in the cos() function
		'''
		a,b,c = determine_sine_curve(expect_test[0],expect_test[1],expect_test[2])
		return [ID1,a,b,c]


	def cost_exp():
		'''
		modification of the global cost_exp function which is independent of theta
		'''
		cost = 0
		n_Y = len(Y)
		n_Y_inv = 1/n_Y
		thetas = np.zeros(n_qubit)
		expect_collect = []
		for i in range(n_Y):
			qc_eval = qc.copy()
			datai = np.array(normalized_Xtrain[i])
			for nf in range(n_feature):
				try:
					qc_eval.assign_parameters({PARAM[nf]: datai[nf]}, inplace = True)
				except: pass
			qc_new = circ_convertSinge(matrix_feature[mf],1,thetas,n_qubit,PARAM)
			qc_eval.compose(qc_new,qubits=list(range(0,n_qubit)),inplace=True)
			st = Statevector(qc_eval)
			expect= st.probabilities()[0]
			expect_collect.append(expect)

		cost=log_loss(Y,expect_collect)
	
		return cost
	
	gateIsDatadependent = True
	for nf in range(n_qubit):
		if '_' in matrix_feature[mf][nf]:
			mf_split = matrix_feature[mf][nf].split('_')[1]
			if mf_split == 'y':
				gateIsDatadependent = False
			
	
	AA=np.zeros(norm_Xtrain_l)
	BB=np.zeros(norm_Xtrain_l)
	CC=np.zeros(norm_Xtrain_l)
	
	if count==0:	
		qc_old_i = qc.copy()
		expect_test = [0,0,0]
		for kt in range(3):	#iterate over the three test angles
			'''
			Calculate the kernel matrix entries explicitly for theta=0,PI/2,-PI/2
			'''
			qc_new_test = circ_convertSinge(matrix_feature[mf],1,theta_test_matrix[kt],n_qubit,PARAM)
			qc_new_test_i = qc_new_test.copy()
			try:
				qc_new_test_i.assign_parameters({PARAM[1]: 1}, inplace = True)
			except: pass
			qc_unite_test_i = qc_old_i.compose(qc_new_test_i,qubits=list(range(0,n_qubit)),inplace=False)
			psi_i = Statevector(qc_unite_test_i)
			expect_test[kt] = psi_i.probabilities()[0]
		'''
		Determine from the three explicit kernels the values for a, b, c in the cos() function
		'''
		a,b,c = determine_sine_curve(expect_test[0],expect_test[1],expect_test[2])
		for nXl in range(norm_Xtrain_l):
			AA[nXl] = a
			BB[nXl] = b
			CC[nXl] = c
	else:
		ABC=Parallel(n_jobs=n_jobs)\
				(delayed(layerParallel)(ID1,mf,AA,BB,CC) for ID1 in range(norm_Xtrain_l))
		l_ABC=len(ABC)
		for labc in range(l_ABC):
			AA[ABC[labc][0]]=ABC[labc][1]
			BB[ABC[labc][0]]=ABC[labc][2]
			CC[ABC[labc][0]]=ABC[labc][3]
	
	
	if gateIsDatadependent:
		opt_theta_feature = []
		opt_cost_feature = []

		nf = featureID
		'''
		Determine the cost function E=-1/T^2\sum y_i y_j k(x_i x_j \theta)
		'''
		opt_theta, opt_cost = find_cost_rec_QNN(AA,BB,CC,normalized_Xtrain,Y,nf,num_samples_classical)
		print(f'Gate {matrix_feature[mf]} Feature {nf} Angle {round(opt_theta,5)} has cost {round(opt_cost,5)}',flush=True)
		
		opt_theta_feature.append(opt_theta)
		opt_cost_feature.append(opt_cost)
	
	else:
		
		opt_cost = cost_exp()
		opt_theta_feature = np.zeros(n_feature)
		opt_cost_feature = []
		print(f'Gate {matrix_feature[mf]} Feature 0-{n_feature-1} has cost {round(opt_cost,5)}',flush=True)
		nf = featureID
		opt_cost_feature.append(opt_cost)
			
		
		
	return [opt_theta_feature,opt_cost_feature,AA,BB,CC]
		
	
def find_cost_rec_QNN(AA,BB,CC,normalized_Xtrain,Y,component,num_samples_classical):
	'''
	Minimizes the cost function
	Args:
		AA: amplitude parameter list
		BB: shift parameter list
		CC:	offset parameter list
		normalized_Xtrain: list with all training features
		Y: list with all training labels
		component: component of the data points
	Returns:
		The angle of the minimal cost, the value of the minimal cost and the point where the optimization started.
	'''
	def cost_rec(theta):
		"""
		Efficient numpy implementation of the cost function (TA).
		Relies on broadcasting and can handle multiple values for theta at once.

		Args:
			theta: weight parameter of the new gate. Can be a single value or a numpy array.
		Returns:
			Value of the cost function (TA)
		"""
		theta = np.array(theta)
		expect = AA*np.cos(np.multiply.outer(theta,normalized_Xtrain[:,component])-BB)+CC
		cost=[]
		for th in range(len(theta)):
			cost.append(log_loss(Y,expect[th]))
		cost = np.array(cost)
		
		assert cost.size == theta.size
		return [cost.item() if theta.size == 1 else cost, expect]

	num_xs = int(num_samples_classical)
	if num_xs%2==0:
		num_xs+=1
	xs = np.linspace(-1, 1, num_xs)
	ys,expect = cost_rec(xs)
	opt_param = xs[np.argmin(ys)]
	opt_y = np.min(ys)


	return opt_param, opt_y  # Invert sign to get cost
	
	



#Read in
test_train,seed,repetition,layer,kernelFile,gateFile,costFile,n_jobs,gateList,accOut = read_data()


normalized_Xtest,normalized_Xtrain,normalized_Xval,Y_test,Y_train,Y_val=test_train

for i in range(len(normalized_Xtest)):
    normalized_Xtest[i][-1] = np.pi
for i in range(len(normalized_Xtrain)):
    normalized_Xtrain[i][-1] = np.pi





if os.path.exists(kernelFile):
	print('File to write kernel to exists already')
	exit(9)
if os.path.exists(gateFile):
	print('File to write gate to exists already')
	exit(9)
if os.path.exists(costFile):
	print('File to write cost to exists already')
	exit(9)



#Define constants
n_feature = len(normalized_Xtrain[0])
n_qubit=n_feature	#5
algorithm_globals.random_seed = seed
norm_Xtrain_l = len(normalized_Xtrain)
norm_Xval_l=len(normalized_Xval)
num_samples_classical = 1e3#1e3

rnd.seed(seed)
PARAM = ParameterVector('x', n_feature)
theta_test = [0,np.pi*0.5,-np.pi*0.5]	#angles to determine a,b,c in ARC
theta_test_matrix = [[],[],[]]	#for each qubit the three test angles, dim: n_feature * 3
for tt in range(len(theta_test)):
	for nf in range(n_qubit):
		theta_test_matrix[tt].append(theta_test[tt])
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
Y_sq=np.full((len(Y_train),len(Y_train)),0)


matrix_feature = gate_pool(gateList,n_qubit)	#matrix with all possible gate combinations for one layer	n_feature
qc = initializeMainCirc(n_qubit)




count=0
cost_tot_list = []
layer_list=[]
opZ = SparsePauliOp("Z"*n_qubit)

while count<100:
	n_feature_coll = []
	new_gate_coll = []
	opt_theta_coll = []
	opt_cost_coll = []
	
	trackAA = []
	trackBB = []
	trackCC = []
	featureID = rnd.randint(0,n_feature-1)
	print('randomly selected feature:',featureID)
	for i in range(len(matrix_feature)):
		opt_theta_feature,opt_cost_feature,AA,BB,CC = constructCircuit_QNN(Y_train,normalized_Xtrain,
			i,matrix_feature,theta_test_matrix,n_feature,PARAM,opZ,count,norm_Xtrain_l,featureID)
		newGate = True
		differenceA = 0
		differenceB = 0
		differenceC = 0
		nG = [0,0,0]
		if len(trackAA)==0: 
			trackAA.append(AA)
			trackBB.append(BB)
			trackCC.append(CC)
		else:
			for a in range(len(trackAA)):
				differenceA = sum(abs(trackAA[a]-AA))
				if differenceA<0.001:	nG[0] = 1
			for b in range(len(trackBB)):
				differenceB = sum(abs(trackBB[b]-BB))
				if differenceB<0.001:	nG[1] = 1
			for c in range(len(trackCC)):
				differenceC = sum(abs(trackCC[c]-CC))
				if differenceC<0.001:	nG[2] = 1
			if sum(nG)==3:
				newGate = False
			else:
				trackAA.append(AA)
				trackBB.append(BB)
				trackCC.append(CC)
		
		if newGate:
			n_feature_coll.append(featureID)
			new_gate_coll.append(matrix_feature[i])
			opt_theta_coll.append(opt_theta_feature[0])
			opt_cost_coll.append(opt_cost_feature[0])

	
	
	ID_best = np.argmin(opt_cost_coll)
	new_gate_select = new_gate_coll[ID_best]
	n_feature_select = n_feature_coll[ID_best]
	opt_theta_select = opt_theta_coll[ID_best]
	opt_cost_select = opt_cost_coll[ID_best]
	
	new_gate_mod = []
	for qu in range(n_qubit):
		if new_gate_select[qu] == '111111': 
			new_gate_mod.append('111111')
		else:
			gate,gate_id = new_gate_select[qu].split('_')
			new_gate_mod.append(f'{gate}_{n_feature_select}')
	
	minCost_layer = circ_convertSinge(new_gate_mod,1,[opt_theta_select]*n_qubit,n_qubit,PARAM)
	qc.compose(minCost_layer,qubits=list(range(n_qubit)),inplace=True)
	
	print(qc)
	print('Summary',count,opt_cost_select,opt_theta_select,flush=True)
	
	count+=1
	cost_tot_list.append(opt_cost_select)
	layer_list.append(count)
	
	write_gates(gateFile,new_gate_select,opt_theta_select,n_feature_select)
	write_cost(costFile,opt_cost_select,count)
	
	## Check if convergence (i.e., no substantial improvement in cost)
	#if count > 1 and cost_tot_list[-2] - cost_tot_list[-1] < 1e-8:	#1e-3:
	#	print('Converged')
	#	break  # Converged!
		
		


