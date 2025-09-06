import argparse
import numpy as np
from sklearn import preprocessing
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

import os.path
import time
from joblib import Parallel, delayed
import sympy as sp

from QNN_circuit import circ_convert
from QNN_cost import determine_sine_curve, find_cost_angle


def firstKernelTest(mf,AA,BB,CC,qc,matrix_feature,theta_test_matrix,n_feature,PARAM,n_qubit,norm_Xtrain_l):
	'''
	Determine a,b,c for the first gate added to the circuit
	Args:
		mf: gate iteration parameter
		AA: amplitude parameter matrix
		BB: shift parameter matrix
		CC:	offset parameter matrix
	Returns:
		Parameters of the cosine functions.
	'''
	qc_old_i = qc.copy()
	qc_old_j = qc.copy()
	kernel_test = [0,0,0]
	for kt in range(3):	#iterate over the three test angles
		'''
		Calculate the kernel matrix entries explicitly for theta=0,PI/2,-PI/2
		'''
		qc_new_test = circ_convert(matrix_feature[mf],1,theta_test_matrix[kt],n_feature,PARAM)
		qc_new_test_i = qc_new_test.copy()
		qc_new_test_j = qc_new_test
		qc_new_test_i.assign_parameters({PARAM[1]: 1}, inplace = True)
		qc_new_test_j.assign_parameters({PARAM[1]: 0}, inplace = True)
		qc_unite_test_i = qc_old_i.compose(qc_new_test_i,qubits=list(range(0,n_qubit)),inplace=False)
		qc_unite_test_j = qc_old_j.compose(qc_new_test_j,qubits=list(range(0,n_qubit)),inplace=False)
		psi_i = Statevector(qc_unite_test_i)
		psi_j = Statevector(qc_unite_test_j)
		k_test_ij = psi_j.inner(psi_i)
		kernel_test[kt] = abs(k_test_ij)**2
	'''
	Determine from the three explicit kernels the values for a, b, c in the cos() function
	'''
	
	a,b,c = determine_sine_curve(kernel_test[0],kernel_test[1],kernel_test[2])
	for ID1 in range(norm_Xtrain_l):
		for ID2 in range(ID1,norm_Xtrain_l):
			AA[ID1][ID2] = a
			AA[ID2][ID1] = a
			BB[ID1][ID2] = b
			BB[ID2][ID1] = b
			CC[ID1][ID2] = c
			CC[ID2][ID1] = c
	return [AA,BB,CC]


def layerParallel(ID1,ID2,mf,AA,BB,CC):
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
	dataj = np.array(normalized_Xtrain[ID2])		
	kernel_test = [0,0,0]
	for kt in range(3):	#iterate over the three test angles len(kernel_test)
		'''
		Calculate the kernel matrix entries explicitly for theta=0,PI/2,-PI/2
		'''
		qc_old_i = qc.copy()
		qc_old_j = qc.copy()
		for nf in range(n_feature):
			try:
				qc_old_i.assign_parameters({PARAM[nf]: datai[nf]}, inplace = True)
				qc_old_j.assign_parameters({PARAM[nf]: dataj[nf]}, inplace = True)
			except: pass
		qc_new_test = circ_convert(matrix_feature[mf],1,theta_test_matrix[kt],n_feature,PARAM)
		qc_new_test_i = qc_new_test.copy()
		qc_new_test_j = qc_new_test
		qc_new_test_i.assign_parameters({PARAM[1]: 1}, inplace = True)
		qc_new_test_j.assign_parameters({PARAM[1]: 0}, inplace = True)
		qc_unite_test_i = qc_old_i.compose(qc_new_test_i,qubits=list(range(0,n_feature)),inplace=False)
		qc_unite_test_j = qc_old_j.compose(qc_new_test_j,qubits=list(range(0,n_feature)),inplace=False)
		psi_i = Statevector(qc_unite_test_i)
		psi_j = Statevector(qc_unite_test_j)
		k_test_ij = psi_j.inner(psi_i)
		kernel_test[kt] = abs(k_test_ij)**2
	'''
	Determine from the three explicit kernels the values for a, b, c in the cos() function
	'''
	a,b,c = determine_sine_curve(lastKernel[ID1][ID2],kernel_test[1],kernel_test[2])
	return [ID1,ID2,a,b,c]
					

def layerBuild(last_feat,qubit_use,qc,norm_Xtrain_l,matrix_feature,lastKernel,
	theta_test_matrix,n_feature,PARAM,n_qubit,diff_X_feature,Y_sq,
	num_samples_classical):
	'''
	Build up the circuit by adding a new gate
	Args:
		last_feat: not used anymore
		qubit_use: not used anymore
	Returns:
		Parameters of the circuit at minimal cost.
	'''
	t0=time.time()
	minCost = np.inf
	minCost_layer = qc
	minCost_angle = 0
	minCost_qubit = 0
	minCost_feature = 0
	minCost_a = np.zeros((norm_Xtrain_l,norm_Xtrain_l))
	minCost_b = np.zeros((norm_Xtrain_l,norm_Xtrain_l))
	minCost_c = np.zeros((norm_Xtrain_l,norm_Xtrain_l))
	l_mat_feat=len(matrix_feature)
	for mf in range(l_mat_feat):	#iterate over all possible gates, independent of their feature
		component = 0
		l_mat_feat_i = len(matrix_feature[mf])
		for mfq in range(l_mat_feat_i):
			if matrix_feature[mf][mfq]!='111111':
				component=mfq
		AA=np.zeros((norm_Xtrain_l,norm_Xtrain_l))
		BB=np.zeros((norm_Xtrain_l,norm_Xtrain_l))
		CC=np.zeros((norm_Xtrain_l,norm_Xtrain_l))
		if len(lastKernel)==0:
			AA,BB,CC=firstKernelTest(mf,AA,BB,CC,qc,matrix_feature,
				theta_test_matrix,n_feature,PARAM,n_qubit,norm_Xtrain_l)
		else:
			ABC=Parallel(n_jobs=n_jobs)\
				(delayed(layerParallel)(ID1,ID2,mf,AA,BB,CC) for ID1 in range(norm_Xtrain_l) for ID2 in range(ID1,norm_Xtrain_l))
			l_ABC=len(ABC)
			for labc in range(l_ABC):
				AA[ABC[labc][0]][ABC[labc][1]]=ABC[labc][2]
				AA[ABC[labc][1]][ABC[labc][0]]=ABC[labc][2]
				BB[ABC[labc][0]][ABC[labc][1]]=ABC[labc][3]
				BB[ABC[labc][1]][ABC[labc][0]]=ABC[labc][3]
				CC[ABC[labc][0]][ABC[labc][1]]=ABC[labc][4]
				CC[ABC[labc][1]][ABC[labc][0]]=ABC[labc][4]
		for nf in range(n_feature):	
			'''
			Determine the cost function E=-1/T^2\sum y_i y_j k(x_i x_j \theta)
			'''
			thetaMin,globalMin = find_cost_angle(AA,BB,CC,diff_X_feature,Y_sq,nf,num_samples_classical)
			print(f'Gate {matrix_feature[mf]} Feature {nf} has cost {globalMin}')
			if globalMin<minCost:
				minCost = globalMin
				theta_list = [thetaMin]*n_feature
				new_gate = []
				for qu in range(n_feature):
					if matrix_feature[mf][qu] == '111111': new_gate.append('111111')
					else:
						gate,gate_id = matrix_feature[mf][qu].split('_')
						new_gate.append(f'{gate}_{nf}')
				minCost_layer = circ_convert(new_gate,1,theta_list,n_feature,PARAM)
				minCost_angle = thetaMin
				minCost_qubit = component
				minCost_feature = nf
				minCost_a = AA
				minCost_b = BB
				minCost_c = CC
				minCost_feat = matrix_feature[mf]
	tend=time.time()
	print('time gate:',tend-t0)
	return [minCost,minCost_layer,minCost_angle,minCost_qubit,minCost_feature,minCost_a,minCost_b,minCost_c,minCost_feat]

