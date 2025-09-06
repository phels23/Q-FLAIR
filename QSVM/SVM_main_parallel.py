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


def read_data():
	'''
	read in the data files and the setup parameters
	Returns:
		Data of starting files.
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--parameter", type=str)
	parser.add_argument("-c","--contin", action='store_true')
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
	if train_length==False:
		train_length=len(allData[1])
		train_start=0
	elif train_start+train_length>len(allData[1]):
		train_length = len(allData[1])-train_start
		print('new length for training dataset:',train_length)
	allData[1] = allData[1][train_start:train_start+train_length]
	allData[4] = allData[4][train_start:train_start+train_length]
	scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi),clip=True)	
	allData[1] =scaler.fit_transform(allData[1])
	allData[0]=scaler.transform(allData[0])
	allData[2] =scaler.transform(allData[2])
	param_open.close()
	return [allData,seed,repetition,layers,kernelFile,gateFile,costFile,n_jobs,gateList,args.contin]

def gate_pool():
	'''
	Provides the list of all allowed gates for the circuit
	Returns:
		A matrix with all gate combinations that can be added to the circuit.
	'''
	possible_gate = gateList
	matrix_feature = []
	for pg in range(len(possible_gate)):
		for nf in range(n_qubit):	#n_feature):
			vector_feature = np.full((1,n_qubit),'111111')[0]
			vector_feature[nf] = possible_gate[pg]
			matrix_feature.append(vector_feature)
	return matrix_feature
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
	Qubits = n_qubit	#n_feature
	Descriptor=np.reshape(list(seq),(Qubits,m))
	Rotation=np.reshape(list(p),(Qubits,m))
	qreg_q = QuantumRegister(Qubits, 'q')
	circuit = QuantumCircuit(qreg_q)
	
	l_descr=len(Descriptor)
	l_descr0=len(Descriptor[0])
	for i in range(l_descr0):
		for j in range(l_descr):
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
def determine_sine_curve(f0,fp,fm):
	'''
	determine a,b, and c values for a cos(theta(xj-xi)-b)+c
	by solving equation for f0=k(theta0),fp=k(thetaPI/2), and fm=k(theta-PI/2)
	Rotosolve implementation adapted from
	[Pennylane](https://docs.pennylane.ai/en/stable/_modules/pennylane/optimize/rotosolve.html#RotosolveOptimizer)
	Args:
		f0: kernel for theta=0
		fp: kernel for theta=+pi/2
		fm: kernel for theta=-pi/2
	Returns:
		Solution for system of three cos equations.
	'''
	c = 0.5 * (fp + fm)
	b = np.arctan2(2 * (f0 - c), fp - fm)
	a = np.sqrt((f0 - c) ** 2 + 0.25 * (fp - fm) ** 2)
	# alter b since rotosolve is implemented for sin and additive shift parameter b:
	b = -b + np.pi / 2
	return a, b, c

def eq_sin(theta,a,b,c):
	'''
	Cosinus function with free parameters
	Args:
		theta: angle
		a: amplitude
		b: shift
		c: offset
	Returns:
		Solution of Cosinus function.
	'''
	return a*np.cos(theta-b)+c
def kernel(theta,AA,BB,CC,diff_X_feature,Y_sq,component):
	'''
	Calculate the full kernel from cos functions
	Args:
		theta: angle of new gate
		AA: amplitude parameter matrix
		BB: shift parameter matrix
		CC:	offset parameter matrix
		diff_X_feature: matrix with difference between all data points
		Y_sq: matrix with all classes multiplied
		component: component of the data points
	Returns:
		The kernel matrix.
	'''
	Kernel = np.zeros((l_Y_sq,l_Y_sq))
	for i in range(l_Y_sq):
		for j in range(i,l_Y_sq):
			Kernel[i][j] = AA[i][j]*np.cos(theta*diff_X_feature[component][i][j]-BB[i][j])+CC[i][j]
			if i!=j:
				Kernel[j][i] = Kernel[i][j]
	return Kernel


def cost(theta,AA,BB,CC,diff_X_feature,Y_sq,component):
	'''
	Calculates the kernel target alignment
	Args:
		theta: angle of new gate
		AA: amplitude parameter matrix
		BB: shift parameter matrix
		CC:	offset parameter matrix
		diff_X_feature: matrix with difference between all data points
		Y_sq: matrix with all classes multiplied
		component: component of the data points
	Returns:
		The value of the cost.
	'''
	T=l_Y_sq
	factor = 1/T**2
	Cost=0
	for i in range(T):
		for j in range(i,T):
			costij = factor*Y_sq[i][j]*(AA[i][j]*np.cos(theta*diff_X_feature[component][i][j]-BB[i][j])+CC[i][j])#Kernel[i][j]			
			Cost-= costij
			if i!=j:
				Cost-= costij
	return Cost

def find_cost_angle(AA,BB,CC,diff_X_feature,Y,component):
	'''
	Minimizes the cost function
	Args:
		AA: amplitude parameter matrix
		BB: shift parameter matrix
		CC:	offset parameter matrix
		diff_X_feature: matrix with difference between all data points
		Y: matrix with all classes multiplied
		component: component of the data points
		repetition: not used anymore
	Returns:
		The angle of the minimal cost, the value of the minimal cost and the point where the optimization started.
	'''

	def fn(theta):
		"""
		Efficient numpy implementation of the cost function (TA).
		Relies on broadcasting and can handle multiple values for theta at once.

		Args:
			theta: weight parameter of the new gate. Can be a single value or a numpy array.
		Returns:
			Value of the cost function (TA)
		"""
		theta = np.array(theta)
		theta = theta.reshape(-1, 1)
		kernel_mat_flat = AA.flatten()*np.cos(theta * diff_X_feature[component].flatten() - BB.flatten()) + CC.flatten()
		costs = np.divide(np.sum(Y.flatten() * kernel_mat_flat, axis=1),np.sqrt(np.sum(kernel_mat_flat**2,axis=1))*norm_Xtrain_l)

		# note: mean instead of sum introduces a normalization factor of 1/T^2
		assert costs.size == theta.size
		return costs.item() if theta.size == 1 else costs
	
	num_xs = int(num_samples_classical)
	if num_xs%2==0:
		num_xs+=1
	xs = np.linspace(-1, 1, num_xs)
	ys = fn(xs)
	
	opt_param = xs[np.argmax(ys)]
	opt_y = np.max(ys)
	return opt_param, -opt_y  # Invert sign to get cost

def cost_sym(theta,AA,BB,CC,diff_X_feature,Y_sq,component):
	T=l_Y_sq
	factor=1/T**2
	Cost=0
	for i in range(T):
		for j in range(i,T):
			costij=factor*Y_sq[i][j]*(AA[i][j]*sp.cos(theta*diff_X_feature[component][i][j]-BB[i][j])+CC[i][j])
			Cost-=costij
			if i!=j: Cost-=costij
	return Cost

def find_cost_angle_sym(AA,BB,CC,diff_X_feature,Y,component):	#solving the cost equation analythically is too expensive !!!!!!
	th = sp.Symbol('th', real=True)
	f=cost_sym(th,AA,BB,CC,diff_X_feature,Y,component)
	d1 = f.diff(th)
	if d1==0:
		th_minimum=0
		val_minimum=cost(th_minimum,AA,BB,CC,diff_X_feature,Y_sq,component)
	else:
		d2 = d1.diff(th)
		search_range=np.pi/minDif_feature
		extrema=sp.solve(d1,th)
		print(extrema)
		th_minimum=0
		val_minimum=cost(th_minimum,AA,BB,CC,diff_X_feature,Y_sq,component)
		for ex in extrema:
			if d2.subs(th,ex).is_positive:
				th_pos=ex.evalf()
				th_cost=cost(th_pos,AA,BB,CC,diff_X_feature,Y_sq,component)
				if th_cost<val_minimum:
					th_minimum=th_pos
					val_minimum=th_cost
				elif th_cost==val_minimum and th_minimum**2>th_pos**2:
						th_minimum=th_pos
						val_minimum=th_cost
	return [th_minimum,val_minimum]
			
			



def initializeMainCirc(contin,oldGates):
	'''
	Setup the main quantum circuit
	Returns:
		Quantum circuit.
	'''
	qreg_qc = QuantumRegister(n_qubit, 'qc')
	qc = QuantumCircuit(qreg_qc)	#initialize quantum circuit
	

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
	
	if contin:
		# read in gates which have been determined previously and add them to the circuit
		gate,angle,feature = read_circuit(oldGates)
		for i in range(0,len(gate)):
			for j in range(len(gate[i])):
				if gate[i][j]!='111111':
					gTyp,fID = gate[i][j].split('_')
					gate[i][j] = f'{gTyp}_{feature[i]}'
			angle_list = [angle[i]]*n_qubit	#n_feature
			newGate = circ_convert(gate[i],1,angle_list)
			qc.compose(newGate,qubits=list(range(0,n_qubit)),inplace=True)
	return qc


def firstKernelTest(mf,AA,BB,CC):
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
		qc_new_test = circ_convert(matrix_feature[mf],1,theta_test_matrix[kt])
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
	print('Test kernel determined')
	a,b,c = determine_sine_curve(kernel_test[0],kernel_test[1],kernel_test[2])
	print('a,b,c determined')
	AA = np.full((norm_Xtrain_l,norm_Xtrain_l),a)
	BB = np.full((norm_Xtrain_l,norm_Xtrain_l),b)
	CC = np.full((norm_Xtrain_l,norm_Xtrain_l),c)
	print('AA,BB,CC determined')
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
	if len(lastKernel) == 0:
		kt0 = 0
	else:
		kt0 = 1
	
	
	for kt in range(kt0,3):	#iterate over the three test angles len(kernel_test)
		'''
		Calculate the kernel matrix entries explicitly for theta=0,PI/2,-PI/2
		'''
		qc_old_i = qc.copy()
		qc_old_j = qc.copy()
		for nf in range(n_feature):	#
			try:
				qc_old_i.assign_parameters({PARAM[nf]: datai[nf]}, inplace = True)
				qc_old_j.assign_parameters({PARAM[nf]: dataj[nf]}, inplace = True)
			except: pass
		qc_new_test = circ_convert(matrix_feature[mf],1,theta_test_matrix[kt])
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
	if len(lastKernel) == 0:
		a,b,c = determine_sine_curve(kernel_test[0],kernel_test[1],kernel_test[2])
	else:
		a,b,c = determine_sine_curve(lastKernel[ID1][ID2],kernel_test[1],kernel_test[2])
	return [ID1,ID2,a,b,c]
					
def layerBuild(last_feat,qubit_use):
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
		t1 = time.time()
		component = 0
		l_mat_feat_i = len(matrix_feature[mf])
		for mfq in range(l_mat_feat_i):
			if matrix_feature[mf][mfq]!='111111':
				component=mfq
			
		AA=np.zeros((norm_Xtrain_l,norm_Xtrain_l))
		BB=np.zeros((norm_Xtrain_l,norm_Xtrain_l))
		CC=np.zeros((norm_Xtrain_l,norm_Xtrain_l))
		t2=time.time()
		print(f'initialize gate determination: t={t2-t1}',flush=True)
		if len(lastKernel)==0 and contin==False:
			AA,BB,CC=firstKernelTest(mf,AA,BB,CC)
			t3=time.time()
			print(f'determined AA,BB,CC, t={t3-t2}',flush=True)
		else:
			ABC=Parallel(n_jobs=n_jobs, verbose=1)\
				(delayed(layerParallel)(ID1,ID2,mf,AA,BB,CC) for ID1 in range(norm_Xtrain_l) for ID2 in range(ID1,norm_Xtrain_l))		#, verbose=1
			t3=time.time()
			print(f'determined ABC, t={t3-t2}',flush=True)
			l_ABC=len(ABC)
			for labc in range(l_ABC):
				AA[ABC[labc][0]][ABC[labc][1]]=ABC[labc][2]
				AA[ABC[labc][1]][ABC[labc][0]]=ABC[labc][2]
				BB[ABC[labc][0]][ABC[labc][1]]=ABC[labc][3]
				BB[ABC[labc][1]][ABC[labc][0]]=ABC[labc][3]
				CC[ABC[labc][0]][ABC[labc][1]]=ABC[labc][4]
				CC[ABC[labc][1]][ABC[labc][0]]=ABC[labc][4]
			t4=time.time()
			print(f'Fill AA,BB,CC, t={t4-t3}',flush=True)

		t5=time.time()
		print('Minimizing theta and cost',flush=True)
		Minimized = Parallel(n_jobs=2, verbose=1)(delayed(find_cost_angle)
			(AA,BB,CC,diff_X_feature,Y_sq,nf) for nf in range(n_feature))
		t6=time.time()
		print(f'Time for minimization: t={t6-t5}',flush=True)
		print('Determine gate with lowest cost',flush=True)
		for nf in range(n_feature):	
			'''
			Determine the cost function E=-1/T^2\sum y_i y_j k(x_i x_j \theta)
			'''
			thetaMin,globalMin = find_cost_angle(AA,BB,CC,diff_X_feature,Y_sq,nf)
			print(f'Gate {matrix_feature[mf]} Feature {nf} has cost {globalMin}')
			if globalMin<minCost:
				minCost = globalMin
				theta_list = [thetaMin]*n_qubit	#n_feature
				new_gate = []
				for qu in range(n_qubit):#n_feature):
					if matrix_feature[mf][qu] == '111111': new_gate.append('111111')
					else:
						gate,gate_id = matrix_feature[mf][qu].split('_')
						new_gate.append(f'{gate}_{nf}')
				minCost_layer = circ_convert(new_gate,1,theta_list)
				minCost_angle = thetaMin
				minCost_qubit = component
				minCost_feature = nf
				minCost_a = AA
				minCost_b = BB
				minCost_c = CC
				minCost_feat = matrix_feature[mf]
		t7=time.time()
		print(f'Time for sorting features: t={t7-t6}',flush=True)
	tend=time.time()
	print('time gate:',tend-t0)

	return [minCost,minCost_layer,minCost_angle,minCost_qubit,minCost_feature,minCost_a,minCost_b,minCost_c,minCost_feat]


def write_kernel(kernel_file,kernel,ID_kernel):
	'''
	write the current kernel to the end of a txt file
	'''
	outfile = open(kernel_file,"a")
	outfile.write(f'Kernel {ID_kernel}\n')
	for i in range(len(kernel)):
		for j in range(len(kernel[0])):
			outfile.write(f'{kernel[i][j]}\t')
		outfile.write("\n")
	outfile.close()

def write_gates(gate_file,newGate,angle,feature):
	'''
	write the current gate configuration to a txt file
	'''
	outfile = open(gate_file,"a")
	for i in range(len(newGate)):
		outfile.write(f'{newGate[i]}\t')
	outfile.write(f'{angle}\t')
	outfile.write(f'{feature}\n')
	outfile.close()

def write_cost(cost_file,cost,ID_gate):
	'''
	write the cost of the newly added gate to a txt file
	'''
	outfile = open(cost_file,"a")
	outfile.write(f'{ID_gate}\t{cost}\n')
	outfile.close()	





#Read in
print('Reading parameters',flush=True)
test_train,seed,repetition,layer,kernelFile,gateFile,costFile,n_jobs,gateList,contin = read_data()
normalized_Xtest,normalized_Xtrain,normalized_Xval,Y_test,Y_train,Y_val=test_train


oldGates = 'a'
if contin:
	oldGates = gateFile
	kernelFile = kernelFile+'_cc'
	gateFile = gateFile+'_cc'
	costFile = costFile+'_cc'
	print('new gate file:',gateFile,flush=True)
	print('new cost file:',costFile,flush=True)
else:
	print('gate file:',gateFile,flush=True)
	print('cost file:',costFile,flush=True)

if os.path.exists(kernelFile):
	print('File to write kernel to exists already')
	exit(9)
if os.path.exists(gateFile):
	print('File to write gate to exists already')
	exit(9)
if os.path.exists(costFile):
	print('File to write cost to exists already')
	exit(9)

print('Reading finished',flush=True)

#Define constants
n_feature = len(normalized_Xtrain[0])
n_qubit=n_feature #5
algorithm_globals.random_seed = seed
norm_Xtrain_l = len(normalized_Xtrain)
norm_Xval_l=len(normalized_Xval)
num_samples_classical = 1e3
rnd.seed(seed)
PARAM = ParameterVector('x', n_feature)
theta_test = [0,np.pi*0.5,-np.pi*0.5]	#angles to determine a,b,c in ARC
theta_test_matrix = [[],[],[]]	#for each qubit the three test angles, dim: n_feature * 3
for tt in range(len(theta_test)):
	for nf in range(n_qubit):
		theta_test_matrix[tt].append(theta_test[tt])
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
print('Determine Y_sq',flush=True)
Y_sq=np.full((len(Y_train),len(Y_train)),0)
#Train data
for yt1 in range(len(Y_train)):
	for yt2 in range(yt1,len(Y_train)):
		Y_sq[yt1][yt2]=Y_train[yt1]*Y_train[yt2]
		Y_sq[yt2][yt1]=Y_sq[yt1][yt2]
print('Determine diff_X_feature',flush=True)
diff_X_feature = []
minDif_feature = np.inf
maxDif_feature = -np.inf
for i in range(n_feature):
	diff_X_feature.append(np.full((norm_Xtrain_l,norm_Xtrain_l),0.000000000000))
	for xt1 in range(norm_Xtrain_l):
		for xt2 in range(xt1,norm_Xtrain_l):
			delta = normalized_Xtrain[xt1][i]-normalized_Xtrain[xt2][i]
			diff_X_feature[i][xt1][xt2] = delta
			diff_X_feature[i][xt2][xt1] = -1*delta
			if delta!=0:
				abs_delta = abs(delta)
				if abs_delta<minDif_feature:
					minDif_feature = abs_delta
				if abs_delta>maxDif_feature:
					maxDif_feature = abs_delta
print(minDif_feature,maxDif_feature,flush=True)
l_Y_sq=len(Y_sq)

print('Set up lists',flush=True)

#Run classification
matrix_feature = gate_pool()	#matrix with all possible gate combinations for one layer
qc = initializeMainCirc(contin,oldGates)
minCost = 1
minCost_feat=['0000']*n_feature
count=0
qubit_use = 0
cost_tot_list=[]
layer_list=[]
THETA=[]
AA = []
BB = []
CC = []
COMPONENT = []
FEATURE = []
sum_cost=0
sum_cost_list=[]

lastKernel=[]
while True:#count<20:
	print(f'Gates determined: {count}',flush=True)
	minCost,minCost_layer,minCost_angle,minCost_qubit,minCost_feature,minCost_a,minCost_b,minCost_c,minCost_feat = layerBuild(minCost_feat,qubit_use)
	qc.compose(minCost_layer,qubits=list(range(n_qubit)),inplace=True)	#n_feature

	Kernel = kernel(minCost_angle,minCost_a,minCost_b,minCost_c,diff_X_feature,Y_sq,minCost_feature)
	lastKernel=Kernel

	print(qc)
	print('Summary',count,minCost,minCost_angle,flush=True)
	count+=1

	write_kernel(kernelFile,Kernel,count)
	write_kernel(kernelFile+'_a',minCost_a,count)
	write_kernel(kernelFile+'_b',minCost_b,count)
	write_kernel(kernelFile+'_c',minCost_c,count)
	write_gates(gateFile,minCost_feat,minCost_angle,minCost_feature)
	write_cost(costFile,minCost,count)


	cost_tot_list.append(minCost)
	layer_list.append(count)

## Check if convergence (i.e., no substantial improvement in cost)
	if count > 1 and cost_tot_list[-2] - cost_tot_list[-1] < 1e-8:  #1e-3:
		print('Converged')
		break  # Converged!
