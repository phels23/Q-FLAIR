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
			args.measure,args.initialize,args.randomangle]
	else:
		return [allData,seed,repetition,layers,kernelFile,gateFile,costFile,n_jobs,gateList,accOut]



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

