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
			costij = factor*Y_sq[i][j]*(AA[i][j]*np.cos(theta*diff_X_feature[component][i][j]-BB[i][j])+CC[i][j])
			Cost-= costij
			if i!=j:
				Cost-= costij
	return Cost

def find_cost_angle(AA,BB,CC,diff_X_feature,Y,component,num_samples_classical):
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
		costs = np.mean(Y.flatten() * kernel_mat_flat, axis=1)

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

def find_cost_angle_sym(AA,BB,CC,diff_X_feature,Y,component):
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
			
