import argparse
import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy as sc
import random as rnd
from qiskit.circuit import Parameter, ParameterVector
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
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
	# alter b since rotosolve is implemented for sin and additive shift parameter b:
	b = -b + np.pi / 2
	return a, b, c
