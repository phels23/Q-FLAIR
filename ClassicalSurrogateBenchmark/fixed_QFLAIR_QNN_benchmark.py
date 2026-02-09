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
from sklearn.metrics import log_loss, balanced_accuracy_score
from qiskit import transpile
from qiskit_aer import AerSimulator
import itertools
from collections import defaultdict
import torch
from tqdm import tqdm
import math
import json


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
		if len(line_split) < 2:
			continue
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
		entries=line.split()
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


def generate_positive_half_space(unique_freqs):
	"""
	Generates ONLY the positive half-space of frequency combinations.
	Excludes the zero vector (origin).

	Args:
		unique_freqs (dict): {feature_idx: [freqs...]}

	Returns:
		torch.Tensor: Shape (N_half, D). The positive frequency vectors.
	"""
	# 1. Prepare sorted tensors
	full_vectors = [
		torch.tensor(sorted(freqs), dtype=torch.float64)
		for _, freqs in sorted(unique_freqs.items())
	]

	n_feats = len(full_vectors)
	chunks = []
	can_have_zero_prefix = True

	# 2. Iterate through dimensions to build disjoint slices
	for i, vec in enumerate(full_vectors):
		# If previous dimensions couldn't be zero, we can't continue the chain
		if not can_have_zero_prefix:
			break

		# Define parts:
		# Prefix: [0, 0, ...] (Fixed zeros for previous dims)
		prefix = [torch.tensor([0.0], dtype=torch.float64)] * i

		# Pivot: The current dimension must be strictly POSITIVE
		pivot = [vec[vec > 0]]

		# Suffix: All subsequent dimensions take ALL values
		suffix = full_vectors[i + 1:]

		# Generate slice ONLY if pivot has positive values
		if pivot[0].numel() > 0:
			slice_components = prefix + pivot + suffix
			chunk = torch.cartesian_prod(*slice_components)
			chunk = chunk.view(-1, n_feats)  # ensure 2D shape
			chunks.append(chunk)

		# Check if we can continue chaining zeros
		if not (vec == 0).any():
			can_have_zero_prefix = False

	# 3. Concatenate (Zero vector is naturally excluded by the pivot > 0 logic)
	if not chunks:
		return torch.empty((0, n_feats), dtype=torch.float64)

	return torch.cat(chunks)


def verify_omega_completeness(half_space_omega, unique_freqs):
	"""
	Sanity Check: Reconstructs the full set (Half U -Half U {0}) and
	compares it against the brute-force Cartesian product.
	"""
	print("Running Sanity Check...", end=" ", flush=True)

	# 1. Generate the Brute Force Full Grid
	full_vectors = [torch.tensor(sorted(freqs), dtype=torch.float64) for _, freqs in sorted(unique_freqs.items())]

	# WARNING: This might OOM if the grid is massive. Only run on small/test sets.
	expected_full = torch.cartesian_prod(*full_vectors)

	# 2. Reconstruct from Half Space
	half = half_space_omega
	neg_half = -1 * half_space_omega

	# Check if Zero Vector should exist (only if ALL input lists contain 0)
	has_zero = all(0 in freqs for freqs in unique_freqs.values())

	parts = [half, neg_half]
	if has_zero:
		zero_vec = torch.zeros((1, len(full_vectors)), dtype=torch.float64)
		parts.append(zero_vec)

	reconstructed = torch.cat(parts)

	# 3. Compare Sets
	# We sort both matrices lexicographically to compare row-by-row
	def sort_rows(matrix):
		# Simple lexsort for PyTorch
		if matrix.numel() == 0: return matrix
		# Convert to numpy for stable easy lexsort
		mat_np = matrix.numpy()
		# Sort by last column, then 2nd to last, etc.
		order = np.lexsort(mat_np.T[::-1])
		return torch.tensor(mat_np[order])

	expected_sorted = sort_rows(expected_full)
	reconstructed_sorted = sort_rows(reconstructed)

	# Check 1: Sizes
	if expected_sorted.shape != reconstructed_sorted.shape:
		raise ValueError(f"Shape Mismatch! Expected {expected_sorted.shape}, Got {reconstructed_sorted.shape}")

	# Check 2: Values
	if not torch.allclose(expected_sorted, reconstructed_sorted):
		raise ValueError("Value Mismatch! The union does not recover the original grid.")

	print("PASS. The logic is mathematically exact.")
	print(f"  - Full Size: {expected_sorted.shape[0]}")
	print(f"  - Half Size: {half.shape[0]}")
	print(f"  - Zero Vec:  {'Included' if has_zero else 'Not present in input'}")


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

### Configurations for classical surrogate model training ###
MAX_FREQS = 100_000_000
N_EPOCHS = 1000
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
FREQ_DECIMALS = 6


# Collect frequencies per feature:
eig_vals = defaultdict(list)
# angle holds the current frequencies (more precisely +- eig vals)
# feature holds the corresponding feature index
# qc is fully parameterized circuit
for a, f in zip(angle, feature):
	eig_vals[f].append({a/2,
						-a/2})  # eigenvalues in pairs (only two distinct eigenvalues per generator, which is 1/2 Pauli)

# test
#eig_vals = {list(eig_vals.keys())[0]: [{-1, 1}, {-1, 1}]}
#print(eig_vals)
selected_features = list(sorted(eig_vals.keys()))

unique_freqs = defaultdict(lambda: {0.0})  # start with zero freq for Minkowski sum
for f, spec_gate in eig_vals.items():
	for eigs in spec_gate:
		for eig_i in eigs:
			for eig_j in eigs:
				freq = eig_j - eig_i
				# minkowski sum with existing freqs:
				print(freq)
				unique_freqs[f] |= {np.round(freq + existing_freq, decimals=FREQ_DECIMALS)
									for existing_freq in unique_freqs[f]}
print("Number of unique features:", len(unique_freqs))
for f in unique_freqs:
	unique_freqs[f] = list(sorted(unique_freqs[f]))
	print(f'Feature {f} has {len(unique_freqs[f])} unique frequencies: {unique_freqs[f]}', flush=True)
print("Number of frequencies estimate:",
	  (n_freq_est := ((int(math.prod(list(map(len, unique_freqs.values())))) - 1) // 2)))
print("Number of Fourier coefficients estimate:", 2 * n_freq_est + 1)

with open(accOut, 'w') as acc_file:
	acc_file.write(f'num_unique_feats {len(unique_freqs)} '
				   f'num_freqs {n_freq_est} all_freqs {json.dumps(unique_freqs, separators=(",", ":"), sort_keys=True)}\n')

MAX_FREQS = 100_000_000
if n_freq_est > MAX_FREQS:
	raise SystemExit("Number of Fourier features too large, aborting to prevent memory issues.")

# Improved torch version: create Omega_torch (positive half-space) directly

Omega_torch = generate_positive_half_space(unique_freqs)
# Verify that Omega half space is correct (comment out for performance)
# verify_omega_completeness(Omega_torch, unique_freqs)

# Previous torch version to create full Omega (now replaced by generate_positive_half_space):
# freq_vectors = [torch.tensor(sorted(freqs), dtype=torch.float64) for _, freqs in sorted(unique_freqs.items())]
# Omega_torch = torch.cartesian_prod(*freq_vectors)

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
print(f"Number of frequencies confirmed: {Omega_torch.size(1)}", flush=True)
X_train_torch = torch.tensor(normalized_Xtrain[:, selected_features], dtype=torch.float64)
Y_train_torch = torch.tensor(Y_train, dtype=torch.float64).view(-1, 1)

X_val_torch = torch.tensor(normalized_Xval[:, selected_features], dtype=torch.float64)
Y_val_torch = torch.tensor(Y_val, dtype=torch.float64).view(-1, 1)


class FourierLinearModel(torch.nn.Module):
	def __init__(self, omega_matrix, out_features=1):
		super().__init__()
		self.register_buffer('Omega', omega_matrix)
		n_fourier_feats = 2 * omega_matrix.shape[1]

		# Linear layer with bias (as bias is no longer included in Fourier features due to positive half-space)
		self.linear = torch.nn.Linear(n_fourier_feats, out_features, bias=True, dtype=torch.float64)

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

#loss_fn = torch.nn.MSELoss(reduction='mean')
loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')  # log loss but numerically stable
# use stochastic gradient descent to learn fourier coeffs
optimizer = torch.optim.Adam(classical_surrogate_model.parameters(), lr=LEARNING_RATE)
# use minibatches
mem_size = max(min(BATCH_SIZE, MAX_FREQS // n_freq_est), 1)  # at least 1 element in mem chunk
print(f'Batch size: {BATCH_SIZE} Using memory chunk size: {mem_size}', flush=True)
# use dataset loader for batching
train_dataset = torch.utils.data.TensorDataset(X_train_torch, Y_train_torch)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(X_val_torch, Y_val_torch)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# after each epoch, evaluate on validation set and print loss + accuracy
# losses = []
for epoch in tqdm(range(N_EPOCHS)):
	classical_surrogate_model.train()  # switch to train mode
	for X_batch, Y_batch in train_loader:
		optimizer.zero_grad()

		# current_batch_loss = 0.0

		# split into chunks to prevent memory blow up:
		for X_chunk, Y_chunk in zip(torch.split(X_batch, mem_size), torch.split(Y_batch, mem_size)):
			Y_pred_chunk = classical_surrogate_model(X_chunk)
			loss_chunk = loss_fn(Y_pred_chunk, Y_chunk)
			loss_chunk = loss_chunk * (X_chunk.size(0) / X_batch.size(0))  # important, assumes mean loss reduction!
			loss_chunk.backward()  # accumulate gradients over chunks into model.weight.grad
			# current_batch_loss += loss_chunk.item()
		optimizer.step()
		# losses.append(current_batch_loss)
	if (epoch + 1) % 100 == 0:
		classical_surrogate_model.eval()  # switch to eval mode
		with torch.no_grad():
			# use batch loader for evaluation too:
			Y_val_pred = torch.cat([classical_surrogate_model(X_val_chunk)
									for X_val_batch, _ in val_loader
									for X_val_chunk in torch.split(X_val_batch, mem_size)],
								   dim=0)
			val_loss = loss_fn(Y_val_pred, Y_val_torch)
			val_acc = balanced_accuracy_score(Y_val, (Y_val_pred >= 0).int().numpy())  # expects logits, not probs
			Y_train_pred, Y_train_true = map(torch.cat, zip(*[(classical_surrogate_model(X_chunk), Y_chunk)
															  for X_b, Y_b in train_loader
															  for X_chunk, Y_chunk in zip(torch.split(X_b, mem_size),
																						  torch.split(Y_b, mem_size))]))
			train_loss = loss_fn(Y_train_pred, Y_train_true)
			train_acc = balanced_accuracy_score(Y_train_true.numpy(), (Y_train_pred >= 0).int().numpy())  # expects logits, not probs
		print(f'\nEpoch {epoch + 1}/{N_EPOCHS}, '
			  f'Training Loss: {train_loss.item():.4f}, Accuracy: {train_acc*100:.2f}%\n'
			  f'Epoch {epoch + 1}/{N_EPOCHS}, '
			  f'Validation Loss: {val_loss.item():.4f}, Accuracy: {val_acc*100:.2f}%\n',
			  flush=True)
		with open(accOut, 'a') as acc_file:
			acc_file.write(f'epoch {epoch + 1} train_loss {train_loss.item():.6f} train_acc {train_acc:.6f} '
						   f'val_loss {val_loss.item():.6f} val_acc {val_acc:.6f}\n')

