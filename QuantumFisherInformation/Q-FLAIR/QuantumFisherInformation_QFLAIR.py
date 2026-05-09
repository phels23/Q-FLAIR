import argparse
import numpy as np
from sklearn import preprocessing
import time
import os
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


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
	allData = [[],[],[],[],[],[]] #X_test,X_train,X_val,Y_test,Y_train,Y_val
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
	scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi),clip=True)			#normalize the data to the range 0 to 2PI
	allData[1] =scaler.fit_transform(allData[1])
	allData[0]=scaler.transform(allData[0])
	allData[2] =scaler.transform(allData[2])
	param_open.close()
	return [allData,seed,repetition,layers,kernelFile,gateFile,costFile,n_jobs,gateList,args.contin]


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

print('Reading parameters',flush=True)
test_train,seed,repetition,layer,kernelFile,gateFile,costFile,n_jobs,gateList,contin = read_data()
normalized_Xtest,normalized_Xtrain,normalized_Xval,Y_test,Y_train,Y_val=test_train
gate, angle, feature = read_circuit(gateFile)
n_feature = len(normalized_Xtrain[0])
n_qubit=7	#n_feature
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
print('Parameter-reading is finished',flush=True)

########################################################### QFLAIR
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

def predict_qc(gate,angle,PARAM,n_qubit):
	qc=initializeMainCirc()
	y_sign_gates=[]
	y_prob_pos_gates=[]
	Kernel_gate = []	
	for i in range(0,len(gate)):
		y_sign=[]
		y_prob_pos=0
		for j in range(len(gate[i])):
			if gate[i][j]!='111111':
				gTyp,fID = gate[i][j].split('_')
				gate[i][j] = f'{gTyp}_{feature[i]}'
		angle_list = [angle[i]]*n_qubit
		newGate = circ_convert(gate[i],1,angle_list)
		qc.compose(newGate,qubits=list(range(0,n_qubit)),inplace=True)
	return qc


n_feature = len(normalized_Xtrain[0])
n_qubit = 5
PARAM = ParameterVector('x', n_feature)
zzfm = predict_qc(gate,angle,PARAM,n_qubit)

print(zzfm)

print('Setup',flush=True)


import numpy as np
try:
    from qiskit_aer import AerSimulator
except ImportError:
    from qiskit.providers.aer import AerSimulator
from tqdm import tqdm
from joblib import Parallel, delayed


# --- Configuration ---
# We split data into small batches (e.g., 10 samples)
batch_size = 10
# Number of parallel jobs (-1 uses all available logical cores)
n_jobs_parallel = 2	#-1


# --- 1. Define Worker Function (Must be at top level) ---
def process_data_chunk(chunk_X, circuit_sv, epsilon):
    """
    Runs QFI calculation for a small chunk of data on a single CPU core.
    """
    # Create a local simulator instance for this process
    # max_parallel_experiments=1 prevents Aer from spinning up its own threads
    backend = AerSimulator(method='statevector', max_parallel_experiments=1)

    n_params = circuit_sv.num_parameters
    batch_binds = []

    # A. Prepare Binds for this chunk
    for x in chunk_X:
        # Base case
        base_bind = {k: [v] for k, v in zip(circuit_sv.parameters, x)}
        batch_binds.append(base_bind)

        # Shifted cases (Finite Difference)
        for i in range(n_params):
            x_shift = x.copy()
            x_shift[i] += epsilon
            shift_bind = {k: [v] for k, v in zip(circuit_sv.parameters, x_shift)}
            batch_binds.append(shift_bind)

    # B. Run Job (C++ Backend)
    batch_circuits = [circuit_sv] * len(batch_binds)
    job = backend.run(batch_circuits, parameter_binds=batch_binds)
    result = job.result()

    # C. Post-Process Gradients
    chunk_qfis = []
    stride = n_params + 1

    for i in range(0, len(batch_binds), stride):
        psi = result.get_statevector(i).data

        grads = []
        for j in range(n_params):
            psi_shift = result.get_statevector(i + 1 + j).data
            d_psi = (psi_shift - psi) / epsilon
            grads.append(d_psi)

        grads = np.array(grads)

        # QFI Algebra
        grads_conj = grads.conj()
        overlap_matrix = np.dot(grads, grads_conj.T)
        psi_grad_overlaps = np.dot(grads, psi.conj())
        term2 = np.outer(psi_grad_overlaps.conj(), psi_grad_overlaps)

        qfi_mat = 4 * np.real(overlap_matrix - term2)
        chunk_qfis.append(qfi_mat)

    return chunk_qfis

# --- 2. Main Execution Block ---
if __name__ == "__main__":
    start_time = time.time()

    # Setup Circuit
    zzfm_sv = zzfm.copy()
    zzfm_sv.remove_final_measurements()
    zzfm_sv.save_statevector()

    # Create chunks
    data_chunks = [
        normalized_Xtest[i: i + batch_size]
        for i in range(0, len(normalized_Xtest), batch_size)
    ]

    print(f"Split {len(normalized_Xtest)} samples into {len(data_chunks)} parallel tasks.",flush=True)

    # --- Run Parallel with TQDM ---
    # return_as="generator" returns a lazy iterator that yields results as they complete
    results_gen = Parallel(n_jobs=n_jobs_parallel, return_as="generator")(
        delayed(process_data_chunk)(chunk, zzfm_sv, 1e-4) for chunk in data_chunks
    )

    final_results_list = []

    # Iterate over the generator with tqdm
    for chunk_result in tqdm(results_gen, total=len(data_chunks), desc="Parallel Processing"):
        final_results_list.extend(chunk_result)

    # Combine
    final_qfi = np.array(final_results_list)

    # average over all samples
    mean_qfi = np.mean(final_qfi, axis=0)

    # store the matrices:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parameter", type=str)
    param_file = parser.parse_args().parameter
    base_name = os.path.splitext(os.path.basename(param_file))[0]
    np.savez_compressed(f"{base_name}_QFI.npz", qfi=final_qfi, mean_qfi=mean_qfi)

    print("\nDone.",flush=True)
    print("Final Shape:", final_qfi.shape,flush=True)

    eigvals = np.linalg.eigvalsh(mean_qfi)
    effective_dimension = 2*np.sum(eigvals / (n_feature/len(final_qfi) + eigvals))
    print(f"Effective Dimension: {effective_dimension:.4f}",flush=True)

    elapsed_time = time.time() - start_time
    print(f"Total time: {elapsed_time:.2f}s",flush=True)
    print(f"Time per sample: {elapsed_time / len(normalized_Xtest):.4f}s",flush=True)
