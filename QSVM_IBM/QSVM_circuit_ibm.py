import numpy as np
from qiskit import QuantumCircuit, QuantumRegister



def gate_pool(gateList,n_feature):
	'''
	Provides the list of all allowed gates for the circuit
	Args:
		gateList: list of gates that will be used in the circuit
		n_feature: number of features
	Returns:
		A matrix with all gate combinations that can be added to the circuit.
	'''
	possible_gate = gateList#['Xn_x','Xm_x','Z_x','Zn_x','Zm_x','Rz_x','H_x','X_x','U1_x','T_x']#['Xn_x','Xm_x','Z_x','Zn_x','Zm_x','Rz_x','H_x','X_x','U1_x','Rx_x','Rxx_x','Ry_x','Ryy_x','Rzx_x','Rzz_x','S_x','Sdg_x','swap_x','Sx_x','Sxdg_x','Y_x']#['H','Z']#['H','X']#['X_x','H_x',]#
	matrix_feature = []
	for pg in range(len(possible_gate)):
		for nf in range(n_feature):
			vector_feature = np.full((1,n_feature),'111111')[0]
			vector_feature[nf] = possible_gate[pg]
			matrix_feature.append(vector_feature)
	return matrix_feature

def circ_convert(seq,m,p,n_feature,PARAM):
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
	Qubits = n_feature
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
				#additional gates
				elif desc == 'T':
					circuit.p(np.pi*0.25, qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Rx':
					circuit.rx(Rotation[j][i],qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Rxx':
					circuit.rxx(Rotation[j][i],qreg_q[j-1],qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Ry':
					circuit.ry(Rotation[j][i],qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Ryy':
					circuit.ryy(Rotation[j][i],qreg_q[j-1],qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Rzx':
					circuit.rzx(Rotation[j][i],qreg_q[j-1],qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Rzz':
					circuit.rzz(Rotation[j][i],qreg_q[j-1],qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
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


def circ_convertSinge(seq,m,p,n_feature,PARAM):
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
	Qubits = n_feature
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
				if ID == 'x' or ID == 'y':
					ID = 1
				ID = int(ID)
				if desc == 'U1':
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'U2':
					circuit.rx(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'U3':
					circuit.ry(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'P':
					circuit.p(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'CP':
					circuit.cp(Rotation[j][i]*PARAM[ID],qreg_q[j-1],qreg_q[j])
				elif desc == 'H':
					circuit.h(qreg_q[j])
				elif desc == 'X':
					circuit.cx(qreg_q[j-1], qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Xn':
					circuit.cx(qreg_q[j-2], qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Xm':
					circuit.cx(qreg_q[j-3], qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				
				elif desc == 'Rz':
					circuit.rz(Rotation[j][i],qreg_q[j])
				elif desc == 'Z':
					circuit.cz(qreg_q[j-1], qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Zn':
					circuit.cz(qreg_q[j-2], qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				elif desc == 'Zm':
					circuit.cz(qreg_q[j-3], qreg_q[j])
					circuit.rz(Rotation[j][i]*PARAM[ID],qreg_q[j])
				#additional gates
				elif desc == 'T':
					circuit.p(np.pi*0.25, qreg_q[j])
				elif desc == 'Rx':
					circuit.rx(Rotation[j][i],qreg_q[j])
				elif desc == 'Rxx':
					circuit.rxx(Rotation[j][i]*PARAM[ID],qreg_q[j-1],qreg_q[j])
				elif desc == 'Ry':
					circuit.ry(Rotation[j][i],qreg_q[j])
				elif desc == 'Ryy':
					circuit.ryy(Rotation[j][i]*PARAM[ID],qreg_q[j-1],qreg_q[j])
				elif desc == 'Rzx':
					circuit.rzx(Rotation[j][i]*PARAM[ID],qreg_q[j-1],qreg_q[j])
				elif desc == 'Rzz':
					circuit.rzz(Rotation[j][i]*PARAM[ID],qreg_q[j-1],qreg_q[j])
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


def initializeMainCirc(n_feature):
	'''
	Setup the main quantum circuit
	Returns:
		Quantum circuit.
	'''
	qreg_qc = QuantumRegister(n_feature, 'qc')
	qc = QuantumCircuit(qreg_qc)	#initialize quantum circuit
	return qc


