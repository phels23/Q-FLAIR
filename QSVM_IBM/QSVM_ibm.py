"""
On the brief history of this file:
This file did not start as a copy of SVM_main_parallel.py but of QNN_logLoss_ibm.py (the IBM version of the QNN code).
This was then modified to implement QSVM instead of QNN.

Note that the experiments were run just before the switch of the IBM platform:
channel="ibm_quantum_platform" -> channel="ibm_cloud" in QiskitRuntimeService
"""


import os.path
import random as rnd
import pickle

import numpy as np
import scipy.linalg
from joblib import Parallel, delayed
from qiskit import generate_preset_pass_manager
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Statevector
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import gen_batches, shuffle
from sklearn.svm import SVC
from sklearn.kernel_approximation import Nystroem

from QSVM_circuit_ibm import gate_pool, circ_convertSinge, initializeMainCirc
from QSVM_cost_ibm import determine_sine_curve
from QSVM_in_out_ibm import read_data, write_gates, write_cost, write_model_outputs, Tee

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerOptions
from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime.options import DynamicalDecouplingOptions
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2 as FakeProvider
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_ibm_runtime.exceptions import IBMRuntimeError

from qiskit_aer import AerProvider

import mthree
import mthree.utils

import sys

def TA_loss(kernel_flat, YY):
	return - np.divide(np.sum(YY.flatten() * kernel_flat, axis=1),
					   np.sqrt(np.sum(kernel_flat**2,axis=1))) / np.sqrt(YY.size)

def constructCircuit_QSVM_IBM(Y, normalized_Xtrain, mf, matrix_feature, theta_test_matrix, n_feature, PARAM, opZ, count, norm_Xtrain_l):
	'''
	
	'''
	pub_qc = None
	pub_param_vecs = []
	global n_nystroem

	def layerParallel(ID1,ID2,mf):
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
		nonlocal pub_qc, pub_param_vecs

		datai = np.array(normalized_Xtrain[ID1])
		dataj = np.array(normalized_Xtrain[ID2])

		p = Parameter('angle')  # Parameter for the angle of the gate we want to sample from for reconstruction
		p_rep = [p] * n_qubit  # replicate along the number of qubits but it will only be applied once
		qc_new_test = circ_convertSinge(matrix_feature[mf], 1, p_rep, n_qubit, [1.0] * len(PARAM))

		qc_old_i = qc.copy()
		global PARAM_i  #PARAM_i = ParameterVector('xi', n_feature)
		qc_old_i.assign_parameters({p: pi for p, pi in zip(PARAM, PARAM_i) if p in qc_old_i.parameters}, inplace=True)

		qc_old_j = qc.copy()
		global PARAM_j
		qc_old_j.assign_parameters({p: pj for p, pj in zip(PARAM, PARAM_j) if p in qc_old_j.parameters}, inplace=True)

		# Use distinct set of parameters for each data point
		qc_kernel_i_j = qc_old_i & qc_new_test & qc_old_j.inverse()

		# Set the quantum circuit (is not necessarily None at the beginning but must always be the same circuit)
		pub_qc = qc_kernel_i_j.copy()

		for kt in range(3):	#iterate over the three test angles len(kernel_test)
			'''
			Calculate the kernel matrix entries explicitly for theta=0,PI/2,-PI/2
			'''

			# Final data mapping for set features
			data_param_mapping = ({PARAM_i[nf]: datai[nf] for nf in range(n_feature)} |
								  {PARAM_j[nf]: dataj[nf] for nf in range(n_feature)})
			# set the sampling positions
			data_param_mapping["angle"] = theta_test_matrix[kt][0]

			# Set the parameters for the current data point and one of the shifts
			pub_param_vecs.append(data_param_mapping)

		return [ID1,0,0,0]

	for nf in range(n_qubit):
		if '_' in matrix_feature[mf][nf]:
			mf_split = matrix_feature[mf][nf].split('_')[1]
			if mf_split == 'y':
				raise ValueError("QSVM construction only uses data dependent gates!")
			
	# the following code used to be for data dependent gates only:

	if n_nystroem > 0:  # Use Nystroem approximation
		train_pairs = ID_pairs(norm_Xtrain_l, n_nystroem, exclude_diag=True)  # skip diagonal for training kernels
	else:
		train_pairs = ID_pairs(norm_Xtrain_l, exclude_diag=True)  # skip diagonal for training kernels

	if count==0:  # First layer, so no data dependence yet
		# call the layerParallel function (for a fixed data point pair 0,0, which is not used, and removed in prepare_submission)
		layerParallel(0, 0, mf)
	else:
		for ID1, ID2 in train_pairs:  # Only unique pairs (triangular matrix)
			layerParallel(ID1, ID2, mf)

	pub_qc = remove_idle_qwires(pub_qc)
	print(pub_qc, len(pub_qc.parameters))
	print("Num parameter configs:", len(pub_param_vecs))
		
	return pub_qc, pub_param_vecs


def evaluateCircuit_QSVM_IBM(Y, normalized_Xtrain, mf, matrix_feature, theta_test_matrix, n_feature, PARAM, opZ, count,
							 norm_Xtrain_l, evs, return_reconstruction_only=False):
	'''

	'''
	def layerParallel(ID_flat, mf, AA, BB, CC):
		'''
		Determine a,b,c for all but the first gate added to the circuit in parallel
		Args:
			ID_flat: data point iteration 1
			ID2: data point iteration 2
			mf: gate iteration parameter
			AA: amplitude parameter matrix
			BB: shift parameter matrix
			CC:	offset parameter matrix
		Returns:
			Parameters of the cosine functions.
		'''
		nonlocal evs

		expect_test = [evs[3 * ID_flat], evs[3 * ID_flat + 1], evs[3 * ID_flat + 2]]
		'''
		Determine from the three explicit kernels the values for a, b, c in the cos() function
		'''
		a, b, c = determine_sine_curve(expect_test[0], expect_test[1], expect_test[2])
		return [ID_flat, a, b, c]

	try:
		global n_nystroem
		n_nystroem
	except NameError:
		n_nystroem = 0

	if n_nystroem > 0:  # Use Nystroem approximation
		norm_Xtrain_l2 = n_nystroem
		train_pairs = ID_pairs(norm_Xtrain_l, n_nystroem, exclude_diag=True)  # skip diagonal for training kernels
	else:
		norm_Xtrain_l2 = norm_Xtrain_l
		train_pairs = ID_pairs(norm_Xtrain_l, exclude_diag=True)  # skip diagonal for training kernels

	# Use nan init for arrays to sanity check later
	AA = np.full((norm_Xtrain_l, norm_Xtrain_l2), np.nan)
	BB = np.full((norm_Xtrain_l, norm_Xtrain_l2), np.nan)
	CC = np.full((norm_Xtrain_l, norm_Xtrain_l2), np.nan)

	XX = normalized_Xtrain[:, np.newaxis, :] - normalized_Xtrain[np.newaxis, :, :]
	XX = np.transpose(XX, axes=(2, 0, 1))
	triu_idx = np.tril_indices(norm_Xtrain_l, k=-1)
	XX[:, triu_idx[0], triu_idx[1]] *= -1
	XX = XX[:, :norm_Xtrain_l, :norm_Xtrain_l2]

	YY = np.outer(np.asarray(Y), np.asarray(Y))

	# the following code used to be for data dependent gates only:
	if count == 0:
		assert len(evs) == 3
		'''
		Determine from the three explicit kernels the values for a, b, c in the cos() function
		'''
		a, b, c = determine_sine_curve(evs[0], evs[1], evs[2])
		AA[:] = a
		BB[:] = b
		CC[:] = c
	else:
		ABC =[layerParallel(ID_flat, mf, AA, BB, CC)
			 for ID_flat in range(len(train_pairs))]
		for ID_flat, a, b, c in ABC: # unpack them
			i, j = train_pairs[ID_flat]
			AA[i, j] = a
			BB[i, j] = b
			CC[i, j] = c
			# Fill the lower triangle of the matrix with the same values (symmetry)
			if (j, i) not in train_pairs and j < norm_Xtrain_l and i < norm_Xtrain_l2:
				AA[j, i] = a
				BB[j, i] = b
				CC[j, i] = c

	# Post process the diagonal (fill with ones):
	for ID in range(min(norm_Xtrain_l, norm_Xtrain_l2)):
		AA[ID, ID] = 0.0		# no amplitude
		BB[ID, ID] = np.pi/2	# shift is arbitrary, but pi/2 is the result of determine_sine_curve(1, 1, 1)
		CC[ID, ID] = 1.0		# offset is 1.0

	assert not np.any(np.isnan(AA)), "AA contains NaN values!"
	assert not np.any(np.isnan(BB)), "BB contains NaN values!"
	assert not np.any(np.isnan(CC)), "CC contains NaN values!"
	assert scipy.linalg.issymmetric(AA[:min(norm_Xtrain_l, norm_Xtrain_l2), :min(norm_Xtrain_l, norm_Xtrain_l2)]), "AA is not symmetric!"
	assert scipy.linalg.issymmetric(BB[:min(norm_Xtrain_l, norm_Xtrain_l2), :min(norm_Xtrain_l, norm_Xtrain_l2)]), "BB is not symmetric!"
	assert scipy.linalg.issymmetric(CC[:min(norm_Xtrain_l, norm_Xtrain_l2), :min(norm_Xtrain_l, norm_Xtrain_l2)]), "CC is not symmetric!"

	if return_reconstruction_only:
		return AA, BB, CC, XX, YY

	opt_theta_feature = []
	opt_cost_feature = []
	opt_kernel_feature = []

	opts = Parallel(n_jobs=n_jobs)(
		delayed(find_cost_rec_QNN)(AA, BB, CC, XX, YY, nf, num_samples_classical)
		for nf in range(n_feature))

	for nf, (opt_theta, opt_cost, opt_kernel) in enumerate(opts):
		if matrix_feature is not None:
			print(f'Gate {matrix_feature[mf][active_qubits]} Feature {nf} Angle {round(opt_theta, 5)} has cost {round(opt_cost, 5)}')

		opt_theta_feature.append(opt_theta)
		opt_cost_feature.append(opt_cost)
		opt_kernel_feature.append(opt_kernel)

	return [opt_theta_feature, opt_cost_feature, opt_kernel_feature]

	
def find_cost_rec_QNN(AA,BB,CC,XX,YY,component,num_samples_classical):
	'''
	Minimizes the cost function
	Args:
		AA: amplitude parameter list
		BB: shift parameter list
		CC:	offset parameter list
		XX: list with all training feature differences
		YY: list with all training label products
		component: component of the data points
	Returns:
		The angle of the minimal cost, the value of the minimal cost and the point where the optimization started.
	'''
	global n_nystroem
	def cost_rec(theta):
		"""
		Efficient numpy implementation of the cost function (TA).
		Relies on broadcasting and can handle multiple values for theta at once.

		Args:
			theta: weight parameter of the new gate. Can be a single value or a numpy array.
		Returns:
			Value of the cost function (TA)
		"""

		theta = np.asarray(theta)
		theta = theta.reshape(-1, 1)
		kernel_mat_flat = AA.flatten()*np.cos(theta * XX[component].flatten() - BB.flatten()) + CC.flatten()

		if n_nystroem > 0:  # Use Nystroem approximation
			kernel_approx_flat = []
			for kernel_matrix in kernel_mat_flat.reshape(-1, *AA.shape):  # kernel_matrix shape (n_samples, n_nystroem)
				nystroem = Nystroem('precomputed', n_components=n_nystroem)
				basis_kernel_matrix = kernel_matrix[:n_nystroem, :n_nystroem]  # shape (n_nystroem, n_nystroem)
				nystroem.fit(basis_kernel_matrix)
				kernel_matrix_transf = nystroem.transform(kernel_matrix)
				full_kernel_approx = kernel_matrix_transf @ kernel_matrix_transf.T
				kernel_approx_flat.append(full_kernel_approx.flatten()) # full shape (n_samples, n_samples)
			kernel_approx_flat = np.clip(kernel_approx_flat, 0.0, 1.0)
			cost = TA_loss(np.asarray(kernel_approx_flat), YY.flatten())
		else:
			kernel_mat_flat = np.clip(kernel_mat_flat, 0.0, 1.0)
			cost = TA_loss(kernel_mat_flat, YY.flatten())
		
		assert cost.size == theta.size
		return [cost.item() if theta.size == 1 else cost, kernel_mat_flat]

	num_xs = int(num_samples_classical)
	if num_xs%2==0:
		num_xs+=1
	xs = np.linspace(-1, 1, num_xs)
	ys,kernel_mat_flat = cost_rec(xs)
	min_idx = np.argmin(ys)
	opt_param = xs[min_idx]
	opt_y = np.min(ys)
	opt_kernel = np.asarray(kernel_mat_flat[min_idx]).reshape(AA.shape)

	return opt_param, opt_y, opt_kernel  # Invert sign to get cost


def remove_idle_qwires(circ):
	dag = circuit_to_dag(circ)

	idle_wires = list(dag.idle_wires())
	dag.remove_qubits(*idle_wires)

	return dag_to_circuit(dag)


def prepare_submission(pubs, backend=None):
	'''
	Prepare the PUBs for submission to the IBM backend.
	Args:
		pubs: List of PUBs to be submitted.
	Returns:
		List of PUBs ready for submission.
	'''
	global USE_SAMPLER

	prepared_pubs = []
	for pub in pubs:
		pub_c, pub_o, pub_ps = pub

		# Transpilation:
		if backend is not None:
			pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend)
			pub_c = pass_manager.run(pub_c)  # Transpile the circuit
			if not USE_SAMPLER:
				pub_o = pub_o.apply_layout(pub_c.layout)  # Apply the layout to the observable

		flattened_pub_ps = []
		for pub_p in pub_ps:

			if len(pub_p) == 0:
				# No parameters to assign, just the circuit and operator
				flattened_pub_p = None
			else:
				# Flatten the parameter vectors for the PUB:
				assert isinstance(pub_p, dict)
				n_params = len(pub_c.parameters)
				pub_p = {(k if isinstance(k, str) else k.name): v for k, v in pub_p.items()}  # Ensure keys are strings
				flat_p_vec = ParameterVector('params', n_params)
				flat_p_vec_map = {}
				flattened_pub_p = np.full(n_params, fill_value=np.nan)
				for idx, p in enumerate(pub_c.parameters):
					k = p.name
					flattened_pub_p[idx] = pub_p[k]
					flat_p_vec_map[k] = flat_p_vec[idx]
				assert np.all(~np.isnan(flattened_pub_p)), "All parameter values should be set throughout flattening"
			flattened_pub_ps.append(flattened_pub_p)

		prepared_pubs.append((pub_c, pub_o, flattened_pub_ps))

	if USE_SAMPLER:  # remove observable
		# also adds measurements if needed
		prepared_pubs = [((pub_c.measure_active(inplace=False) if len(pub_c.cregs) == 0 else pub_c), pub_ps)
						 for pub_c, _, pub_ps in prepared_pubs]

	return prepared_pubs


service = None
def load_backend(backend_key):
	if backend_key == "simulator":  # simulator backend
		# No service needed, use Aer simulator
		backend = AerProvider().get_backend("aer_simulator")
		backend.set_options(shots=SHOTS, method="statevector")
	else:  # service needed
		global service  # Load service only once
		if service is None:
			service = QiskitRuntimeService(channel='ibm_quantum_platform',
										   instance='pinq-quebec-hub/university-of-br/roman-krems')
		match backend_key:
			case _ if backend_key.startswith("fake_"):  # Note that this is also a simulated backend
				backend = FakeProvider().backend(backend_key)
				backend.refresh(service)  # Refresh the backend to get the latest data
			case "least_busy":
				# Get the least busy backend from the service
				# Use Eagle architecture as of now
				backend = service.least_busy(filters=lambda b: b.n_qubits <= 127)
			case _ if backend_key.startswith("ibm_"):
				backend = service.backend(backend_key)

			case _:
				raise ValueError(f"Unsupported backend: {backend_key}. Please use 'exact', 'shot_noise', 'fake_[backend name]', "
								 f"'least_busy', ibm_[backend name]'.")
		backend.set_options(shots=SHOTS)

	assert backend is not None, "Backend could not be initialized!"

	return backend

def mask_sub_pool(gate_pool, qubit_subset):
	if isinstance(qubit_subset, int):
		qubit_rest = slice(qubit_subset, None)
	else:
		qubit_rest = sorted(set(range(len(gate_pool[0]))) - set(qubit_subset))
	gate_pool = np.asarray(gate_pool)
	rest_pool = gate_pool[:, qubit_rest]
	identities = (rest_pool == "111111")
	active_mask = np.all(identities, axis=1)
	return active_mask

def remove_cyclic_gates(gate_pool, active_pool, active_qubits):
	for idx, (is_active, gate) in enumerate(zip(active_pool, gate_pool)):
		if not is_active:
			continue
		if set(gate[min(active_qubits)]) != {'1'}:  # Check if gate is applied on lowest active qubit
	 		active_pool[idx] = False

	return active_pool

def submit_jobs(pubs, filename_prefix, sv_pubs=None):
	global backend, USE_SAMPLER, SHOTS, costFile, sv_sampler, sv_estimator, sampler, sv_estimator, session

	if sv_pubs is None:
		sv_pubs = pubs

	# Save the PUBs to a pickle file for debugging or later use
	pubs_orig_file_path = os.path.join(os.path.dirname(costFile), "ibm_results", f"{filename_prefix}_pubs_orig.pickle")
	pubs_file_path = os.path.join(os.path.dirname(costFile), "ibm_results", f"{filename_prefix}_pubs.pickle")
	os.makedirs(os.path.dirname(pubs_file_path), exist_ok=True)  # ensure directory exists
	with open(pubs_orig_file_path, "wb") as file:
		pickle.dump(pubs_orig, file, protocol=pickle.HIGHEST_PROTOCOL)
	with open(pubs_file_path, "wb") as file:
		pickle.dump(pubs, file, protocol=pickle.HIGHEST_PROTOCOL)

	# Run the exact simulation using the StatevectorEstimator (and store for reference)
	if USE_SAMPLER:
		sv_job = sv_sampler.run(sv_pubs)
	else:
		sv_job = sv_estimator.run(sv_pubs)
	sv_result = sv_job.result()
	# Store raw PUBs results in a JSON file
	sv_res_file_path = os.path.join(os.path.dirname(costFile), "ibm_results", f"sv_{filename_prefix}_{sv_job.job_id()}.json")
	os.makedirs(os.path.dirname(sv_res_file_path), exist_ok=True)  # ensure directory exists
	#with open(sv_res_file_path, "w") as file:  # store
	#	json.dump(sv_result, file, cls=RuntimeEncoder)

	# Run the estimator with the list of PUBs
	multiple_jobs = False
	if sum(len(pub[-1]) for pub in pubs) * SHOTS > 10_000_000:  # Limit the number of shots per job to 10 million
		print("Multiple jobs will be used to avoid exceeding the 10 million shots limit.")
		multiple_jobs = True
		# Split the PUBs into smaller batches
		pubs_list = [[]]
		current_shots = 0
		for pub in pubs:
			pub_shots = len(pub[-1]) * SHOTS
			if current_shots + pub_shots > 10_000_000:
				# Start a new batch if the current one exceeds the limit
				pubs_list.append([])
				current_shots = 0
			current_shots += pub_shots
			pubs_list[-1].append(pub)
		print(f"Split {sum(len(pub[-1]) for pub in pubs) * SHOTS} shots into {len(pubs_list)} jobs")

	if SHOTS is not None:
		for _ in range(10):  # Wait for the job to complete, retrying on failure (10 times in total)
			if USE_SAMPLER:

				try:
					if multiple_jobs:
						jobs = [sampler.run(pubs) for pubs in pubs_list]
					else:
						job = sampler.run(pubs)
				except IBMRuntimeError as e:
					if session.status() is None or session.status() in ['Closed', 'None']:
						session = Session(backend)  # Open a new session if the current one is closed
						print(f"Old session closed, opened new session ({session.session_id})")
						sampler = init_sampler()  # Reinitialize the sampler with a new session
						if multiple_jobs:
							jobs = [sampler.run(pubs) for pubs in pubs_list]
						else:
							job = sampler.run(pubs)
						# Also re-calibrate the mitigation:
						calibrate_mit()
					else:
						raise e
			else:
				job = estimator.run(pubs)

			if multiple_jobs:
				for p, job in zip(pubs_list, jobs):
					print(f"Iteration {filename_prefix}: Submitted job {job.job_id()} with {len(p)} PUBs")
			else:
				print(f"Iteration {filename_prefix}: Submitted job {job.job_id()} with {len(pubs)} PUBs")

			try:
				if multiple_jobs:
					# Wait for all jobs to complete
					pubs_result = []
					for job in jobs:
						pubs_result.extend(job.result())
				else:
					# Wait for the single job to complete
					pubs_result = job.result()  # This will block until the job is done
				break
			except Exception as e:
				print(f"Error while waiting for job {job.job_id()} to complete an error occured: {e}")
				print(job.error_message())
				print(f"Resubmitting")

		if not job.done():
			raise RuntimeError(f"Job {job.job_id()} did not complete successfully: {job.error_message()}")
		print(f"Iteration {filename_prefix}: Job {job.job_id()} completed successfully in {job.metrics()['usage']}")
	else:
		print("Running in exact simulation mode!")
		job, pubs_result = sv_job, sv_result  # Use the statevector result directly

	# Store raw PUBs results in a JSON file
	pubs_res_file_path = os.path.join(os.path.dirname(costFile), "ibm_results", f"{filename_prefix}_{job.job_id()}.json")
	os.makedirs(os.path.dirname(pubs_res_file_path), exist_ok=True)  # ensure directory exists
	#with open(pubs_res_file_path, "w") as file:  # store
	#	json.dump(pubs_result, file, cls=RuntimeEncoder)

	return pubs_result, sv_result


def model_outputs_from_pub_results(pub_res, mitigation=None, mitigation_mapping=None, true_labels=None):
	if USE_SAMPLER:
		# Extract probabilities from the raw sampler results (for reference, not used later):
		raw_evs = []
		for m in pub_res.data.meas:
			counts = m.get_int_counts().get(0, 0.0)  # All zero counts (first .get)
			raw_ev = counts / m.num_shots  # normalize counts
			raw_evs.append(raw_ev)
		raw_evs = np.array(raw_evs)
		# Extract probabilities via the M3 mitigation (used afterward for the reconstructions):
		if mitigation is None:  # skip mitigation if not provided
			evs = raw_evs
		else:
			evs = []
			for m in pub_res.data.meas:
				raw_counts_dict = m.get_counts()  # mthree only compatible with bitstring key counts dictionary
				quasis = mitigation.apply_correction(raw_counts_dict, mitigation_mapping)
				probs = quasis.nearest_probability_distribution()
				zero_bitstring = '0' * len(list(probs.keys())[0])
				ev = probs.get(zero_bitstring, 0.0)  # gets the probability of the all-zero state, if not available, returns 0.0
				evs.append(ev)
			evs = np.array(evs)
	else:
		evs = np.clip((pub_res.data.evs+1)/2, min=0.0, max=1.0)
		raw_evs = evs

	if true_labels is not None:
		acc = np.nan
		#acc = balanced_accuracy_score((true_labels + 1)/2, np.round(evs))  # Convert to [0, 1] range for accuracy
		return evs, raw_evs, acc
	else:
		return evs, raw_evs

def init_sampler():
	global session, SHOTS
	new_sampler = Sampler(mode=session,
						  options=SamplerOptions(default_shots=SHOTS,
												 dynamical_decoupling=DynamicalDecouplingOptions(enable=True)))
	return new_sampler

def calibrate_mit():
	global backend, transpiled_bit_mappings, session, mit, sampler
	for _ in range(10):  # Retry calibration up to 10 times
		try:
			try:
				mit = mthree.M3Mitigation(backend)  # resets the calibration
				mit.cals_from_system(transpiled_bit_mappings, runtime_mode=session)  # <- calibration happens here
			except IBMRuntimeError as e:
				if session.status() is None or session.status() in ['Closed', 'None']:
					session = Session(backend)  # Open a new session if the current one is closed
					print(f"Old session closed, opened new session ({session.session_id})")
					sampler = init_sampler()  # Reinitialize the sampler with a new session
					mit = mthree.M3Mitigation(backend)  # resets the calibration
					mit.cals_from_system(transpiled_bit_mappings, runtime_mode=session)  # <- calibration happens here
				else:
					raise e
			break
		except Exception as e:
			print(f"Error while waiting for calibration to complete an error occured: {e}")
			print(f"Resubmitting")

def ID_pairs(len_1, len_2=None, exclude_diag=None):
	'''
	Generate all unique pairs of IDs from two lists of given lengths.
	Args:
		len_1: Length of the first list.
		len_2: Length of the second list (optional, defaults to len_1).
	Returns:
		List of tuples containing all unique pairs of IDs.
	'''
	if len_2 is None:
		len_2 = len_1
	if exclude_diag is None:  # Exclude diagonal pairs (i, i) if same length by default
		exclude_diag = (len_1 == len_2)  # Include diagonal pairs if lengths are different by default
	ID_pairs = []
	for i in range(len_1):
		for j in range(len_2):
			if not (exclude_diag and i == j):
				if (j, i) not in ID_pairs:  # Avoid duplicates (i, j) and (j, i)
					ID_pairs.append((i, j))
	return ID_pairs

def ID_all_pairs(len_1, len_2=None):
	'''
	Generate all pairs of IDs from two lists of given lengths.
	Args:
		len_1: Length of the first list.
		len_2: Length of the second list (optional, defaults to len_1).
	Returns:
		List of tuples containing all pairs of IDs.
	'''
	if len_2 is None:
		len_2 = len_1
	ID_pairs = []
	for i in range(len_1):
		for j in range(len_2):
			ID_pairs.append((i, j))
	return ID_pairs

def kernel_mat_from_flat(vals, len_1, len_2, pairs):
	assert len(vals) == len_1 * len_2, "Length of values does not match the expected size of the kernel matrix!"
	kernel_mat = np.full((len_1, len_2), np.nan)  # Initialize with NaN to check for unassigned values later

	for (i, j), val in zip(pairs, vals):
		kernel_mat[i, j] = val

	assert not np.any(np.isnan(vals)), "Values contain NaN values!"
	return kernel_mat

if __name__ == "__main__":

	#Read in
	test_train,seed,repetition,layer,kernelFile,gateFile,costFile,n_jobs,gateList,accOut,batch_size,SHOTS,BACKEND,USE_SAMPLER,n_nystroem = read_data()

	# Set up logging (forward all prints into file and console)
	log_file_path = os.path.join(os.path.dirname(costFile), "output.log")
	os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # ensure directory exists
	log_file = open(log_file_path, 'w')
	tee = Tee(sys.__stdout__, log_file)
	sys.stdout = tee
	sys.stderr = tee

	trainModelOutFile = os.path.join(os.path.dirname(costFile), "TrainModelOut.txt")
	testModelOutFile = os.path.join(os.path.dirname(costFile), "TestModelOut.txt")
	svTestModelOutFile = os.path.join(os.path.dirname(costFile), "SVTestModelOut.txt")

	normalized_Xtest,normalized_Xtrain,normalized_Xval,Y_test,Y_train,Y_val=test_train

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
	n_qubit=30
	active_qubits = [n_qubit // 2 - 1, n_qubit // 2] # init with 2 center qubits, will be adjusted/extended iteratively

	norm_Xtrain_l = len(normalized_Xtrain)
	norm_Xval_l=len(normalized_Xval)
	num_samples_classical = 1e2

	rnd.seed(seed)
	PARAM = ParameterVector('x', n_feature)
	PARAM_i = ParameterVector('xi', n_feature)  # first data point in kernel circuit
	PARAM_j = ParameterVector('xj', n_feature)  # second data point in kernel circuit
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


	matrix_feature = gate_pool(gateList,n_qubit)	#matrix with all possible gate combinations for one layer
	active_pool = mask_sub_pool(matrix_feature, active_qubits)  # Keep track of active gates in the pool
	qc = initializeMainCirc(n_qubit)

	backend = load_backend(BACKEND)  # Load the backend based on the configuration

	if USE_SAMPLER:
		default_shots = SHOTS if SHOTS is not None else 8192  # Default shots for the sampler
		# StatevectorSampler requires shots to be set, default is 8192
		sv_sampler = StatevectorSampler(default_shots=default_shots)  # For exact simulation, use StatevectorSampler
		sampler = None  # To be initialized in session
		mit = mthree.M3Mitigation(backend)  # Use M3 mitigation sampler
	else:
		sv_estimator = StatevectorEstimator()  # For exact simulation, use StatevectorEstimator
		estimator = None  # To be initialized in session
		mit = None  # no explicit error mitigation used

	# Each iteration of the algorithm will be submitted as a list of pubs because they can all be run in parallel

	# Filter gate pool to only include gates that can have an effect in the first iteration:
	# Filter out data independent gates
	active_pool = [is_active and not any('_y' in g for g in gate) for is_active, gate in zip(active_pool, matrix_feature)]  # Keep only active gates
	# Filter symmetric  applied to the second qubit (use set("111") == {'1'} to identify)
	active_pool = remove_cyclic_gates(matrix_feature, active_pool, active_qubits)  # Remove cyclic gates that are not applied to the lowest active qubit

	session = None  # IBM runtime session, to be initialized/started later
	transpiled_bit_mappings = None # Bit mappings for the transpiled circuits, to be initialized later
	count=0
	cost_tot_list = []
	layer_list=[]
	opZ = SparsePauliOp("Z"+"I"*(n_qubit-1))  # Not used anymore! (from old QNN code)
	batch_size = min(batch_size, norm_Xtrain_l)  # Ensure batch size does not exceed training data length
	mini_batch_slices = iter([])
	orig_normalized_Xtrain = normalized_Xtrain.copy()  # Keep original training data for batching
	orig_Y_train = Y_train.copy()  # Keep original training labels for batching
	orig_norm_Xtrain_l = norm_Xtrain_l  # Keep original length of training data for batching

	shuffled_normalized_Xtrain = normalized_Xtrain.copy()  # Keep original training data for batching
	shuffled_Y_train = Y_train.copy()

	with Session(backend=backend) as session:  # Use a session to submit jobs

		while True:

			if count > 0:  # Update gate pool in case it was filtered in the first iteration
				active_pool = mask_sub_pool(matrix_feature, active_qubits)  # Update active pool based on the current number of qubits
				active_pool = remove_cyclic_gates(matrix_feature, active_pool, active_qubits)  # Remove cyclic gates that are not applied to the lowest active qubit

			try:
				mini_batch_slice = next(mini_batch_slices)
			except StopIteration:
				print(count, "New epoch")
				mini_batch_slices = list(gen_batches(len(normalized_Xtrain), batch_size))
				if mini_batch_slices[-1].stop - mini_batch_slices[-1].start < batch_size:
					mini_batch_slices = mini_batch_slices[:-1]  # drop last incomplete batch if it exists
				mini_batch_slices = iter(mini_batch_slices)
				mini_batch_slice = next(mini_batch_slices)
				shuffled_normalized_Xtrain, shuffled_Y_train = shuffle(orig_normalized_Xtrain, orig_Y_train, random_state=count)

			normalized_Xtrain = shuffled_normalized_Xtrain[mini_batch_slice]
			Y_train = shuffled_Y_train[mini_batch_slice]
			norm_Xtrain_l = batch_size
			# Due to shuffling, we can always take the first nNystroem training samples for the Nystroem approximation
			if n_nystroem > 0:  # However, check if samples from both classes are present among nNystroem
				labels_present = set(Y_train[:n_nystroem].astype(int))
				if len(labels_present) < 2:  # If not both classes are present, move samples around
					missing_label = list({-1, 1} - labels_present)[0]
					missing_mask = np.isclose(Y_train, missing_label)
					if np.any(missing_mask):  # only if such label present in entire batch!  (likely)
						missing_idx = np.where(missing_mask)[0][0]
						Y_train[[0, missing_idx]] = Y_train[[missing_idx, 0]]
						normalized_Xtrain[[0, missing_idx]] = normalized_Xtrain[[missing_idx, 0]]


			# Store the minibatch as numpy npz archive
			mini_batch_path = os.path.join(os.path.dirname(costFile), "train_minibatches", f"{count:03d}.npz")
			os.makedirs(os.path.dirname(mini_batch_path), exist_ok=True)  # ensure directory exists
			np.savez_compressed(mini_batch_path, normalized_Xtrain=normalized_Xtrain, Y_train=Y_train)

			n_feature_coll = []
			new_gate_coll = []
			new_gate_idx_coll = []
			opt_theta_coll = []
			opt_cost_coll = []
			opt_kernel_coll = []
			pubs = []  # List of PUBs for this iteration
			for i in range(len(matrix_feature)):  # Loop over gate pool to construct corresponding IBM PUBs
				# uncomment for explicit calculation without reconstruction
				#exp_cost_collect = []
				#for j in range(n_feature):
				#	opt_theta_exp, opt_cost_exp = find_cost_exp_QNN(qc,Y_train,normalized_Xtrain,j,matrix_feature[i],n_feature,PARAM,opZ)
				#	print(f'Expl {matrix_feature[i]} Feature {j} Angle {round(opt_theta_exp,5)} has cost {round(opt_cost_exp,5)}')
				#	exp_cost_collect.append(opt_cost_exp)
			#		n_feature_coll.append(j)
			#		new_gate_coll.append(matrix_feature[i])
			#		opt_theta_coll.append(opt_theta_exp)
			#		opt_cost_coll.append(opt_cost_exp)
				if not active_pool[i]:  # Skip inactive gates
					continue

				pub_c,pub_p = constructCircuit_QSVM_IBM(Y_train, normalized_Xtrain,
														i, matrix_feature, theta_test_matrix, n_feature, PARAM, opZ, count, norm_Xtrain_l)

				pub = (pub_c, opZ, pub_p)
				pubs.append(pub)  # Add the PUB to the list of PUBs for this iteration

			# Prepare the PUBs for submission
			pubs_orig = pubs
			pubs = prepare_submission(pubs_orig, backend=backend) # Prepare the PUBs for submission to the backend
			sv_pubs = prepare_submission(pubs_orig, backend=None)  # Prepare the PUBs for state vector simulation
			print(f"{len(pubs)} circuts with a "
				  f"{sum(len(pub[-1]) for pub in pubs)} total circuit-parameter configurations to run")

			filename_prefix = f"{count:03d}"

			store_raw_evs = []
			store_evs = []

			# Set up estimator / sampler primitive (with session execution mode):
			if not USE_SAMPLER:
				estimator = Estimator(mode=session)
				raise NotImplementedError("Estimator not implemented, please use USE_SAMPLER=True")
			else:
				sampler = init_sampler()

				# Calibrate the error mitigation if using a sampler
				print("Calibrating error mitigation...")
				transpiled_bit_mappings = mthree.utils.final_measurement_mapping([pub[0] for pub in pubs])
				print(transpiled_bit_mappings)
				calibrate_mit()

				# Store calibration result
				mit_cals_file_path = os.path.join(os.path.dirname(costFile), "mthree", f'{filename_prefix}_cals.json')
				os.makedirs(os.path.dirname(mit_cals_file_path), exist_ok=True)  # ensure directory exists
				mit.cals_to_file(mit_cals_file_path)
				print("Error mitigation calibration completed.")

			pubs_result, _ = submit_jobs(pubs, filename_prefix, sv_pubs=sv_pubs)  # Submit the PUBs and get the results

			assert len(pubs_result) == len(pubs) == sum(active_pool), "Result length does not match PUBs length"

			for i in range(len(matrix_feature)):  # Loop over gate pool to process and collect results

				if not active_pool[i]:  # Skip inactive gates
					continue

				pub_idx = sum(active_pool[:i])  # Get the index of the current gate in the active pool
				pub_res = pubs_result[pub_idx]
				mapping = transpiled_bit_mappings[pub_idx] if USE_SAMPLER else None  # Get the mapping for the current PUB

				# Post-process the raw results (evs are the probabilities in the sampler case)
				evs, raw_evs = model_outputs_from_pub_results(pub_res=pub_res,
															  mitigation=mit,
															  mitigation_mapping=mapping)
				store_evs.append(evs)
				store_raw_evs.append(raw_evs)
				#print(f"Raw evs {raw_evs} vs M3 mitigated evs {evs} for gate {matrix_feature[i][active_qubits]}")
				# Save raw expectation values for the training minibatch
				ev_path = os.path.join(os.path.dirname(costFile), "ibm_results", f"{filename_prefix}_evs.npz")
				os.makedirs(os.path.dirname(ev_path), exist_ok=True)  # ensure directory exists
				np.savez_compressed(ev_path,
									**{f"raw_evs[{sre_idx}]": sre for sre_idx, sre in enumerate(store_raw_evs)},
									**{f"evs[{se_idx}]": se for se_idx, se in enumerate(store_evs)})

				# Evaluate (reconstruct) the circuit with the obtained expectation values
				opt_theta_feature, opt_cost_feature, opt_kernel_feature = evaluateCircuit_QSVM_IBM(Y_train,
																								   normalized_Xtrain,
																								   i, matrix_feature,
																								   theta_test_matrix,
																								   n_feature,
																								   PARAM, opZ, count,
																								   norm_Xtrain_l, evs)
				for j in range(n_feature):
					#if abs(exp_cost_collect[j]-opt_cost_feature[j])>1e-5:
					#	print(f'deviation between explicit and reconstruction in cost (Feature {j}) {exp_cost_collect[j]} {opt_cost_feature[j]} {abs(exp_cost_collect[j]-opt_cost_feature[j])}')
					#	exit(9)
					#print(matrix_feature[i],j,opt_theta_feature[j],opt_cost_feature[j])
					n_feature_coll.append(j)
					new_gate_coll.append(matrix_feature[i])
					new_gate_idx_coll.append(i)  # Store the index of the gate in the pool
					opt_theta_coll.append(opt_theta_feature[j])
					opt_cost_coll.append(opt_cost_feature[j])
					opt_kernel_coll.append(opt_kernel_feature[j])

			ID_best = np.argmin(opt_cost_coll)
			new_gate_select = new_gate_coll[ID_best]
			new_gate_idx_select = new_gate_idx_coll[ID_best]
			n_feature_select = n_feature_coll[ID_best]
			opt_theta_select = opt_theta_coll[ID_best]
			opt_cost_select = opt_cost_coll[ID_best]
			opt_kernel_select = opt_kernel_coll[ID_best]
			pub_idx_select = np.sum(active_pool[:new_gate_idx_select])
			pub_select = pubs[pub_idx_select]  # get the actual best PUB to rerun the exact transpiled circuit for test eval
			print(f"Select pub idx {pub_idx_select} with gate {new_gate_select} ")
			assert scipy.linalg.issymmetric(opt_kernel_select[:min(*opt_kernel_select.shape), :min(*opt_kernel_select.shape)])

			svm = SVC(kernel='precomputed')
			kernel_path = os.path.join(os.path.dirname(costFile), "ibm_results", f"{filename_prefix}_kernel.npz")
			os.makedirs(os.path.dirname(kernel_path), exist_ok=True)  # ensure directory exists
			if n_nystroem > 0:  # use Nystroem kernel approximation
				# the training kernel opt_kernel_select has then the shape len(normalized_Xtrain) x n_nystroem
				nystroem = Nystroem(kernel='precomputed', n_components=n_nystroem)
				nystroem.fit(opt_kernel_select[:n_nystroem, :n_nystroem])  # Fit the Nystroem approximation
				pre_nystroem_kernel = opt_kernel_select.copy()
				opt_kernel_select_transf = nystroem.transform(opt_kernel_select)
				opt_kernel_select_approx = opt_kernel_select_transf @ opt_kernel_select_transf.T
				#opt_kernel_select_approx = np.clip(opt_kernel_select_approx, 0.0, 1.0)
				np.savez_compressed(kernel_path, kernel=opt_kernel_select_approx, kernel_red_nystroem=opt_kernel_select)
				opt_kernel_select = opt_kernel_select_approx
			else:
				np.savez_compressed(kernel_path, kernel=opt_kernel_select)  # Save the kernel matrix for the current iteration

			svm.fit(X=opt_kernel_select, y=Y_train)

			pred_train = svm.predict(X=opt_kernel_select)  # Fit the SVM with the kernel matrix
			train_acc = balanced_accuracy_score(Y_train, pred_train)
			print(f"Train accuracy: {train_acc}")

			new_gate_mod = []
			for qu in range(n_qubit):
				if new_gate_select[qu] == '111111':
					new_gate_mod.append('111111')
				else:
					gate,gate_id = new_gate_select[qu].split('_')
					new_gate_mod.append(f'{gate}_{n_feature_select}')

			minCost_layer = circ_convertSinge(new_gate_mod,1,[opt_theta_select]*n_qubit,n_qubit,PARAM)
			qc.compose(minCost_layer,qubits=list(range(n_qubit)),inplace=True)
			# check how many qubits are unused (if none free, add new qubit):
			unused_qubits = {qc.find_bit(wire).index for wire in circuit_to_dag(qc).idle_wires()}
			used_qubits = set(range(n_qubit)) - unused_qubits
			n_used_qubits = len(used_qubits)
			unused_active = set(active_qubits) - used_qubits
			if len(unused_active) < 2:
			#if n_used_qubits == len(active_qubits):
				print(f'Add new qubit to active ones ({len(active_qubits)} -> {len(active_qubits) + 1})')
				if min(active_qubits) in used_qubits:
					active_qubits.append((min(active_qubits) - 1) % n_qubit)  # Add a new qubit
				if max(active_qubits) in used_qubits:
					active_qubits.append((max(active_qubits) + 1) % n_qubit)
				active_qubits.sort()
				#active_qubits += 1
				#qc_expand = QuantumCircuit(n_qubit)
				#qc_expand.compose(qc, qubits=range(n_qubit), inplace=True)  # append qubit to the end
				#qc = qc_expand
				active_pool = mask_sub_pool(matrix_feature, active_qubits)  # Expand/regenerate the gate pool
				opZ = SparsePauliOp("Z" + "I" * (n_qubit - 1))  # Update the operator for the new number of qubits
				#assert qc.num_qubits == n_qubit, f"Expected {n_qubit} qubits, got {qc.num_qubits}"


			print(remove_idle_qwires(qc))
			print('Summary',count,opt_cost_select,opt_theta_select)

			# Test circuit (re-use transpilation and mitigation):
			test_qc = pub_select[0].copy()
			# Replace the angle parameter by the feature parameter and the weight parameter
			test_qc.assign_parameters({'angle': opt_theta_select * (PARAM_i[n_feature_select] - PARAM_j[n_feature_select])},
									  inplace=True)

			# Test:
			# Create the full train kernel matrix and check fidelity with Nystroem approximation
			verify_train_kernel = np.full((norm_Xtrain_l, norm_Xtrain_l), np.nan)  # Initialize the train kernel matrix
			for i in range(norm_Xtrain_l):
				for j in range(norm_Xtrain_l):
					tq = test_qc.assign_parameters({PARAM_i[nf]: normalized_Xtrain[j, nf] for nf in range(n_feature) if
													PARAM_i[nf] in test_qc.parameters} |
												   {PARAM_j[nf]: normalized_Xtrain[i, nf] for nf in range(n_feature) if
													PARAM_i[nf] in test_qc.parameters})
					tq.remove_final_measurements()
					verify_train_kernel[i, j] = Statevector(remove_idle_qwires(tq)).probabilities()[0]

			assert not np.any(np.isnan(opt_kernel_select[i, j])), f"Kernel value at ({i}, {j}) is NaN"
			print("Train kernel matrix verification complete.")
			rel_err = np.linalg.norm(verify_train_kernel - opt_kernel_select) / np.linalg.norm(verify_train_kernel)
			print("Relative error in train kernel matrix:", rel_err)

			# index j iterates over the train data points!
			if n_nystroem > 0:  # use Nystroem kernel approximation
				test_kernel_shape = (len(normalized_Xtest), n_nystroem)  # Test kernel shape for Nystroem approximation
			else:
				test_kernel_shape = (len(normalized_Xtest), len(normalized_Xtrain))
			test_ID_pairs = ID_all_pairs(*test_kernel_shape)
			test_pub_ps = [({PARAM_i[nf]: normalized_Xtest[i, nf] for nf in range(n_feature)} |
							{PARAM_j[nf]: normalized_Xtrain[j, nf] for nf in range(n_feature)})
						   for i, j in test_ID_pairs]  # Create parameter sets for the test PUBs
			test_pub = (test_qc, opZ, test_pub_ps)  # Create a test PUB with the current circuit and operator
			sv_qc = (qc.assign_parameters({p: pi for p, pi in zip(PARAM, PARAM_i) if p in qc.parameters}, inplace=False) &
					 qc.assign_parameters({p: pj for p, pj in zip(PARAM, PARAM_j) if p in qc.parameters}, inplace=False).inverse())
			sv_test_pub = (remove_idle_qwires(sv_qc).copy(), opZ, test_pub_ps)
			test_pubs = prepare_submission([test_pub], backend=None)  # backend=None skips transpilation (since already transpiled)
			sv_test_pubs = prepare_submission([sv_test_pub], backend=None)  # Prepare the test PUB for state vector simulation
			filename_prefix = f"test_{count:03d}"
			test_pubs_result, sv_test_pubs_result = submit_jobs(test_pubs, filename_prefix, sv_pubs=sv_test_pubs)

			# Process test results
			test_model_outputs, test_model_outputs_raw, test_acc = (
				model_outputs_from_pub_results(test_pubs_result[0],
											   mitigation=mit, mitigation_mapping=transpiled_bit_mappings[pub_idx_select],
											   true_labels=Y_test))
			sv_test_model_outputs, _, sv_test_acc = model_outputs_from_pub_results(sv_test_pubs_result[0], true_labels=Y_test)

			# Reshape the test model outputs into a kernel matrix
			test_kernel = kernel_mat_from_flat(test_model_outputs, test_kernel_shape[0], test_kernel_shape[1], pairs=test_ID_pairs)
			sv_test_kernel = kernel_mat_from_flat(sv_test_model_outputs, test_kernel_shape[0], test_kernel_shape[1], pairs=test_ID_pairs)

			# Store test kernel matrix (and SV result)
			kernel_path = os.path.join(os.path.dirname(costFile), "ibm_results", f"{filename_prefix}_kernel.npz")
			os.makedirs(os.path.dirname(kernel_path), exist_ok=True)  # ensure directory exists
			sv_kernel_path = os.path.join(os.path.dirname(costFile), "ibm_results", f"sv_{filename_prefix}_kernel.npz")
			os.makedirs(os.path.dirname(sv_kernel_path), exist_ok=True)  # ensure directory exists

			if n_nystroem > 0:  # use Nystroem kernel approximation
				test_kernel_transf = nystroem.transform(test_kernel)
				test_kernel_approx = test_kernel_transf @ opt_kernel_select_transf.T

				sv_test_kernel_transf = nystroem.transform(sv_test_kernel)
				sv_test_kernel_approx = sv_test_kernel_transf @ opt_kernel_select_transf.T

				np.savez_compressed(kernel_path, kernel=test_kernel_approx, kernel_red_nystroem=test_kernel)
				np.savez_compressed(sv_kernel_path, kernel=sv_test_kernel_approx, kernel_red_nystroem=sv_test_kernel)
				# Use the approximated kernel for SVM prediction
				test_kernel = test_kernel_approx
				sv_test_kernel = sv_test_kernel_approx
			else:
				np.savez_compressed(kernel_path, kernel=test_kernel)  # Save the kernel matrix for the current iteration
				np.savez_compressed(sv_kernel_path, kernel=sv_test_kernel)  # Save the kernel matrix for the current iteration

			# SVM prediction on test data
			pred_test = svm.predict(test_kernel)  # Fit the SVM with the test kernel matrix
			sv_pred_test = svm.predict(sv_test_kernel)  # Fit the SVM with the statevector test kernel matrix
			test_acc = balanced_accuracy_score(Y_test, pred_test)
			sv_test_acc = balanced_accuracy_score(Y_test, sv_pred_test)

			print(f"{count:03d}: {test_acc} test accuracy ({sv_test_acc} simulated test accuracy)")

			# Save raw expectation values for test results
			ev_path = os.path.join(os.path.dirname(costFile), "ibm_results", f"{filename_prefix}_evs.npz")
			os.makedirs(os.path.dirname(ev_path), exist_ok=True)  # ensure directory exists
			np.savez_compressed(ev_path,
								raw_evs=np.asarray(test_model_outputs_raw), evs=np.asarray(test_model_outputs),
								sv_evs=np.asarray(sv_test_model_outputs))

			count+=1
			cost_tot_list.append(opt_cost_select)
			layer_list.append(count)

			write_gates(gateFile,new_gate_select,opt_theta_select,n_feature_select)
			write_cost(costFile,opt_cost_select,count)
			write_model_outputs(trainModelOutFile, train_acc, opt_kernel_select.flatten(), count)
			write_model_outputs(testModelOutFile, test_acc, test_model_outputs, count)
			write_model_outputs(svTestModelOutFile, sv_test_acc, sv_test_model_outputs, count)

			## Check if convergence (i.e., no substantial improvement in cost)
			if count >= 100 and cost_tot_list[-2] - cost_tot_list[-1] < 1e-8:	#1e-3:
				print('Converged')
				break  # Converged!



	log_file.close()
