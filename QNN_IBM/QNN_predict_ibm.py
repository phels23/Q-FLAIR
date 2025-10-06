from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Statevector
from sklearn import preprocessing
from tqdm import tqdm

import numpy as np
import argparse
from sklearn.metrics import log_loss as sk_log_loss, roc_curve
import os.path
from collections import OrderedDict
import pickle

from QNN_logLoss_ibm import evaluateCircuit_QNN_IBM
from QNN_circuit_ibm import circ_convertSinge

def remove_idle_qwires(circ):
	dag = circuit_to_dag(circ)

	idle_wires = list(dag.idle_wires())
	dag.remove_qubits(*idle_wires)

	return dag_to_circuit(dag)

def read_data():
	'''
	read in the data files and the setup parameters
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--parameter", type=str)
	args = parser.parse_args()
	param_file = args.parameter
	allData = [[],[],[],[],[],[]] #X_test,X_train,X_val,Y_test,Y_train,Y_val
	seed=0
	repetition=0
	layers=0
	kernelFile = 'a'
	gateFile = 'a'
	costFile = 'a'
	train_start = 0
	train_length=False
	accOut = 'a'
	nTest = 0
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
		elif line_split[0].strip() == 'kernelFile': kernelFile = line_split[1].strip()
		elif line_split[0].strip() == 'gateFile': gateFile = line_split[1].strip()
		elif line_split[0].strip() == 'costFile': costFile = line_split[1].strip()
		elif line_split[0].strip() == 'train_start': train_start = int(line_split[1].strip())
		elif line_split[0].strip() == 'train_length': train_length = int(line_split[1].strip())
		elif line_split[0].strip() == 'accOut': accOut = line_split[1].strip()
		elif line_split[0].strip() == 'nTest': nTest = int(line_split[1].strip())
		elif line_split[0].strip() == 'train_batch': train_batch = int(line_split[1].strip())
		elif line_split[0].strip() == 'shots': shots = int(line_split[1].strip())
	if train_length==False:
		train_length=len(allData[1])
		train_start=0
	elif train_start+train_length>len(allData[1]):
		train_length = len(allData[1])-train_start
		print('new length for training dataset:',train_length)
	allData[0] = allData[0][:nTest]
	allData[3] = allData[3][:nTest]
	scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi),clip=True)			#normalize the data to the range 0 to 2PI
	allData[1] =scaler.fit_transform(allData[1])
	allData[0]=scaler.transform(allData[0])
	allData[2] =scaler.transform(allData[2])

	# cut train data:
	allData[1] = allData[1][train_start:train_start+train_length]
	allData[4] = allData[4][train_start:train_start+train_length]

	param_open.close()
	return [allData,seed,repetition,layers,kernelFile,gateFile,costFile,accOut,train_batch,shots]

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


def initializeMainCirc():
	qreg_qc = QuantumRegister(n_qubit, 'qc')
	qc = QuantumCircuit(qreg_qc)	#initialize quantum circuit
	return qc


def evaluation(y_pred,Y_real):#,P_pred):
    '''
    Calculating statistical criteria for the proposed model.

    Args:
        y_pred: Predicted labels.
        Y_real: Real labels in the dataset.
        P_pred: Predicted probability of correct label.

    Returns:
        acc0:  True Positive Rate (TPR)
        acc1: True Negative Rate (TNR)
        Low_acc: Min(TPR,TNR)
        BIC: Bayesian Information Criteria.
        f1: F1-score
    '''
    count0=0
    count1=0
    t=0
    r=0
    Yr=Y_real
    for j in range(len(y_pred)):
        if Yr[j]==0:
            t+=1
            if y_pred[j]==Yr[j]:
                count0+=1
        else:
            r+=1
            if y_pred[j]==Yr[j]:
                count1+=1
    acc0=count0/t
    acc1=count1/r
    Low_acc = min(acc0,acc1)
    BIC=0
    f1 = 0
    return (acc0,acc1,Low_acc,BIC,f1)


def predict_qc(X_dataset):
	inv_n_data = 1 / len(X_dataset)
	qc = initializeMainCirc()

	y_sign_gates = []
	y_prob_pos_gates = []
	expect_collect = []
	gate_cutoff = min([150, len(gate)])  # 60
	X_dataset_orig = np.asarray(X_dataset)
	for i in tqdm(range(0, gate_cutoff)):
		if X_dataset_orig.ndim == 3:  # different batches per iteration
			X_dataset = X_dataset_orig[i]
		y_sign = []
		y_prob_pos = 0
		for j in range(len(gate[i])):
			if gate[i][j] != '111111':
				gTyp, fID = gate[i][j].split('_')
				gate[i][j] = f'{gTyp}_{feature[i]}'
		angle_list = [angle[i]] * n_qubit
		newGate = circ_convertSinge(gate[i], 1, angle_list, n_qubit, PARAM)
		qc.compose(newGate, qubits=list(range(0, n_qubit)), inplace=True)
		# for i in range(len(gate)-1,len(gate)):
		# print(qc)
		expect_collect.append([])
		qc_red = remove_idle_qwires(qc)
		for dp in range(len(X_dataset)):
			# print('Dataset',dp)
			y_pred = 0
			qc_test = qc_red.copy()
			for nf in range(n_feature):
				try:
					qc_test.assign_parameters({PARAM[nf]: X_dataset[dp][nf]}, inplace=True)
				except:
					pass

			st = Statevector(qc_test)
			expect = st.probabilities()[0] # no rescaling!
			expect_collect[-1].append(expect)

			expect_sign = np.sign(expect)
			if expect_sign == 0: expect_sign = 1
			y_sign.append(expect_sign)
			if y_sign[dp] > 0: y_prob_pos += inv_n_data

		y_sign_gates.append(y_sign)
		y_prob_pos_gates.append(y_prob_pos)
	return [y_sign_gates, y_prob_pos_gates, expect_collect]


def sv_evaluate_pub(pub):
	qc, _, ps = pub
	evs = []
	for p in ps:
		qc_p = qc.copy()
		for k, v in p.items():
			try:
				qc_p.assign_parameters({k: v}, inplace=True)
			except:
				pass
		ev = Statevector(qc_p).probabilities()[0]
		evs.append(ev)
	return evs


test_train,seed,repetition,layer,kernelFile,gateFile,costFile,accOut,train_batch,shots = read_data()
gate, angle, feature = read_circuit(gateFile)
normalized_Xtest,normalized_Xtrain,normalized_Xval,Y_test,Y_train,Y_val=test_train
accOut = os.path.splitext(accOut)[0]

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

# Make sure that constant column is present:
assert np.isclose(np.std(normalized_Xtest[:,-1]), 0.)  # Ensure last column is constant
for i in range(len(normalized_Xtest)):
	normalized_Xtest[i][-1] = np.pi
assert np.isclose(np.std(normalized_Xtrain[:,-1]), 0.)  # Ensure last column is constant
for i in range(len(normalized_Xtrain)):
	normalized_Xtrain[i][-1] = np.pi

# Load mini batches:
normalized_Xtrain_mini = []
Y_train_mini = []
for i in range(len(gate)):
	train_mini_i = np.load(os.path.join(os.path.dirname(accOut), 'train_minibatches', f'{i:03d}.npz'))
	normalized_Xtrain_mini.append(train_mini_i['normalized_Xtrain'])
	Y_train_mini.append(train_mini_i['Y_train'])


print('read in finished',flush=True)
n_feature = len(normalized_Xtrain[0])
n_qubit=30
PARAM = ParameterVector('x', n_feature)

expect_predict_train = np.asarray(predict_qc(normalized_Xtrain)[-1])
expect_predict = np.asarray(predict_qc(normalized_Xtest)[-1])
losses = [sk_log_loss(Y_test, p) for p in expect_predict]
losses_train = [sk_log_loss(Y_train, p) for p in expect_predict_train]
y_predicted = []
y_predicted_train = []
for i in range(len(gate)):  # for each gate
	# ROC analysis for threshold optimization on training data:
	fpr,tpr,thresholds = roc_curve(np.asarray(Y_train),expect_predict_train[i])
	tnr = 1 - fpr
	accs = (tpr + tnr) * 0.5
	maxID = np.argmax(accs)
	max_threshold = thresholds[maxID]
	#max_threshold = 0.5
	y_predicted_threshold = np.where(expect_predict[i] >= max_threshold, 1, 0)
	y_predicted_threshold_train = np.where(expect_predict_train[i] >= max_threshold, 1, 0)
	y_predicted.append(y_predicted_threshold)
	y_predicted_train.append(y_predicted_threshold_train)
expect_predict_mini = np.array(expect_predict)  # copied for later use
del expect_predict_train, expect_predict  # avoids accidental reuse of the predictions


# IBM backend
expect_predict_train_ibm = []  # train predictions will be later retrieved from the IBM reconstructions
test_model_out = np.genfromtxt(os.path.join(os.path.dirname(accOut), 'TestModelOut.txt'),
									 dtype=float, skip_header=0)
expect_predict_ibm = np.clip((test_model_out[:, 2:] + 1) / 2, 0., 1.)  # rescale to [0,1]
assert expect_predict_ibm.shape[1] == len(Y_test), "Mismatch in number of predictions and test labels"
assert expect_predict_ibm.shape[0] == len(gate), "Mismatch in number of gates and predictions"
assert np.allclose(test_model_out[:, 0], np.arange(len(gate)) + 1), "Gate indices do not match"



# Mini batches (exact simulation); only difference for test data is the threshold optimized on the training mini batches
expect_predict_train_mini = np.asarray(predict_qc(normalized_Xtrain_mini)[-1])
losses_mini = losses
losses_train_mini = [sk_log_loss(Yt, p) for Yt, p in zip(Y_train_mini, expect_predict_train_mini)]
y_predicted_mini = []
y_predicted_train_mini = []
for i in range(len(gate)):  # for each gate
	# ROC analysis for threshold optimization on training data:
	fpr,tpr,thresholds = roc_curve(np.asarray(Y_train_mini[i]),expect_predict_train_mini[i])
	tnr = 1 - fpr
	accs = (tpr + tnr) * 0.5
	maxID = np.argmax(accs)
	max_threshold = thresholds[maxID]
	#max_threshold = 0.5
	y_predicted_threshold = np.where(expect_predict_mini[i] >= max_threshold, 1, 0)
	y_predicted_threshold_train = np.where(expect_predict_train_mini[i] >= max_threshold, 1, 0)
	y_predicted_mini.append(y_predicted_threshold)
	y_predicted_train_mini.append(y_predicted_threshold_train)
del expect_predict_train_mini, expect_predict_mini  # avoids accidental reuse of the predictions

# We load the evs from the IBM backend (with and without error mitigation contained in .npz) and determine the error mitigation effect relative to the exact simulation
# Skip loss calculation, only model output RMSE
# load npz:
rmses = []
for i in tqdm(range(len(gate)), desc="Reconstruction quality evaluation"):
	evs_ibm_backend = np.load(os.path.join(os.path.dirname(accOut), 'ibm_results', f'{i:03d}_evs.npz'))
	evs_pool = [v for k, v in sorted(evs_ibm_backend.items()) if k.startswith('evs')]  # mitigated
	raw_evs_pool = [v for k, v in sorted(evs_ibm_backend.items()) if k.startswith('raw')]  # raw (unmitigated)
	# For exact simulation, load pubs_orig.pickle:
	with open(os.path.join(os.path.dirname(accOut), 'ibm_results', f'{i:03d}_pubs_orig.pickle'), 'rb') as f:
		pubs_orig = pickle.load(f)
	# Evaluate the circuits with exact simulation (to obtain sv_evs):
	sv_evs_pool = []
	for pub in pubs_orig:
		sv_evs = sv_evaluate_pub(pub)
		sv_evs_pool.append(sv_evs)
	assert len(sv_evs_pool) == len(evs_pool) == len(raw_evs_pool) == len(pubs_orig)
	# For lists, add tuple where first element is w.r.t the mitigated evs, second element is w.r.t. the raw evs
	rmses.append([])
	num_samples = int(1e2)
	opt_pool_cost = np.inf
	opt_pool_expect_predict_train_ibm = None
	for evs, raw_evs, sv_evs in zip(evs_pool, raw_evs_pool, sv_evs_pool):  # iterate over the pool

		(AA, BB, CC) = \
			evaluateCircuit_QNN_IBM(Y_train_mini[i], normalized_Xtrain_mini[i],
									 0, None, None, 1, None, None,
									 i, len(normalized_Xtrain_mini[i]), evs, return_reconstruction_only=True)
		(AA_raw, BB_raw, CC_raw) = \
			evaluateCircuit_QNN_IBM(Y_train_mini[i], normalized_Xtrain_mini[i],
									 0, None, None, 1, None, None,
									 i, len(normalized_Xtrain_mini[i]), raw_evs, return_reconstruction_only=True)
		(AA_sv, BB_sv, CC_sv) = \
			evaluateCircuit_QNN_IBM(Y_train_mini[i], normalized_Xtrain_mini[i],
									 0, None, None, 1, None, None,
									 i, len(normalized_Xtrain_mini[i]), sv_evs, return_reconstruction_only=True)

		# Reconstruct the ibm training data predictions:
		expect_pred_train_ibm_rec = np.clip((AA.flatten() * np.cos(angle[i]*normalized_Xtrain_mini[i][:, feature[i]] - BB.flatten()) + CC.flatten() + 1) / 2, 0., 1.)
		expect_pred_train_ibm_rec_cost = sk_log_loss(Y_train_mini[i], expect_pred_train_ibm_rec)
		if expect_pred_train_ibm_rec_cost < opt_pool_cost:  # new, better gate in the pool
			opt_pool_cost = expect_pred_train_ibm_rec_cost
			opt_pool_expect_predict_train_ibm = expect_pred_train_ibm_rec

		# Sample reconstructions
		alpha = np.linspace(-np.pi, np.pi, num_samples)
		alpha = alpha.reshape(-1, 1)
		kernel_mat_flat = np.clip((AA.flatten() * np.cos(alpha - BB.flatten()) + CC.flatten() + 1) / 2, 0.0, 1.0)
		kernel_mat_flat_raw = np.clip((AA_raw.flatten() * np.cos(alpha - BB_raw.flatten()) + CC_raw.flatten() + 1) / 2, 0.0, 1.0)
		kernel_mat_flat_sv = np.clip((AA_sv.flatten() * np.cos(alpha - BB_sv.flatten()) + CC_sv.flatten() + 1) / 2, 0.0, 1.0)

		# Calculate the RMSE for the mitigated and raw evs w.r.t. the sv_evs
		rmse = np.sqrt(np.mean((kernel_mat_flat - kernel_mat_flat_sv) ** 2, axis=0))
		rmse_raw = np.sqrt(np.mean((kernel_mat_flat_raw - kernel_mat_flat_sv) ** 2, axis=0))
		rmses[-1].extend(zip(rmse, rmse_raw))

	expect_predict_train_ibm.append(opt_pool_expect_predict_train_ibm)
assert len(expect_predict_train_ibm) == len(gate), "Mismatch in number of gates and predictions for IBM train data"

# Compute metrics given the IBM train predictions
losses_ibm = [sk_log_loss(Y_test, p) for p in expect_predict_ibm]
losses_train_ibm = [sk_log_loss(Yt, p) for Yt, p in zip(Y_train_mini, expect_predict_train_ibm)]
y_predicted_ibm = []
y_predicted_train_ibm = []
for i in range(len(gate)):  # for each gate
	# ROC analysis for threshold optimization on training data:
	fpr,tpr,thresholds = roc_curve(np.asarray(Y_train_mini[i]),expect_predict_train_ibm[i])
	tnr = 1 - fpr
	accs = (tpr + tnr) * 0.5
	maxID = np.argmax(accs)
	max_threshold = thresholds[maxID]
	y_predicted_threshold_ibm = np.where(expect_predict_ibm[i] >= max_threshold, 1, 0)
	y_predicted_threshold_train_ibm = np.where(expect_predict_train_ibm[i] >= max_threshold, 1, 0)
	y_predicted_ibm.append(y_predicted_threshold_ibm)
	y_predicted_train_ibm.append(y_predicted_threshold_train_ibm)

del expect_predict_ibm, expect_predict_train_ibm

metrics_keys = ['Acc0', 'Acc1', 'AccLow', 'Avg-Acc', 'loss']
keys = metrics_keys.copy()
keys.extend([k + '_tr' for k in metrics_keys])
keys.extend([k + '_ibm' for k in metrics_keys])
keys.extend([k + '_ibm_tr' for k in metrics_keys])
keys.extend([k + '_mini' for k in metrics_keys])
keys.extend([k + '_mini_tr' for k in metrics_keys])
keys.extend(['RMSE_mean', 'RMSE_std', 'RMSE_mean_raw', 'RMSE_std_raw'])
outfile=open(accOut + '.txt','w')
outfile.write('\t'.join(keys) + '\n')  # write header with keys
for i in range(len(y_predicted)):
	# Compute all metrics
	acc0, acc1, Low, BIC, F1 = evaluation(y_predicted[i], Y_test)
	acc0_tr, acc1_tr, Low_tr, BIC_tr, F1_tr = evaluation(y_predicted_train[i], Y_train)
	acc0_ibm, acc1_ibm, Low_ibm, BIC_ibm, F1_ibm = evaluation(y_predicted_ibm[i], Y_test)
	acc0_ibm_tr, acc1_ibm_tr, Low_ibm_tr, BIC_ibm_tr, F1_ibm_tr = evaluation(y_predicted_train_ibm[i], Y_train_mini[i])
	acc0_mini, acc1_mini, Low_mini, BIC_mini, F1_mini = evaluation(y_predicted_mini[i], Y_test)
	acc0_mini_tr, acc1_mini_tr, Low_mini_tr, BIC_mini_tr, F1_mini_tr = evaluation(y_predicted_train_mini[i], Y_train_mini[i])
	print((acc0_ibm + acc1_ibm) / 2)

	# Build ordered result
	result = OrderedDict([
		('Acc0', acc0), ('Acc1', acc1), ('AccLow', Low), ('Avg-Acc', (acc0 + acc1) / 2), ('loss', losses[i]),
		('Acc0_tr', acc0_tr), ('Acc1_tr', acc1_tr), ('AccLow_tr', Low_tr), ('Avg-Acc_tr', (acc0_tr + acc1_tr) / 2), ('loss_tr', losses_train[i]),
		('Acc0_ibm', acc0_ibm), ('Acc1_ibm', acc1_ibm), ('AccLow_ibm', Low_ibm), ('Avg-Acc_ibm', (acc0_ibm + acc1_ibm) / 2), ('loss_ibm', losses_ibm[i]),
		('Acc0_ibm_tr', acc0_ibm_tr), ('Acc1_ibm_tr', acc1_ibm_tr), ('AccLow_ibm_tr', Low_ibm_tr), ('Avg-Acc_ibm_tr', (acc0_ibm_tr + acc1_ibm_tr) / 2), ('loss_ibm_tr', losses_train_ibm[i]),
		('Acc0_mini', acc0_mini), ('Acc1_mini', acc1_mini), ('AccLow_mini', Low_mini), ('Avg-Acc_mini', (acc0_mini + acc1_mini) / 2), ('loss_mini', losses_mini[i]),
		('Acc0_mini_tr', acc0_mini_tr), ('Acc1_mini_tr', acc1_mini_tr), ('AccLow_mini_tr', Low_mini_tr), ('Avg-Acc_mini_tr', (acc0_mini_tr + acc1_mini_tr) / 2), ('loss_mini_tr', losses_train_mini[i]),
		('RMSE_mean', np.mean([r[0] for r in rmses[i]])), ('RMSE_std', np.std([r[0] for r in rmses[i]])),
		('RMSE_mean_raw', np.mean([r[1] for r in rmses[i]])), ('RMSE_std_raw', np.std([r[1] for r in rmses[i]])),
	])

	# Write to file and print
	outfile.write('\t'.join(str(result[k]) for k in keys) + '\n')
	print()  # Blank line between results
outfile.close()
