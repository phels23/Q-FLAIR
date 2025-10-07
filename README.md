# Q-FLAIR 
***(Quantum Feature-Map Learning via Analytic Iterative Reconstructions)***

Code and data for the article: 
    
J. Jäger, P. Elsässer, and E. Torabian, "Quantum feature-map learning with reduced resource overhead"

(available as an [arXiv preprint arXiv:2510.03389](https://arxiv.org/abs/2510.03389))

The learning of quantum features by Q-FLAIR was demonstrated for quantum neural networks and quantum support vector machines. The code for the quantum neural network may be found in 'QNN', that for the quantum support vector machines in 'QSVM'.

The programs in QNN are:

- 'QNN_logLoss.py' is the basic algorithm for the training of a QNN
- 'QNN_logLoss_randomAngle.py' is a modification in which the optimization of the weight parameters (angles) is replaced by the selection of a random value for which gates and features are selected
- 'QNN_logLoss_randomAngle_MultiPara.py' is a modification in which the optimization of weight parameters (angles) is replaced by random values. The value varies for each gate/feature combination
- 'QNN_logLoss_randomFeature.py' an optimal gate/weight parameter combination is determined for a randomly selected feature
- 'QNN_logLoss_randomGate.py' an optimal weight parameter/feature combination is determined for a randomly selected gate
- 'QNN_predict.py' is the code that is used to predict average accuracies on test datasets
- 'QNN_circuit.py' contains definitions for the gate pool
- 'QNN_circuitConstruction.py' contains definitions for building the quantum circuit (mostly not in use)
- 'QNN_cost.py' contains definitions for calculations of the cost
- 'QNN_in_out.py' contains definitions for reading in parameters and writing out of results
- 'OptimizedParameters' is a directory with programs for re-optimizing fully trained quantum circuits.
    - 'pofc_bfgs.py' optimizes the model with L-BFGS-B and uses as initial configuration the iteratively optimized parameters
    - 'pofc_bfgs_rnd.py' optimizes the model with L-BFGS-B and uses as initial configuration the random parameters
    - 'pofc_cobyla.py' optimizes the model with cobyla and uses as initial configuration the iteratively optimized parameters
    - 'pofc_cobyla_rnd.py' optimizes the model with cobyla and uses as initial configuration the random parameters

The programs in QNN_IBM are:

- 'QNN_logLoss_ibm.py' is the basic algorithm for the training of a QNN for the IBM quantum computing benchmark
- 'QNN_predict_ibm.py' is the code that is used to predict average accuracies on the train and test datasets, given both the IBM benchmark model results and also numerical simulations as a reference
- Remaining files are copies, potentially with slight modifications specific to IBM benchmarking, of the helper helper files (e.g., containing definitions) in QNN 

The programs in QSVM are:

- 'SVM_main_parallel.py' is the basic algorithm for the training of a QSVM
- 'SVM_predict.py' is the code that is used to predict average accuracies on test datasets
  
The programs in QSVM_IBM are:

- 'QSVM_ibm.py' is the basic algorithm for the training of a QSVM for the IBM quantum computing benchmark
- 'QSVM_predict_ibm.py' is the code that is used to predict average accuracies on the train and test datasets, given both the IBM benchmark model results and also numerical simulations as a reference
- Remaining files are copies, potentially with slight modifications specific to IBM benchmarking, of the helper helper files (e.g., containing definitions) in QNN (not QSVM, since the QSVM_IBM code was adopted from QNN_IBM)

The functions 'read_data' contain the options for the input files.

The programs in Datasets are:

- 'DataGeneration.py' is the general dataset generator for other datasets
- 'DataGeneration_MNIST.py' is the MNIST dataset generator with the desired dimension
- 'mnist_pixel_processing.py' is the code to pre-process and downscale the MNIST dataset

The data in Results is split in data for experiments and simulations of quantum neural networks (QNN) 'Data_QNN' and quantum support vector machines (QSVM) 'Data_TA'. The programs in the directory plot the figures in the publication:

- 'PlotAcc_IBM_paper.ipynb' plots the overview over the accuracies from QNN experiments on a quantum computer
- 'PlotAcc_Ta_IBM_paper.ipynb' plots the overview over the accuracies for QSVM experiments on a quantum computer
- 'PlotAcc_TA_paper.ipynb' plots the overview over the accuracies for the QSVM simulations
- 'PlotAcc_paper.ipynb' plots the overview over the accuracies for the QNN simulations
- 'PlotMNIST-MNIST_pca_paper.ipynb' plots the comparison between the MNIST and the MNIST pca datasets
- 'PlotMinGateNeeded.ipynb' plots the comparison between QSVM and QNN with regard to required gates for a given accuracy
- 'PlotOptim.ipynb' plots the results for post-optimization
- 'PlotRandomGate_paper.ipynb' plots the result of the ablation study

