# Q-FLAIR
Code and data for the article: J. Jäger, P. Elsässer, and E. Torabian, "Quantum feature-map learning with reduced resource overhead"

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

The programs in QSVM are:

- 'SVM_main_parallel.py' is the basic algorithm for the training of a QSVM
- 'SVM_predict.py' is the code that is used to predict average accuracies on test datasets

The programs in Datasets are:

- 'DataGeneration_MNIST.py' is the MNIST dataset generator with the desired dimension
- 'DataGeneration.py' is the general dataset generator for other datasets
- 'mnist_pixel_processing.py' is the code to process and downscale the MNIST dataset 

The functions 'read_data' contain the options for the input files.
