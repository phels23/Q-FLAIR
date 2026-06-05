[![paper](https://img.shields.io/badge/Phys.%20Rev.%20Research-Paper-0274b5.svg)](https://doi.org/10.1103/v29j-rh32)
[![arXiv](https://img.shields.io/badge/arXiv-2510.03389-b31b1b.svg)](https://arxiv.org/abs/2510.03389)
[![DOI](https://zenodo.org/badge/1040833456.svg)](https://doi.org/10.5281/zenodo.20101654)

# Q-FLAIR 
***(Quantum Feature-Map Learning via Analytic Iterative Reconstructions)***

Code and data for the article:

J. Jäger, P. Elsässer, and E. Torabian. **"Quantum feature-map learning with reduced resource overhead."** *Phys. Rev. Research* 8(2), 023247 (2026). Available at [DOI: 10.1103/v29j-rh32](https://doi.org/10.1103/v29j-rh32)

<details>
<summary><b>Click to expand BibTeX</b></summary>

```bibtex
@article{jaeger2026qflair,
  title = {Quantum feature-map learning with reduced resource overhead},
  author = {J\"ager, Jonas and Els\"asser, Philipp and Torabian, Elham},
  journal = {Phys. Rev. Res.},
  volume = {8},
  issue = {2},
  pages = {023247},
  numpages = {23},
  year = {2026},
  month = {Jun},
  publisher = {American Physical Society},
  doi = {10.1103/v29j-rh32},
  url = {https://link.aps.org/doi/10.1103/v29j-rh32}
}
```
</details>

The learning of quantum features by Q-FLAIR was demonstrated for quantum neural networks and quantum support vector machines. The code for the quantum neural network may be found in 'QNN', that for the quantum support vector machines in 'QSVM'.

Installation of the Python environment (Python 3.11 recommended) to run the code can be achieved as follows:
- For numerical simulations, use 'requirements.txt' and install via `pip install -r requirements.txt`
- For IBM quantum computers, use 'requirements_ibm.txt' and install via `pip install -r requirements_ibm.txt`

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
- Remaining files are copies, potentially with slight modifications specific to IBM benchmarking, of the helper files (e.g., containing definitions) in QNN 

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

ClassicalSurrogateBenchmark contains an additional study on de-quantization robustness of Q-FLAIR against direct classical modeling (Fourier linear models).
- 'QFLAIR_QNN_benchmark.py' is the core PyTorch training script of classical surrogates and corresponding Q-FLAIR iteration.
- 'QFLAIR_QNN_benchmark_plot_figure.ipynb' plots the benchmark results for the bars & stripes dataset over incrementally growing ansatz depth, comparing the test accuracies of Q-FLAIR QNNs and the trained classical surrogate models, as well as visualizes the parameter count required for the classical surrogate model to match the quantum circuit at each Q-FLAIR iteration. 
- 'QFLAIR_QNN_benchmark_table.ipynb' devises a table comparing classical surrogate and Q-FLAIR QNN accuracies.
- 'run_benchmark.sh' runs the benchmark for a single dataset and selected Q-FLAIR iterations.
- 'run_benchmarks.sh' automatically discovers all directories/datasets containing a params file and runs benchmarks for them.

QuantumFisherInformation contains an additional study in which the Quantum Fisher Information Matrix (QFIM) between the ZZFeatureMap and Q-FLAIR is compared.
- 'QuantumFisherInformation_QFLAIR.py' in the subdirectory 'Q-FLAIR' calculates the QFIM for Q-FLAIR. The results for different sizes of MNIST are found in the same subdirectory
- 'QuantumFisherInformation.py' in the subdirectory 'ZZFeatureMap' does the same for ZZFeatureMap. The results are again in the same directory.
- 'plotAccuracy_Q-FLAIR_ZZfeature.py' is the script for plotting the results.

