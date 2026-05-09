# Classical Surrogate Benchmarks
Scripts to evaluate Q-FLAIR against classical surrogate models (Fourier Linear Models).

### Files
- QFLAIR_QNN_benchmark.py: Core PyTorch training script of classical surrogates and corresponding Q-FLAIR iteration.
- run_benchmark.sh: Runs the benchmark for a single dataset and selected Q-FLAIR iterations.
- run_benchmarks.sh: Auto-discovers all directories/datasets containing a params file and runs benchmarks for them.
