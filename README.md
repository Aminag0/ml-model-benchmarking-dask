# ml-model-benchmarking-dask
Benchmarking ML models using Dask with CPU/GPU acceleration
# ML Model Benchmarking with Dask (CPU/GPU Acceleration)

This project benchmarks the performance of various ML classifiers using parallel computing with Dask, supporting both CPU and GPU execution. It evaluates models like RandomForest, XGBoost, LightGBM, and CatBoost on accuracy, F1 score, training time, and system resource usage.

## Features
- Preprocessing large datasets
- Serial vs parallel vs GPU model training
- Training time and resource monitoring (CPU, RAM, GPU)
- Visual performance comparison
- Confusion matrix and F1 score evaluation

## Tech Stack
- Python, Scikit-learn, XGBoost, LightGBM, CatBoost
- Dask (LocalCluster and CUDACluster)
- Matplotlib & Seaborn

## How to Run
1. Clone the repo:
```bash
git clone https://github.com/yourusername/ml-model-benchmarking-dask.git
cd ml-model-benchmarking-dask
