# ============================
# ML Model Benchmarking Script
# ============================

import warnings
import time
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import psutil
try:
    import pynvml
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "nvidia-ml-py3"], check=True)
    import pynvml
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class EnhancedPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model_metrics = {}
        self.feature_importances = {}
        self.resource_usage = {}
        self.serial_time = None
        self.client = None

    def _start_cluster(self, gpu=False):
        if self.client:
            self.client.close()
            gc.collect()
        if gpu:
            try:
                from dask_cuda import LocalCUDACluster
                cluster = LocalCUDACluster()
            except ImportError:
                print("dask_cuda not available, using CPU cluster")
                cluster = LocalCluster()
        else:
            cluster = LocalCluster()
        self.client = Client(cluster)
        print(f"Started {'GPU' if gpu else 'CPU'} cluster with {len(self.client.nthreads())} workers")

    def monitor_resources(self):
        return {
            'cpu': psutil.cpu_percent(interval=1),
            'ram': psutil.virtual_memory().percent,
            'gpu': self._get_gpu_usage()
        }

    def _get_gpu_usage(self):
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        except:
            return 0

    def preprocess_data(self):
        try:
            df = pd.read_csv(self.data_path)
            if 'target' not in df.columns:
                raise ValueError("Data must contain 'target' column")
            numeric_cols = df.select_dtypes(include=['number']).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            df = df.dropna()
            for col in categorical_cols:
                if col != 'target':
                    df[col] = LabelEncoder().fit_transform(df[col])
            numeric_cols = [col for col in numeric_cols if col != 'target']
            if numeric_cols:
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            return df
        except Exception as e:
            print(f"Data loading failed: {str(e)}")
            return None

    def benchmark_serial(self, X_train, y_train):
        model = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=1, random_state=42)
        start = time.time()
        model.fit(X_train, y_train)
        self.serial_time = time.time() - start
        return model

    def train_evaluate(self, X_train, X_test, y_train, y_test, model_type='random_forest'):
        model_config = {
            'random_forest': (RandomForestClassifier, {'n_estimators': 100, 'max_depth': 12, 'n_jobs': -1, 'random_state': 42}),
            'gradient_boosting': (GradientBoostingClassifier, {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}),
            'xgboost_cpu': (XGBClassifier, {'n_estimators': 100, 'max_depth': 6, 'tree_method': 'hist', 'random_state': 42}),
            'xgboost_gpu': (XGBClassifier, {'n_estimators': 100, 'max_depth': 6, 'tree_method': 'gpu_hist', 'device': 'cuda', 'random_state': 42}),
            'lightgbm_gpu': (LGBMClassifier, {'n_estimators': 100, 'max_depth': 6, 'device': 'gpu', 'random_state': 42}),
            'catboost_gpu': (CatBoostClassifier, {'iterations': 100, 'depth': 6, 'task_type': 'GPU', 'random_seed': 42, 'verbose': 0})
        }

        model_class, params = model_config[model_type]
        model = model_class(**params)
        resources_before = self.monitor_resources()
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        resources_after = self.monitor_resources()
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = {
            'accuracy': accuracy_score(y_test, preds),
            'f1': f1_score(y_test, preds),
            'roc_auc': roc_auc_score(y_test, probas) if probas is not None else None,
            'precision': precision_score(y_test, preds),
            'recall': recall_score(y_test, preds),
            'confusion_matrix': confusion_matrix(y_test, preds),
            'train_time': train_time,
            'resource_usage': {
                'before': resources_before,
                'after': resources_after,
                'delta': {
                    'cpu': resources_after['cpu'] - resources_before['cpu'],
                    'ram': resources_after['ram'] - resources_before['ram'],
                    'gpu': resources_after['gpu'] - resources_before['gpu']
                }
            }
        }
        if hasattr(model, 'feature_importances_'):
            self.feature_importances[model_type] = model.feature_importances_
        return metrics

    def run_benchmark(self):
        try:
            print("\\n=== DATA PREPROCESSING ===")
            df = self.preprocess_data()
            if df is None or len(df) == 0:
                raise ValueError("Data preprocessing failed")
            X = df.drop('target', axis=1)
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            print("\\n=== SERIAL BASELINE ===")
            serial_model = self.benchmark_serial(X_train, y_train)
            serial_preds = serial_model.predict(X_test)
            self.model_metrics['serial'] = {
                'accuracy': accuracy_score(y_test, serial_preds),
                'train_time': self.serial_time
            }

            print("\\n=== PARALLEL CPU MODELS ===")
            for model in ['random_forest', 'gradient_boosting', 'xgboost_cpu']:
                self._start_cluster(gpu=False)
                metrics = self.train_evaluate(X_train, X_test, y_train, y_test, model)
                self.model_metrics[model] = metrics
                self.client.close()
                gc.collect()

            print("\\n=== GPU ACCELERATED MODELS ===")
            for model in ['xgboost_gpu', 'lightgbm_gpu', 'catboost_gpu']:
                self._start_cluster(gpu=True)
                metrics = self.train_evaluate(X_train, X_test, y_train, y_test, model)
                self.model_metrics[model] = metrics
                self.client.close()
                gc.collect()

            self.calculate_speedups()
            self.generate_visualizations()
            return self.model_metrics
        except Exception as e:
            print(f"Benchmark failed: {str(e)}")
            return None

    def calculate_speedups(self):
        if not self.serial_time:
            print("Warning: No serial baseline recorded")
            return
        print("\\n=== SPEEDUP ANALYSIS ===")
        for model, metrics in self.model_metrics.items():
            if model == 'serial':
                continue
            speedup = self.serial_time / metrics['train_time']
            reduction = 1 - (metrics['train_time'] / self.serial_time)
            status = "✅" if reduction >= 0.7 else "❌"
            print(f"{status} {model}: {speedup:.1f}x speedup ({reduction:.1%} reduction)")

    def generate_visualizations(self):
        if not self.model_metrics:
            return
        vis_data = []
        for model, metrics in self.model_metrics.items():
            vis_data.append({
                'Model': model.replace('_', ' ').title(),
                'Accuracy': metrics['accuracy'],
                'F1 Score': metrics.get('f1', 0),
                'Training Time': metrics['train_time'],
                'Type': 'GPU' if 'gpu' in model else ('Serial' if model == 'serial' else 'CPU')
            })

        df = pd.DataFrame(vis_data)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        sns.barplot(data=df, x='Model', y='Accuracy', hue='Type')
        plt.title('Model Accuracy Comparison')
        plt.xticks(rotation=45)
        plt.subplot(1, 2, 2)
        sns.barplot(data=df, x='Model', y='Training Time', hue='Type')
        plt.title('Training Time Comparison (Log Scale)')
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('performance_comparison.png', bbox_inches='tight')
        plt.close()

        resource_data = []
        for model, metrics in self.model_metrics.items():
            if 'resource_usage' in metrics:
                resource_data.append({
                    'Model': model.replace('_', ' ').title(),
                    'CPU Usage': metrics['resource_usage']['after']['cpu'],
                    'RAM Usage': metrics['resource_usage']['after']['ram'],
                    'GPU Usage': metrics['resource_usage']['after']['gpu']
                })

        if resource_data:
            res_df = pd.DataFrame(resource_data)
            plt.figure(figsize=(12, 6))
            res_df.set_index('Model').plot(kind='bar', rot=45)
            plt.title('Resource Usage During Training')
            plt.ylabel('Percentage')
            plt.tight_layout()
            plt.savefig('resource_usage.png', bbox_inches='tight')
            plt.close()

    def print_results(self):
        print("\\n=== FINAL RESULTS ===")
        for model, metrics in self.model_metrics.items():
            print(f"\\n{model.replace('_', ' ').title()}:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            if 'f1' in metrics:
                print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"Training Time: {metrics['train_time']:.2f}s")
            if 'confusion_matrix' in metrics:
                print("Confusion Matrix:")
                print(metrics['confusion_matrix'])

if __name__ == "__main__":
    try:
        import xgboost
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "xgboost", "lightgbm", "catboost", "dask", "dask-ml"], check=True)

    data_file = 'pdc_dataset_with_target.csv'
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
    else:
        print(f"\\nStarting benchmark with {data_file}")
        pipeline = EnhancedPipeline(data_file)
        df = pipeline.preprocess_data()
        if df is not None:
            print("Data preprocessing successful!")
            print(f"Data shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            results = pipeline.run_benchmark()
            if results:
                pipeline.print_results()
                print("\\nVisualizations saved:")
                print("- performance_comparison.png")
                print("- resource_usage.png")
