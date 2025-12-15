#!/usr/bin/env python3
"""
Utility functions for Fall Detection Experiment
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')


def load_and_preprocess_data(train_dir='data/train', test_dir='data/test'):
    """
    Load and preprocess training and test data
    
    Args:
        train_dir: Directory containing training CSV files
        test_dir: Directory containing test CSV files
    
    Returns:
        Tuple of (train_data, test_data) DataFrames
    """
    print(f"Loading data from {train_dir} and {test_dir}...")
    
    # Load training data
    train_files = glob.glob(os.path.join(train_dir, '*.csv'))
    if not train_files:
        raise FileNotFoundError(f"No CSV files found in {train_dir}")
    
    train_dfs = []
    for file in train_files:
        df = pd.read_csv(file)
        train_dfs.append(df)
    
    train_data = pd.concat(train_dfs, ignore_index=True)
    
    # Load test data
    test_files = glob.glob(os.path.join(test_dir, '*.csv'))
    if not test_files:
        raise FileNotFoundError(f"No CSV files found in {test_dir}")
    
    test_dfs = []
    for file in test_files:
        df = pd.read_csv(file)
        test_dfs.append(df)
    
    test_data = pd.concat(test_dfs, ignore_index=True)
    
    print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
    
    return train_data, test_data


def visualize_3d_data(data, sample_size=5000, save_path='3d_visualization.png'):
    """
    Create 3D scatter plot of the data
    
    Args:
        data: DataFrame with x, y, z coordinates and anomaly label
        sample_size: Number of samples to plot (for performance)
        save_path: Path to save the figure
    """
    print(f"Creating 3D visualization with {sample_size} samples...")
    
    # Sample data for visualization
    if len(data) > sample_size:
        data_sample = data.sample(n=sample_size, random_state=42)
    else:
        data_sample = data
    
    # Separate normal and anomaly data
    normal_data = data_sample[data_sample['anomaly'] == 0.0]
    anomaly_data = data_sample[data_sample['anomaly'] == 1.0]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot normal data
    ax.scatter(normal_data['x'], normal_data['y'], normal_data['z'],
               c='blue', marker='o', s=10, alpha=0.3, label='Normal')
    
    # Plot anomaly data
    ax.scatter(anomaly_data['x'], anomaly_data['y'], anomaly_data['z'],
               c='red', marker='^', s=30, alpha=0.8, label='Fall')
    
    ax.set_xlabel('X-axis', fontsize=12)
    ax.set_ylabel('Y-axis', fontsize=12)
    ax.set_zlabel('Z-axis', fontsize=12)
    ax.set_title('3D Visualization of Fall Detection Data', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"3D visualization saved to '{save_path}'")
    plt.close()


def print_data_summary(data):
    """
    Print summary statistics of the dataset
    
    Args:
        data: DataFrame to summarize
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print(f"\nDataset shape: {data.shape}")
    print(f"Number of features: {len(data.columns) - 1}")  # Excluding label
    
    print("\nClass distribution:")
    class_counts = data['anomaly'].value_counts()
    for label, count in class_counts.items():
        label_name = 'Normal' if label == 0.0 else 'Fall'
        percentage = (count / len(data)) * 100
        print(f"  {label_name}: {count} ({percentage:.2f}%)")
    
    print("\nFeature statistics:")
    print(data[['x', 'y', 'z']].describe())
    
    print("\nMissing values:")
    print(data.isnull().sum())
    
    print("="*60 + "\n")


def save_results_to_csv(results, filename='experiment_results.csv'):
    """
    Save experiment results to CSV file
    
    Args:
        results: Dictionary of results from experiments
        filename: Output CSV filename
    """
    rows = []
    
    # Process binary classification results
    if 'binary_classification' in results:
        for model_name, metrics in results['binary_classification'].items():
            row = {
                'Approach': 'Binary Classification',
                'Model': model_name,
                'CV_Recall': metrics['best_score'],
                'Test_Accuracy': metrics['test_metrics']['accuracy'],
                'Test_Recall': metrics['test_metrics']['recall']
            }
            rows.append(row)
    
    # Process anomaly detection results
    if 'anomaly_detection' in results:
        for model_name, metrics in results['anomaly_detection'].items():
            row = {
                'Approach': 'Anomaly Detection',
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1']
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to '{filename}'")


if __name__ == "__main__":
    # Test data loading
    try:
        train_data, test_data = load_and_preprocess_data()
        print_data_summary(pd.concat([train_data, test_data]))
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure data files are in 'data/train' and 'data/test' directories")

