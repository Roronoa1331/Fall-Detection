#!/usr/bin/env python3
"""
Fall Detection Experiment - Main Script
This script runs both binary classification and anomaly detection approaches
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Import custom modules
from binary_classification import run_binary_classification
from anomaly_detection import run_anomaly_detection
from utils import load_and_preprocess_data, visualize_3d_data


def main():
    """Main experimental pipeline"""
    
    print("="*80)
    print("FALL DETECTION EXPERIMENT")
    print("="*80)
    
    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing data...")
    train_data, test_data = load_and_preprocess_data()
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"\nClass distribution in training data:")
    print(train_data['anomaly'].value_counts())
    
    # Step 2: Normalize features
    print("\n[Step 2] Normalizing features...")
    scaler = MinMaxScaler()
    
    # Fit scaler on training data
    feature_cols = ['x', 'y', 'z']
    train_data[feature_cols] = scaler.fit_transform(train_data[feature_cols])
    test_data[feature_cols] = scaler.transform(test_data[feature_cols])
    
    normalized_data = pd.concat([train_data, test_data], ignore_index=True)
    print("Normalization complete!")
    
    # Step 3: Visualize data (optional)
    print("\n[Step 3] Generating 3D visualization...")
    try:
        visualize_3d_data(normalized_data)
        print("3D visualization saved!")
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    # Step 4: Run Binary Classification Approach
    print("\n" + "="*80)
    print("APPROACH 1: BINARY CLASSIFICATION")
    print("="*80)
    
    bc_results = run_binary_classification(normalized_data)
    
    print("\n[Binary Classification Results]")
    for model_name, metrics in bc_results.items():
        print(f"\n{model_name}:")
        print(f"  Best Parameters: {metrics['best_params']}")
        print(f"  Best CV Recall: {metrics['best_score']:.4f}")
        if 'test_metrics' in metrics:
            print(f"  Test Accuracy: {metrics['test_metrics']['accuracy']:.4f}")
            print(f"  Test Recall: {metrics['test_metrics']['recall']:.4f}")
            print(f"  Test AUC: {metrics['test_metrics']['auc']:.4f}")
    
    # Step 5: Run Anomaly Detection Approach
    print("\n" + "="*80)
    print("APPROACH 2: ANOMALY DETECTION")
    print("="*80)
    
    ad_results = run_anomaly_detection(normalized_data)
    
    print("\n[Anomaly Detection Results]")
    for model_name, metrics in ad_results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
    
    # Step 6: Summary and Recommendations
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    # Find best models
    best_bc_model = max(bc_results.items(), 
                        key=lambda x: x[1]['best_score'])
    best_ad_model = max(ad_results.items(), 
                        key=lambda x: x[1]['f1'])
    
    print(f"\nBest Binary Classification Model: {best_bc_model[0]}")
    print(f"  CV Recall: {best_bc_model[1]['best_score']:.4f}")
    
    print(f"\nBest Anomaly Detection Model: {best_ad_model[0]}")
    print(f"  F1-Score: {best_ad_model[1]['f1']:.4f}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
1. For production deployment:
   - Use {bc_name} for primary detection (high recall)
   - Use {ad_name} as backup system (handles imbalanced data)
   
2. For further improvement:
   - Collect more fall event samples
   - Add temporal features (velocity, acceleration)
   - Consider ensemble methods
   
3. Monitoring:
   - Track false positive/negative rates
   - Regularly retrain with new data
   - Implement A/B testing for model updates
    """.format(bc_name=best_bc_model[0], ad_name=best_ad_model[0]))
    
    print("\nExperiment completed successfully!")
    print("="*80)
    
    return {
        'binary_classification': bc_results,
        'anomaly_detection': ad_results,
        'best_bc_model': best_bc_model[0],
        'best_ad_model': best_ad_model[0]
    }


if __name__ == "__main__":
    results = main()

