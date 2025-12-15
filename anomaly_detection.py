#!/usr/bin/env python3
"""
Anomaly Detection Approach for Fall Detection
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.pca import PCA as PCA_AD
import warnings

warnings.filterwarnings('ignore')


def run_anomaly_detection(normalized_data, contamination=0.05):
    """
    Run anomaly detection experiments with multiple algorithms
    
    Args:
        normalized_data: Preprocessed and normalized dataset
        contamination: Expected proportion of anomalies
    
    Returns:
        Dictionary with results for each algorithm
    """
    print("\n[Anomaly Detection] Preparing data...")
    
    # Prepare features and labels
    X = normalized_data[['x', 'y', 'z']].values
    y_true = normalized_data['anomaly'].values
    
    # Define anomaly detection models
    models = {
        'ABOD': ABOD(contamination=contamination),
        'CBLOF': CBLOF(contamination=contamination, n_clusters=8, random_state=42),
        'IForest': IForest(contamination=contamination, random_state=42),
        'KNN': KNN(contamination=contamination),
        'Average KNN': KNN(contamination=contamination, method='mean'),
        'LOF': LOF(contamination=contamination),
        'PCA': PCA_AD(contamination=contamination)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n[Anomaly Detection] Fitting {name}...")
        
        try:
            # Fit the model
            model.fit(X)

            # Predict anomalies (0 = normal, 1 = anomaly)
            y_pred = model.predict(X)

            # Get anomaly scores for AUC calculation
            y_scores = model.decision_function(X)

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            auc = roc_auc_score(y_true, y_scores)

            results[name] = {
                'model': model,
                'predictions': y_pred,
                'scores': y_scores,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'y_true': y_true
            }

            print(f"Results for {name}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  AUC:       {auc:.4f}")
            
        except Exception as e:
            print(f"Error fitting {name}: {e}")
            continue
    
    # Plot performance comparison
    plot_performance_comparison(results)

    # Plot ROC curves
    plot_roc_curves_ad(results)

    return results


def plot_performance_comparison(results):
    """Plot performance comparison across all anomaly detection models"""
    
    if not results:
        print("[Anomaly Detection] No results to plot")
        return
    
    # Prepare data for plotting
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    data = {metric: [results[model][metric] for model in models] 
            for metric in metrics}
    
    # Create bar plot
    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(16, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = width * (i - 2)
        ax.bar(x + offset, data[metric], width, label=metric.upper() if metric == 'auc' else metric.capitalize(),
               color=color, alpha=0.8)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Anomaly Detection Models Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_comparison.png', dpi=300, bbox_inches='tight')
    print("\n[Anomaly Detection] Performance comparison saved to 'anomaly_detection_comparison.png'")
    plt.close()
    
    # Create detailed metrics table
    fig, ax = plt.subplots(figsize=(10, len(models) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for model in models:
        row = [model]
        for metric in metrics:
            row.append(f"{results[model][metric]:.4f}")
        table_data.append(row)
    
    table = ax.table(cellText=table_data,
                     colLabels=['Model'] + [m.upper() if m == 'auc' else m.capitalize() for m in metrics],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(metrics) + 1):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows alternately
    for i in range(1, len(models) + 1):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(metrics) + 1):
            table[(i, j)].set_facecolor(color)
    
    plt.savefig('anomaly_detection_metrics_table.png', dpi=300, bbox_inches='tight')
    print("[Anomaly Detection] Metrics table saved to 'anomaly_detection_metrics_table.png'")
    plt.close()


def plot_roc_curves_ad(results):
    """Plot ROC curves for all anomaly detection models"""
    plt.figure(figsize=(12, 8))

    for name, metrics in results.items():
        y_true = metrics['y_true']
        y_scores = metrics['scores']

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = metrics['auc']

        # Plot
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.4f})')

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Anomaly Detection Models', fontsize=14)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_curves_anomaly_detection.png', dpi=300, bbox_inches='tight')
    print("[Anomaly Detection] ROC curves saved to 'roc_curves_anomaly_detection.png'")
    plt.close()

