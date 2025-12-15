#!/usr/bin/env python3
"""
Binary Classification Approach for Fall Detection
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import warnings

warnings.filterwarnings('ignore')

# Try to import LightGBM, skip if not available
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: LightGBM not available: {e}")
    print("Skipping LightGBM classifier...")
    LIGHTGBM_AVAILABLE = False


def prepare_balanced_data(data, sample_size=6000, random_state=42):
    """
    Balance the dataset by undersampling the majority class
    
    Args:
        data: DataFrame with features and anomaly label
        sample_size: Number of normal samples to keep
        random_state: Random seed for reproducibility
    
    Returns:
        Balanced DataFrame
    """
    class_0_data = data[data['anomaly'] == 0.0].sample(
        n=sample_size, random_state=random_state
    )
    class_1_data = data[data['anomaly'] == 1.0]
    
    balanced_data = pd.concat([class_0_data, class_1_data]).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    
    return balanced_data


def run_binary_classification(normalized_data, test_size=0.2, random_state=42):
    """
    Run binary classification experiments with multiple models
    
    Args:
        normalized_data: Preprocessed and normalized dataset
        test_size: Proportion of data for testing
        random_state: Random seed
    
    Returns:
        Dictionary with results for each model
    """
    print("\n[Binary Classification] Preparing balanced dataset...")
    balanced_data = prepare_balanced_data(normalized_data)
    
    print(f"Balanced dataset shape: {balanced_data.shape}")
    print(f"Class distribution:\n{balanced_data['anomaly'].value_counts()}")
    
    # Prepare features and target
    X = balanced_data[['x', 'y', 'z']]
    y = balanced_data['anomaly']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifiers and search spaces
    classifiers = [
        ('SVC', SVC(), {
            'C': Real(1, 10, prior='log-uniform'),
            'kernel': Categorical(['linear'])
        }),
        ('KNN', KNeighborsClassifier(), {
            'n_neighbors': Integer(3, 15)
        }),
        ('Logistic Regression', LogisticRegression(solver='saga', max_iter=5000), {
            'C': Real(1e-2, 1e2, prior='log-uniform'),
            'penalty': Categorical(['l1', 'l2'])
        })
    ]

    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        classifiers.insert(1, ('LightGBM', LGBMClassifier(verbose=-1), {
            'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
            'n_estimators': Integer(100, 300),
            'max_depth': Integer(3, 10)
        }))
    
    results = {}
    
    # Perform Bayesian Search for each classifier
    for name, model, search_space in classifiers:
        print(f"\n[Binary Classification] Running Bayesian Optimization for {name}...")
        
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=search_space,
            n_iter=10,
            scoring='recall',
            cv=5,
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )
        
        bayes_search.fit(X_train_scaled, y_train)

        # Evaluate on test set
        y_pred = bayes_search.predict(X_test_scaled)

        # Get prediction probabilities for AUC calculation
        if hasattr(bayes_search.best_estimator_, 'predict_proba'):
            y_pred_proba = bayes_search.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(bayes_search.best_estimator_, 'decision_function'):
            y_pred_proba = bayes_search.decision_function(X_test_scaled)
        else:
            y_pred_proba = y_pred

        # Calculate AUC
        auc_score = roc_auc_score(y_test, y_pred_proba)

        results[name] = {
            'best_params': bayes_search.best_params_,
            'best_score': bayes_search.best_score_,
            'model': bayes_search.best_estimator_,
            'test_metrics': {
                'accuracy': accuracy_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'auc': auc_score,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'y_pred_proba': y_pred_proba,
                'y_test': y_test
            }
        }

        print(f"\nResults for {name}:")
        print(f"  Best parameters: {bayes_search.best_params_}")
        print(f"  Best cross-validated recall: {bayes_search.best_score_:.4f}")
        print(f"  Test accuracy: {results[name]['test_metrics']['accuracy']:.4f}")
        print(f"  Test recall: {results[name]['test_metrics']['recall']:.4f}")
        print(f"  Test AUC: {results[name]['test_metrics']['auc']:.4f}")
    
    # Plot confusion matrices
    plot_confusion_matrices(results)

    # Plot ROC curves
    plot_roc_curves(results)

    return results


def plot_confusion_matrices(results):
    """Plot confusion matrices for all models"""
    n_models = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (name, metrics) in enumerate(results.items()):
        cm = metrics['test_metrics']['confusion_matrix']
        ax = axes[idx]
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Normal', 'Fall'],
               yticklabels=['Normal', 'Fall'],
               title=f'{name}\nConfusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("\n[Binary Classification] Confusion matrices saved to 'confusion_matrices.png'")
    plt.close()


def plot_roc_curves(results):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))

    for name, metrics in results.items():
        y_test = metrics['test_metrics']['y_test']
        y_pred_proba = metrics['test_metrics']['y_pred_proba']

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = metrics['test_metrics']['auc']

        # Plot
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.4f})')

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Binary Classification Models', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    print("[Binary Classification] ROC curves saved to 'roc_curves.png'")
    plt.close()

