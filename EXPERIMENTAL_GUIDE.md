# Fall Detection Experimental Guide

## Overview
This experiment demonstrates two different approaches to fall detection using sensor data (x, y, z coordinates):
1. **Binary Classification Approach** - Treating fall detection as a supervised learning problem
2. **Anomaly Detection Approach** - Treating falls as anomalies in normal behavior patterns

## Dataset Description
- **Training Data**: Located in `data/train/` directory
- **Test Data**: Located in `data/test/` directory
- **Features**: 
  - `x`, `y`, `z`: 3D spatial coordinates
  - `anomaly`: Binary label (0.0 = normal, 1.0 = fall)
- **Class Distribution**: Highly imbalanced (~95% normal, ~5% falls)

## Experimental Setup

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib plotly lightgbm pyod
```

### Data Preprocessing
1. **Data Loading**: Merge all CSV files from training directory
2. **Feature Selection**: Extract only x, y, z coordinates and anomaly label
3. **Normalization**: Apply MinMax scaling to coordinate features
4. **Class Balancing**: Sample 6,000 normal instances to balance with fall instances

## Approach 1: Binary Classification

### Objective
Train supervised learning models to classify sensor readings as normal or fall events.

### Models Evaluated
1. **Support Vector Classifier (SVC)** - Linear kernel
2. **LightGBM** - Gradient boosting classifier
3. **K-Nearest Neighbors (KNN)** - Distance-based classifier
4. **Logistic Regression** - Linear probabilistic classifier

### Hyperparameter Optimization
- **Method**: Bayesian Optimization using `BayesSearchCV`
- **Scoring Metric**: Recall (to minimize false negatives)
- **Cross-Validation**: 5-fold CV
- **Iterations**: 10 per model

### Expected Results
- **Best Model**: KNN typically achieves highest recall (~96%)
- **Trade-offs**: Balance between precision and recall
- **Evaluation Metrics**: 
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)
  - ROC-AUC Curves

## Approach 2: Anomaly Detection

### Objective
Detect falls as anomalies without relying on balanced training data.

### Models Evaluated
1. **ABOD** (Angle-Based Outlier Detection)
2. **CBLOF** (Cluster-Based Local Outlier Factor)
3. **IForest** (Isolation Forest)
4. **KNN** (K-Nearest Neighbors for anomaly detection)
5. **Average KNN**
6. **LOF** (Local Outlier Factor)
7. **PCA** (Principal Component Analysis)

### Methodology
- Train on normal data only (class 0)
- Detect falls as outliers/anomalies
- Compare performance across different algorithms

### Expected Results
- Varying performance across algorithms
- No single best algorithm for all scenarios
- Useful for imbalanced datasets

## Running the Experiments

### Quick Start
```bash
# Run the complete experiment
python run_experiment.py

# Or run individual approaches
python binary_classification.py
python anomaly_detection.py
```

### Output
- Model performance metrics
- Visualization plots (3D scatter plots, ROC curves)
- Saved model files (optional)

## Key Findings

### Binary Classification
- **Advantages**: High accuracy when data is balanced
- **Disadvantages**: Requires labeled data, sensitive to class imbalance
- **Best Use Case**: When sufficient labeled fall data is available

### Anomaly Detection
- **Advantages**: Works with imbalanced data, no need for fall examples
- **Disadvantages**: May have higher false positive rate
- **Best Use Case**: When fall events are rare and hard to collect

## Recommendations

1. **For Production Systems**: 
   - Use ensemble of both approaches
   - Implement real-time monitoring with KNN classifier
   - Set up anomaly detection as backup system

2. **For Research**:
   - Collect more fall event data
   - Experiment with deep learning approaches
   - Consider temporal features (velocity, acceleration)

3. **Data Collection**:
   - Ensure diverse fall scenarios
   - Include edge cases (near-falls, sitting down quickly)
   - Maintain balanced test set for proper evaluation

## Troubleshooting

### Common Issues
1. **Memory Error**: Reduce sample size in data balancing step
2. **Slow Training**: Reduce number of Bayesian optimization iterations
3. **Poor Performance**: Check data normalization and feature scaling

## References
- PyOD Documentation: https://pyod.readthedocs.io/
- Scikit-learn: https://scikit-learn.org/
- LightGBM: https://lightgbm.readthedocs.io/

