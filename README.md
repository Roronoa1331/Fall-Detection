# 🧠 Fall Detection using Binary Classification and Anomaly Detection

## 📌 Project Overview

This project investigates **fall detection** using 3D sensor data (`x, y, z`) by comparing two fundamentally different machine learning approaches:

1. **Binary Classification (Supervised Learning)**
2. **Anomaly Detection (Unsupervised / Semi-supervised Learning)**

The goal is to determine **which approach is more suitable for fall detection**, especially under **highly imbalanced data conditions**, where fall events are rare but critical.

------

## 🎯 Problem Motivation

Falls are rare events but have **severe consequences**, especially for:

- elderly monitoring systems
- wearable health devices
- smart healthcare and IoT systems

The key challenge is:

> **Detecting falls reliably without missing them (high recall), even when fall data is scarce.**

------

## 📊 Dataset Description

- **Features**:
  - `x`, `y`, `z`: 3D sensor coordinates (e.g., accelerometer)
  - `anomaly`: label (`0` = normal activity, `1` = fall)
- **Class Distribution**:
  - ~95% normal
  - ~5% fall
- **Directories**:
  - `data/train/`
  - `data/test/`

This strong imbalance makes fall detection a **non-trivial classification problem**.

------

## ⚙️ Data Preprocessing

The following preprocessing steps were applied:

1. Merge all training CSV files
2. Select only `x`, `y`, `z` and `anomaly`
3. Normalize features using **MinMaxScaler**
4. Create a **balanced training set** by:
   - keeping all fall samples
   - randomly sampling 6,000 normal samples

Balanced dataset shape:

```
(14183, 8)
Fall samples:   8183
Normal samples: 6000
```

------

## 🚀 Approach 1: Binary Classification

### 🔍 Methodology

Fall detection is treated as a **supervised binary classification problem**:

> “Is this sensor reading a fall or a normal movement?”

### Models Evaluated

- Support Vector Classifier (Linear)
- Logistic Regression
- LightGBM
- K-Nearest Neighbors (KNN)

### Hyperparameter Optimization

- **Bayesian Optimization** (`BayesSearchCV`)
- **Scoring metric**: Recall (fall class)
- **Cross-validation**: 5-fold

### 📈 Results

| Model               | Recall    | Accuracy  | AUC       |
| ------------------- | --------- | --------- | --------- |
| SVC                 | 0.803     | 0.710     | 0.777     |
| Logistic Regression | 0.754     | 0.693     | 0.784     |
| LightGBM            | 0.947     | 0.876     | **0.942** |
| **KNN**             | **0.977** | **0.913** | 0.939     |

### 🏆 Best Model

**KNN (k=3)** achieved the highest recall:

> **97.7% of fall events were correctly detected**

This makes KNN the most suitable supervised model for this dataset.

------

## 🚨 Approach 2: Anomaly Detection

### 🔍 Methodology

Falls are treated as **anomalies** in normal movement patterns.

- Models are trained using **only normal data**
- Falls are detected as outliers

### Models Evaluated

- ABOD
- CBLOF
- Isolation Forest
- KNN (anomaly)
- Average KNN
- LOF
- PCA

### 📉 Results

| Model     | Recall    | Precision | F1-score | AUC  |
| --------- | --------- | --------- | -------- | ---- |
| ABOD      | 0.096     | 0.087     | 0.09     | 0.67 |
| **CBLOF** | **0.301** | **0.300** | **0.30** | 0.60 |
| IForest   | 0.153     | 0.153     | 0.15     | 0.55 |
| PCA       | 0.245     | 0.245     | 0.24     | 0.61 |
| LOF       | 0.055     | 0.065     | 0.06     | 0.51 |

### ⚠️ Observation

Although anomaly detection models achieve high **accuracy**, their **recall for falls is low**.

> This indicates that falls in this dataset are **not always strong statistical outliers**, making pure anomaly detection insufficient as a standalone solution.

------

## 🔎 Comparison of Approaches

| Approach                  | Best Model | Recall    |
| ------------------------- | ---------- | --------- |
| **Binary Classification** | **KNN**    | **0.977** |
| Anomaly Detection         | CBLOF      | 0.301     |

### ✅ Key Insight

> **Binary classification significantly outperforms anomaly detection for fall detection when labeled fall data is available.**

------

## 🧠 Discussion

- Supervised models can learn **complex, non-linear motion patterns**
- Anomaly detection struggles because:
  - not all falls are extreme outliers
  - some falls resemble normal fast movements
- Accuracy alone is misleading in imbalanced datasets
- **Recall is the most critical metric for fall detection**

------

## 🧪 Final Conclusion

> **Fall detection is better modeled as a supervised classification problem rather than a pure anomaly detection task for this dataset.**

### Recommended System Design:

- ✅ Primary system: **Binary Classification (KNN / LightGBM)**
- 🛟 Backup system: **Anomaly Detection** for unseen or novel fall patterns

------

## 🛠️ How to Run

```
pip install numpy pandas scikit-learn matplotlib plotly lightgbm pyod scikit-optimize
python binary_classification.py
python anomaly_detection.py
```# Fall-Detection
