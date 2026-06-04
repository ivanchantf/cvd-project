# ❤️ Heart Failure Prediction Using Statistical Machine Learning

This repository contains the source code, data preprocessing pipelines, and trained machine learning frameworks developed for the early detection and risk management of Cardiovascular Diseases (CVDs). 



## 📁 Repository Directory Matrix

| File Name | Description |
| :--- | :--- |
| **`main.ipynb`** | Exploratory Data Analysis (EDA), primary clinical data preprocessing, and **Logistic Regression** modeling. |
| **`main_rf.ipynb`** | Hyperparameter grid optimization, ensemble training, and feature evaluation for **Random Forest**. |
| **`main_gbm.ipynb`** | Vectorized feature encoding pipeline setup and training loops for **Gradient Boosting Machine (GBM)**. |
| **`main-svm.ipynb`** | Linear, Polynomial, and RBF multi-kernel search space profiling for **Support Vector Machine (SVM)**. |

---

## 🛠️ Data Preprocessing & Cleaning Pipeline
From an original combined dataset of **918 observations** sourced via Kaggle ([https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)), clinical anomalies were identified and purged during Exploratory Data Analysis:
*   **Resting Blood Pressure (`RestingBP`)**: Removed 1 clinically impossible row tracking a value of `0`. Severe hypertensive readings (180–200 mmHg) were retained.
*   **Serum Cholesterol**: Purged **171 instances** recording biological impossible levels of `0`, eliminating an artificial bimodal distribution to establish an authentic training baseline.

---

## 📊 Benchmarking & Model Performance
Models were evaluated using an 80/20 train-test separation matrix. All classifiers achieved excellent diagnostic accuracy and strong generalization profiles:

| Model Machine Learning | Test Accuracy | Test ROC-AUC | F1-Score | Architectural Setup |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest (RF)** | **90.67%** | 0.9553 | **0.9091** | `max_depth=11`, `n_estimators=200` |
| **Logistic Regression (LR)** | 89.00% | **0.9600** | 0.9000 | Baseline Intercept / Stratified 10-Fold CV |
| **Gradient Boosting (GBM)** | 86.00% | 0.9293 | 0.8500 | Symmetrical precision/recall ($0.85$ balanced split) |
| **Support Vector Machine (SVM)** | 85.33% | 0.9240 | 0.8550 | **RBF Kernel** ($C=0.05$, $\gamma=\text{scale}$) |

---

## 🧬 Diagnostic Feature Insights
To ensure clinical utility, tree-based models were decoupled via Feature Importance tracking to rank critical bio-indicators:
1.  **`ST_Slope_Up`** (GBM Importance: **0.4842**, RF Importance: **0.1572**): Ranked as the primary predictor across algorithms. An atypical peak exercise ST segment slope indicates myocardial hypoxia or coronary blockage.
2.  **`MaxHR`** (RF Importance: **0.1258**): Acts as a reliable proxy for overall cardiovascular fitness.
3.  **`ExerciseAngina_Y`** (RF Importance: **0.1131**): Exertion-induced chest pain serves as a direct clinical warning of high-risk ischemic coronary behavior.