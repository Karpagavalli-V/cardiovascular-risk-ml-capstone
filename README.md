# Final Machine Learning Project: Capstone Submission

**Course**: Machine Learning Algorithms — FullStack + Gen AI  
**Project Title**: End-to-End Predictive Health Analytics & Cardiovascular Risk Stratification Platform  
**Target Domain**: Clinical Informatics, Preventive Cardiology, and Automated Triage  

---

## 1. Project Overview & Architectural Scope
This project represents the cumulative Capstone submission for the Machine Learning module. It delivers a comprehensive, production-ready Machine Learning system engineered to ingest multidimensional patient biometrics and provide real-time cardiovascular risk stratification.

### Core Engineering Deliverables Included:
1. `reports/capstone_project_report.pdf`: Formal 3-page publication-quality technical report generated via ReportLab detailing clinical objectives, benchmark tables, embedded figures, and deployment recommendations.
2. `reports/capstone_project_report.md`: Complete Markdown technical documentation.
3. `notebooks/final_machine_learning_project_capstone.ipynb`: End-to-end Jupyter Notebook with data exploration, mathematical derivations, cross-validation, and diagnostic plots.
4. `src/train_and_evaluate.py`: Multi-model training and cross-validation pipeline comparing 5 supervised classifiers alongside unsupervised K-Means clustering.
5. `src/inference_service.py`: Low-latency real-time inference service with probability calibration and risk tier classification.
6. `app/web_dashboard.py` & `app/dashboard.html`: Interactive web dashboard demonstrating live patient predictions.
7. `test_capstone_system.py`: Comprehensive test suite verifying all 15 core deliverables and inference logic.
8. `models/`: Exported serialized pipelines (`best_model_gradient_boosting.pkl`, `random_forest_model.pkl`, `logistic_regression_model.pkl`, `kmeans_patient_clusters.pkl`).
9. `figures/`: Diagnostic figures (`model_comparison_benchmark.png`, `roc_curves_comparison.png`, `confusion_matrices_grid.png`, `feature_importance_rf.png`, `patient_risk_stratification_clusters.png`).

---

## 2. Multi-Model Benchmark Results

| Classifier Algorithm | 3-Fold CV ROC-AUC | Test Accuracy | Precision | Recall | F1-Score | Holdout ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression ($L_2$)** | **0.8241** | **87.40%** | **46.15%** | **14.63%** | **0.2222** | **0.7984** | **0.3942** |
| **Gradient Boosting (Tuned)** | 0.7926 | 86.70% | 40.38% | 17.07% | 0.2400 | 0.7584 | 0.3289 |
| **Random Forest (Ensemble)** | 0.7895 | 88.20% | 77.78% | 5.69% | 0.1061 | 0.7527 | 0.3322 |
| **Support Vector Machine (RBF)** | 0.7127 | 88.10% | 75.00% | 4.88% | 0.0916 | 0.7007 | 0.3196 |
| **Decision Tree (Cart)** | 0.7153 | 87.00% | 37.93% | 8.94% | 0.1447 | 0.6995 | 0.2347 |

---

## 3. Patient Risk Stratification Tiers
- **Low Risk ($P < 20\%$)**: Healthy baseline. Routine annual physical and wellness checkup.
- **Moderate Risk ($20\% \le P < 50\%$)**: Pre-clinical warning. Lifestyle counseling, 6-month lipid/glucose monitoring, smoking cessation.
- **High Risk ($P \ge 50\%$)**: Critical clinical alert. Immediate cardiology referral, 12-lead ECG, echocardiogram, and pharmacological evaluation.

---

## 4. Execution & Verification Guide

### 1. Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib reportlab joblib
```

### 2. Run Model Training & Evaluation Benchmark
```bash
python src/train_and_evaluate.py
```

### 3. Run Real-Time Patient Inference
```bash
python src/inference_service.py
```

### 4. Run End-to-End System Verification Test
```bash
python test_capstone_system.py
```

### 5. Launch Interactive HTML Dashboard
Open `app/dashboard.html` in your browser.

