# End-to-End Predictive Health Analytics & Cardiovascular Risk Intelligence Platform
### Final Machine Learning Capstone Project Report
**Course**: Machine Learning Algorithms — FullStack + Gen AI  
**Author**: Machine Learning Systems Engineer  

---

## 1. Executive Summary & Clinical Background
Cardiovascular diseases (CVDs) remain the leading cause of adult morbidity and mortality globally, claiming approximately 17.9 million lives each year according to the World Health Organization. Early non-invasive risk detection enables targeted clinical interventions before catastrophic cardiac events occur.

This Capstone project engineers a complete, enterprise-grade Machine Learning system that:
1. Ingests and standardizes complex multivariable patient clinical telemetry across demographics, vital signs, lipid profiles, and lifestyle behaviors.
2. Benchmarks five primary supervised classification architectures: Logistic Regression, Random Forest, Gradient Boosting, Support Vector Classifier, and Decision Trees.
3. Implements unsupervised K-Means clustering and PCA for patient phenotyping and population risk stratification.
4. Delivers a production-ready real-time inference service, interactive dashboard, and actionable clinical intervention workflows.

---

## 2. Dataset Architecture & Clinical Biomarkers
The project utilizes a clinically validated cohort of **5,000 patient records** spanning 14 core features:
- **Demographics**: `Age` (25 - 80), `Gender` (Male, Female).
- **Vital Signs**: `Systolic_BP` (mmHg), `Diastolic_BP` (mmHg), `BMI` ($\text{kg/m}^2$), `Resting_Heart_Rate` (bpm).
- **Laboratory Panels**: `Cholesterol` (mg/dL), `Fasting_Glucose` (mg/dL).
- **Lifestyle & Medical History**: `Smoking` (0/1), `Alcohol_Intake` (0/1), `Physical_Activity` (Sedentary, Moderate, Active), `Family_History_CVD` (0/1).
- **Target Label**: `Cardiovascular_Disease` (0 = Negative, 1 = Positive).

---

## 3. Data Preprocessing & Pipeline Engineering
- **Imputation**: Median imputation for continuous biomarker values to prevent skewness from extreme lab outliers; mode imputation for categorical attributes.
- **Encoding**: Scikit-learn `OneHotEncoder(drop='first', sparse_output=False)` for categorical columns (`Gender`, `Physical_Activity`).
- **Standardization**: `StandardScaler` applied across continuous metrics to normalize feature variance for distance and gradient-based algorithms.
- **Split & Validation**: 80% Training / 20% Holdout Test split stratified on the target class. 3-Fold Stratified Cross-Validation was conducted across all candidate models.

---

## 4. Multi-Model Benchmark & Statistical Results

| Classifier Algorithm | 3-Fold CV ROC-AUC | Test Accuracy | Precision | Recall | F1-Score | Holdout ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression ($L_2$)** | **0.8241** | **87.40%** | **46.15%** | **14.63%** | **0.2222** | **0.7984** | **0.3942** |
| **Gradient Boosting (Tuned)** | 0.7926 | 86.70% | 40.38% | 17.07% | 0.2400 | 0.7584 | 0.3289 |
| **Random Forest (Ensemble)** | 0.7895 | 88.20% | 77.78% | 5.69% | 0.1061 | 0.7527 | 0.3322 |
| **Support Vector Machine (RBF)** | 0.7127 | 88.10% | 75.00% | 4.88% | 0.0916 | 0.7007 | 0.3196 |
| **Decision Tree (Cart)** | 0.7153 | 87.00% | 37.93% | 8.94% | 0.1447 | 0.6995 | 0.2347 |

---

## 5. Unsupervised Patient Phenotyping (K-Means & PCA)
Unsupervised K-Means clustering ($K=3$) was performed on the standardized clinical feature space to reveal natural patient clusters:
- **Cluster 0: Low Risk / Baseline Cohort**: Younger demographic, normal blood pressure (< 120/80 mmHg), active lifestyle, low lipid levels.
- **Cluster 1: Moderate Risk / Metabolic Warning**: Borderline hypertension (130-140 mmHg systolic), elevated BMI (> 28), sedentary tendency, moderate cholesterol.
- **Cluster 2: High Risk / Severe Clinical Warning**: Advanced age (> 60), stage 2 hypertension (> 160 mmHg systolic), elevated fasting glucose, positive family history, and smoking history.

---

## 6. Real-Time Clinical Inference & Triage Strategy
The inference engine maps continuous predicted probabilities to three clinical risk tiers:
- **Low Risk ($P < 20\%$)**: Annual wellness checkup, standard preventative diet and exercise.
- **Moderate Risk ($20\% \le P < 50\%$)**: 6-month lipid/glucose monitoring, nutritional DASH counseling, smoking cessation.
- **High Risk ($P \ge 50\%$)**: Urgent clinical referral to cardiologist, 12-lead ECG, echocardiogram, and pharmacological therapy evaluation.

