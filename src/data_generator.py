"""
Final Machine Learning Project - Dataset Generator
Domain: Clinical Informatics & Cardiovascular Disease Risk Prediction
Generates clinically grounded patient biometric and lifestyle records.
"""

import numpy as np
import pandas as pd
import os

def generate_cardiovascular_dataset(n_samples=5000, random_state=42):
    np.random.seed(random_state)

    # 1. Demographics
    age = np.random.normal(53, 10, n_samples).clip(25, 80).round(0).astype(int)
    gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.52, 0.48])

    # 2. Vitals & Biometrics
    # Blood pressure (Systolic & Diastolic)
    systolic_bp = (np.random.normal(128, 18, n_samples) + (age - 50) * 0.4).clip(90, 200).round(0).astype(int)
    diastolic_bp = (np.random.normal(82, 11, n_samples) + (systolic_bp - 120) * 0.35).clip(60, 130).round(0).astype(int)

    # BMI (Normal, Overweight, Obese)
    bmi = np.random.normal(27.5, 4.8, n_samples).clip(18.5, 48.0).round(1)

    # Resting Heart Rate (bpm)
    resting_hr = np.random.normal(74, 10, n_samples).clip(48, 120).round(0).astype(int)

    # 3. Laboratory Biomarkers
    # Total Cholesterol (mg/dL): <200 Normal, 200-239 Borderline, >=240 High
    cholesterol = (np.random.normal(210, 42, n_samples) + (bmi - 25) * 2.2).clip(120, 380).round(0).astype(int)

    # Fasting Blood Glucose (mg/dL): 70-99 Normal, 100-125 Prediabetes, >=126 Diabetes
    glucose = (np.random.normal(105, 30, n_samples) + (bmi - 25) * 1.8).clip(70, 300).round(0).astype(int)

    # 4. Lifestyle & Medical History
    smoking = np.random.choice([0, 1], size=n_samples, p=[0.72, 0.28])
    alcohol_intake = np.random.choice([0, 1], size=n_samples, p=[0.78, 0.22])
    physical_activity = np.random.choice(['Sedentary', 'Moderate', 'Active'], size=n_samples, p=[0.42, 0.38, 0.20])
    family_history_cvd = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])

    # 5. Ground Truth Disease Probability (Multivariable Clinical Risk Equation)
    risk_score = (
        -7.5
        + 0.052 * age
        + np.where(gender == 'Male', 0.35, 0.0)
        + 0.028 * (systolic_bp - 120)
        + 0.018 * (diastolic_bp - 80)
        + 0.014 * (cholesterol - 190)
        + 0.016 * (glucose - 95)
        + 0.055 * (bmi - 24)
        + 0.58 * smoking
        + 0.25 * alcohol_intake
        + 0.75 * family_history_cvd
    )

    # Physical activity mitigation
    activity_factor = np.where(physical_activity == 'Sedentary', 0.40, np.where(physical_activity == 'Active', -0.65, 0.0))
    risk_score += activity_factor

    # Gender coefficient vector
    male_bonus = np.where(gender == 'Male', 0.35, -0.15)
    risk_score += male_bonus

    # Stochastic variation
    risk_score += np.random.normal(0, 0.40, size=n_samples)

    # Sigmoid mapping to probability
    prob_cvd = 1.0 / (1.0 + np.exp(-risk_score))
    cvd_target = (np.random.rand(n_samples) < prob_cvd).astype(int)

    df = pd.DataFrame({
        'Patient_ID': [f"PT_{i+1:06d}" for i in range(n_samples)],
        'Age': age,
        'Gender': gender,
        'Systolic_BP': systolic_bp,
        'Diastolic_BP': diastolic_bp,
        'BMI': bmi,
        'Resting_Heart_Rate': resting_hr,
        'Cholesterol': cholesterol,
        'Fasting_Glucose': glucose,
        'Smoking': smoking,
        'Alcohol_Intake': alcohol_intake,
        'Physical_Activity': physical_activity,
        'Family_History_CVD': family_history_cvd,
        'Cardiovascular_Disease': cvd_target
    })

    # Introduce realistic 1% missing values in clinical lab metrics
    for col in ['Cholesterol', 'Fasting_Glucose', 'BMI']:
        mask = np.random.rand(n_samples) < 0.012
        df.loc[mask, col] = np.nan

    return df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'cardiovascular_patient_data.csv')

    df = generate_cardiovascular_dataset(n_samples=5000, random_state=42)
    df.to_csv(csv_path, index=False)
    print(f"Generated dataset at: {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Target Distribution:\n{df['Cardiovascular_Disease'].value_counts(normalize=True)}")
    print("\nHead:\n", df.head(3))
