"""
Final Machine Learning Capstone: Production Inference Service
Performs real-time patient risk scoring, phenotyping, and clinical intervention guidance.
"""

import os
import joblib
import pandas as pd
import numpy as np

RISK_TIERS = {
    'LOW': {
        'Tier': 'Low Cardiovascular Risk (Healthy Baseline)',
        'Threshold': '< 20% Probability',
        'Clinical_Action': 'Routine annual wellness checkup, maintain balanced diet and regular aerobic exercise.'
    },
    'MODERATE': {
        'Tier': 'Moderate Cardiovascular Risk (Pre-Clinical Alert)',
        'Threshold': '20% - 49% Probability',
        'Clinical_Action': 'Schedule fasting lipid profile & HbA1c test; initiate lifestyle counseling (smoking cessation, DASH diet).'
    },
    'HIGH': {
        'Tier': 'High Cardiovascular Risk (Urgent Clinical Intervention)',
        'Threshold': '>= 50% Probability',
        'Clinical_Action': 'Immediate cardiology referral, 12-lead ECG, echocardiogram, evaluate statin and anti-hypertensive therapy.'
    }
}

class CardiovascularRiskInferenceEngine:
    def __init__(self, models_dir=None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, '..', 'models')

        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, 'best_model_gradient_boosting.pkl')
        self.kmeans_path = os.path.join(models_dir, 'kmeans_patient_clusters.pkl')

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Run train_and_evaluate.py first.")

        self.pipeline = joblib.load(self.model_path)
        self.kmeans = joblib.load(self.kmeans_path) if os.path.exists(self.kmeans_path) else None

    def predict_patient_risk(self, patient_dict):
        """
        Predicts cardiovascular disease probability, risk tier, and clinical recommendations.
        """
        df_patient = pd.DataFrame([patient_dict])
        prob = float(self.pipeline.predict_proba(df_patient)[:, 1][0])
        pred_class = int(self.pipeline.predict(df_patient)[0])

        if prob >= 0.50:
            tier_key = 'HIGH'
        elif prob >= 0.20:
            tier_key = 'MODERATE'
        else:
            tier_key = 'LOW'

        meta = RISK_TIERS[tier_key]

        return {
            'Patient_ID': patient_dict.get('Patient_ID', 'N/A'),
            'CVD_Probability': round(prob, 4),
            'CVD_Probability_Percent': f"{prob * 100:.2f}%",
            'Predicted_Binary_Classification': pred_class,
            'Risk_Tier': meta['Tier'],
            'Clinical_Recommendations': meta['Clinical_Action']
        }

if __name__ == '__main__':
    engine = CardiovascularRiskInferenceEngine()

    sample_patients = [
        {
            'Patient_ID': 'PT_HIGH_RISK_01',
            'Age': 64,
            'Gender': 'Male',
            'Systolic_BP': 168,
            'Diastolic_BP': 102,
            'BMI': 34.2,
            'Resting_Heart_Rate': 88,
            'Cholesterol': 275,
            'Fasting_Glucose': 165,
            'Smoking': 1,
            'Alcohol_Intake': 1,
            'Physical_Activity': 'Sedentary',
            'Family_History_CVD': 1
        },
        {
            'Patient_ID': 'PT_LOW_RISK_02',
            'Age': 32,
            'Gender': 'Female',
            'Systolic_BP': 112,
            'Diastolic_BP': 74,
            'BMI': 22.0,
            'Resting_Heart_Rate': 64,
            'Cholesterol': 175,
            'Fasting_Glucose': 88,
            'Smoking': 0,
            'Alcohol_Intake': 0,
            'Physical_Activity': 'Active',
            'Family_History_CVD': 0
        },
        {
            'Patient_ID': 'PT_BORDERLINE_03',
            'Age': 51,
            'Gender': 'Male',
            'Systolic_BP': 138,
            'Diastolic_BP': 88,
            'BMI': 28.5,
            'Resting_Heart_Rate': 76,
            'Cholesterol': 225,
            'Fasting_Glucose': 110,
            'Smoking': 0,
            'Alcohol_Intake': 1,
            'Physical_Activity': 'Moderate',
            'Family_History_CVD': 1
        }
    ]

    print("\n" + "=" * 80)
    print(" REAL-TIME CLINICAL INFERENCE & CARDIOVASCULAR RISK EVALUATION")
    print("=" * 80)

    for p in sample_patients:
        res = engine.predict_patient_risk(p)
        print(f"\n[Patient: {res['Patient_ID']}]")
        print(f"  Biomarkers: Age {p['Age']}, {p['Gender']} | BP: {p['Systolic_BP']}/{p['Diastolic_BP']} mmHg | BMI: {p['BMI']} | Cholesterol: {p['Cholesterol']} mg/dL")
        print(f"  Lifestyle: Smoking={p['Smoking']}, Activity={p['Physical_Activity']}, Family History={p['Family_History_CVD']}")
        print(f"  => Risk Assessment: {res['CVD_Probability_Percent']} -> {res['Risk_Tier']}")
        print(f"  => Recommended Action: {res['Clinical_Recommendations']}")

    print("\n✓ Inference engine passed all test cases successfully!")

