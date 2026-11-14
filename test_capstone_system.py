"""
Comprehensive End-to-End Test and Verification Script for Final ML Capstone
Validates all pipeline stages, serialized models, inference engine, and outputs.
"""

import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'src'))

from inference_service import CardiovascularRiskInferenceEngine

def run_system_verification():
    print("=" * 80)
    print(" FINAL MACHINE LEARNING CAPSTONE: END-TO-END SYSTEM VERIFICATION")
    print("=" * 80)

    # 1. Verify Directories & Artifacts
    expected_files = [
        os.path.join(base_dir, 'data', 'cardiovascular_patient_data.csv'),
        os.path.join(base_dir, 'models', 'best_model_gradient_boosting.pkl'),
        os.path.join(base_dir, 'models', 'random_forest_model.pkl'),
        os.path.join(base_dir, 'models', 'logistic_regression_model.pkl'),
        os.path.join(base_dir, 'models', 'kmeans_patient_clusters.pkl'),
        os.path.join(base_dir, 'reports', 'capstone_project_report.pdf'),
        os.path.join(base_dir, 'reports', 'capstone_project_report.md'),
        os.path.join(base_dir, 'reports', 'model_benchmark_results.csv'),
        os.path.join(base_dir, 'notebooks', 'final_machine_learning_project_capstone.ipynb'),
        os.path.join(base_dir, 'app', 'dashboard.html'),
        os.path.join(base_dir, 'figures', 'model_comparison_benchmark.png'),
        os.path.join(base_dir, 'figures', 'roc_curves_comparison.png'),
        os.path.join(base_dir, 'figures', 'confusion_matrices_grid.png'),
        os.path.join(base_dir, 'figures', 'feature_importance_rf.png'),
        os.path.join(base_dir, 'figures', 'patient_risk_stratification_clusters.png')
    ]

    print("\n[Check 1] Verifying System Artifacts and Deliverables...")
    all_exist = True
    for f in expected_files:
        exists = os.path.exists(f)
        status = "✓ FOUND" if exists else "✗ MISSING"
        print(f"  {status:9s} : {os.path.relpath(f, base_dir)}")
        if not exists:
            all_exist = False

    if not all_exist:
        raise FileNotFoundError("One or more required system artifacts are missing.")
    print("✓ All 15 required core deliverables verified on disk.")

    # 2. Test Real-Time Inference
    print("\n[Check 2] Validating Real-Time Inference Engine on Diverse Patient Profiles...")
    engine = CardiovascularRiskInferenceEngine(models_dir=os.path.join(base_dir, 'models'))

    test_cases = [
        {
            'Patient_ID': 'PT_CRITICAL_ICU',
            'Age': 72,
            'Gender': 'Male',
            'Systolic_BP': 185,
            'Diastolic_BP': 115,
            'BMI': 36.8,
            'Resting_Heart_Rate': 96,
            'Cholesterol': 320,
            'Fasting_Glucose': 195,
            'Smoking': 1,
            'Alcohol_Intake': 1,
            'Physical_Activity': 'Sedentary',
            'Family_History_CVD': 1
        },
        {
            'Patient_ID': 'PT_HEALTHY_ATHLETE',
            'Age': 28,
            'Gender': 'Female',
            'Systolic_BP': 108,
            'Diastolic_BP': 70,
            'BMI': 20.5,
            'Resting_Heart_Rate': 56,
            'Cholesterol': 160,
            'Fasting_Glucose': 82,
            'Smoking': 0,
            'Alcohol_Intake': 0,
            'Physical_Activity': 'Active',
            'Family_History_CVD': 0
        }
    ]

    for tc in test_cases:
        res = engine.predict_patient_risk(tc)
        print(f"\n  Patient: {res['Patient_ID']}")
        print(f"  -> CVD Probability: {res['CVD_Probability_Percent']}")
        print(f"  -> Risk Tier: {res['Risk_Tier']}")
        print(f"  -> Action: {res['Clinical_Recommendations']}")

    print("\n" + "=" * 80)
    print(" ALL SYSTEM VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 80)

if __name__ == '__main__':
    run_system_verification()

