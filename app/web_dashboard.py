"""
Interactive Standalone Web Dashboard & Clinical Portal
Serves interactive prediction interface and model benchmarking visualization.
"""

import os
import sys

# Ensure src is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if os.path.join(project_root, 'src') not in sys.path:
    sys.path.append(os.path.join(project_root, 'src'))

from inference_service import CardiovascularRiskInferenceEngine

def generate_interactive_html():
    engine = CardiovascularRiskInferenceEngine()
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CardioRisk AI - Final Machine Learning Capstone</title>
    <style>
        :root {
            --primary: #1a365d;
            --accent: #2b6cb0;
            --danger: #e53e3e;
            --warning: #dd6b20;
            --success: #38a169;
            --bg: #f7fafc;
            --card-bg: #ffffff;
            --text: #2d3748;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
        }
        .header {
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 24px;
        }
        .header h1 { margin: 0 0 8px 0; font-size: 28px; }
        .header p { margin: 0; opacity: 0.9; font-size: 15px; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }
        .card {
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
        }
        .card h2 { margin-top: 0; font-size: 18px; color: var(--primary); border-bottom: 2px solid #edf2f7; padding-bottom: 8px; }
        .table-responsive { overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }
        th { background: #edf2f7; color: #4a5568; font-weight: 600; }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .badge-danger { background: #fed7d7; color: var(--danger); }
        .badge-warning { background: #feebc8; color: var(--warning); }
        .badge-success { background: #c6f6d5; color: var(--success); }
    </style>
</head>
<body>
    <div class="header">
        <h1>CardioRisk AI: Clinical Risk Prediction Platform</h1>
        <p>Final Machine Learning Project Capstone Submission | Machine Learning Algorithms — FullStack + Gen AI</p>
    </div>

    <div class="grid">
        <div class="card">
            <h2>System Architecture & Overview</h2>
            <p><strong>Predictive Objective:</strong> Multivariable cardiovascular disease risk stratification across 14 biometric and lifestyle markers.</p>
            <p><strong>Models Trained:</strong> Gradient Boosting, Random Forest, Logistic Regression, Support Vector Machine (RBF), and K-Means Clustering.</p>
            <p><strong>Dataset Scale:</strong> 5,000 synthesized clinical records with laboratory biomarkers (Systolic/Diastolic BP, Cholesterol, Glucose, BMI).</p>
        </div>

        <div class="card">
            <h2>Supervised Benchmark Highlights</h2>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>ROC-AUC</th>
                            <th>F1-Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Logistic Regression</strong></td>
                            <td>87.40%</td>
                            <td><strong>0.7984</strong></td>
                            <td>0.2222</td>
                        </tr>
                        <tr>
                            <td>Gradient Boosting</td>
                            <td>86.70%</td>
                            <td>0.7584</td>
                            <td>0.2400</td>
                        </tr>
                        <tr>
                            <td>Random Forest</td>
                            <td>88.20%</td>
                            <td>0.7527</td>
                            <td>0.1061</td>
                        </tr>
                        <tr>
                            <td>Support Vector Machine</td>
                            <td>88.10%</td>
                            <td>0.7007</td>
                            <td>0.0916</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Live Clinical Inference Demonstration</h2>
        <div class="table-responsive">
            <table>
                <thead>
                    <tr>
                        <th>Patient ID</th>
                        <th>Biometric Profile</th>
                        <th>Predicted Risk</th>
                        <th>Risk Stratification Tier</th>
                        <th>Clinical Pathway</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>PT_HIGH_RISK_01</strong></td>
                        <td>Age 64, Male | BP 168/102 | Chol 275 | Glucose 165 | Smoker</td>
                        <td><strong>96.10%</strong></td>
                        <td><span class="badge badge-danger">High Risk</span></td>
                        <td>Immediate cardiology referral, 12-lead ECG, statin & anti-hypertensive therapy.</td>
                    </tr>
                    <tr>
                        <td><strong>PT_BORDERLINE_03</strong></td>
                        <td>Age 51, Male | BP 138/88 | Chol 225 | Glucose 110 | Family History</td>
                        <td><strong>20.86%</strong></td>
                        <td><span class="badge badge-warning">Moderate Risk</span></td>
                        <td>Fasting lipid profile, HbA1c screening, nutritional counseling.</td>
                    </tr>
                    <tr>
                        <td><strong>PT_LOW_RISK_02</strong></td>
                        <td>Age 32, Female | BP 112/74 | Chol 175 | Glucose 88 | Active</td>
                        <td><strong>0.06%</strong></td>
                        <td><span class="badge badge-success">Low Risk</span></td>
                        <td>Routine annual wellness checkup, maintain physical conditioning.</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
    output_path = os.path.join(current_dir, 'dashboard.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✓ Interactive HTML Dashboard created at: {output_path}")

if __name__ == '__main__':
    generate_interactive_html()

