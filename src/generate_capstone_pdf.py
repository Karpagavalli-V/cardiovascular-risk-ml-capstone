"""
Generates publication-quality Capstone Project PDF report for Final ML Project using reportlab.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    pdf_path = os.path.join(project_root, 'reports', 'capstone_project_report.pdf')
    figures_dir = os.path.join(project_root, 'figures')

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36, leftMargin=36,
        topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=19,
        leading=23,
        textColor=colors.HexColor('#1a365d'),
        alignment=1,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#2b6cb0'),
        alignment=1,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2c5282'),
        spaceBefore=10,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#2d3748'),
        leftIndent=12,
        spaceAfter=3
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("End-to-End Predictive Health Analytics & Cardiovascular Risk Stratification", title_style))
    story.append(Paragraph("Final Machine Learning Capstone Project | Machine Learning Algorithms — FullStack + Gen AI", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#cbd5e0'), spaceAfter=8))

    # Section 1: Executive Summary
    story.append(Paragraph("1. Executive Summary & Clinical Problem Scope", h1_style))
    story.append(Paragraph(
        "Cardiovascular diseases represent the single largest contributor to premature adult mortality globally. "
        "Standard epidemiological risk estimators often fail to capture complex non-linear interactions across multivariable biomarkers. "
        "This Capstone project engineers a production-ready Machine Learning system integrating multi-model supervised classification, "
        "unsupervised patient clustering, robust data imputation, and real-time clinical pathway recommendation.",
        body_style
    ))

    # Section 2: Data Engineering & Pipeline
    story.append(Paragraph("2. Data Architecture & Preprocessing Pipeline", h1_style))
    story.append(Paragraph(
        "• <b>Cohort Scale:</b> 5,000 synthesized patient records modeled on NHANES and Framingham epidemiological distributions.<br/>"
        "• <b>Features:</b> Demographics (Age, Gender), Vitals (Systolic/Diastolic BP, BMI, Resting HR), Lab Panels (Cholesterol, Fasting Glucose), and Behavioral Attributes (Smoking, Alcohol, Physical Activity, Family History).<br/>"
        "• <b>Preprocessing:</b> Automated ColumnTransformer pipeline with median imputation for continuous parameters, most frequent imputation for categorical variables, OneHotEncoding with drop-first, and StandardScaler normalization.",
        bullet_style
    ))

    # Section 3: Benchmark Table
    story.append(Paragraph("3. Model Benchmark & Statistical Performance", h1_style))
    metrics_data = [
        ['Classifier Algorithm', 'CV ROC-AUC', 'Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        ['Logistic Regression (L2)', '0.8241', '87.40%', '46.15%', '14.63%', '0.2222', '0.7984'],
        ['Gradient Boosting (Tuned)', '0.7926', '86.70%', '40.38%', '17.07%', '0.2400', '0.7584'],
        ['Random Forest (Ensemble)', '0.7895', '88.20%', '77.78%', '5.69%', '0.1061', '0.7527'],
        ['Support Vector Machine (RBF)', '0.7127', '88.10%', '75.00%', '4.88%', '0.0916', '0.7007'],
        ['Decision Tree (Cart)', '0.7153', '87.00%', '37.93%', '8.94%', '0.1447', '0.6995']
    ]
    t = Table(metrics_data, colWidths=[140, 65, 70, 60, 55, 55, 65])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#edf2f7')])
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # Section 4: Visualizations
    story.append(Paragraph("4. Model Diagnostic Suite & Risk Stratification", h1_style))
    roc_img = os.path.join(figures_dir, 'roc_curves_comparison.png')
    if os.path.exists(roc_img):
        story.append(Image(roc_img, width=480, height=210))
        story.append(Spacer(1, 6))

    cluster_img = os.path.join(figures_dir, 'patient_risk_stratification_clusters.png')
    if os.path.exists(cluster_img):
        story.append(Image(cluster_img, width=480, height=190))
        story.append(Spacer(1, 6))

    # Section 5: Clinical Deployment Recommendations
    story.append(Paragraph("5. Production Inference & Clinical Pathway Guidelines", h1_style))
    story.append(Paragraph(
        "<b>• Low Risk Tier (< 20% Probability):</b> Maintain standard annual wellness checkups and primary preventative habits.<br/>"
        "<b>• Moderate Risk Tier (20% - 49% Probability):</b> High-risk lifestyle counseling, intensive dietary modification (DASH protocol), smoking cessation, and 6-month lipid/glucose re-evaluation.<br/>"
        "<b>• High Risk Tier (>= 50% Probability):</b> Urgent clinical escalation: 12-lead resting electrocardiogram (ECG), echocardiogram, coronary artery calcium (CAC) scoring, and pharmacological evaluation (statins, ACE inhibitors).",
        bullet_style
    ))

    doc.build(story)
    print(f"✓ Publication-grade Capstone PDF report generated: {pdf_path}")

if __name__ == '__main__':
    generate_pdf()

