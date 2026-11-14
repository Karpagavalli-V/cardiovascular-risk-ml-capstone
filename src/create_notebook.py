import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
notebook_path = os.path.join(base_dir, '..', 'notebooks', 'final_machine_learning_project_capstone.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Final Machine Learning Project: Capstone Submission\n",
    "## Domain: **End-to-End Clinical Informatics & Cardiovascular Disease Risk Prediction**\n",
    "**Course**: Machine Learning Algorithms — FullStack + Gen AI  \n",
    "\n",
    "---\n",
    "## 1. Project Abstract\n",
    "Cardiovascular diseases (CVDs) remain the leading cause of global mortality. Early, non-invasive risk stratification allows healthcare systems to intervene prior to acute myocardial infarction or stroke.  \n",
    "This Capstone project delivers a comprehensive, production-grade Machine Learning system:\n",
    "- **Data Engineering**: Synthesis and preprocessing of 5,000 clinical records across 14 biometric, laboratory, and behavioral attributes.\n",
    "- **Supervised Learning**: Comparative benchmarking of **Logistic Regression**, **Random Forest**, **Gradient Boosting**, **Support Vector Machines (RBF)**, and **Decision Trees**.\n",
    "- **Unsupervised Learning**: Patient phenotyping via **K-Means Clustering** ($K=3$) and Principal Component Analysis (PCA).\n",
    "- **Statistical Evaluation**: Cross-validated ROC-AUC, PR-AUC, Precision, Recall, F1-Score, and Gini Feature Importance.\n",
    "- **Clinical Deployment**: Exported scikit-learn pipelines, REST service, and real-time risk tiering."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1: Ingesting Clinical Telemetry Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "df = pd.read_csv('../data/cardiovascular_patient_data.csv')\n",
    "print(f\"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\")\n",
    "print(f\"Target Prevalence: {df['Cardiovascular_Disease'].mean()*100:.2f}%\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Exploratory Data Analysis & Biomarker Distributions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n",
    "axes[0].hist(df['Systolic_BP'], bins=30, color='#2b6cb0', edgecolor='black')\n",
    "axes[0].set_title('Systolic BP Distribution')\n",
    "axes[1].hist(df['Cholesterol'].dropna(), bins=30, color='#38a169', edgecolor='black')\n",
    "axes[1].set_title('Total Cholesterol Distribution')\n",
    "axes[2].hist(df['BMI'].dropna(), bins=30, color='#dd6b20', edgecolor='black')\n",
    "axes[2].set_title('BMI Distribution')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3: Multi-Model Supervised Training & Cross-Validation Benchmark\n",
    "\n",
    "Mathematical formulations:\n",
    "- **Logistic Regression Logit**:\n",
    "  $$\\ln\\left(\\frac{p}{1-p}\\right) = \\beta_0 + \\sum_{j=1}^p \\beta_j X_j$$\n",
    "- **Random Forest Ensemble**:\n",
    "  $$\\hat{y} = \\text{mode}\\{T_1(x), T_2(x), \\dots, T_B(x)\\}$$\n",
    "- **Gradient Boosting Loss Minimization**:\n",
    "  $$F_m(x) = F_{m-1}(x) + \\gamma_m h_m(x)$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "benchmark_df = pd.read_csv('../reports/model_benchmark_results.csv')\n",
    "print(\"=== SUPERVISED MODEL BENCHMARK RESULTS ===\")\n",
    "benchmark_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4: Diagnostic Visualizations (ROC Curves & Feature Importance)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.image as mpimg\n",
    "roc_img = mpimg.imread('../figures/roc_curves_comparison.png')\n",
    "plt.figure(figsize=(10, 8))\n",
    "plt.imshow(roc_img)\n",
    "plt.axis('off')\n",
    "plt.title('Multi-Model ROC Curves', fontsize=14, fontweight='bold')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5: Unsupervised Patient Phenotyping via K-Means"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "cluster_img = mpimg.imread('../figures/patient_risk_stratification_clusters.png')\n",
    "plt.figure(figsize=(10, 7))\n",
    "plt.imshow(cluster_img)\n",
    "plt.axis('off')\n",
    "plt.title('Patient Risk Stratification Clusters', fontsize=14, fontweight='bold')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f"✓ Jupyter Notebook created successfully: {notebook_path}")

