"""
Final Machine Learning Capstone: Model Training & Evaluation Engine
Trains and benchmarks 5 Supervised Classifiers + Unsupervised K-Means Risk Phenotyping.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

from data_generator import generate_cardiovascular_dataset
from pipeline import get_preprocessor

def run_capstone_training():
    print("=" * 85)
    print(" FINAL MACHINE LEARNING PROJECT: END-TO-END PREDICTIVE HEALTH INTELLIGENCE ENGINE")
    print(" DOMAIN: CARDIOVASCULAR DISEASE RISK PREDICTION & PATIENT PHENOTYPING")
    print("=" * 85)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))

    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, 'models')
    figures_dir = os.path.join(project_root, 'figures')
    reports_dir = os.path.join(project_root, 'reports')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, 'cardiovascular_patient_data.csv')

    # 1. Dataset Ingestion
    if not os.path.exists(csv_path):
        print("\n[Step 1] Synthesizing comprehensive clinical dataset (5,000 patient records)...")
        df = generate_cardiovascular_dataset(n_samples=5000, random_state=42)
        df.to_csv(csv_path, index=False)
    else:
        print(f"\n[Step 1] Loading existing dataset from {csv_path}...")
        df = pd.read_csv(csv_path)

    print(f"✓ Dataset loaded successfully: {df.shape[0]} patients, {df.shape[1]} clinical biomarkers.")
    target_dist = df['Cardiovascular_Disease'].value_counts()
    print(f"Target Distribution: Non-CVD (0) = {target_dist[0]}, CVD Positive (1) = {target_dist[1]} (Prevalence: {df['Cardiovascular_Disease'].mean()*100:.1f}%)")

    # 2. Features and Target Definition
    preprocessor, num_cols, bin_cols, cat_cols = get_preprocessor()
    all_features = num_cols + bin_cols + cat_cols
    target = 'Cardiovascular_Disease'

    X = df[all_features]
    y = df[target]

    # 3. Train-Test Split (Stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[Step 2] Train-Test Split: Train = {len(X_train)} samples, Test = {len(X_test)} samples (Stratified).")

    # 4. Supervised Multi-Model Benchmark Suite
    candidate_classifiers = {
        'Logistic Regression': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, min_samples_split=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=8, random_state=42, n_jobs=1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42),
        'Support Vector Machine (RBF)': SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
    }

    print("\n[Step 3] Training & Cross-Validating Candidate Machine Learning Models...")
    benchmark_records = []
    trained_pipelines = {}

    for name, clf in candidate_classifiers.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # 3-Fold Stratified Cross-Validation on Train
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=1)
        mean_cv_auc = np.mean(cv_scores)

        # Fit model on full training set
        pipe.fit(X_train, y_train)

        # Predict on holdout test set
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        benchmark_records.append({
            'Model': name,
            'CV_ROC_AUC': round(mean_cv_auc, 4),
            'Test_Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4),
            'ROC_AUC': round(roc_auc, 4),
            'PR_AUC': round(pr_auc, 4)
        })

        trained_pipelines[name] = (pipe, y_pred, y_prob)
        print(f"  ✓ {name:30s} | Test Acc: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} (CV AUC: {mean_cv_auc:.4f})")

    benchmark_df = pd.DataFrame(benchmark_records).sort_values(by='ROC_AUC', ascending=False)
    print("\n" + "=" * 85)
    print(" MODEL PERFORMANCE BENCHMARK MATRIX")
    print("=" * 85)
    print(benchmark_df.to_string(index=False))

    benchmark_csv = os.path.join(reports_dir, 'model_benchmark_results.csv')
    benchmark_df.to_csv(benchmark_csv, index=False)

    # 5. Model Serialization
    best_model_name = benchmark_df.iloc[0]['Model']
    best_pipe = trained_pipelines[best_model_name][0]
    print(f"\n[Step 4] Best Performing Classifier: '{best_model_name}' (ROC-AUC = {benchmark_df.iloc[0]['ROC_AUC']})")

    # Save best model and individual pipelines
    best_model_file = os.path.join(models_dir, 'best_model_gradient_boosting.pkl')
    joblib.dump(best_pipe, best_model_file)

    joblib.dump(trained_pipelines['Random Forest'][0], os.path.join(models_dir, 'random_forest_model.pkl'))
    joblib.dump(trained_pipelines['Logistic Regression'][0], os.path.join(models_dir, 'logistic_regression_model.pkl'))
    print(f"✓ Saved serialized production models to {models_dir}")

    # 6. Unsupervised Patient Risk Clustering (K-Means)
    print("\n[Step 5] Training Unsupervised K-Means for Patient Risk Stratification (k=3)...")
    # Extract preprocessed training features
    preprocessed_X = best_pipe.named_steps['preprocessor'].fit_transform(X)
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=20, random_state=42)
    cluster_labels = kmeans.fit_predict(preprocessed_X)
    joblib.dump(kmeans, os.path.join(models_dir, 'kmeans_patient_clusters.pkl'))
    print("✓ Saved K-Means patient clustering model.")

    # 7. Diagnostic Visualizations Suite
    print("\n[Step 6] Generating High-Resolution Diagnostic Visualizations...")

    # Plot 1: Model Benchmark Multi-Bar Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_to_plot = ['Test_Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    x = np.arange(len(benchmark_df))
    width = 0.15

    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + (i - 2) * width, benchmark_df[metric], width, label=metric.replace('_', ' '), alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Score (0.0 - 1.0)', fontsize=12)
    ax.set_title('Supervised Classification Model Benchmark Comparison', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmark_df['Model'], rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'model_comparison_benchmark.png'), dpi=200)
    plt.close()

    # Plot 2: ROC Curves Comparison
    plt.figure(figsize=(9, 7))
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, (m_name, (_, _, y_prob)) in enumerate(trained_pipelines.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, lw=2.2, color=palette[idx], label=f'{m_name} (AUC = {auc_val:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'roc_curves_comparison.png'), dpi=200)
    plt.close()

    # Plot 3: Confusion Matrix Grid
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    top_models = ['Gradient Boosting', 'Random Forest', 'Logistic Regression']
    for i, mod_name in enumerate(top_models):
        _, y_p, _ = trained_pipelines[mod_name]
        cm = confusion_matrix(y_test, y_p)
        axes[i].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[i].set_title(f'{mod_name}', fontweight='bold', fontsize=12)
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(['Pred No CVD', 'Pred CVD'])
        axes[i].set_yticklabels(['Actual No CVD', 'Actual CVD'])
        for r in range(2):
            for c in range(2):
                axes[i].text(c, r, f"{cm[r, c]}", ha='center', va='center',
                             color='white' if cm[r, c] > cm.max()/2 else 'black', fontweight='bold', fontsize=12)
    plt.suptitle('Confusion Matrix Comparison Across Top Classifiers', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'confusion_matrices_grid.png'), dpi=200)
    plt.close()

    # Plot 4: Feature Importance (Random Forest & Gradient Boosting)
    rf_clf = trained_pipelines['Random Forest'][0].named_steps['classifier']
    cat_enc = trained_pipelines['Random Forest'][0].named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
    cat_names = list(cat_enc.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + bin_cols + cat_names

    importances = rf_clf.feature_importances_
    feat_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(feat_df['Feature'], feat_df['Importance'], color='#2b6cb0', edgecolor='black', alpha=0.85)
    plt.title('Clinical Feature Importance for Cardiovascular Risk (Random Forest)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Gini Feature Importance', fontsize=11)
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'feature_importance_rf.png'), dpi=200)
    plt.close()

    # Plot 5: Patient Risk Stratification Clusters (PCA 2D)
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(preprocessed_X)
    plt.figure(figsize=(10, 7))
    cluster_labels_named = ['Low Risk / Baseline Cohort', 'Moderate Risk / Metabolic Warning', 'High Risk / Severe Clinical Warning']
    cluster_colors = ['#2ca02c', '#ff7f0e', '#d62728']

    for c in range(3):
        mask = (cluster_labels == c)
        plt.scatter(pca_coords[mask, 0], pca_coords[mask, 1], c=cluster_colors[c], alpha=0.55, s=25, label=f'{cluster_labels_named[c]} (n={np.sum(mask)})')

    plt.title('Unsupervised Patient Risk Stratification Clusters (PCA 2D)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'patient_risk_stratification_clusters.png'), dpi=200)
    plt.close()

    print(f"✓ All diagnostic charts generated in: {figures_dir}")
    print("=" * 85)
    print(" CAPSTONE TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 85)

if __name__ == '__main__':
    run_capstone_training()
