"""
Preprocessing Pipeline Definition
Handles clinical feature imputation, categorical encoding, and feature scaling.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor():
    numerical_features = [
        'Age', 'Systolic_BP', 'Diastolic_BP', 'BMI',
        'Resting_Heart_Rate', 'Cholesterol', 'Fasting_Glucose'
    ]

    binary_features = [
        'Smoking', 'Alcohol_Intake', 'Family_History_CVD'
    ]

    categorical_features = [
        'Gender', 'Physical_Activity'
    ]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    bin_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_features),
            ('bin', bin_pipeline, binary_features),
            ('cat', cat_pipeline, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor, numerical_features, binary_features, categorical_features

