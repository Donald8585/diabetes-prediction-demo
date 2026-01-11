#!/usr/bin/env python3
"""
Train diabetes prediction model from Kaggle notebook

Steps:
1. Download dataset from Kaggle: https://www.kaggle.com/competitions/playground-series-s5e12
2. Place train.csv in this directory
3. Run: python train_model.py
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

print("🔬 Training Diabetes Prediction Model...")

# Load data
df = pd.read_csv('train.csv')
print(f"Loaded {len(df)} samples")

# Encode categorical variables
categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split features and target
X = df_encoded.drop(['id', 'diagnosed_diabetes'], axis=1)
y = df_encoded['diagnosed_diabetes']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model (using best parameters from GridSearch)
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.2,
    min_child_weight=3,
    random_state=42,
    eval_metric='logloss'
)

print("Training model...")
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import classification_report, roc_auc_score

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nPerformance:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Save model
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✅ Model saved as diabetes_model.pkl")
print("\n🚀 Run: python app.py to test the demo")
