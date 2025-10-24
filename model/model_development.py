# model_development.py
"""
Train regression models on YouTube data to predict views or engagement rate.

Uses:
 - Random Forest Regressor
 - XGBoost Regressor

Compares performance on:
 - Scraped data only
 - API data only
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import matplotlib.pyplot as plt

# ---------------------------
# Load feature-engineered data
# ---------------------------
df = pd.read_csv("feature-engineering/data/features.csv")

# ---------------------------
# Choose target and features
# ---------------------------
target_col = "views_norm"  # or "engagement_rate_norm"
feature_cols = [
    "duration_minutes_norm",
    "days_since_upload_norm",
    "title_length_norm",
    "description_length_norm",
    "keyword_count_norm"
]

# ---------------------------
# Split datasets
# ---------------------------
# Scraped only
df_scraped = df[df['was_scraped_scraped'] == True]
X_scraped = df_scraped[feature_cols]
y_scraped = df_scraped[target_col]

# API only
df_api = df[df['was_api_api'] == True]
X_api = df_api[feature_cols]
y_api = df_api[target_col]

# ---------------------------
# Impute missing values
# ---------------------------
imputer = SimpleImputer(strategy="median")  # fill NaNs with median
X_scraped = pd.DataFrame(imputer.fit_transform(X_scraped), columns=feature_cols)
X_api = pd.DataFrame(imputer.fit_transform(X_api), columns=feature_cols)

# ---------------------------
# Standardize features
# ---------------------------
scaler = StandardScaler()
X_scraped_scaled = scaler.fit_transform(X_scraped)
X_api_scaled = scaler.fit_transform(X_api)

# ---------------------------
# Train/test split helper
# ---------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

X_train_s, X_test_s, y_train_s, y_test_s = split_data(X_scraped_scaled, y_scraped)
X_train_a, X_test_a, y_train_a, y_test_a = split_data(X_api_scaled, y_api)

# ---------------------------
# Train and evaluate helper
# ---------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - MSE: {mse:.6f}, R2: {r2:.6f}")
    
    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        plt.figure(figsize=(8,4))
        plt.bar(feature_cols, importances)
        plt.title(f"{model_name} Feature Importances")
        plt.show()
    
    return model

# ---------------------------
# Initialize models
# ---------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)

# ---------------------------
# Train/evaluate on scraped data
# ---------------------------
print("=== Scraped Data ===")
train_and_evaluate(X_train_s, X_test_s, y_train_s, y_test_s, rf_model, "Random Forest")
train_and_evaluate(X_train_s, X_test_s, y_train_s, y_test_s, xgb_model, "XGBoost")

# ---------------------------
# Train/evaluate on API data
# ---------------------------
print("\n=== API Data ===")
train_and_evaluate(X_train_a, X_test_a, y_train_a, y_test_a, rf_model, "Random Forest")
train_and_evaluate(X_train_a, X_test_a, y_train_a, y_test_a, xgb_model, "XGBoost")