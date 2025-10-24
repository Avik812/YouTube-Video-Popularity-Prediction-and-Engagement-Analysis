"""
Model Development Script
------------------------
Trains regression models on YouTube data to predict views or engagement rate.

Models:
 - Random Forest Regressor
 - XGBoost Regressor

Evaluations:
 - Scraped data only
 - API data only

Visualizations:
 - Feature importance for each model
 - Model performance comparison
 - Engagement trends by category, length, and upload time
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb

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
# Split datasets by source
# ---------------------------
df_scraped = df[df.get('was_scraped_scraped', False) == True]
df_api = df[df.get('was_api_api', False) == True]

X_scraped = df_scraped[feature_cols]
y_scraped = df_scraped[target_col]
X_api = df_api[feature_cols]
y_api = df_api[target_col]

# ---------------------------
# Handle missing values
# ---------------------------
imputer = SimpleImputer(strategy="median")
X_scraped = pd.DataFrame(imputer.fit_transform(X_scraped), columns=feature_cols)
X_api = pd.DataFrame(imputer.fit_transform(X_api), columns=feature_cols)

# ---------------------------
# Standardize features
# ---------------------------
scaler = StandardScaler()
X_scraped_scaled = scaler.fit_transform(X_scraped)
X_api_scaled = scaler.fit_transform(X_api)

# ---------------------------
# Split into train/test sets
# ---------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

X_train_s, X_test_s, y_train_s, y_test_s = split_data(X_scraped_scaled, y_scraped)
X_train_a, X_test_a, y_train_a, y_test_a = split_data(X_api_scaled, y_api)

# ---------------------------
# Initialize models
# ---------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)

# ---------------------------
# Train, evaluate, and collect metrics
# ---------------------------
results = []

def train_and_collect_metrics(X_train, X_test, y_train, y_test, model, model_name, dataset_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Dataset": dataset_name,
        "Model": model_name,
        "MSE": mse,
        "R2": r2
    })

    print(f"{dataset_name} - {model_name}: MSE={mse:.6f}, R2={r2:.6f}")

    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        plt.figure(figsize=(8,4))
        sns.barplot(x=feature_cols, y=importances, palette="coolwarm")
        plt.title(f"{model_name} Feature Importances ({dataset_name})")
        plt.ylabel("Importance")
        plt.xlabel("Feature")
        plt.show()

    return model

# ---------------------------
# Train both models on both datasets
# ---------------------------
print("=== MODEL TRAINING START ===\n")

train_and_collect_metrics(X_train_s, X_test_s, y_train_s, y_test_s, rf_model, "Random Forest", "Scraped")
train_and_collect_metrics(X_train_s, X_test_s, y_train_s, y_test_s, xgb_model, "XGBoost", "Scraped")
train_and_collect_metrics(X_train_a, X_test_a, y_train_a, y_test_a, rf_model, "Random Forest", "API")
train_and_collect_metrics(X_train_a, X_test_a, y_train_a, y_test_a, xgb_model, "XGBoost", "API")

print("\n=== MODEL TRAINING COMPLETE ===")

# ---------------------------
# Compare model performance
# ---------------------------
perf_df = pd.DataFrame(results)
plt.figure(figsize=(8,5))
sns.barplot(data=perf_df, x="Dataset", y="R2", hue="Model", palette="viridis")
plt.title("Model R² Comparison by Dataset")
plt.ylabel("R² Score")
plt.xlabel("Dataset Type")
plt.legend(title="Model")
plt.show()

# ---------------------------
# Engagement trend visualizations
# ---------------------------

# Ensure required columns exist
if "engagement_rate" not in df.columns and "engagement_rate_norm" in df.columns:
    df["engagement_rate"] = df["engagement_rate_norm"]

# Category vs Engagement
if "categoryId" in df.columns:
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df, x="categoryId", y="engagement_rate")
    plt.title("Engagement Rate by Video Category")
    plt.xlabel("Category ID")
    plt.ylabel("Engagement Rate")
    plt.show()

# Video Length vs Engagement
if "duration_minutes" in df.columns:
    df["length_bin"] = pd.cut(df["duration_minutes"],
                              bins=[0,5,10,20,60,200],
                              labels=["0-5","5-10","10-20","20-60","60+"])
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="length_bin", y="engagement_rate")
    plt.title("Engagement Rate by Video Length")
    plt.xlabel("Video Length (minutes)")
    plt.ylabel("Engagement Rate")
    plt.show()

# Upload Month vs Engagement
if "publishedAt" in df.columns:
    df["upload_month"] = pd.to_datetime(df["publishedAt"], errors='coerce').dt.month
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df, x="upload_month", y="engagement_rate")
    plt.title("Engagement Rate by Upload Month")
    plt.xlabel("Month")
    plt.ylabel("Engagement Rate")
    plt.show()

print("\n✅ Visualization complete!")