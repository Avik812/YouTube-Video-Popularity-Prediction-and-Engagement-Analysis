# feature_engineering.py
# -------------------------
# Create engineered features for YouTube Video Popularity Prediction
# -------------------------

import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------
# Helper functions
# -------------------------

def safe_len(x):
    return len(str(x)) if pd.notna(x) else 0

def count_hashtags(x):
    return str(x).count('#') if pd.notna(x) else 0

def keyword_count(text, keywords):
    if pd.isna(text):
        return 0
    text = text.lower()
    return sum(text.count(kw.lower()) for kw in keywords)

# -------------------------
# Load data
# -------------------------

infile = Path("data-preprocessing/data/combined_canonical.csv")
df = pd.read_csv(infile)

print(f"Loaded {len(df)} rows from {infile}")

# -------------------------
# Feature Engineering
# -------------------------

# Title and description features
df['title_length'] = df['title_scraped'].combine_first(df['title_api']).apply(safe_len)
df['num_hashtags'] = df['title_scraped'].combine_first(df['title_api']).apply(count_hashtags)
df['description_length'] = df['description'].fillna('').apply(safe_len)

# Keyword counts (example keywords)
keywords = ['trending', 'fun', 'tutorial', 'challenge', 'news', 'music', 'gaming']
df['keyword_count'] = df['title_scraped'].combine_first(df['title_api']).apply(lambda x: keyword_count(x, keywords))

# Numeric features
df['duration_minutes'] = df['duration_minutes_scraped'].combine_first(df['duration_minutes_api'])
df['days_since_upload'] = df['days_since_upload_scraped'].combine_first(df['days_since_upload_api'])
df['views'] = df['views'].combine_first(df['viewCount'])
df['likeCount'] = df['likeCount'].fillna(0)
df['commentCount'] = df['commentCount'].fillna(0)
df['engagement_rate'] = (df['likeCount'] + df['commentCount']) / df['views'].replace(0, np.nan)

# Category feature
df['categoryId'] = df['categoryId'].fillna(-1)

# Optional: normalize numeric features (Min-Max)
numeric_cols = ['duration_minutes', 'days_since_upload', 'title_length', 'description_length', 'keyword_count', 'views', 'engagement_rate']
for col in numeric_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    if max_val > min_val:
        df[col + '_norm'] = (df[col] - min_val) / (max_val - min_val)
    else:
        df[col + '_norm'] = df[col]

# -------------------------
# Save features
# -------------------------

outfile = "feature-engineering/data/features.csv"
df.to_csv(outfile, index=False)
print(f"Feature engineering complete! Saved features to {outfile}")

# -------------------------
# Optional: preview
# -------------------------
print("\nSample features:")
print(df[numeric_cols + ['keyword_count', 'views', 'engagement_rate']].head(5))