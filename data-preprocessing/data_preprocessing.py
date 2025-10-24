import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta, timezone
import isodate  # for parsing ISO 8601 durations from API

# -------------------------------
# 1. Helper functions
# -------------------------------

# Convert views like "9.5M views" to integer
def parse_views(views_text):
    if pd.isna(views_text) or views_text == "":
        return np.nan
    views_text = views_text.replace("views", "").strip()
    multiplier = 1
    if views_text.endswith("K"):
        multiplier = 1_000
        views_text = views_text[:-1]
    elif views_text.endswith("M"):
        multiplier = 1_000_000
        views_text = views_text[:-1]
    try:
        return int(float(views_text) * multiplier)
    except:
        return np.nan

# Convert "2 months ago", "16 hours ago" to days
def parse_upload_text(upload_text):
    if pd.isna(upload_text) or upload_text == "":
        return np.nan
    nums = re.findall(r"\d+", upload_text)
    if not nums:
        return np.nan
    num = int(nums[0])
    if "day" in upload_text:
        return num
    elif "hour" in upload_text:
        return num / 24
    elif "month" in upload_text:
        return num * 30
    elif "year" in upload_text:
        return num * 365
    else:
        return np.nan

# Convert duration "12:10" to minutes
def parse_duration_text(duration_text):
    if pd.isna(duration_text) or duration_text == "":
        return np.nan
    try:
        parts = duration_text.split(":")
        if len(parts) == 2:
            minutes, seconds = int(parts[0]), int(parts[1])
            return minutes + seconds/60
        elif len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            return hours*60 + minutes + seconds/60
        else:
            return np.nan
    except:
        return np.nan

# For API ISO 8601 duration to minutes
def parse_api_duration(duration_iso):
    try:
        td = isodate.parse_duration(duration_iso)
        return td.total_seconds() / 60
    except:
        return np.nan

# Engagement rate
def engagement_rate(likes, comments, views):
    if views == 0 or np.isnan(views):
        return np.nan
    return (likes + comments) / views

# -------------------------------
# 2. Preprocess scraped dataset
# -------------------------------

def preprocess_scraped(df):
    df = df.copy()
    df['views'] = df['views_text'].apply(parse_views)
    df['days_since_upload'] = df['upload_text'].apply(parse_upload_text)
    df['duration_minutes'] = df['duration_text'].apply(parse_duration_text)
    df['title_length'] = df['title'].apply(lambda x: len(x))
    df['num_hashtags'] = df['title'].apply(lambda x: x.count("#"))
    # Optional: fill missing durations with median
    df['duration_minutes'].fillna(df['duration_minutes'].median(), inplace=True)
    return df[['title', 'channel', 'views', 'days_since_upload', 'duration_minutes', 'title_length', 'num_hashtags', 'url']]

# -------------------------------
# 3. Preprocess API dataset
# -------------------------------

def preprocess_api(df):
    df = df.copy()
    df['duration_minutes'] = df['duration'].apply(parse_api_duration)
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df['days_since_upload'] = (datetime.now(timezone.utc) - df['publishedAt']).dt.days    
    df['title_length'] = df['title'].apply(lambda x: len(x))
    df['description_length'] = df['description'].apply(lambda x: len(x) if pd.notna(x) else 0)
    df['tag_count'] = df['tags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['engagement_rate'] = df.apply(lambda row: engagement_rate(row['likeCount'], row['commentCount'], row['viewCount']), axis=1)
    return df[['videoId', 'title', 'channelTitle', 'categoryId', 'viewCount', 'likeCount', 'commentCount',
               'days_since_upload', 'duration_minutes', 'title_length', 'description_length', 'tag_count', 'engagement_rate']]

# -------------------------------
# 4. Load data and apply preprocessing
# -------------------------------

# Example: load from JSON files
scraped_df = pd.read_json("scraped_trending.jsonl", lines=True)
api_df = pd.read_json("api_videos.jsonl", lines=True)

scraped_clean = preprocess_scraped(scraped_df)
api_clean = preprocess_api(api_df)

# -------------------------------
# 5. Save cleaned datasets
# -------------------------------

scraped_clean.to_csv("scraped_clean.csv", index=False)
api_clean.to_csv("api_clean.csv", index=False)

print("Preprocessing complete! Clean CSVs saved.")