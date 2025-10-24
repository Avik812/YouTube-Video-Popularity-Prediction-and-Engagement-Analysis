"""
data_preprocessing.py
Preprocess scraped and API YouTube datasets for the
YouTube Video Popularity Prediction and Engagement Analysis project.

Run directly:
    (.venv) python data_preprocessing.py

This will:
 - Load scraped.jsonl and api.jsonl
 - Clean and engineer features
 - Align and merge datasets
 - Save cleaned CSVs to ./cleaned_output/
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
import isodate
import numpy as np
import pandas as pd
from dateutil import parser as dtparser

# -------------------------
# Helper parsing functions
# -------------------------

def parse_views_text(views_text):
    if pd.isna(views_text):
        return np.nan
    s = str(views_text).lower().replace('views', '').strip()
    if s in ['', 'no views']:
        return 0
    s = s.replace(',', '').replace(' ', '')
    try:
        if s.endswith('k'):
            return int(float(s[:-1]) * 1_000)
        if s.endswith('m'):
            return int(float(s[:-1]) * 1_000_000)
        if s.endswith('b'):
            return int(float(s[:-1]) * 1_000_000_000)
        return int(float(s))
    except:
        return np.nan

def parse_upload_ago(upload_text):
    if pd.isna(upload_text):
        return np.nan
    s = str(upload_text).lower()
    m = re.search(r'(\d+)\s*(hour|hr|h)s?', s)
    if m: return int(m.group(1)) / 24
    m = re.search(r'(\d+)\s*(minute|min|m)s?', s)
    if m: return int(m.group(1)) / (24 * 60)
    m = re.search(r'(\d+)\s*(day|d)s?', s)
    if m: return int(m.group(1))
    m = re.search(r'(\d+)\s*(week|w)s?', s)
    if m: return int(m.group(1)) * 7
    m = re.search(r'(\d+)\s*(month|mo)s?', s)
    if m: return int(m.group(1)) * 30
    m = re.search(r'(\d+)\s*(year|y)s?', s)
    if m: return int(m.group(1)) * 365
    m = re.search(r'(\d+)', s)
    if m: return float(m.group(1))
    return np.nan

def parse_duration_mm(duration_text):
    if pd.isna(duration_text): return np.nan
    parts = str(duration_text).split(':')
    try:
        if len(parts) == 2:
            m, s = map(int, parts)
            return m + s/60
        elif len(parts) == 3:
            h, m, s = map(int, parts)
            return h*60 + m + s/60
        elif str(duration_text).isdigit():
            return float(duration_text)/60
    except:
        return np.nan
    return np.nan

def parse_iso_duration_minutes(iso_duration):
    try:
        td = isodate.parse_duration(str(iso_duration))
        return td.total_seconds() / 60
    except:
        return np.nan

def extract_video_id_from_url(url):
    if pd.isna(url): return None
    url = str(url)
    for pat in [r'/shorts/([A-Za-z0-9_-]{8,})', r'youtu\.be/([A-Za-z0-9_-]{8,})', r'v=([A-Za-z0-9_-]{8,})']:
        m = re.search(pat, url)
        if m: return m.group(1)
    return None

def safe_len(x): return len(str(x)) if pd.notna(x) else 0
def count_hashtags(t): return str(t).count('#') if pd.notna(t) else 0

# -------------------------
# Preprocessing functions
# -------------------------

def preprocess_scraped(df):
    df['views'] = df['views_text'].apply(parse_views_text)
    df['days_since_upload'] = df['upload_text'].apply(parse_upload_ago)
    df['duration_minutes'] = df['duration_text'].apply(parse_duration_mm)
    df['videoId'] = df['url'].apply(extract_video_id_from_url)
    df['title_length'] = df['title'].apply(safe_len)
    df['num_hashtags'] = df['title'].apply(count_hashtags)
    df['was_scraped'] = True
    df['was_api'] = False
    return df

def preprocess_api(df):
    df['duration_minutes'] = df['duration'].apply(parse_iso_duration_minutes)
    df['publishedAt_dt'] = pd.to_datetime(df['publishedAt'], utc=True, errors='coerce')
    now = datetime.now(timezone.utc)
    df['days_since_upload'] = (now - df['publishedAt_dt']).dt.total_seconds() / (24*3600)
    df['viewCount'] = pd.to_numeric(df['viewCount'], errors='coerce')
    df['likeCount'] = pd.to_numeric(df['likeCount'], errors='coerce').fillna(0)
    df['commentCount'] = pd.to_numeric(df['commentCount'], errors='coerce').fillna(0)
    df['engagement_rate'] = (df['likeCount'] + df['commentCount']) / df['viewCount'].replace(0, np.nan)
    df['title_length'] = df['title'].apply(safe_len)
    df['tag_count'] = df['tags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['was_scraped'] = False
    df['was_api'] = True
    return df

def align_and_merge(scraped, api):
    merged = pd.merge(scraped, api, on='videoId', how='outer', suffixes=('_scraped', '_api'))
    merged['views_canonical'] = merged['viewCount'].fillna(merged['views'])
    return merged

# -------------------------
# Main
# -------------------------

def main():
    in_scraped = Path("data-collection/data/scraped_data.jsonl")
    in_api = Path("data-collection/data/api_data.jsonl")
    outdir = Path("cleaned_output")
    outdir.mkdir(exist_ok=True)

    print("Loading input data...")
    df_scraped = pd.read_json(in_scraped, lines=True)
    df_api = pd.read_json(in_api, lines=True)

    print(f"Scraped rows: {len(df_scraped)}, API rows: {len(df_api)}")

    print("Preprocessing scraped data...")
    scraped_clean = preprocess_scraped(df_scraped)
    print("Preprocessing API data...")
    api_clean = preprocess_api(df_api)

    print("Merging datasets...")
    combined = align_and_merge(scraped_clean, api_clean)

    scraped_clean.to_csv("data-preprocessing/data/scraped_clean.csv", index=False)
    api_clean.to_csv("data-preprocessing/data/api_clean.csv", index=False)
    combined.to_csv("data-preprocessing/data/combined_canonical.csv", index=False)

    print("\nPreprocessing complete!")
    print(f"Cleaned files saved in: {outdir.resolve()}")
    print("\nSamples:")
    print(scraped_clean.head(2))
    print(api_clean.head(2))
    print(combined.head(2))

if __name__ == "__main__":
    main()