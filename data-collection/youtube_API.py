import os, time, json
from googleapiclient.discovery import build
from tqdm import tqdm

# Read the API key from an environment variable. Do NOT hardcode keys in source.
# Set in zsh: export YOUTUBE_API_KEY="YOUR_API_KEY"
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

def get_video_ids_from_search(q, max_results=50, regionCode="US"):
    ids = []
    nextToken = None
    while len(ids) < max_results:
        resp = youtube.search().list(
            q=q, part="id", type="video", maxResults=min(50, max_results - len(ids)),
            pageToken=nextToken, regionCode=regionCode
        ).execute()
        for it in resp.get("items", []):
            ids.append(it["id"]["videoId"])
        nextToken = resp.get("nextPageToken")
        if not nextToken:
            break
        time.sleep(0.1)
    return ids

def fetch_videos(video_ids):
    results = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        resp = youtube.videos().list(part="snippet,contentDetails,statistics", id=",".join(chunk)).execute()
        for v in resp.get("items", []):
            d = {}
            snip = v.get("snippet", {})
            stats = v.get("statistics", {})
            cd = v.get("contentDetails", {})
            d.update({
                "videoId": v.get("id"),
                "title": snip.get("title"),
                "description": snip.get("description"),
                "tags": snip.get("tags", []),
                "publishedAt": snip.get("publishedAt"),
                "channelId": snip.get("channelId"),
                "channelTitle": snip.get("channelTitle"),
                "categoryId": snip.get("categoryId"),
                "duration": cd.get("duration"),
                "viewCount": int(stats.get("viewCount", 0)),
                "likeCount": int(stats.get("likeCount", 0)) if "likeCount" in stats else None,
                "commentCount": int(stats.get("commentCount", 0)) if "commentCount" in stats else None
            })
            results.append(d)
        time.sleep(0.1)
    return results

if __name__ == "__main__":
    queries = ["trending", "music", "gaming", "news"]
    all_vids = []
    for q in queries:
        ids = get_video_ids_from_search(q, max_results=10)  # tune to reach 3000
        vids = fetch_videos(ids)
        all_vids.extend(vids)
    with open("data-collection/data/api_data.jsonl", "w", encoding="utf-8") as f:
        for row in all_vids:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Saved", len(all_vids))
