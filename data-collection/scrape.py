import os
import time, json, traceback
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from tqdm import tqdm

def init_driver():
    print("init_driver: creating Chrome webdriver with headless options")
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64)")
    driver = webdriver.Chrome(options=opts)
    print("init_driver: webdriver created")
    return driver

def parse_video_card(card):
    # card: bs4 element for a video tile from a search/trending page
    title_el = card.select_one("#video-title")
    title = title_el.text.strip() if title_el else None
    url = ("https://www.youtube.com" + title_el['href']) if title_el and title_el.has_attr('href') else None
    meta = card.select_one("#metadata-line")
    meta_text = meta.text.strip() if meta else ""
    # Extract approximate view counts and upload time heuristically
    # YouTube often separates metadata with a bullet character '•' or newlines.
    views = None
    upload = None
    if meta:
        # split on bullets and newlines, keep non-empty parts
        raw_parts = []
        for part in meta_text.replace('\u2022', '\n').split('\n'):
            p = part.strip()
            if p and p != '•':
                raw_parts.append(p)
        # first non-bullet part is usually views (e.g., '324K views' or '17M views')
        if len(raw_parts) >= 1:
            views = raw_parts[0]
        if len(raw_parts) >= 2:
            upload = raw_parts[1]
    # duration
    # duration: look for the thumbnail overlay time label which may include 'SHORTS'
    duration = None
    # multiple possible selectors depending on markup
    dur = card.select_one("ytd-thumbnail-overlay-time-status-renderer span") or card.select_one("span.ytd-thumbnail-overlay-time-status-renderer")
    if dur:
        duration = dur.text.strip()
        # normalize weird unicode bullets
        if duration == '•':
            duration = None
    channel = None
    ch = card.select_one("ytd-channel-name a")
    if ch:
        channel = ch.text.strip()
    return {"title": title, "url": url, "views_text": views, "upload_text": upload, "duration_text": duration, "channel": channel}

def scrape_search(query, max_results=10, out_file="scraped_videos.jsonl"):
    print(f"scrape_search: starting for query={query!r}, max_results={max_results}")
    driver = None
    video_data = []
    try:
        driver = init_driver()
        base = f"https://www.youtube.com/results?search_query={query}"
        print("scrape_search: opening", base)
        driver.get(base)
        print("scrape_search: page opened, sleeping 3s to let content load")
        time.sleep(3)
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        print("scrape_search: initial page height:", last_height)
        while len(video_data) < max_results:
            print(f"scrape_search: loop start - collected {len(video_data)} / {max_results}")
            soup = BeautifulSoup(driver.page_source, "html.parser")
            cards = soup.select("ytd-video-renderer, ytd-grid-video-renderer")
            print(f"scrape_search: found {len(cards)} candidate cards on page")
            for c in cards:
                item = parse_video_card(c)
                if item['url'] and item not in video_data:
                    title_preview = (item['title'][:70] + '...') if item['title'] and len(item['title']) > 70 else item['title']
                    video_data.append(item)
                    print(f"scrape_search: added video: {title_preview!r} total={len(video_data)}")
                    if len(video_data) >= max_results:
                        print("scrape_search: reached max_results")
                        break
            # scroll
            print("scrape_search: scrolling to bottom to load more results")
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            print(f"scrape_search: new page height: {new_height} (last: {last_height})")
            if new_height == last_height:
                print("scrape_search: page height unchanged after scroll — no more content, breaking")
                break
            last_height = new_height
    except Exception as e:
        print("scrape_search: exception encountered:", repr(e))
        traceback.print_exc()
    finally:
        if driver:
            try:
                print("scrape_search: quitting driver")
                driver.quit()
            except Exception:
                print("scrape_search: error quitting driver")
                traceback.print_exc()

    try:
        with open(out_file, "w", encoding="utf-8") as f:
            for row in video_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("Saved", len(video_data), "rows to", out_file)
    except Exception as e:
        print("Error saving results to file:", repr(e))
        traceback.print_exc()

if __name__ == "__main__":
    os.makedirs("data-collection/data", exist_ok=True)
    scrape_search("trending", max_results=10, out_file="data-collection/data/scraped_trending.jsonl") # working example usage for 10 trending videos
