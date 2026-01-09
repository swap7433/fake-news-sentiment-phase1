# src/rss_fetch.py
import feedparser
import pandas as pd
from pathlib import Path
import datetime

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = DATA_DIR / "live_news.csv"

RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml"
]

def fetch_news_from_rss():
    rows = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            rows.append({
                "source": feed.feed.get("title", url),
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "published": entry.get("published", ""),
                "link": entry.get("link", "")
            })
    df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["link"], inplace=True)
    df["fetched_at"] = datetime.datetime.utcnow().isoformat()
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(df)} live news articles to {OUT_CSV}")
    return df

if __name__ == "__main__":
    fetch_news_from_rss()
