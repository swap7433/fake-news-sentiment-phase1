# src/live_fetch.py
import snscrape.modules.twitter as sntwitter
import pandas as pd
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = DATA_DIR / "live_tweets.csv"

def fetch_tweets(query: str, max_tweets: int = 500, since=None, until=None):
    """
    Example query: 'covid vaccine since:2023-01-01 until:2023-12-31'
    If since/until provided, you can append to query string.
    """
    tweets = []
    i = 0
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if i >= max_tweets:
            break
        tweets.append({
            "id": tweet.id,
            "date": tweet.date.strftime("%Y-%m-%d %H:%M:%S"),
            "user": tweet.user.username,
            "content": tweet.content,
            "replyCount": tweet.replyCount,
            "retweetCount": tweet.retweetCount,
            "likeCount": tweet.likeCount,
            "quoteCount": tweet.quoteCount,
            "url": tweet.url
        })
        i += 1
    df = pd.DataFrame(tweets)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(df)} tweets to {OUT_CSV}")
    return df

if __name__ == "__main__":
    # example usage
    q = "fake news OR misinformation OR hoax lang:en"
    fetch_tweets(q, max_tweets=200)
