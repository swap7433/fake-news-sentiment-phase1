# src/tweepy_fetch.py
import os
import pandas as pd
from pathlib import Path
import tweepy

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = DATA_DIR / "live_tweets.csv"

BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")  # recommended to set in env

def fetch_tweets_bearer(query="fake news OR misinformation OR hoax lang:en", max_results=500):
    if BEARER_TOKEN is None:
        raise ValueError("Set environment variable TWITTER_BEARER_TOKEN or edit this file with your token.")
    client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)
    tweets = []
    # use Paginator to fetch multiple pages (max 100 per request)
    for response in tweepy.Paginator(client.search_recent_tweets,
                                     query=query,
                                     tweet_fields=['created_at','public_metrics','lang','text'],
                                     max_results=100):
        if not response or not getattr(response, "data", None):
            continue
        for t in response.data:
            if t.lang != "en":
                continue
            pm = t.public_metrics
            tweets.append({
                "id": t.id,
                "date": t.created_at.isoformat(),
                "text": t.text,
                "retweets": pm.get("retweet_count"),
                "replies": pm.get("reply_count"),
                "likes": pm.get("like_count"),
                "quotes": pm.get("quote_count")
            })
            if len(tweets) >= max_results:
                break
        if len(tweets) >= max_results:
            break

    df = pd.DataFrame(tweets)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(df)} tweets to {OUT_CSV}")
    return df

if __name__ == "__main__":
    fetch_tweets_bearer()
