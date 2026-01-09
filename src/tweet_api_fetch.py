# src/tweet_api_fetch.py
import tweepy
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR

OUT_CSV = Path(DATA_DIR) / "tweet_api_tweets.csv"

def fetch_with_api(bearer_token, query, max_results=100):
    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    tweets = []
    # recent_search returns up to 100 per request; we can paginate (Paginator)
    for response in tweepy.Paginator(client.search_recent_tweets,
                                     query=query,
                                     tweet_fields=['created_at','public_metrics','text','lang'],
                                     max_results=100):
        for t in response.data or []:
            if t.lang != 'en':
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
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} tweets to {OUT_CSV}")
    return df
