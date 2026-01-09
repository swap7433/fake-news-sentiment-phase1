# src/reddit_fetch.py
import praw
import pandas as pd
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT = DATA_DIR / "reddit_posts.csv"

def fetch_reddit(client_id, client_secret, user_agent, subreddit='news', limit=200):
    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent)
    posts = []
    for p in reddit.subreddit(subreddit).hot(limit=limit):
        posts.append({
            "id": p.id,
            "title": p.title,
            "selftext": p.selftext,
            "score": p.score,
            "num_comments": p.num_comments,
            "url": p.url,
            "created_utc": p.created_utc
        })
    df = pd.DataFrame(posts)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} reddit posts to {OUT}")
    return df
