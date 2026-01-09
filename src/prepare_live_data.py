# src/prepare_live_data.py
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INFILE = DATA_DIR / "live_news.csv"
OUTFILE = DATA_DIR / "live_tweets.csv"  # train_live.py expects this filename

FAKE_KEYWORDS = ["fake", "hoax", "rumor", "misinformation", "false"]

def weak_label(text):
    t = str(text).lower()
    for k in FAKE_KEYWORDS:
        if k in t:
            return "fake"
    return "real"

def prepare():
    df = pd.read_csv(INFILE)
    df["content"] = df["title"].fillna("") + " " + df["summary"].fillna("")
    df["label"] = df["content"].apply(weak_label)
    df[["content", "label"]].to_csv(OUTFILE, index=False, encoding="utf-8")
    print(f"Prepared training file saved → {OUTFILE}")

if __name__ == "__main__":
    prepare()
