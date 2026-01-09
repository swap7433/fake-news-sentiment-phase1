import os

import pandas as pd
from sklearn.model_selection import train_test_split
from .config import FAKE_CSV, SENTIMENT_CSV

def load_fake_data(path=FAKE_CSV):
    df = pd.read_csv(path)
    # Expect columns 'text' and 'label' ('fake' or 'real')
    df = df.dropna(subset=['text','label'])
    return df

def load_sentiment_data(path=SENTIMENT_CSV):
    df = pd.read_csv(path)
    df = df.dropna(subset=['text','sentiment'])
    return df

def split_df(df, text_col='text', label_col='label', test_size=0.2, random_state=42):
    X = df[text_col].astype(str)
    y = df[label_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# add function
def load_live_tweets(path=None):
    if path is None:
        from .config import DATA_DIR
        path = os.path.join(DATA_DIR, "live_tweets.csv")
    return pd.read_csv(path)
