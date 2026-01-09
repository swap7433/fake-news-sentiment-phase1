# src/train_sentiment.py
import joblib
from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path
from src.config import MODEL_DIR, SENT_MODEL_PATH

def train_sentiment_model():
    sia = SentimentIntensityAnalyzer()

    Path(MODEL_DIR).mkdir(exist_ok=True)
    joblib.dump(sia, SENT_MODEL_PATH)

    print(f"Sentiment model saved at {SENT_MODEL_PATH}")

if __name__ == "__main__":
    train_sentiment_model()
