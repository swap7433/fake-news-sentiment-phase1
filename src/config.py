import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FAKE_CSV = os.path.join(DATA_DIR, "fake_news.csv")
SENTIMENT_CSV = os.path.join(DATA_DIR, "sentiment.csv")

TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
FAKE_MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_clf.pkl")
SENT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_clf.pkl")
