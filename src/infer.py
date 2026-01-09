# src/infer.py (debug-ready)
import joblib
import sys
from src.preprocess import clean_text
from src.features import load_vectorizer
from src.utils import load_model
from src.config import FAKE_MODEL_PATH, SENT_MODEL_PATH
from nltk.sentiment import SentimentIntensityAnalyzer

# Heuristic keywords (expandable)
FAKE_KEYWORDS = {
    "hoax", "fake", "scam", "misinformation", "false",
    "rumor", "fraud", "not true", "conspiracy", "lies",
    "viral message", "whatsapp forward", "unverified", "fake news", "no evidence"
}

def heuristic_is_fake(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    # check whole keywords first (multi-word)
    for kw in sorted(FAKE_KEYWORDS, key=lambda x: -len(x)):
        if kw in t:
            return True
    return False

def predict_fake_news(text: str):
    cleaned = clean_text(text)
    vec = load_vectorizer()
    X = vec.transform([cleaned])

    clf = load_model(FAKE_MODEL_PATH)

    # Probabilities if available
    fake_conf = 1.0
    proba = None
    try:
        proba = clf.predict_proba(X)[0]
        fake_conf = float(max(proba))
    except Exception:
        try:
            # fallback to decision_function magnitude if available
            if hasattr(clf, "decision_function"):
                score = clf.decision_function(X)
                fake_conf = float(abs(score[0]))  # not normalized
        except Exception:
            fake_conf = 1.0

    fake_pred = clf.predict(X)[0]
    return fake_pred, fake_conf, cleaned, proba

def predict_sentiment(text: str):
    # load VADER
    sia: SentimentIntensityAnalyzer = joblib.load(SENT_MODEL_PATH)
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return sentiment, abs(compound), scores

def predict_all(text: str):
    # Debug: print entry
    print("=== DEBUG PREDICT_ALL ===")
    print("Original text (first 200 chars):", repr(text[:200]))

    # 1) Heuristic check on original text
    heur = heuristic_is_fake(text)
    print("Heuristic check result:", heur)

    if heur:
        cleaned = clean_text(text)
        print("Heuristic triggered. Returning fake with high confidence.")
        return {
            "fake_pred": "fake",
            "fake_confidence": 0.90,
            "sentiment_pred": "neutral",
            "sentiment_confidence": 0.5,
            "cleaned_text": cleaned
        }

    # 2) ML-based prediction
    fake_pred, fake_conf, cleaned, proba = predict_fake_news(text)
    print("Cleaned text used for model:", cleaned)
    if proba is not None:
        print("Model predict_proba:", proba)
    print("Model prediction:", fake_pred, "confidence:", fake_conf)

    # 3) Sentiment
    sent_pred, sent_conf, sent_scores = predict_sentiment(text)
    print("Sentiment:", sent_pred, "scores:", sent_scores)
    print("=== END DEBUG ===")

    return {
        "fake_pred": fake_pred,
        "fake_confidence": fake_conf,
        "sentiment_pred": sent_pred,
        "sentiment_confidence": sent_conf,
        "cleaned_text": cleaned
    }
