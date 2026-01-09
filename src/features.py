from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from .config import TFIDF_PATH
import os

def build_tfidf(corpus, max_features=10000, ngram_range=(1,2)):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vec.fit_transform(corpus)
    # ensure model dir exists
    joblib.dump(vec, TFIDF_PATH)
    return vec, X

def load_vectorizer(path=TFIDF_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vectorizer not found at {path}. Train models first.")
    return joblib.load(path)

def transform_texts(vectorizer, texts):
    return vectorizer.transform(texts)
