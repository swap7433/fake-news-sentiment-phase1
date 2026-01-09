# src/explain.py
import joblib
from src.features import load_vectorizer
from src.config import FAKE_MODEL_PATH, TFIDF_PATH
import numpy as np

def clean_and_return(text):
    from src.preprocess import clean_text
    return clean_text(text)

def top_contributing_words(text, topn=10):
    vec = load_vectorizer()
    X = vec.transform([text])
    clf = joblib.load(FAKE_MODEL_PATH)
    if not hasattr(clf, "coef_"):
        return []
    coefs = clf.coef_[0]
    arr = X.toarray()[0]
    idxs = np.where(arr>0)[0]
    pairs = []
    feat_names = vec.get_feature_names_out()
    for i in idxs:
        pairs.append((feat_names[i], coefs[i]*arr[i]))
    pairs = sorted(pairs, key=lambda x: -abs(x[1]))[:topn]
    return pairs

def get_word_weight_map(text):
    pairs = top_contributing_words(text, topn=200)
    # convert to dict with normalized weights
    d = {w: float(s) for w,s in pairs}
    return d

def get_global_top_words(n=20):
    vec = load_vectorizer()
    feat = vec.get_feature_names_out()
    clf = joblib.load(FAKE_MODEL_PATH)
    coefs = clf.coef_[0]
    words_coefs = list(zip(feat, coefs))
    top_fake = sorted(words_coefs, key=lambda x: -x[1])[:n]
    top_real = sorted(words_coefs, key=lambda x: x[1])[:n]
    return top_fake, top_real
