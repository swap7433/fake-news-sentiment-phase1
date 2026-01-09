# src/debug_infer.py
import sys
import joblib
import numpy as np
from src.preprocess import clean_text
from src.features import load_vectorizer
from src.config import FAKE_MODEL_PATH, TFIDF_PATH

def top_features_for_text(model, vectorizer, X, topn=10):
    # only works for linear models (coef_)
    if not hasattr(model, "coef_"):
        return []
    coefs = model.coef_[0]  # binary
    # get non-zero indices in X
    if hasattr(X, "toarray"):
        vec = X.toarray()[0]
    else:
        vec = np.array(X)[0]
    idxs = np.where(vec > 0)[0]
    scores = [(vectorizer.get_feature_names_out()[i], coefs[i] * vec[i]) for i in idxs]
    scores = sorted(scores, key=lambda x: -abs(x[1]))
    return scores[:topn]

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.debug_infer \"Your text here\"")
        return
    text = sys.argv[1]
    print("ORIGINAL:", text)
    cleaned = clean_text(text)
    print("CLEANED:", cleaned)

    vec = load_vectorizer()
    X = vec.transform([cleaned])

    clf = joblib.load(FAKE_MODEL_PATH)
    print("MODEL:", type(clf).__name__)
    try:
        probs = clf.predict_proba(X)[0]
        print("PRED PROBS:", probs)
    except Exception:
        print("predict_proba not available for this model.")

    pred = clf.predict(X)[0]
    print("PRED:", pred)

    top = top_features_for_text(clf, vec, X, topn=15)
    print("\nTop contributing features (feature, score):")
    for f, s in top:
        print(f"{f}\t{round(s,4)}")

if __name__ == "__main__":
    main()
