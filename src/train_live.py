# src/train_live.py (robust tolerant version)
import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# local imports
from src.preprocess import clean_text
from src.features import build_tfidf, load_vectorizer
from src.utils import save_model
from src.config import MODEL_DIR, TFIDF_PATH, FAKE_MODEL_PATH, DATA_DIR

OUT_CSV = Path(DATA_DIR) / "live_tweets.csv"

# Simple weak-label keywords (used elsewhere)
FAKE_KEYWORDS = {"fake","hoax","scam","misinformation","false","rumor","fraud"}

def fetch_not_applicable():
    # placeholder if you want to fetch when missing
    return None

def load_live_tweets_or_error():
    if not OUT_CSV.exists():
        raise FileNotFoundError(f"{OUT_CSV} not found. Run fetch/prepare scripts first.")
    df = pd.read_csv(OUT_CSV)
    return df

def ensure_min_class_counts(df, min_count=2, inplace=True):
    """
    Ensure each label class has at least min_count samples by upsampling minority classes.
    Returns a new dataframe.
    """
    df = df.copy()
    if 'label' not in df.columns:
        raise ValueError("No 'label' column found in dataset.")
    vc = df['label'].value_counts()
    to_concat = [df]
    changed = False
    for lbl, cnt in vc.items():
        if cnt < min_count:
            needed = min_count - cnt
            # sample with replacement from existing samples of this class
            samples = df[df['label'] == lbl]
            if len(samples) == 0:
                # nothing to sample, skip
                continue
            dup = samples.sample(n=needed, replace=True, random_state=42)
            to_concat.append(dup)
            changed = True
            print(f"[upsample] Label '{lbl}' had {cnt} rows; duplicated {needed} rows.")
    if changed:
        newdf = pd.concat(to_concat, ignore_index=True)
        return newdf
    return df

def prepare_and_label(df):
    # ensure content column exists
    if 'content' not in df.columns and 'text' in df.columns:
        df['content'] = df['text']
    if 'content' not in df.columns and 'title' in df.columns:
        df['content'] = df['title']
    if 'content' not in df.columns:
        # fallback: first column as content
        df['content'] = df.iloc[:,0].astype(str)
    # fill labels NaN if any
    if 'label' not in df.columns:
        raise ValueError("No 'label' column found in the dataset. Please label data before training.")
    df = df.dropna(subset=['content','label']).copy()
    # clean
    df['clean_text'] = df['content'].astype(str).map(clean_text)
    return df

def train_from_df(df):
    # df expected to have 'clean_text' and 'label'
    if 'clean_text' not in df.columns or 'label' not in df.columns:
        raise ValueError("DataFrame must contain 'clean_text' and 'label' columns.")
    X = df['clean_text']
    y = df['label']

    # Ensure minimal counts by upsampling very small classes (>=2)
    df2 = ensure_min_class_counts(df, min_count=2)
    if len(df2) != len(df):
        print("[info] Dataset augmented to ensure minimal class counts.")
        X = df2['clean_text']
        y = df2['label']

    # Now decide whether stratified split is possible
    vc = y.value_counts()
    min_class_count = vc.min()
    stratify_arg = y if min_class_count >= 2 else None
    if stratify_arg is None:
        print("[warn] Some classes still have <2 samples. Proceeding without stratified split (stratify=None).")
    else:
        print("[info] Stratified split enabled.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)

    # Build TF-IDF on training corpus
    vectorizer, X_train_vec = build_tfidf(X_train, max_features=8000)
    X_test_vec = vectorizer.transform(X_test)

    # Train classifier
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    print("[train] Fake News Classifier - evaluation on test set:")
    try:
        print(classification_report(y_test, preds))
    except Exception:
        print("Could not print classification report (unexpected).")

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_model(clf, FAKE_MODEL_PATH)
    print(f"[save] Model saved to {FAKE_MODEL_PATH}")
    return clf

def main():
    df = load_live_tweets_or_error()
    print(f"[load] Found existing {OUT_CSV}")
    # use existing labels
    print(f"[label] Using existing 'label' column with {df['label'].notna().sum()} labels.")
    df = prepare_and_label(df)
    clf = train_from_df(df)
    print("[done] training complete. Models saved to models/")

if __name__ == "__main__":
    main()
