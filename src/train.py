import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

from .data_loader import load_fake_data, load_sentiment_data, split_df
from .preprocess import clean_text
from .features import build_tfidf
from .utils import save_model
from .config import TFIDF_PATH, FAKE_MODEL_PATH, SENT_MODEL_PATH, MODEL_DIR

def prepare_corpus(df, text_col='text', label_col='label'):
    df = df.copy()
    df['clean_text'] = df[text_col].astype(str).map(clean_text)
    return df

def train_fake_model():
    df = load_fake_data()
    df = prepare_corpus(df, text_col='text', label_col='label')
    X_train, X_test, y_train, y_test = split_df(df, text_col='clean_text', label_col='label')
    # build vectorizer on combined corpus
    vectorizer, X_train_vec = build_tfidf(X_train)
    X_test_vec = vectorizer.transform(X_test)
    # simple classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)
    print("Fake News Classifier:")
    print(classification_report(y_test, preds))
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_model(clf, FAKE_MODEL_PATH)

def train_sentiment_model():
    df = load_sentiment_data()
    df = prepare_corpus(df, text_col='text', label_col='sentiment')
    X_train, X_test, y_train, y_test = split_df(df, text_col='clean_text', label_col='sentiment')
    # reuse vectorizer if exists else build new
    if os.path.exists(TFIDF_PATH):
        vectorizer = joblib.load(TFIDF_PATH)
    else:
        vectorizer, _ = build_tfidf(X_train)
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)
    print("Sentiment Classifier:")
    print(classification_report(y_test, preds))
    save_model(clf, SENT_MODEL_PATH)

if __name__ == "__main__":
    train_fake_model()
    train_sentiment_model()
    print("Training complete. Models saved to 'models/'")
