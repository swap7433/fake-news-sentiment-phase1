from sklearn.metrics import classification_report, accuracy_score
from .data_loader import load_fake_data, load_sentiment_data, split_df
from .preprocess import clean_text
from .features import load_vectorizer
from .utils import load_model
import numpy as np

def evaluate_fake():
    df = load_fake_data()
    df['clean_text'] = df['text'].astype(str).map(clean_text)
    X_train, X_test, y_train, y_test = split_df(df, text_col='clean_text', label_col='label')
    vec = load_vectorizer()
    X_test_vec = vec.transform(X_test)
    clf = load_model()
