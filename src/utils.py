import joblib
import os
from .config import MODEL_DIR, FAKE_MODEL_PATH, SENT_MODEL_PATH

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)

def save_models(fake_clf, sent_clf):
    save_model(fake_clf, FAKE_MODEL_PATH)
    save_model(sent_clf, SENT_MODEL_PATH)
