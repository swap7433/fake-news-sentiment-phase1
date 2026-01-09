# src/preprocess.py
import re
import nltk
from nltk.corpus import stopwords

# -------------------------
# SAFE NLTK DATA LOADING
# -------------------------
def ensure_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

# Ensure resources are available at import time
ensure_nltk_data()

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove non-alphabetic chars
    text = re.sub(r"[^a-z\s]", " ", text)

    # tokenize
    tokens = nltk.word_tokenize(text)

    # remove stopwords & short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    return " ".join(tokens)
