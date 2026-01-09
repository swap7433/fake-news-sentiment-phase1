# src/preprocess.py
import os
import re
import nltk
from nltk.corpus import stopwords

# -------------------------------------------------
# FORCE NLTK DATA DIRECTORY (Streamlit Cloud safe)
# -------------------------------------------------
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

# -------------------------------------------------
# ENSURE REQUIRED NLTK RESOURCES
# -------------------------------------------------
def ensure_nltk_data():
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
    }

    for pkg, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, download_dir=NLTK_DATA_DIR)

# Run once at import
ensure_nltk_data()

STOP_WORDS = set(stopwords.words("english"))

# -------------------------------------------------
# EXISTING FUNCTIONALITY — UNCHANGED
# -------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    # ⬇️ DO NOT CHANGE
    tokens = nltk.word_tokenize(text)

    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)
