# src/preprocess.py
import re
import nltk
from nltk.corpus import stopwords

# -------------------------
# ENSURE REQUIRED NLTK DATA
# -------------------------
def ensure_nltk_data():
    resources = [
        "tokenizers/punkt",
        "tokenizers/punkt_tab",
        "corpora/stopwords",
    ]

    for res in resources:
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(res.split("/")[-1])

# call once at import
ensure_nltk_data()

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    # ⬇️ UNCHANGED FUNCTIONALITY
    tokens = nltk.word_tokenize(text)

    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)
