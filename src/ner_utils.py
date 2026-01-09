# src/ner_utils.py
import spacy

def load_spacy_model():
    """
    Safely load spaCy model.
    Downloads it if not present (needed for Streamlit Cloud).
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

# load lazily (only once)
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = load_spacy_model()
    return _nlp

def extract_entities(text):
    nlp = get_nlp()
    doc = nlp(text)
    rows = []
    for ent in doc.ents:
        rows.append({"text": ent.text, "label": ent.label_})
    return rows
