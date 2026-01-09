# src/ner_utils.py
import spacy
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    rows = []
    for ent in doc.ents:
        rows.append({"text": ent.text, "label": ent.label_})
    return rows
