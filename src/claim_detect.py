# src/claim_detect.py
CLAIM_PATTERNS = [
    "confirmed", "officially announced", "leaked", "reported", "has announced",
    "shocking", "must share", "no evidence", "this is true", "forward this", "viral message",
    "click here", "guaranteed", "scientists confirm", "doctors confirm"
]

def find_claim_phrases(text):
    t = str(text).lower()
    found = [p for p in CLAIM_PATTERNS if p in t]
    return found
