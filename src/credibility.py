# src/credibility.py
from src.infer import predict_all
from src.claim_detect import find_claim_phrases
from nltk.sentiment import SentimentIntensityAnalyzer
from src.preprocess import clean_text

def compute_credibility(text):
    # simple heuristic scoring (0-100)
    out = predict_all(text)
    score = 50.0
    components = {}
    # fake model penalty
    if str(out['fake_pred']).lower() == 'fake':
        score -= 30
        components['fake_penalty'] = -30
    else:
        components['fake_penalty'] = 0
    # sentiment neutrality + extreme negativity reduces credibility slightly (example)
    if out['sentiment_pred'] == 'negative':
        score -= 5
        components['sentiment_penalty'] = -5
    else:
        components['sentiment_penalty'] = 0
    # claim phrases penalty
    claims = find_claim_phrases(text)
    components['claims_found'] = claims
    if claims:
        score -= 10
        components['claims_penalty'] = -10
    else:
        components['claims_penalty'] = 0
    # length bonus (very short messages lower credibility)
    if len(text.split()) < 8:
        score -= 10
        components['length_penalty'] = -10
    else:
        components['length_penalty'] = 0
    final = max(0, min(100, int(score)))
    components['base'] = 50
    components['final_score'] = final
    return {'score': final, 'components': components}
