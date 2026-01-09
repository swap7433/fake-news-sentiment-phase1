# src/summarizer.py
try:
    from transformers import pipeline
    SUM_PIPE = pipeline("summarization", model="t5-small", tokenizer="t5-small")
except Exception:
    SUM_PIPE = None

def summarize(text, max_length=120):
    if SUM_PIPE is None:
        # fallback: return first 3 sentences quickly
        return " ".join(text.split(".")[:3]) + "..."
    out = SUM_PIPE(text, max_length=max_length, min_length=30, do_sample=False)
    return out[0]['summary_text']
