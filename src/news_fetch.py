# src/news_fetch.py
import requests
import pandas as pd
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT = DATA_DIR / "newsapi_articles.csv"

def fetch_news(api_key, q='fake news', page_size=100, language='en'):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': q,
        'pageSize': page_size,
        'language': language,
        'sortBy': 'relevancy',
        'apiKey': api_key
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    items = r.json().get('articles', [])
    rows = []
    for a in items:
        rows.append({
            "source": a["source"]["name"],
            "title": a["title"],
            "description": a["description"],
            "content": a["content"] or "",
            "url": a["url"],
            "publishedAt": a["publishedAt"]
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} articles to {OUT}")
    return df
