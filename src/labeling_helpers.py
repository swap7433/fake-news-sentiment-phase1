# src/labeling_helpers.py
import pandas as pd
UNRELIABLE_DOMAINS = {"theonion.com","fakenews.example"}  # extend with real lists

def label_by_domain(df, url_col='url'):
    def label_row(u):
        if pd.isna(u):
            return None
        for d in UNRELIABLE_DOMAINS:
            if d in u:
                return "fake"
        return "real"
    df['label'] = df[url_col].apply(label_row)
    return df
