# app/components.py
import streamlit as st
import plotly.graph_objects as go
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np

def colored_result(label, is_fake):
    st.subheader("Final Interpretation (Confidence-based)")

if ui_color == "red":
    st.markdown(
        f"<div style='padding:12px;background:#ffe6e6;border-left:6px solid #ff0000;'>"
        f"<b>{ui_label}</b><br>"
        f"Fake confidence: <b>{fake_conf*100:.2f}%</b>"
        f"</div>",
        unsafe_allow_html=True
    )

elif ui_color == "yellow":
    st.markdown(
        f"<div style='padding:12px;background:#fff8e1;border-left:6px solid #ffb300;'>"
        f"<b>{ui_label}</b><br>"
        f"Fake confidence: <b>{fake_conf*100:.2f}%</b>"
        f"</div>",
        unsafe_allow_html=True
    )

else:
    st.markdown(
        f"<div style='padding:12px;background:#e8f5e9;border-left:6px solid #2e7d32;'>"
        f"<b>{ui_label}</b><br>"
        f"Fake confidence: <b>{fake_conf*100:.2f}%</b>"
        f"</div>",
        unsafe_allow_html=True
    )


def show_confidence_bar(title, value):
    fig = go.Figure(go.Bar(x=[value*100], y=[title], orientation='h', text=[f"{value*100:.1f}%"], textposition='inside'))
    fig.update_layout(xaxis=dict(range=[0,100], showgrid=False), height=70, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

# visuals
def generate_wordcloud(texts, max_words=150):
    if not texts:
        return None
    joined = " ".join([str(t) for t in texts if isinstance(t,str)])
    if not joined.strip():
        return None
    wc = WordCloud(width=800, height=400, max_words=max_words, background_color="white").generate(joined)
    return wc.to_image()

def show_wordcloud(texts, caption="Wordcloud"):
    img = generate_wordcloud(texts)
    if img is None:
        st.info("No text available for wordcloud.")
        return
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    st.image(buffer, caption=caption, use_container_width=True)

def plot_class_distribution_from_df(df, label_col="label"):
    if df is None or label_col not in df.columns:
        st.info("No labeled data to plot class distribution.")
        return
    counts = df[label_col].value_counts().to_dict()
    labels = list(counts.keys())
    values = list(counts.values())
    fig = go.Figure([go.Bar(x=labels, y=values, text=values, textposition="auto")])
    fig.update_layout(title="Class Distribution", xaxis_title="Class", yaxis_title="Count", height=300)
    # use a unique key per-figure instance
    st.plotly_chart(fig, use_container_width=True, key=f"class_dist_{id(fig)}")


def plot_confidence_hist(confidences, title="Confidence distribution"):
    if not confidences:
        st.info("No confidence scores to display.")
        return
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=[c*100 for c in confidences], nbinsx=20))
    fig.update_layout(title=title, xaxis_title="Confidence (%)", yaxis_title="Count", height=300)
    st.plotly_chart(fig, use_container_width=True, key=f"conf_hist_{id(fig)}")


# explainability helpers
def show_feature_heatmap(word_weight_map):
    # word_weight_map: list of tuples (word, weight) or dict
    if not word_weight_map:
        st.info("No explanation available.")
        return
    # render as highlighted HTML
    html = "<div style='line-height:1.8'>"
    for w, weight in word_weight_map.items():
        color = "#ffcccc" if weight>0 else "#ccffcc"
        html += f"<span style='background:{color};padding:3px;margin:2px;border-radius:4px;' title='{weight:.4f}'>{w}</span> "
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def show_top_words_bar(dict_of_lists_or_list):
    if isinstance(dict_of_lists_or_list, list):
        dict_of_lists = {"Top words": dict_of_lists_or_list}
    elif isinstance(dict_of_lists_or_list, dict):
        dict_of_lists = dict_of_lists_or_list
    else:
        st.info("No top words to display.")
        return

    for cls, items in dict_of_lists.items():
        st.subheader(f"Top words: {cls}")
        words = [w for w, s in items]
        scores = [abs(s) for w, s in items]
        if not words:
            st.info(f"No words found for {cls}.")
            continue
        fig = go.Figure([go.Bar(x=words, y=scores, text=[f"{s:.3f}" for s in scores], textposition="auto")])
        fig.update_layout(height=320, margin=dict(t=30,b=10))
        # unique key per figure instance
        st.plotly_chart(fig, use_container_width=True, key=f"topwords_{cls}_{id(fig)}")



def show_sentiment_timeseries(df):
    if 'published' not in df.columns or 'content' not in df.columns:
        st.info("No published/content columns available for timeseries.")
        return
    # simple: compute VADER sentiment per row (if sentiment model exists it should be used)
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    df = df.copy()
    df['published'] = pd.to_datetime(df['published'], errors='coerce')
    df = df.dropna(subset=['published'])
    df['compound'] = df['content'].astype(str).map(lambda t: sia.polarity_scores(t)['compound'])
    df = df.set_index('published').resample('D').mean().reset_index()
    fig = go.Figure([go.Scatter(x=df['published'], y=df['compound'], mode='lines+markers')])
    fig.update_layout(title="Average sentiment (daily)", xaxis_title="Date", yaxis_title="Compound sentiment")
    st.plotly_chart(fig, use_container_width=True)
