# app/app.py
import sys, pathlib, os
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from src.infer import predict_all
from src import explain, ner_utils, topic_utils, summarizer, claim_detect, credibility
from app.components import (
    colored_result, show_confidence_bar,
    show_wordcloud, plot_class_distribution_from_df, plot_confidence_hist,
    show_feature_heatmap, show_top_words_bar, show_sentiment_timeseries
)
from src.config import TFIDF_PATH, FAKE_MODEL_PATH, SENT_MODEL_PATH
from app.components import render_confidence_result
from app.components import render_confidence_result, colored_confidence_bar

from pathlib import Path
import time

st.set_page_config(page_title="Fake News Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("Fake News Detection & Sentiment Analysis — Dashboard (Phase 1)")

# Sidebar quick actions
with st.sidebar:
    st.header("Quick actions")
    st.write("Use these to manage data & models")
    if st.button("Fetch RSS (live)"):
        os.system(f'python -m src.rss_fetch')
        st.success("Fetched RSS into data/live_news.csv")
    if st.button("Prepare live data (weak labels)"):
        os.system(f'python -m src.prepare_live_data')
        st.success("Prepared data/live_tweets.csv")
    if st.button("Train fake+sentiment models"):
        # trains fake model; sentiment uses VADER trainer
        os.system(f'python -m src.train_live')
        os.system(f'python -m src.train_sentiment')
        st.success("Training complete (check terminal for details).")
    st.markdown("---")
    st.write("Model files:")
    st.write(f"TF-IDF: {'OK' if Path(TFIDF_PATH).exists() else 'Missing'}")
    st.write(f"Fake model: {'OK' if Path(FAKE_MODEL_PATH).exists() else 'Missing'}")
    st.write(f"Sentiment model: {'OK' if Path(SENT_MODEL_PATH).exists() else 'Missing'}")
    st.markdown("---")
    st.write("Notes:")
    st.write("- Summarization is optional (heavy).")
    st.write("- Use manual labeling for best results.")

# Tabs for dashboard
tabs = st.tabs(["Home", "Explainability", "Visuals", "NER & Claims", "Topics & Trends", "Timeline", "Summarize", "Credibility", "Model Eval", "Data"])

# --------------------------
# HOME: single text analysis
# --------------------------
with tabs[0]:
    st.header("Analyze single article / paste text")
    text = st.text_area("Paste article or claim here", height=300)
    col1, col2 = st.columns([2,1])
    with col1:
        if st.button("Analyze"):
            if not text.strip():
                st.error("Please paste some text.")
            else:
                out = predict_all(text)
                st.subheader("Prediction")
                #is_fake = str(out['fake_pred']).lower() in ['fake','1','true','yes']
                #colored_result(fake_label, is_fake)
                #show_confidence_bar("Fake news confidence", out['fake_confidence'])
                # -----------------------------------
                    # SENTIMENT-BASED UI COLOR LOGIC
                    # -----------------------------------
                # -----------------------------------
# CONFIDENCE-THRESHOLD BASED COLOR
# -----------------------------------
                conf = float(out.get("sentiment_confidence", 0.0))  # using sentiment confidence
                
                if conf < 0.30:
                    ui_color = "red"
                    ui_label = "Low Confidence / Potentially Risky Content"
                
                elif conf < 0.65:
                    ui_color = "yellow"
                    ui_label = "Medium Confidence / Needs Verification"
                
                else:
                    ui_color = "green"
                    ui_label = "High Confidence / Likely Benign Content"
                
                st.subheader("Final Interpretation (Confidence-based)")
                render_confidence_result(ui_label, ui_color, conf)
                
                st.markdown("### News Confidence")
                colored_confidence_bar(conf, ui_color)

                st.markdown("---")
                st.subheader("Sentiment")
                st.write(f"**{out['sentiment_pred'].title()}**")
                show_confidence_bar("Sentiment confidence", out['sentiment_confidence'])
                st.markdown("---")
                st.subheader("Cleaned text")
                st.write(out['cleaned_text'])
                st.markdown("---")
                # quick local explainability
                if Path(FAKE_MODEL_PATH).exists():
                    st.subheader("Top contributing words (local explain)")
                    feats = explain.top_contributing_words(out['cleaned_text'], topn=10)
                    show_top_words_bar({"local_explain": feats})
    with col2:
        st.subheader("Quick Visuals")
        sample_df = None
        sample_path = Path("data/live_tweets.csv")
        if sample_path.exists():
            try:
                sample_df = pd.read_csv(sample_path)
            except Exception:
                sample_df = None
        if sample_df is None:
            st.info("No sample data found. Use RSS fetch / prepare scripts or paste an article to test.")
        else:
            st.markdown(f"Sample size: {len(sample_df)}")
            show_wordcloud(sample_df['content'].astype(str).head(200).tolist(), caption="Wordcloud (sample)")
            if 'label' in sample_df.columns:
                plot_class_distribution_from_df(sample_df, label_col='label')
            # model confidence histogram
            if Path(FAKE_MODEL_PATH).exists() and Path(TFIDF_PATH).exists():
                st.markdown("Model confidence histogram (small sample)")
                sample_texts = sample_df['content'].astype(str).sample(min(200,len(sample_df)), random_state=1).tolist()
                confidences = []
                preds = []
                prog = st.progress(0)
                for i, txt in enumerate(sample_texts):
                    res = predict_all(txt)
                    confidences.append(float(res.get('fake_confidence', 1.0)))
                    preds.append(res.get('fake_pred'))
                    prog.progress(int((i+1)/len(sample_texts)*100))
                    time.sleep(0.01)
                prog.empty()
                plot_confidence_hist(confidences)
                # predicted counts
                try:
                    import pandas as pd
                    s = pd.Series(preds).value_counts()
                    st.table(s.rename_axis('class').reset_index(name='count'))
                except Exception:
                    pass

# --------------------------
# Explainability tab
# --------------------------
with tabs[1]:
    st.header("Model explainability")
    st.write("Per-text highlights + global top words for Fake / Real classes.")
    text_for_explain = st.text_area("Enter text to explain (or leave blank to use sample)", height=200)
    target_text = text_for_explain if text_for_explain.strip() else None
    if st.button("Explain text"):
        if not target_text:
            st.error("Please paste text to explain.")
        else:
            cleaned = explain.clean_and_return(target_text)
            # per-word weights
            heat = explain.get_word_weight_map(cleaned)
            st.subheader("Highlighted text (red = fake, green = real)")
            show_feature_heatmap(heat)
            st.subheader("Top weighted words (global)")
            top_fake, top_real = explain.get_global_top_words(20)
            show_top_words_bar({"fake": top_fake, "real": top_real})

# --------------------------
# Visuals tab
# --------------------------
with tabs[2]:
    st.header("Visual analytics")
    sample_path = Path("data/live_tweets.csv")
    if sample_path.exists():
        df = pd.read_csv(sample_path)
        st.subheader("Wordcloud")
        show_wordcloud(df['content'].astype(str).tolist())
        if 'label' in df.columns:
            st.subheader("Class distribution")
            plot_class_distribution_from_df(df, label_col='label')
        st.subheader("Confidence histogram")
        # compute confidences if model exists
        if Path(FAKE_MODEL_PATH).exists():
            sample_texts = df['content'].astype(str).sample(min(300,len(df)), random_state=2).tolist()
            confidences = [predict_all(t)['fake_confidence'] for t in sample_texts]
            plot_confidence_hist(confidences)
    else:
        st.info("No dataset found in data/live_tweets.csv")

# --------------------------
# NER & Claims tab
# --------------------------
with tabs[3]:
    st.header("Named Entities & Claim Detection")
    txt = st.text_area("Paste text for NER & claim detection", height=250)
    if st.button("Run NER & Claims"):
        if not txt.strip():
            st.error("Please paste text.")
        else:
            ents = ner_utils.extract_entities(txt)
            st.subheader("Entities")
            st.table(ents)
            st.subheader("Detected suspicious claim phrases")
            claims = claim_detect.find_claim_phrases(txt)
            if claims:
                for c in claims:
                    st.warning(c)
            else:
                st.success("No suspicious claim patterns found.")

# --------------------------
# Topics & Trends tab
# --------------------------
with tabs[4]:
    st.header("Topic Modeling (LDA)")
    st.write("Build topics from sample dataset (runs locally).")
    if Path("data/live_tweets.csv").exists():
        if st.button("Build topics (may take 10-30s)"):
            df = pd.read_csv("data/live_tweets.csv")
            topics = topic_utils.build_topics(df['content'].astype(str).tolist(), num_topics=6)
            st.write("Topics (top keywords):")
            for i, t in enumerate(topics):
                st.write(f"Topic {i}: {', '.join(t)}")
            st.success("Topics built. Use assign topic on single text below.")
    # assign single text
    ttxt = st.text_area("Assign topic to text", height=120)
    if st.button("Assign topic"):
        if not ttxt.strip():
            st.error("Paste text to assign.")
        else:
            top = topic_utils.assign_topic(ttxt)
            st.write("Assigned topic:", top)

# --------------------------
# Timeline & Sentiment tab
# --------------------------
with tabs[5]:
    st.header("Timeline & Sentiment (from dataset)")
    if Path("data/live_tweets.csv").exists():
        df = pd.read_csv("data/live_tweets.csv")
        if 'published' in df.columns:
            st.subheader("Sentiment over time (sample)")
            show_sentiment_timeseries(df)
        else:
            st.info("No 'published' column in dataset. Use RSS fetch to get timestamps.")
    else:
        st.info("No dataset found.")

# --------------------------
# Summarize tab (optional heavy)
# --------------------------
with tabs[6]:
    st.header("Summarize text (optional heavy)")
    sm_text = st.text_area("Paste text to summarize (longer is better)", height=200)
    if st.button("Summarize"):
        if not sm_text.strip():
            st.error("Please add text.")
        else:
            s = summarizer.summarize(sm_text)
            st.subheader("Summary")
            st.write(s)

# --------------------------
# Credibility tab
# --------------------------
with tabs[7]:
    st.header("Credibility score")
    txtc = st.text_area("Paste text to compute credibility", height=200)
    if st.button("Compute credibility"):
        if not txtc.strip():
            st.error("Please paste text.")
        else:
            cred = credibility.compute_credibility(txtc)
            st.metric("Credibility score (0-100)", cred['score'])
            st.write("Breakdown:")
            st.json(cred['components'])

# --------------------------
# Model Eval tab
# --------------------------
with tabs[8]:
    st.header("Model evaluation (on data/live_tweets.csv if labeled)")
    if Path("data/live_tweets.csv").exists():
        df = pd.read_csv("data/live_tweets.csv")
        if 'label' in df.columns:
            st.write("Run quick evaluation (train/test split used)")
            if st.button("Evaluate"):
                os.system("python -m src.evaluate")
        else:
            st.info("Dataset has no 'label' column. Add labels to evaluate.")
    else:
        st.info("No dataset found to evaluate.")

# --------------------------
# Data tab
# --------------------------
with tabs[9]:
    st.header("Data manager")
    st.write("View / edit a few rows (manual labeling recommended)")
    if Path("data/live_tweets.csv").exists():
        df = pd.read_csv("data/live_tweets.csv")
        st.dataframe(df.head(500))
        st.download_button("Download dataset", df.to_csv(index=False).encode('utf-8'), "live_tweets.csv")
    else:
        st.info("No dataset available. Use RSS fetch or other fetch scripts.")
