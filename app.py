"""
AI Personal Knowledge & Research Assistant
============================================
A self-contained Streamlit app combining Pandas/NumPy EDA, Scikit-learn ML,
a PyTorch neural network, NLP text analysis, SQLite storage, and a local
retrieval-based AI assistant. No paid APIs, no external uploads, no backend server.

Run with:  streamlit run app.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np

from src import database as db
from src import data_generator as dg
from src import eda
from src import ml_models as ml
from src import neural_net as nn_mod
from src import nlp_analysis as nlp
from src.assistant import KnowledgeAssistant

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------------------
# Global CSS - modern dark dashboard styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 10px; }
    div[data-testid="stMetricValue"] { color: #8b5cf6; }
    h1, h2, h3 { color: #e6e6e6; }
    .stButton>button {
        background: linear-gradient(90deg, #7c3aed, #06b6d4);
        color: white; border: none; border-radius: 8px; font-weight: 600;
    }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

db.init_db()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "sklearn_bundle" not in st.session_state:
    st.session_state.sklearn_bundle = None
if "nn_bundle" not in st.session_state:
    st.session_state.nn_bundle = None
if "reg_bundle" not in st.session_state:
    st.session_state.reg_bundle = None

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🧠 Knowledge Assistant")
st.sidebar.caption("Personal AI Research & Learning Tracker")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "✍️ Add Entry", "📊 EDA & Charts", "🤖 ML Prediction",
     "🔥 Neural Network", "📝 NLP Analysis", "💬 AI Assistant", "⚙️ Data Management"],
)

df = db.fetch_all_df()
st.sidebar.markdown("---")
st.sidebar.metric("Total Entries", len(df))
if not df.empty:
    st.sidebar.metric("Total Hours Logged", f"{df['hours_spent'].sum():.1f}")

# ---------------------------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("AI Personal Knowledge & Research Assistant")
    st.caption("Pandas • NumPy • Scikit-learn • PyTorch • NLP • Streamlit — 100% local, no paid APIs")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Entries", len(df))
    c2.metric("⏱️ Hours Logged", f"{df['hours_spent'].sum():.1f}" if not df.empty else "0.0")
    c3.metric("🏷️ Categories", df["category"].nunique() if not df.empty else 0)
    c4.metric("⭐ Avg Priority", f"{df['priority'].mean():.2f}" if not df.empty else "0.0")

    st.markdown("---")
    if df.empty:
        st.info("No data yet. Go to **⚙️ Data Management** to generate sample data, or **✍️ Add Entry** to add your own.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(eda.category_distribution_fig(df), use_container_width=True)
        with col2:
            st.plotly_chart(eda.hours_over_time_fig(df), use_container_width=True)

        st.subheader("Recent Entries")
        st.dataframe(df[["title", "category", "priority", "hours_spent", "created_at"]].head(10),
                     use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# PAGE: Add Entry
# ---------------------------------------------------------------------------
elif page == "✍️ Add Entry":
    st.title("Add a Knowledge Base Entry")
    st.caption("All data is stored locally in SQLite — nothing leaves your machine.")

    with st.form("add_entry_form", clear_on_submit=True):
        title = st.text_input("Title", placeholder="e.g. Understanding gradient descent")
        category = st.selectbox("Category", dg.CATEGORIES)
        content = st.text_area("Notes / Content", placeholder="What did you learn? Write freely...", height=150)
        col1, col2 = st.columns(2)
        priority = col1.slider("Priority (1=low, 5=high)", 1, 5, 3)
        hours_spent = col2.number_input("Hours Spent", min_value=0.0, max_value=24.0, value=1.0, step=0.25)
        submitted = st.form_submit_button("💾 Save Entry")

        if submitted:
            if not title.strip() or not content.strip():
                st.error("Please provide both a title and content before saving.")
            else:
                try:
                    db.insert_note(title.strip(), category, content.strip(), priority, hours_spent)
                    st.success(f"Saved '{title}' to your knowledge base!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save entry: {e}")

    if not df.empty:
        st.markdown("---")
        st.subheader("All Entries")
        st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# PAGE: EDA & Charts
# ---------------------------------------------------------------------------
elif page == "📊 EDA & Charts":
    st.title("Exploratory Data Analysis")

    if df.empty or len(df) < 3:
        st.warning("Add at least 3 entries (or generate sample data) to see meaningful analysis.")
    else:
        st.subheader("Summary Statistics")
        st.dataframe(eda.summary_stats(df), use_container_width=True)

        st.subheader("Category Averages")
        st.dataframe(eda.category_avg_hours(df), use_container_width=True)

        tab1, tab2, tab3, tab4 = st.tabs(["Category Distribution", "Hours Over Time", "Priority vs Hours", "Correlation"])
        with tab1:
            st.plotly_chart(eda.category_distribution_fig(df), use_container_width=True)
        with tab2:
            st.plotly_chart(eda.hours_over_time_fig(df), use_container_width=True)
        with tab3:
            st.plotly_chart(eda.priority_vs_hours_fig(df), use_container_width=True)
        with tab4:
            st.plotly_chart(eda.correlation_heatmap_fig(df), use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: ML Prediction (Scikit-learn)
# ---------------------------------------------------------------------------
elif page == "🤖 ML Prediction":
    st.title("Machine Learning — Priority Prediction")
    st.caption("Random Forest Classifier (priority bucket) + Linear Regression (hours estimate), built with Scikit-learn.")

    if df.empty or len(df) < 20:
        st.warning("Need at least 20 entries to train a reliable model. Generate sample data in ⚙️ Data Management.")
    else:
        if st.button("🚀 Train / Retrain Models"):
            with st.spinner("Training Random Forest classifier..."):
                st.session_state.sklearn_bundle = ml.train_priority_classifier(df)
            with st.spinner("Training Linear Regression..."):
                st.session_state.reg_bundle = ml.train_hours_regressor(df)
            st.success("Models trained!")

        bundle = st.session_state.sklearn_bundle
        reg_bundle = st.session_state.reg_bundle

        if bundle is not None:
            st.subheader("Classifier Performance")
            c1, c2 = st.columns(2)
            c1.metric("Test Accuracy", f"{bundle['accuracy']*100:.1f}%")
            c2.write("**Feature Importances**")
            c2.dataframe(bundle["feature_importances"], use_container_width=True)

            st.write("**Classification Report**")
            report_df = pd.DataFrame(bundle["report"]).T
            st.dataframe(report_df, use_container_width=True)

            st.write("**Confusion Matrix**")
            cm_df = pd.DataFrame(bundle["confusion_matrix"], index=bundle["classes"], columns=bundle["classes"])
            st.dataframe(cm_df, use_container_width=True)

            st.markdown("---")
            st.subheader("Try a Live Prediction")
            col1, col2, col3 = st.columns(3)
            p_cat = col1.selectbox("Category", dg.CATEGORIES, key="pred_cat")
            p_title = col2.text_input("Title", "New learning topic", key="pred_title")
            p_hours = col3.number_input("Hours Spent", 0.0, 24.0, 2.0, key="pred_hours")
            p_content = st.text_area("Content", "Some notes about this topic...", key="pred_content")

            if st.button("Predict Priority"):
                pred, proba = ml.predict_priority(bundle, p_cat, p_title, p_content, p_hours, bundle["label_encoder"])
                st.success(f"Predicted Priority Bucket: **{pred}**")
                st.bar_chart(pd.Series(proba))

        if reg_bundle is not None:
            st.markdown("---")
            st.subheader("Hours Regression Performance")
            c1, c2 = st.columns(2)
            c1.metric("MAE", f"{reg_bundle['mae']:.2f} hrs")
            c2.metric("R² Score", f"{reg_bundle['r2']:.3f}")
            comp_df = pd.DataFrame({"Actual": reg_bundle["y_test"], "Predicted": reg_bundle["y_pred"]})
            st.line_chart(comp_df.reset_index(drop=True))

# ---------------------------------------------------------------------------
# PAGE: Neural Network (PyTorch)
# ---------------------------------------------------------------------------
elif page == "🔥 Neural Network":
    st.title("Deep Learning — PyTorch Neural Network")
    st.caption("A small feed-forward network (2 hidden layers, dropout) trained from scratch on your data.")

    if df.empty or len(df) < 20:
        st.warning("Need at least 20 entries to train the neural network. Generate sample data in ⚙️ Data Management.")
    else:
        epochs = st.slider("Training Epochs", 10, 200, 60, step=10)
        if st.button("🔥 Train Neural Network"):
            with st.spinner(f"Training for {epochs} epochs..."):
                st.session_state.nn_bundle = nn_mod.train_neural_net(df, epochs=epochs)
            st.success("Neural network trained!")

        bundle = st.session_state.nn_bundle
        if bundle is not None:
            c1, c2 = st.columns(2)
            c1.metric("Final Test Accuracy", f"{bundle['final_accuracy']*100:.1f}%")
            c2.write(f"Architecture: `Linear(5→32) → ReLU → Dropout(0.2) → Linear(32→16) → ReLU → Linear(16→3)`")

            loss_df = pd.DataFrame({
                "Train Loss": bundle["train_losses"],
                "Test Loss": bundle["test_losses"],
            })
            st.subheader("Loss Curve")
            st.line_chart(loss_df)

            st.subheader("Test Accuracy Over Training")
            st.line_chart(pd.Series(bundle["test_accuracies"], name="Accuracy"))

            st.markdown("---")
            st.subheader("Try a Live Prediction")
            col1, col2, col3 = st.columns(3)
            p_cat = col1.selectbox("Category", dg.CATEGORIES, key="nn_cat")
            p_title = col2.text_input("Title", "New learning topic", key="nn_title")
            p_hours = col3.number_input("Hours Spent", 0.0, 24.0, 2.0, key="nn_hours")
            p_content = st.text_area("Content", "Some notes about this topic...", key="nn_content")

            if st.button("Predict with Neural Net"):
                pred, probs = nn_mod.predict_with_net(bundle, p_cat, p_title, p_content, p_hours)
                st.success(f"Predicted Priority Bucket: **{pred}**")
                st.bar_chart(pd.Series(probs))

# ---------------------------------------------------------------------------
# PAGE: NLP Analysis
# ---------------------------------------------------------------------------
elif page == "📝 NLP Analysis":
    st.title("NLP Text Analysis")
    st.caption("Tokenization, TF-IDF keyword extraction, and lexicon-based sentiment — fully offline, no downloads.")

    if df.empty:
        st.warning("No entries yet. Add data first.")
    else:
        tab1, tab2, tab3 = st.tabs(["Word Frequency", "Keyword Extraction (TF-IDF)", "Sentiment Analysis"])

        with tab1:
            freq_df = nlp.word_frequency(df, top_n=20)
            st.bar_chart(freq_df.set_index("word"))
            st.dataframe(freq_df, use_container_width=True, hide_index=True)

        with tab2:
            kw_df = nlp.extract_keywords_tfidf(df, top_n=15)
            st.bar_chart(kw_df.set_index("keyword"))
            st.dataframe(kw_df, use_container_width=True, hide_index=True)

        with tab3:
            sent_df = nlp.sentiment_for_df(df)
            dist = sent_df["sentiment_label"].value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("Positive", int(dist.get("Positive", 0)))
            c2.metric("Neutral", int(dist.get("Neutral", 0)))
            c3.metric("Negative", int(dist.get("Negative", 0)))
            st.dataframe(
                sent_df[["title", "category", "sentiment_label", "sentiment_score"]],
                use_container_width=True, hide_index=True
            )

            st.subheader("Try It Yourself")
            sample_text = st.text_area("Enter text to analyze", "I struggled at first but the breakthrough felt great.")
            if st.button("Analyze Sentiment"):
                score, label = nlp.sentiment_score(sample_text)
                st.info(f"Sentiment: **{label}** (score: {score})")

# ---------------------------------------------------------------------------
# PAGE: AI Assistant
# ---------------------------------------------------------------------------
elif page == "💬 AI Assistant":
    st.title("AI Assistant — Ask About Your Knowledge Base")
    st.caption("Local TF-IDF retrieval + rule-based reasoning. No external API calls.")

    if df.empty:
        st.warning("Your knowledge base is empty. Add entries or generate sample data first.")
    else:
        assistant = KnowledgeAssistant(df)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.info("Try asking: *'How many entries do I have?'*, *'What category do I spend the most time on?'*, "
                "*'What should I study next?'*, or ask about any topic in your notes.")

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

        user_q = st.chat_input("Ask anything about your knowledge base...")
        if user_q:
            st.session_state.chat_history.append(("user", user_q))
            try:
                answer = assistant.answer(user_q)
            except Exception as e:
                answer = f"Sorry, I hit an error answering that: {e}"
            st.session_state.chat_history.append(("assistant", answer))
            st.rerun()

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ---------------------------------------------------------------------------
# PAGE: Data Management
# ---------------------------------------------------------------------------
elif page == "⚙️ Data Management":
    st.title("Data Management")

    st.subheader("Generate Sample Dataset")
    st.caption("Creates realistic synthetic learning-log entries so you can explore every feature immediately.")
    n_samples = st.slider("Number of sample entries", 20, 300, 120, step=10)
    if st.button("🎲 Generate Sample Data"):
        try:
            rows = dg.generate_sample_data(n=n_samples)
            db.insert_many(rows)
            st.success(f"Inserted {n_samples} sample entries!")
            st.rerun()
        except Exception as e:
            st.error(f"Error generating data: {e}")

    st.markdown("---")
    st.subheader("Export Data")
    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download as CSV", csv, "knowledge_base_export.csv", "text/csv")
    else:
        st.caption("No data to export yet.")

    st.markdown("---")
    st.subheader("⚠️ Danger Zone")
    confirm = st.checkbox("I understand this will permanently delete all entries")
    if st.button("🗑️ Clear All Data", disabled=not confirm):
        try:
            db.clear_all()
            st.session_state.sklearn_bundle = None
            st.session_state.nn_bundle = None
            st.session_state.reg_bundle = None
            st.success("All data cleared.")
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing data: {e}")

    st.markdown("---")
    st.subheader("Raw Table")
    st.dataframe(df, use_container_width=True, hide_index=True)
