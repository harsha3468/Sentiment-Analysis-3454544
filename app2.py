import streamlit as st
import os
import re

# --- DEFENSIVE IMPORT CHECK ---
try:
    import nltk
    import pandas as pd
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Sentiment AI Pro", page_icon="📊", layout="wide")

# --- 2. NLTK DATA LOADING ---
@st.cache_resource
def load_resources():
    if NLTK_AVAILABLE:
        nltk.download('vader_lexicon')
        return SentimentIntensityAnalyzer()
    return None

analyzer = load_resources()

# --- 3. HELPER FUNCTIONS ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentiment(text):
    if analyzer:
        score = analyzer.polarity_scores(text)['compound']
        if score >= 0.05: return "Positive", "😊", "#d4edda"
        elif score <= -0.05: return "Negative", "😟", "#f8d7da"
        else: return "Neutral", "😐", "#e2e3e5"
    return "Error", "❓", "#ffffff"

# --- 4. MAIN UI ---
st.title("🧠 AI Sentiment Analyzer")

# If NLTK failed to import, show a clear instruction box
if not NLTK_AVAILABLE:
    st.error("### ⚠️ Library Missing: NLTK")
    st.write("The server has not installed the required tools yet. Please follow these steps:")
    st.info("""
    1. Go to your **requirements.txt** on GitHub.
    2. Ensure it contains exactly these 3 lines:
       ```
       streamlit
       pandas
       nltk
       ```
    3. If they are already there, go to **Manage App** (bottom right) -> **Three Dots** -> **Reboot App**.
    """)
    st.stop() # Stops the app from crashing further

# --- 5. SEARCH & DATA LOGIC (Runs only if NLTK is found) ---
user_input = st.text_input("Test a sentence:", placeholder="Enter text...")

if user_input:
    label, emoji, color = get_sentiment(clean_text(user_input))
    st.markdown(f"<div style='background-color:{color}; padding:20px; border-radius:10px; text-align:center;'><h2>{emoji} {label}</h2></div>", unsafe_allow_html=True)

st.divider()

# Load CSV
if os.path.exists("reviews.csv"):
    df = pd.read_csv("reviews.csv")
    st.subheader("📋 Dataset Analysis")
    if "Review" in df.columns:
        df["Sentiment"] = df["Review"].apply(lambda x: get_sentiment(clean_text(x))[0])
        col1, col2 = st.columns([1, 2])
        with col1:
            st.bar_chart(df["Sentiment"].value_counts())
        with col2:
            st.dataframe(df[["Review", "Sentiment"]], use_container_width=True)
else:
    st.warning("Upload 'reviews.csv' to GitHub to see batch results.")