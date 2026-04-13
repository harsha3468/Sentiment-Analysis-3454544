import streamlit as st
import pandas as pd
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- 1. INITIAL SETUP ---
st.set_page_config(page_title="Sentiment Dashboard", page_icon="📊", layout="wide")

@st.cache_resource
def setup_analyzer():
    nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

analyzer = setup_analyzer()

# --- 2. HELPER FUNCTIONS ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05: return "Positive", "😊"
    elif score <= -0.05: return "Negative", "😟"
    else: return "Neutral", "😐"

# --- 3. LOAD YOUR CSV ---
@st.cache_data
def load_csv_data():
    if os.path.exists("reviews.csv"):
        df = pd.read_csv("reviews.csv")
        # Ensure the column exists
        if "Review" in df.columns:
            df["Cleaned_Review"] = df["Review"].apply(clean_text)
            df["Sentiment_Label"] = df["Cleaned_Review"].apply(lambda x: get_sentiment(x)[0])
            return df
    return None

import os
data = load_csv_data()

# --- 4. DASHBOARD UI ---
st.title("🧠 Customer Review Sentiment Analysis")

if data is not None:
    # Top Row Metrics
    col1, col2, col3 = st.columns(3)
    counts = data["Sentiment_Label"].value_counts()
    
    col1.metric("Total Reviews", len(data))
    col2.metric("Positive Reviews", counts.get("Positive", 0))
    col3.metric("Negative Reviews", counts.get("Negative", 0))
    
    st.divider()
    
    # Interactive Input Section
    st.subheader("🔥 Test New Input")
    user_input = st.text_input("Enter a new sentence to predict sentiment:", placeholder="Type here...")
    
    if user_input:
        cleaned = clean_text(user_input)
        label, emoji = get_sentiment(cleaned)
        
        if label == "Positive": st.success(f"Result: {label} {emoji}")
        elif label == "Negative": st.error(f"Result: {label} {emoji}")
        else: st.info(f"Result: {label} {emoji}")

    st.divider()

    # Data Display
    st.subheader("📋 Dataset Overview (reviews.csv)")
    st.dataframe(data[["Review", "Sentiment_Label"]], use_container_width=True)
    
    # Simple Chart
    st.bar_chart(counts)

else:
    st.error("Error: 'reviews.csv' not found. Please upload it to your GitHub repository.")