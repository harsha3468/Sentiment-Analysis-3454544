import streamlit as st
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Web & Text Sentiment AI", page_icon="🌐", layout="wide")

# --- 2. HELPER FUNCTIONS ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentiment_textblob(text):
    # TextBlob returns polarity from -1 (Neg) to 1 (Pos)
    analysis = TextBlob(text)
    score = analysis.sentiment.polarity
    if score > 0.1: return "Positive", "😊", "#d4edda"
    elif score < -0.1: return "Negative", "😟", "#f8d7da"
    else: return "Neutral", "😐", "#e2e3e5"

# --- 3. MAIN UI ---
st.title("🌐 Web Scraper & Sentiment AI")
st.markdown("Analyze sentiment from **text**, **CSV files**, or **live websites**.")

# Tab Selection
tab1, tab2 = st.tabs(["🔍 Text/Web Analysis", "📋 CSV Batch Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Option A: Manual Text")
        user_input = st.text_area("Enter text to analyze:", height=150)
        
    with col2:
        st.subheader("Option B: Web Scraper")
        url_input = st.text_input("Paste a URL to scrape (e.g., a news article):")
        if st.button("Scrape & Analyze"):
            try:
                response = requests.get(url_input)
                soup = BeautifulSoup(response.text, 'html.parser')
                # Grab all paragraph text
                paragraphs = soup.find_all('p')
                user_input = " ".join([p.get_text() for p in paragraphs])
                st.success("Webpage scraped successfully!")
            except Exception as e:
                st.error(f"Could not scrape URL: {e}")

    if user_input:
        label, emoji, color = get_sentiment_textblob(user_input)
        st.markdown(f"""
            <div style="background-color:{color}; padding:30px; border-radius:15px; text-align:center; border: 1px solid #ccc;">
                <h1>{emoji} {label}</h1>
                <p>Analyzed Text Preview: {user_input[:200]}...</p>
            </div>
        """, unsafe_allow_html=True)

with tab2:
    st.subheader("Upload reviews.csv")
    uploaded_file = st.file_uploader("Choose your CSV file", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Review" in df.columns:
            df["Sentiment"] = df["Review"].apply(lambda x: get_sentiment_textblob(clean_text(x))[0])
            st.bar_chart(df["Sentiment"].value_counts())
            st.dataframe(df[["Review", "Sentiment"]], use_container_width=True)