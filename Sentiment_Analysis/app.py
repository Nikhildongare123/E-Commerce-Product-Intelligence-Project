import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import joblib
import io
from sklearn.feature_extraction.text import CountVectorizer

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('sentiment_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load vectorizer
@st.cache_resource
def load_vectorizer():
    try:
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return vectorizer
    except FileNotFoundError:
        st.error("❌ Vectorizer file 'vectorizer.pkl' not found!")
        st.info("Please run 'python extract_model.py' first to create the model files.")
        return None

# Clean text function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Predict function
def predict_sentiment(text, model, vectorizer):
    if model is None or vectorizer is None:
        return None, None
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Vectorize
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Predict
    pred = model.predict(text_vectorized)
    proba = model.predict_proba(text_vectorized)
    
    # Get class names
    class_names = model.classes_
    pred_label = class_names[pred[0]]
    pred_prob = np.max(proba[0])
    
    return pred_label, pred_prob

# Main app
def main():
    st.set_page_config(page_title="Sentiment Analysis App", page_icon="📊", layout="wide")
    
    st.title("📊 Sentiment Analysis App")
    st.markdown("Analyze sentiment of product reviews using a trained Naive Bayes model.")
    
    # Check if model files exist
    import os
    if not os.path.exists('sentiment_model.pkl') or not os.path.exists('vectorizer.pkl'):
        st.error("""
        ❌ **Model files not found!**
        
        Please run the following command first to create the model:
