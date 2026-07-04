import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import joblib
import io
import os
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
        st.error("Vectorizer file 'vectorizer.pkl' not found!")
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
    if not os.path.exists('sentiment_model.pkl') or not os.path.exists('vectorizer.pkl'):
        st.error("Model files not found!")
        st.write("Please run 'python extract_model.py' first to create the model files.")
        st.write("This will create:")
        st.write("- sentiment_model.pkl - The trained Naive Bayes model")
        st.write("- vectorizer.pkl - The text vectorizer")
        return
    
    # Load model and vectorizer
    model = load_model()
    vectorizer = load_vectorizer()
    
    if model is None or vectorizer is None:
        st.error("Failed to load model or vectorizer. Please check the files.")
        return
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "Sentiment Analysis Model\n\n"
        "- Algorithm: Multinomial Naive Bayes\n"
        "- Classes: Negative, Neutral, Positive\n"
        "- Trained on product reviews\n\n"
        "How to use:\n"
        "1. Enter text in the Single Text tab\n"
        "2. Or upload a CSV for bulk analysis"
    )
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Single Text", "Upload CSV", "About"])
    
    with tab1:
        st.header("Analyze a Single Review")
        
        # Text input
        text_input = st.text_area(
            "Enter your review text:",
            height=150,
            placeholder="Type or paste a product review here..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.button("Analyze", type="primary")
        
        if analyze_button:
            if text_input:
                with st.spinner("Analyzing sentiment..."):
                    sentiment, confidence = predict_sentiment(text_input, model, vectorizer)
                    
                    if sentiment is None:
                        st.error("Prediction failed. Please try again.")
                    else:
                        # Display results
                        st.subheader("Results")
                        
                        # Color mapping
                        if sentiment.lower() == "positive":
                            emoji = "😊"
                        elif sentiment.lower() == "negative":
                            emoji = "😞"
                        else:
                            emoji = "😐"
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.metric("Sentiment", f"{emoji} {sentiment.capitalize()}")
                        with col2:
                            st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Progress bar for confidence
                        st.progress(confidence)
                        
                        # Display cleaned text
                        with st.expander("Show cleaned text"):
                            st.code(clean_text(text_input), language="text")
            else:
                st.warning("Please enter some text to analyze.")
    
    with tab2:
        st.header("Bulk Sentiment Analysis")
        st.markdown("Upload a CSV file with reviews to analyze multiple reviews at once.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                # Let user select text column
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                if text_columns:
                    selected_column = st.selectbox(
                        "Select the column containing review text:",
                        text_columns
                    )
                    
                    if st.button("Analyze All Reviews", type="primary"):
                        with st.spinner("Analyzing all reviews..."):
                            # Process each review
                            sentiments = []
                            confidences = []
                            
                            for text in df[selected_column]:
                                if pd.notna(text):
                                    sentiment, confidence = predict_sentiment(str(text), model, vectorizer)
                                    sentiments.append(sentiment if sentiment else "neutral")
                                    confidences.append(confidence if confidence else 0.0)
                                else:
                                    sentiments.append("neutral")
                                    confidences.append(0.0)
                            
                            # Add results to dataframe
                            df['sentiment'] = sentiments
                            df['confidence'] = confidences
                            
                            # Display results
                            st.subheader("Analysis Results")
                            
                            # Summary statistics
                            sentiment_counts = df['sentiment'].value_counts()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Positive", sentiment_counts.get('positive', 0))
                            with col2:
                                st.metric("Neutral", sentiment_counts.get('neutral', 0))
                            with col3:
                                st.metric("Negative", sentiment_counts.get('negative', 0))
                            
                            # Display full results
                            st.dataframe(df)
                            
                            # Download button for results
                            csv_buffer = io.StringIO()
                            df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="Download Results CSV",
                                data=csv_buffer.getvalue(),
                                file_name="sentiment_analysis_results.csv",
                                mime="text/csv"
                            )
                            
                            # Chart
                            if len(sentiment_counts) > 0:
                                st.bar_chart(sentiment_counts)
                else:
                    st.warning("No text columns found in the uploaded file.")
                    
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with tab3:
        st.header("About This App")
        st.markdown("""
        ### Sentiment Analysis Model
        
        This app uses a Multinomial Naive Bayes classifier trained on product reviews to predict sentiment.
        
        **Model Details:**
        - Algorithm: Multinomial Naive Bayes
        - Classes: Negative, Neutral, Positive
        - Features: Text preprocessing with stopword removal
        
        **How It Works:**
        1. The app cleans the input text (lowercase, removes punctuation and stopwords)
        2. The text is vectorized using a CountVectorizer
        3. The trained model predicts the sentiment class
        4. Results are displayed with confidence scores
        
        **Usage:**
        - Analyze single reviews by typing or pasting text
        - Upload CSV files for bulk analysis
        - Results can be downloaded as CSV
        """)
        
        # Display sample reviews
        with st.expander("Test with Sample Reviews"):
            sample_reviews = [
                "This product is amazing! I absolutely love it. The quality is outstanding and it exceeded all my expectations.",
                "It's an okay product. Nothing special but it gets the job done. I might upgrade later.",
                "Terrible quality, waste of money. The product broke within the first week of use.",
            ]
            
            for review in sample_reviews:
                sentiment, confidence = predict_sentiment(review, model, vectorizer)
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📝 {review[:100]}...")
                with col2:
                    emoji = "😊" if sentiment == "positive" else "😞" if sentiment == "negative" else "😐"
                    st.write(f"{emoji} **{sentiment.capitalize()}**")
                with col3:
                    st.write(f"**{confidence:.2%}**")
                st.divider()

if __name__ == "__main__":
    main()
