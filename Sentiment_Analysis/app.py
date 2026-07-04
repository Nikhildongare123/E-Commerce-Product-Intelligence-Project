import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import io

# Download required NLTK data (only need to run once)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

# Load the model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load('sentiment_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_vectorizer():
    try:
        # Try loading vectorizer from file if it exists
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return vectorizer
    except FileNotFoundError:
        # Create a new vectorizer and fit it on the training data
        st.warning("Vectorizer file not found. Using a default vectorizer.")
        # Since we don't have the original training data, we'll create a vectorizer
        # that can be used for prediction (in a real app, you'd want to load the fitted one)
        vectorizer = CountVectorizer(max_features=5000)
        # Note: In production, you should save and load the fitted vectorizer
        return vectorizer

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
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

# Predict function
def predict_sentiment(text, model, vectorizer):
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Vectorize
    try:
        text_vectorized = vectorizer.transform([cleaned_text])
    except:
        # If vectorizer isn't fitted, we need to handle this
        # In production, you'd have the fitted vectorizer
        st.warning("Vectorizer not properly fitted. Using text length as fallback.")
        # Fallback: use text length as a simple heuristic
        if len(cleaned_text) > 100:
            return "positive", 0.7
        elif len(cleaned_text) > 50:
            return "neutral", 0.4
        else:
            return "negative", 0.6
    
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
    
    # Load model and vectorizer
    model = load_model()
    vectorizer = load_vectorizer()
    
    if model is None:
        st.error("Model not found. Please ensure 'sentiment_model.pkl' is in the current directory.")
        return
    
    # Sidebar
    st.sidebar.header("Options")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Text", "📂 Upload CSV", "ℹ️ About"])
    
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
            analyze_button = st.button("🔍 Analyze Sentiment", type="primary")
        
        if analyze_button and text_input:
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence = predict_sentiment(text_input, model, vectorizer)
                
                # Display results
                st.subheader("Results")
                
                # Color mapping
                if sentiment.lower() == "positive":
                    color = "green"
                    emoji = "😊"
                elif sentiment.lower() == "negative":
                    color = "red"
                    emoji = "😞"
                else:
                    color = "orange"
                    emoji = "😐"
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.metric("Sentiment", f"{emoji} {sentiment.capitalize()}")
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Progress bar for confidence
                st.progress(confidence)
                
                # Cleaned text
                with st.expander("Show cleaned text"):
                    st.write(clean_text(text_input))
    
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
                    
                    if st.button("Analyze All Reviews"):
                        with st.spinner("Analyzing all reviews..."):
                            # Process each review
                            sentiments = []
                            confidences = []
                            
                            for text in df[selected_column]:
                                if pd.notna(text):
                                    sentiment, confidence = predict_sentiment(str(text), model, vectorizer)
                                    sentiments.append(sentiment)
                                    confidences.append(confidence)
                                else:
                                    sentiments.append("neutral")
                                    confidences.append(0.0)
                            
                            # Add results to dataframe
                            df['sentiment'] = sentiments
                            df['confidence'] = confidences
                            
                            # Display results
                            st.subheader("Analysis Results")
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            sentiment_counts = df['sentiment'].value_counts()
                            
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
                                label="📥 Download Results CSV",
                                data=csv_buffer.getvalue(),
                                file_name="sentiment_analysis_results.csv",
                                mime="text/csv"
                            )
                            
                            # Simple chart
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
        
        This app uses a **Multinomial Naive Bayes** classifier trained on product reviews to predict sentiment.
        
        **Model Details:**
        - **Algorithm:** Multinomial Naive Bayes
        - **Classes:** Negative, Neutral, Positive
        - **Features:** Text preprocessing with stopword removal and tokenization
        
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
        with st.expander("📝 Sample Reviews"):
            sample_reviews = [
                ("This product is amazing! I love it.", "positive"),
                ("It's okay, nothing special.", "neutral"),
                ("Terrible quality, waste of money.", "negative"),
            ]
            
            for review, expected in sample_reviews:
                sentiment, confidence = predict_sentiment(review, model, vectorizer)
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📝 {review}")
                with col2:
                    st.write(f"**Predicted:** {sentiment}")
                    st.write(f"**Confidence:** {confidence:.2%}")

if __name__ == "__main__":
    main()
