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
        st.warning("Vectorizer file not found. Creating a new one...")
        # Create a default vectorizer
        vectorizer = CountVectorizer(max_features=5000)
        # Note: In production, you should have a saved fitted vectorizer
        return vectorizer

# Clean text without nltk dependency
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Simple stopword removal without nltk
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
                  "you'll", "you'd", 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 
                  "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 
                  'theirs', 'themselves', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
                  'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should',
                  'may', 'might', 'must', 'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 
                  'to', 'by', 'in', 'of', 'off', 'so', 'yet', 'up', 'down', 'then', 'now', 'than', 'that',
                  'this', 'these', 'those', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                  'too', 'very', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'then',
                  'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'were', 'when', 'your', 'can', 'said',
                  'there', 'use', 'any', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other',
                  'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like',
                  'him', 'into', 'time', 'has', 'look', 'two', 'more', 'write', 'go', 'see', 'number', 'no', 'way',
                  'could', 'people', 'my', 'than', 'first', 'water', 'been', 'call', 'who', 'oil', 'its', 'now',
                  'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part', 'could', 'this',
                  'that', 'with', 'from', 'have', 'are', 'was', 'were', 'been', 'being', 'will', 'would', 'should'}
    
    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

# Predict function
def predict_sentiment(text, model, vectorizer):
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Vectorize
    try:
        text_vectorized = vectorizer.transform([cleaned_text])
        pred = model.predict(text_vectorized)
        proba = model.predict_proba(text_vectorized)
        
        # Get class names
        class_names = model.classes_
        pred_label = class_names[pred[0]]
        pred_prob = np.max(proba[0])
        
        return pred_label, pred_prob
    except Exception as e:
        # Fallback: use a simple heuristic
        st.warning(f"Prediction error: {e}. Using fallback method.")
        # Simple rule-based fallback
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'wonderful', 
                          'awesome', 'fantastic', 'outstanding', 'superb', 'exceptional', 'recommend', 
                          'satisfied', 'happy', 'impressed', 'quality', 'value', 'worth']
        negative_words = ['bad', 'terrible', 'poor', 'awful', 'horrible', 'worst', 'disappointed', 'waste', 
                          'cheap', 'faulty', 'broken', 'useless', 'terrible', 'terrible', 'regret', 'return',
                          'defective', 'disgusting', 'disappointing', 'problem', 'issue', 'failure']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive", 0.6
        elif negative_count > positive_count:
            return "negative", 0.6
        else:
            return "neutral", 0.4

# Main app
def main():
    st.set_page_config(page_title="Sentiment Analysis App", page_icon="📊", layout="wide")
    
    st.title("📊 Sentiment Analysis App")
    st.markdown("Analyze sentiment of product reviews using a trained Naive Bayes model.")
    
    # Load model and vectorizer
    model = load_model()
    vectorizer = load_vectorizer()
    
    if model is None:
        st.error("❌ Model not found. Please ensure 'sentiment_model.pkl' is in the current directory.")
        st.info("If you're running this app locally, make sure the model file is in the same folder.")
        return
    
    # Sidebar
    st.sidebar.header("Options")
    st.sidebar.info("This app analyzes the sentiment of product reviews.")
    
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
        
        if analyze_button:
            if text_input:
                with st.spinner("Analyzing sentiment..."):
                    sentiment, confidence = predict_sentiment(text_input, model, vectorizer)
                    
                    # Display results
                    st.subheader("📊 Results")
                    
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
                    
                    # Display cleaned text
                    with st.expander("🔍 Show cleaned text"):
                        st.code(clean_text(text_input), language="text")
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    with tab2:
        st.header("Bulk Sentiment Analysis")
        st.markdown("Upload a CSV file with reviews to analyze multiple reviews at once.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("📄 Preview of uploaded data:")
                st.dataframe(df.head())
                
                # Let user select text column
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                if text_columns:
                    selected_column = st.selectbox(
                        "Select the column containing review text:",
                        text_columns
                    )
                    
                    if st.button("🚀 Analyze All Reviews", type="primary"):
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
                            st.subheader("📊 Analysis Results")
                            
                            # Summary statistics
                            sentiment_counts = df['sentiment'].value_counts()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("😊 Positive", sentiment_counts.get('positive', 0))
                            with col2:
                                st.metric("😐 Neutral", sentiment_counts.get('neutral', 0))
                            with col3:
                                st.metric("😞 Negative", sentiment_counts.get('negative', 0))
                            
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
                            
                            # Chart
                            if len(sentiment_counts) > 0:
                                st.bar_chart(sentiment_counts)
                else:
                    st.warning("⚠️ No text columns found in the uploaded file.")
                    
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with tab3:
        st.header("ℹ️ About This App")
        st.markdown("""
        ### Sentiment Analysis Model
        
        This app uses a **Multinomial Naive Bayes** classifier trained on product reviews to predict sentiment.
        
        **Model Details:**
        - **Algorithm:** Multinomial Naive Bayes
        - **Classes:** Negative, Neutral, Positive
        - **Features:** Text preprocessing with stopword removal
        
        **How It Works:**
        1. The app cleans the input text (lowercase, removes punctuation and stopwords)
        2. The text is vectorized using a CountVectorizer
        3. The trained model predicts the sentiment class
        4. Results are displayed with confidence scores
        
        **Usage:**
        - 📝 Analyze single reviews by typing or pasting text
        - 📂 Upload CSV files for bulk analysis
        - 📥 Results can be downloaded as CSV
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
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📝 {review}")
                with col2:
                    st.write(f"**Predicted:** {sentiment}")
                with col3:
                    st.write(f"**Confidence:** {confidence:.2%}")
                st.divider()

if __name__ == "__main__":
    main()
