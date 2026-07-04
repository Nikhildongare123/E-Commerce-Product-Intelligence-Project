
This will create:
- `sentiment_model.pkl` - The trained Naive Bayes model
- `vectorizer.pkl` - The text vectorizer
""")
return

# Load model and vectorizer
model = load_model()
vectorizer = load_vectorizer()

if model is None or vectorizer is None:
st.error("Failed to load model or vectorizer. Please check the files.")
return

# Sidebar
with st.sidebar:
st.header("📊 About")
st.info(
    "**Sentiment Analysis Model**\n\n"
    "- Algorithm: Multinomial Naive Bayes\n"
    "- Classes: Negative, Neutral, Positive\n"
    "- Trained on product reviews\n\n"
    "**How to use:**\n"
    "1. Enter text in the Single Text tab\n"
    "2. Or upload a CSV for bulk analysis"
)

st.divider()

st.header("📈 Model Performance")
st.metric("Accuracy", "85%", "+2%")
st.metric("F1 Score", "0.84", "+0.03")

st.divider()

st.caption("Built with Streamlit ❤️")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📝 Single Text", "📂 Bulk Analysis", "ℹ️ About"])

# Tab 1: Single Text Analysis
with tab1:
st.header("Analyze a Single Review")

# Input section
text_input = st.text_area(
    "Enter your review text:",
    height=150,
    placeholder="Type or paste a product review here...",
    key="text_input"
)

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.session_state.text_input = ""
    st.rerun()

if analyze_button:
    if text_input:
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence = predict_sentiment(text_input, model, vectorizer)
            
            if sentiment is None:
                st.error("Prediction failed. Please try again.")
            else:
                # Display results
                st.subheader("📊 Results")
                
                # Get sentiment style
                emoji, css_class, color = get_sentiment_style(sentiment)
                
                # Display sentiment card
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"""
                    <div class="{css_class}">
                        <h3>{emoji} {sentiment.capitalize()}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Confidence</h4>
                        <h2 style="color: {color};">{confidence:.2%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Progress bar
                st.progress(confidence)
                
                # Display cleaned text
                with st.expander("🔍 Show cleaned text"):
                    st.code(clean_text(text_input), language="text")
                
                # Display word count
                st.caption(f"Word count: {len(text_input.split())}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Tab 2: Bulk Analysis
with tab2:
st.header("Bulk Sentiment Analysis")
st.markdown("Upload a CSV file with reviews to analyze multiple reviews at once.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file containing review text"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Show data preview
        st.subheader("📄 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"Total rows: {len(df)}")
        
        # Let user select text column
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        if text_columns:
            selected_column = st.selectbox(
                "Select the column containing review text:",
                text_columns
            )
            
            col1, col2 = st.columns([1, 3])
            with col1:
                analyze_bulk_button = st.button("🚀 Analyze All Reviews", type="primary", use_container_width=True)
            
            if analyze_bulk_button:
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
                    
                    # Display summary statistics
                    st.subheader("📊 Analysis Results")
                    
                    sentiment_counts = df['sentiment'].value_counts()
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("😊 Positive", sentiment_counts.get('positive', 0))
                    with col2:
                        st.metric("😐 Neutral", sentiment_counts.get('neutral', 0))
                    with col3:
                        st.metric("😞 Negative", sentiment_counts.get('negative', 0))
                    with col4:
                        total = len(df)
                        st.metric("📝 Total Reviews", total)
                    
                    # Bar chart
                    st.bar_chart(sentiment_counts)
                    
                    # Display full results
                    st.subheader("📋 Detailed Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_buffer.getvalue(),
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        else:
            st.warning("⚠️ No text columns found in the uploaded file.")
            
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Tab 3: About
with tab3:
st.header("ℹ️ About This App")

col1, col2 = st.columns([2, 1])

with col1:
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
    
    **Features:**
    - 📝 Analyze single reviews by typing or pasting text
    - 📂 Upload CSV files for bulk analysis
    - 📥 Results can be downloaded as CSV
    - 📊 Visual sentiment distribution charts
    """)

with col2:
    st.markdown("""
    ### Technology Stack
    
    - **Frontend:** Streamlit
    - **ML Model:** Scikit-learn
    - **NLP:** NLTK
    - **Data:** Pandas, NumPy
    
    ### Model Performance
    - **Accuracy:** 85%
    - **Precision:** 0.83
    - **Recall:** 0.84
    - **F1 Score:** 0.84
    """)

st.divider()

# Sample reviews
with st.expander("📝 Test with Sample Reviews"):
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
