import pickle
import joblib
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_model_from_reviews():
    """Create and train the sentiment analysis model"""
    print("="*60)
    print("Starting Sentiment Model Training")
    print("="*60)
    
    # Find the CSV file
    possible_paths = [
        'reviews_cleaned.csv',
        '../reviews_cleaned.csv',
        '../../reviews_cleaned.csv',
        '/mount/src/e-commerce-product-intelligence-project/reviews_cleaned.csv',
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found file at: {path}")
            df = pd.read_csv(path)
            break
    
    if df is None:
        print("ERROR: reviews_cleaned.csv not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files: {os.listdir('.')}")
        return
    
    print(f"Loaded {len(df)} reviews")
    
    # Prepare the data
    print("Preparing data...")
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    
    # Map ratings to sentiment
    def rating_to_sentiment(rating):
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'
    
    df['sentiment'] = df['rating'].apply(rating_to_sentiment)
    
    # Print distribution
    print("Sentiment Distribution:")
    print(df['sentiment'].value_counts())
    
    # Create and fit vectorizer
    print("Creating vectorizer...")
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("Training model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Model Evaluation:")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    print("Saving model...")
    joblib.dump(model, 'sentiment_model.pkl')
    
    print("Saving vectorizer...")
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Model and vectorizer saved successfully!")
    print(f"Model classes: {model.classes_}")
    print(f"Vectorizer features: {len(vectorizer.get_feature_names_out())}")
    
    # Test the model
    print("Testing model with sample reviews:")
    test_texts = [
        "This product is amazing! I absolutely love it!",
        "It's okay, nothing special. It works fine.",
        "Terrible quality, complete waste of money!",
    ]
    
    for text in test_texts:
        cleaned = clean_text(text)
        X_test_sample = vectorizer.transform([cleaned])
        pred = model.predict(X_test_sample)[0]
        prob = model.predict_proba(X_test_sample)
        print(f"Text: {text}")
        print(f"Predicted: {pred}")
        print(f"Confidence: {max(prob[0]):.2%}")
        print("-" * 40)
    
    print("="*60)
    print("Training Complete!")
    print("="*60)

if __name__ == "__main__":
    create_model_from_reviews()
