import pickle
import joblib
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Clean text function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_model_from_reviews():
    print("Loading reviews data...")
    # Load the reviews data
    df = pd.read_csv('reviews_cleaned.csv')
    
    # Prepare the data
    print("Preparing data...")
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    
    # Map ratings to sentiment
    # 1-2: negative, 3: neutral, 4-5: positive
    def rating_to_sentiment(rating):
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'
    
    df['sentiment'] = df['rating'].apply(rating_to_sentiment)
    
    # Create and fit vectorizer
    print("Creating vectorizer...")
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['sentiment']
    
    # Train model
    print("Training model...")
    model = MultinomialNB()
    model.fit(X, y)
    
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
    test_texts = [
        "This product is amazing! I love it!",
        "It's okay, nothing special.",
        "Terrible quality, waste of money."
    ]
    
    print("\nTesting model:")
    for text in test_texts:
        cleaned = clean_text(text)
        X_test = vectorizer.transform([cleaned])
        pred = model.predict(X_test)[0]
        prob = model.predict_proba(X_test)
        print(f"Text: {text}")
        print(f"Predicted: {pred}")
        print(f"Confidence: {max(prob[0]):.2%}")
        print("-" * 40)

if __name__ == "__main__":
    create_model_from_reviews()
