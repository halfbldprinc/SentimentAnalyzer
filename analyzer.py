import joblib
from preprocess import preprocess_text
import nltk

# Load trained model
model = joblib.load('sentiment_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

# Run the model with argument 
def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vec = tfidf.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    return {
        'negative': float(round(proba[0] * 100, 2)),
        'positive': float(round(proba[1] * 100, 2))
    }

if __name__ == "__main__":
    print("Sentiment Analyzer")
    while True:
        user_input = input("\nEnter text (type 'exit' to quit) : ")
        if user_input.lower() == "exit":
            break
        result = predict_sentiment(user_input)
        print(f"Prediction: {result}")
