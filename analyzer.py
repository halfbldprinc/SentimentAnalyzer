import joblib
from preprocess import preprocess_text
import json

# Load trained model
model = joblib.load('sentiment_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

# Load Word List (key - value json)
with open("sentiment_wordlist.json") as f: 
    sentiment_weights = json.load(f) 

# Run the model with argument 
def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vec = tfidf.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    return {
        'negative': float(round(proba[0] * 100, 2)),
        'positive': float(round(proba[1] * 100, 2))
    }
def weighted_sentiment(text):
    cleaned = preprocess_text(text)
    words = cleaned.split()
    pos_score = 0.0
    neg_score = 0.0
# can change the value from - to +
    negations = {"not", "no", "never", "none", "cannot", "can't", "don't", "doesn't", "isn't", "wasn't", "aren't", "won't", "shouldn't", "wouldn't", "couldn't"}
    boosters = {"very": 1.5, "extremely": 2.0, "really": 1.4, "so": 1.3, "too": 1.2, "quite": 1.1, "somewhat": 0.9, "slightly": 0.8, "barely": 0.7}

    negation_count = 0
    booster_multiplier = 1.0

    i = 0
    while i < len(words):
        word = words[i]
        lowered = word.lower()

        if lowered in negations:
            negation_count += 1
            booster_multiplier = 1.0
            i += 1
            continue

        elif lowered in boosters:
            booster_multiplier *= boosters[lowered]
            i += 1
            continue

        else:
            weight = sentiment_weights.get(lowered, 0.0)
            if weight != 0.0:
                if negation_count % 2 == 1:
                    weight = -weight * 0.7

                weight *= booster_multiplier
                weight = max(min(weight, 10.0), -10.0)

                if weight > 0:
                    pos_score += weight
                else:
                    neg_score += abs(weight)

                negation_count = 0
                booster_multiplier = 1.0
            else:
                booster_multiplier = 1.0
                negation_count = 0

            i += 1

   
    return {
        "negative": neg_score,
        "positivee": pos_score,
    }


if __name__ == "__main__":
    print("Sentiment Analyzer")
    while True:
        user_input = input("\nEnter text (type 'exit' to quit) : ")
        if user_input.lower() == "exit":
            break
        print("\nChoose an analysis method:")
        print("1 - Machine Learning Model")
        print("2 - Weigthing Rule-Based")
        print("3 - Both")
        choice = input("Your choice (1/2/3): ") 
        if choice == "1":
            model_result = predict_sentiment(user_input)
            print(f"\nML Model : {model_result}") 
        elif choice == "2":
            lexicon_result = weighted_sentiment(user_input)
            print(f"\nWeighted Scoring: {lexicon_result}") 
        elif choice == "3":
            model_result = predict_sentiment(user_input)
            lexicon_result = weighted_sentiment(user_input)
            print(f"\nModel Prediction: {model_result}") 
            print(f"Weighted Scoring: {lexicon_result}") 
        else:
            print("Invalid choice, please type 1, 2, or 3.") 
