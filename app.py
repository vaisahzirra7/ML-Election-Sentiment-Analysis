from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load your saved model/vectorizer
model = joblib.load("model/sentiment_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")

# Text cleaning (same as your preprocessing pipeline)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Remove non-alphabetical characters
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

# Predict sentiment for the cleaned tweet
def predict_sentiment(tweet):
    cleaned_tweet = clean_text(tweet)
    vect = tfidf.transform([cleaned_tweet])
    sentiment = model.predict(vect)[0]
    return sentiment

# Detect candidate from the tweet using dynamic keyword matching
def detect_candidate(tweet, candidate_keywords):
    if not isinstance(tweet, str):  # Ensure tweet is a string
        return "Unknown"
    tweet = tweet.lower()
    for candidate, keywords in candidate_keywords.items():
        for kw in keywords:
            if kw in tweet:
                return candidate
    return "Unknown"  # If no candidate is detected

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    csv_result = None
    election_winner = None

    # Dynamically generate candidate keywords (could be loaded from a config or database in the future)
    candidate_keywords = {
        "Candidate A": ["keyword1", "keyword2", "candidate a", "A"],
        "Candidate B": ["keyword3", "keyword4", "candidate b", "B"],
        # Add more candidates and keywords dynamically as needed
    }

    # Sentiment mapping (can be adjusted easily for other labels in the future)
    sentiment_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
    emoji_map = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}

    if request.method == "POST":
        # Single-tweet analysis
        if "tweet" in request.form:
            tweet = request.form["tweet"]
            sentiment = predict_sentiment(tweet)
            sentiment_str = sentiment_map[sentiment]
            prediction = f"{emoji_map[sentiment_str]} {sentiment_str.capitalize()}"

        # CSV batch analysis
        elif "file" in request.files:
            file = request.files["file"]
            df = pd.read_csv(file)
            
            # Apply text cleaning and sentiment prediction
            df["cleaned"] = df["Tweets"].apply(clean_text)
            df["Predicted_Sentiment"] = df["cleaned"].apply(predict_sentiment)
            df["Candidate"] = df["Tweets"].apply(lambda tweet: detect_candidate(tweet, candidate_keywords))
            
            # Group by candidate and analyze sentiments
            candidate_sentiment = df.groupby("Candidate")["Predicted_Sentiment"].value_counts().unstack(fill_value=0)

            # Rename the sentiment columns to ensure consistency
            candidate_sentiment = candidate_sentiment.rename(columns={1: "Positive", 0: "Neutral", -1: "Negative"})
            
            # Ensure all sentiment columns exist (Positive, Neutral, Negative)
            candidate_sentiment = candidate_sentiment.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
            
            # Calculate Net Sentiment (Positive - Negative)
            candidate_sentiment["Net Sentiment"] = candidate_sentiment["Positive"] - candidate_sentiment["Negative"]
            
            # Find the candidate with the highest net sentiment
            election_winner = candidate_sentiment["Net Sentiment"].idxmax()

            # Prepare the results for display and download
            csv_result = df[["Tweets", "Candidate", "Predicted_Sentiment"]].to_dict(orient="records")
            df[["Tweets", "Candidate", "Predicted_Sentiment"]].to_csv("static/predicted.csv", index=False)

    return render_template("index.html", prediction=prediction, csv_result=csv_result, election_winner=election_winner)

if __name__ == "__main__":
    app.run(debug=True)