import pandas as pd
import streamlit as st
import joblib
import os

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
other_vectorizer = open("resources/CountVect.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)
other_cv = joblib.load(other_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")

sentiment_desc = {"-1": "Anti", "0": "Neutral", "1": "Pro", "2": "News"}

# Takes a tweet and classifies it with a given model


def select_model(model_name):
    tweet_text = st.text_area("Enter Text", "Type Here")
    if st.button("Classify"):
        # Load your .pkl file with the model of your choice + make predictions
        # Try loading in multiple models to give the user a choice
        if model_name == "Logistic_regression":
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            predictor = joblib.load(
                open(os.path.join(f"resources/{model_name}.pkl"), "rb"))
            prediction = predictor.predict(vect_text)
            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success(f"Your tweet has been categorized as: {sentiment_desc[str(prediction[0])]}")

        elif model_name == "SVM":
            # Transforming user input with vectorizer
            vect_text = other_cv.transform([tweet_text]).toarray()
            predictor = joblib.load(
                open(os.path.join(f"resources/{model_name}.pkl"), "rb"))
            prediction = predictor.predict(vect_text)
            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success(f"Your tweet has been categorized as: {sentiment_desc[str(prediction[0])]}")
        else:
            st.fail('Classification model missing')
