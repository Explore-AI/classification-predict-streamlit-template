import streamlit as st
import joblib, os
import time
import functions.preprocessor as cleanText
from nlppreprocess import NLP
nlp = NLP()

vectorizer = open('./resources/tfidfvect.pkl','rb')   ##  will be replaced by the cleaning and preprocessing function
tweet_cv = joblib.load(vectorizer)

# getting the predictions
def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key
def load_prediction_models(model_file):
    loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_models

def classify(model_choice, input_text, prediction_labels):
    text = cleanText.cleaner(input_text)
    vect_text = tweet_cv.transform([text]).toarray()
    
    if model_choice == 'Logistic Regression':
        predictor = load_prediction_models("./resources/Logistic_regression.pkl")
        prediction = predictor.predict(vect_text)
    elif model_choice == 'Random Forest':
        predictor = load_prediction_models("./resources/Random_model.pkl")
        prediction = predictor.predict(vect_text)
    elif model_choice == 'Decision Tree':
        predictor = load_prediction_models("./resources/Dec_tree_model.pkl")
        prediction = predictor.predict(vect_text)

    classified = get_keys(prediction, prediction_labels)
    my_bar = st.progress(0)

    with st.spinner('Classifying Tweet...'):
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
    my_bar.empty()

    return '{}'.format(classified)