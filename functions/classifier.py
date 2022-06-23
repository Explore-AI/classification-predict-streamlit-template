
# importing neccessary modules and packages
import streamlit as st
import joblib, os
import time
import functions.preprocessor as cleanText
from nlppreprocess import NLP
nlp = NLP()


vectorizer = open('./resources/tfidfvect.pkl','rb')   ##  will be replaced by the cleaning and preprocessing function
tweet_cv = joblib.load(vectorizer) # loading vectorizer

# getting the predictions
def get_keys(val,my_dict):

    '''This function will take the values predicted, and return their corresponding keys'''

    for key,value in my_dict.items():
        if val == value:
            return key


def load_prediction_models(model_file): 

    '''
    This function will allow us to load the given model_file to use 
    when making predictions 
    '''

    loaded_models = joblib.load(open(os.path.join(model_file),"rb")) # setting variable "loaded_models" to selected model 
    return loaded_models


def classify(model_choice, input_text, prediction_labels):

    '''This function takes input from the user, cleans the text, removes noise and predicts the sentiment of the text by predicting with 
       one of the 3 models below which the user will select'''

    text = cleanText.cleaner(input_text) # cleaning the text using preprocessing function
    vect_text = tweet_cv.transform([text]).toarray() # transforming cleaned text to an array, setting it to "vect_text"

    # prediction will be made on which model user selects
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

    # adding loading feature so that user can track prediction progress
    with st.spinner('Classifying Tweet...'):
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
    my_bar.empty()

    return '{}'.format(classified)