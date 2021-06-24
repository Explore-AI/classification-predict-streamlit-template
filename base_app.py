"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
import os

import joblib
# Data dependencies
import pandas as pd
# Streamlit dependencies
import streamlit as st
import cv2


# Preprocessing dependencies

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import io
from PIL import Image

# nltk.download('punkt')
# nltk.download('stopwords')

# Vectorizer
# @st.cache
def load_vectorizer(file_path):
    vector = open(file_path, "rb")
    loaded_vector = joblib.load(vector)
    return loaded_vector


#Noise removal:
def remove_punctuation_numbers(post):
    punc_numbers = string.punctuation + '0123456789'
    return ''.join([l for l in post if l not in punc_numbers])


def remove_urls(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    return text


def rm_stopwords(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    return tokens_without_sw


def removeNonAscii(s): 
    new_string = ""

    for i in s:
        if ord(i)<128:
            new_string += (" " + i)

    return new_string


def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += (" " + ele)  
    
    # return string  
    return str1 

# loading your vectorizer from the pkl file
tfidf_vect = load_vectorizer('resources/pickles/TfidfVectorizer.pkl')

count_vect = CountVectorizer(stop_words='english', min_df= .01)

lr_model = load_vectorizer("resources/pickles/lr_model.pkl")
lsvc_model = load_vectorizer("resources/pickles/lsvc_model.pkl")
svc_model = load_vectorizer("resources/pickles/svc_model.pkl")
rfc_model = load_vectorizer("resources/pickles/rfc_model.pkl")
mnb_model = load_vectorizer("resources/pickles/mnb_model.pkl")




# Load your raw data
raw = pd.read_csv("resources/train.csv")


# The main function where we will build the actual app


def main():
    """Tweet Classifier App with Streamlit """ 
    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Tweet Classifer")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Tutorial", "Model and Data Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Defining model descriptions
    model_paths = {"Logistic Regression": lr_model,
                   "LinearSVC": lsvc_model, 
                   "Support Vector Classifier": svc_model,
                   "MultinomialNB" : mnb_model,
                   "Random Forest": rfc_model}

    model_descriptions = {
                          "Logistic Regression": "Logistic regression is a statistical model that makes use of a logistic function to model \
                           a binary dependent variable, however, multiclass classification with logistic regression can be done through the \
                           one-vs-rest scheme in which a separate model is trained for each class to predict whether an observation is that \
                           class or not (thus making it a binary classification problem)",

                          "LinearSVC":  "The objective of a Linear Support Vector Classifier is to return a 'best fit' hyperplane that categorises\
                           the data. It is similar to SVC with the kernel parameter set to ’linear’, but it is implemented in terms of liblinear rather\
                           than libsvm, so it has more flexibility in the choice of penalties and loss functions and can scale better to large numbers of samples.",

                          "Support Vector Classifier" : "A Support Vector Classifier is a discriminative classifier formally defined by a separating hyperplane.\
                           When labelled training data is passed to the model, also known as supervised learning, the algorithm outputs an optimal hyperplane\
                           which categorizes new data",

                          "MultinomialNB": "The multinomial Naive Bayes classifier is suitable for classification with discrete features\
                           (e.g., word counts for text classification)",

                          "Random Forest": "Random forest models are an example of an ensemble method that is built on decision trees\
                           (i.e. it relies on aggregating the results of an ensemble of decision trees). \
                           Decision tree machine learning models represent data by partitioning it into different sections based on questions asked \
                           of independent variables in the data. Training data is placed at the root node and is then partitioned into smaller subsets \
                           which form the 'branches' of the tree. In random forest models, the trees are randomized and the model returns the mean prediction\
                           of all the individual trees"
                        }

    
    model_accuracies = {"Logistic Regression": 0.7605,
                        "LinearSVC": 0.7593, 
                        "Support Vector Classifier": 0.7491,
                        "MultinomialNB" : 0.7042,
                        "Random Forest": 0.6905}
    
    
    if selection == "Model and Data Information":
        
        chosen = st.radio('Model Selection', (list(model_descriptions.keys())))
        st.subheader(f"{chosen} (Accuracy: {round(model_accuracies[chosen]*100)}%)")
        st.markdown(model_descriptions[chosen])

        image = Image.open('resources\imgs\Accuracy_scores.png')
        st.image(image, caption='Model Accuracies (F1-score)')
        st.markdown("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            # will write the df to the page
            st.write(raw[['sentiment', 'message']])
        

    # Building out the "Information" page
    if selection == "Tutorial": 

        st.title("Tutorial")
        

        st.markdown("Watch the video below to learn how to use the Tweet Classifier")

        video_file = open('resources\Tutorial.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)


    # Building out the predication page
    if selection == "Prediction":
            st.subheader("Climate change tweet classification")
            st.info("Prediction with a selection of ML Models")
            chosen = st.radio('Model Selection', (list(model_paths.keys())))
			# Creating a text box for user input
            st.subheader(f"{chosen} (Accuracy: {round(model_accuracies[chosen]*100)}%)")
            tweet_text = st.text_area("Type or paste your tweet here:", "Type Here")
            if st.button("Classify"):
                # Transforming user input with vectorizer
                
                text = tweet_text.lower()
                
                text = remove_urls(text)

                text = remove_punctuation_numbers(text)

                vect_text = tfidf_vect.transform([text])

                # Load your .pkl file with the model of your choice + make predictions
                # Try loading in multiple models to give the user a choice
                predictor = model_paths[f"{chosen}"]
                prediction = predictor.predict(vect_text)
                # st.markdown(prediction)

                # When model has successfully run, will print prediction
                # You can use a dictionary or similar structure to make this output
                # more human interpretable.
                if prediction == 2:
                    st.warning("This tweet links to factual news about climate change.")

                elif prediction == 1:
                    st.success("This tweet supports the belief of a man-made climate change.")

                elif prediction == 0:
                    st.info("This tweet neither supports nor refutes the belief of a man-made climate change.")

                elif prediction == -1:
                    st.error("This tweet does not believe in a man made climate change")

            






# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
