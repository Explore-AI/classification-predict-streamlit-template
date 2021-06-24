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
# Streamlit dependencies
import re
import nltk
import joblib,os
import streamlit as st

#import contractions
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
import warnings

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings(action = 'ignore') 

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Data dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Vectorizer
news_vectorizer = open("resources/count_vect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

def display_prediction(input_text):
    if input_text[0]==-1:
        output="Anti"
        st.error(f"{output} Sentiment Predicted")
        st.error('Tweets that do not support the belief of man-made climate change.')
    elif input_text[0]==0:
        output="Neutral"
        st.info(f"{output} Sentiment Predicted")
        st.info("Tweets that neither support nor refuse beliefs of climate change.")
    elif input_text[0]==1:
        output ="Pro"
        st.success(f"{output} Sentiment Predicted")
        st.success("Tweets that support the belief of man-made climate change")
    else:
        output = "News"
        st.warning(f"{output} Sentiment Predicted")
        st.warning("Tweets linked to factual news about climate change.")
    
# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifier")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["About The App", "Prediction", "Data Visualisation", "Model Performance"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "About The App":
		st.image('resources/imgs/changes.gif', caption='Climate Change',use_column_width=True)
		st.subheader("About the App")
		st.info("The entire app is built using Machine Learning models that is able to classify whether or not a person believes in climate change, based on their novel tweet data.")
		# You can read a markdown file from supporting resources folder
		st.markdown("Below is just the small portion of dataset that has been used to train the models")

		st.subheader("Introduction")
		st.markdown(
			"""It is undeniable that climate change is one of the most talked topics of our times and one of the biggest challenges the world is facing today. 
			In the past few years, we have seen a steep rise on the Earth's temperature, causing a spike in wild fires, drought, rise of sea levels due to melting glaciers, rainfall pattern shifts, flood disasters.
			"""
			)

		st.subheader("Why App Was Created ?")
		st.markdown('The aim of this App is to gauge the public perception of climate change using twitter data')

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	if selection == "Prediction":
		# Creating a text box for user input
		models_used = ["LogisticRegression", "RidgeClassifier", "LinearSVC", "SGDClassifier", "Support Vector Machine"]
		selected_model = st.selectbox('Choose Your Favourite Model', models_used)

		if selected_model =="RidgeClassifier":
				st.subheader('Model Info Below')
				st.info("This classifier first converts the target values into {-1, 1} and then treats the problem as a regression task (multi-output regression in the multiclass case)")
				written_text = st.text_area("Write Text Below", "Type Here")
				if st.button("Predict Text"):
					predictor = joblib.load(open(os.path.join("resources/RidgeClassifier.pkl"),"rb"))
					predict = predictor.predict([written_text])
					display_prediction(predict)

		if selected_model =="SGDClassifier":
				st.subheader('Model Info Below')
				st.info("This estimator implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate)")
				written_text = st.text_area("Write Text Below", "Type Here")
				if st.button("Predict Text"):
					predictor = joblib.load(open(os.path.join("resources/SGDClassifier.pkl"),"rb"))
					predict = predictor.predict([written_text])
					display_prediction(predict)

		if selected_model == "LinearSVC":
				st.subheader('Model Info Below')
				st.info("This Classifier fit to the data you provide, returning a `best fit` hyperplane that divides, or categorizes, your data. From there, after getting the hyperplane, you can then feed some features to your classifier to see what the `predicted` class is.")
				written_text = st.text_area("Write Text Below", "Type Here")
				if st.button("Predict Text"):
					predictor = joblib.load(open(os.path.join("resources/LinearSVC.pkl"),"rb"))
					predict = predictor.predict([written_text])
					display_prediction(predict)

		if selected_model =="Support Vector Machine":
				st.subheader('Model Info Below')
				st.info("This Classifier finds a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.")
				written_text = st.text_area("Write Text Below", "Type Here")
				if st.button("Predict Text"):
					predictor = joblib.load(open(os.path.join("resources/Support_Vector_Machine.pkl"),"rb"))
					predict = predictor.predict([written_text])
					display_prediction(predict)

		if selected_model =="LogisticRegression":
				st.subheader('Model Info Below')
				st.info("Logistic regression models the probabilities for classification problems with two possible outcomes. It's an extension of the linear regression model for classification problems.")
				written_text = st.text_area("Write Text Below", "Type Here")
				if st.button("Predict Text"):
					predictor = joblib.load(open(os.path.join("resources/LogisticRegression.pkl"),"rb"))
					predict = predictor.predict([written_text])
					display_prediction(predict)
		
	if selection == "Data Visualisation":
		pass

	if selection == "Model Performance":
		model_selected = ["LogisticRegression", "RidgeClassifier", "LinearSVC", "SGDClassifier", "Support Vector Machine"]
		selected_model = st.selectbox("Choose Model Metrics By Model Type", model_selected)
		if selected_model =="LinearSVC":
			pass

		if selected_model =="Support Vector Machine":
			pass

		if selected_model =="RidgeClassifier":
			pass

		if selected_model =="LogisiticRegression":
			pass

		if selected_model =="SGDClassifier":
			pass

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
