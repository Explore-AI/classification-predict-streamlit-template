"""
    Simple Streamlit webserver application for serving developed classification
	models.
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
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
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.write("# Climate Change Tweet Classifer")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home page", "Prediction", "Overview", "Deniers", "Neutrals", "Believers", "Factuals"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the Home page
	if selection == "Home page":
		st.write("Identifying your audience's stance on climate change" +
				 " may reveal important insights about them, such as their " +
				 "personal values, their political inclination, and web behaviour.")
		st.write("This tool allows you to imput sample text from your target audience "+
				 " and select a machine learning model to predict whether the author of "+
				 " that text")
		st.write("* Believes in climate change")
		st.write("* Denies climate change")
		st.write("* Is neutral about climate change")
		st.write("* Provided a factual link to a news site")
		st.write("You can also view an exploratory analysis about each category to gain deeper insights "+
				 "about each category.")
		st.write("Select Prediction in the side bar to get started.")

	# Building out the "Information" page
	if selection == "Overview":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("info")
		st.write(sns.countplot(x='sentiment', data = raw))

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
		

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter sample text of your audience","Type Here")

		# Allow user to select algorithm
		algorithm = st.selectbox("Select an algorithm to make the prediction",
							['Support Vector Classifier', 'Random Forest',
							'K-nearest Neighbours', 'Logistic Regression'])
		
		# Classify using SVC
		if algorithm=='Support Vector Classifier':
			if st.button("Predict using Support Vector Classifier"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == 0:
					st.success('Neutral')
					#st.success("Text Categorized as: {}".format(prediction))
				if prediction == -1:
					st.success('Climate change denier')
				if prediction == 2:
					st.success('Provides link to factual news source')
				if prediction == 1:
					st.success('Climate change believer')

		# Classify using Random Forest
		if algorithm=='Random Forest':
			if st.button("Predict using Random Forest"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == 0:
					st.success('Neutral')
					#st.success("Text Categorized as: {}".format(prediction))
				if prediction == -1:
					st.success('Climate change denier')
				if prediction == 2:
					st.success('Provides link to factual news source')
				if prediction == 1:
					st.success('Climate change believer')
		
		# Classify using K-nearest Neighbours
		if algorithm=='K-nearest Neighbours':
			if st.button("Predict using k-nearest neighbours"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == 0:
					st.success('Neutral')
					#st.success("Text Categorized as: {}".format(prediction))
				if prediction == -1:
					st.success('Climate change denier')
				if prediction == 2:
					st.success('Provides link to factual news source')
				if prediction == 1:
					st.success('Climate change believer')

		# Classify using Logistic Regression
		if algorithm=='Logistic Regression':
			if st.button("Predict using Logistic Regression"):
				# Transforming user input with vectorizer
				#vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/log_model.pkl"),"rb"))
				tweet_text = [tweet_text]
				prediction = predictor.predict(tweet_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == 0:
					st.success('Neutral')
					#st.success("Text Categorized as: {}".format(prediction))
				if prediction == -1:
					st.success('Climate change denier')
				if prediction == 2:
					st.success('Provides link to factual news source')
				if prediction == 1:
					st.success('Climate change believer')
		
		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()