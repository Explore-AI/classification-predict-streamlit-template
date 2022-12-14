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
# import images
from PIL import Image

# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Vectorizer 1
news_vectorizer = open("resources/tf_vect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Vectorizer 2
#news_vectorizer = open("resources/tfidfvect.pkl","rb")
#tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train2.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Climate Change Belief Analysis")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["About Us", "Classification 1", "Classification 2", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Home" page
	if selection == "About Us":
		image = Image.open('climate_UN.jpeg')
		st.image(image, caption='Global warming')

		st.info("Team Information - Experts and their Roles")
		# You can read a markdown file from supporting resources folder
		st.markdown("This Web app has been adapted and developed by the J.GAD AI - a group of       \
			five students from the July 2022 cohort of the Explore AI Academy Data Science course.")

		st.subheader("Meet the Team")
		if st.button('Aniedi'): # information is hidden if button is clicked
			st.markdown('Aniedi Oboho-Etuk is a J.GAD AI Developer')
		if st.button('David'): # information is hidden if button is clicked
			st.markdown('David Mugambi is the J.GAD AI Project Manager')
		if st.button('Gavriel'): # information is hidden if button is clicked
			st.markdown('Gavriel Leibovitz is a J.GAD Developer/Strategist')
		if st.button('Josiah'): # information is hidden if button is clicked
			st.markdown('Josiah Aramide is the J.GAD AI CEO')
		if st.button('Joy'): # information is hidden if button is clicked
			st.markdown('Joy Obukohwo is the J.GAD AI Product Owner')

	# Building out the "Information" page
	if selection == "Information":
		st.info("Project Overview")
		# You can read a markdown file from supporting resources folder
		st.markdown("Many companies are built around lessening one\’s environmental impact or carbon footprint. \
		They offer products and services that are environmentally friendly and sustainable,             \
		in line with their values and ideals. They would like to determine how people perceive           \
		climate change and whether or not they believe it is a real threat. This would add to their       \
		market research efforts in gauging how their product/service may be received.                     \n \
		Providing an accurate and robust solution to this task gives companies access to a                \
		broad base of consumer sentiment, spanning multiple demographic and geographic categories          \
		- thus increasing their insights and informing future marketing strategies.")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the prediction 1 page
	if selection == "Classification 1":
		st.info("Classification with Base ML Model")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/lgr_base.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if prediction == 0:
				st.success(f"Text Categorized as: Neutral or Class {prediction}")
			if prediction == 1:
				st.success(f"Text Categorized as: Pro-climate change or Class {prediction}")
			if prediction == 2:
				st.success(f"Text Categorized as: News or Class {prediction}")
			if prediction == -1:
				st.success(f"Text Categorized as: Anti-climate change or Class {prediction}")

	# Building out the prediction 2 page
	if selection == "Classification 2":
		st.info("Classification with Improved ML Model")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/lgr_bal.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			dict_pred = {-1: 'Anti-climate change', 0: 'Neutral', 1: 'Pro-climate change', 2: 'News'}
			for key in dict_pred.keys():
				if prediction == key:
					st.success(f"Text Categorized as: {dict_pred[key]}")
				#st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
