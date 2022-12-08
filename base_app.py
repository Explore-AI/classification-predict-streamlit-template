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
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import pandas as pd

# import nltk 
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

#from nltk.stem import WordNetLemmatizer
#from nltk.corpus import stopwords

import pickle

# Vectorizer
news_vectorizer = open("resources/count_vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
df_train = pd.read_csv("resources/train_2.csv")
df_test=pd.read_csv("resources/test_with_no_labels.csv")

pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
subs_url = r'url-web'
df_train['message'] = df_train['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)
df_test['message'] = df_test['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)

df_train['message'] = df_train['message'].str.lower()
df_test['message'] = df_test['message'].str.lower()

import string
def remove_punctuation(post):
    return ''.join([l for l in post if l not in string.punctuation])

df_train['message'] = df_train['message'].apply(remove_punctuation)
df_test['message'] = df_test['message'].apply(remove_punctuation)

df_train['message'].str.replace("rt","")
df_train['message'].str.replace("@","")

df_test['message'].str.replace("rt","")
df_test['message'].str.replace("@","")

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(df_train['message'])
vect = CountVectorizer()
vect.fit(df_train['message'])
vect = CountVectorizer(stop_words='english')
vect = CountVectorizer(ngram_range=(1, 2))



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# This is our company logo
	st.image("resources/imgs/LeafLogo.png")
	
	# Creates a main title and subheader on your page -
	# these are static across all pages

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["About Us", "Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "About Us" page
	if selection == "About Us":

		# You can read a markdown file from supporting resources folder
		st.title("Who Are We?")
		st.subheader("Enviro")
		st.markdown("Some information here")


	# Building out the "Information" page
	if selection == "Information":
		st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification")
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification")
		st.info("Prediction with ML Models")
		#st.markdown('<div style="text-align: center;">Prediction with ML Models</div>', unsafe_allow_html=True)

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Enter any text here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/logistic_regression_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Your text was categorized as: {}".format(prediction))

			if st.checkbox('See Category Meanings'):
				st.markdown(f"""
						**THE MEANING OF THESE CATEGORIES?**
						- Category **-1** = Negative
						- Category **0** = Neutral
						- Category **1** = Pro
						- Category **2** = Factual News
						""")
	

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
