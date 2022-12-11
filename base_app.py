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
from streamlit_option_menu import option_menu

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
	st.image("resources/imgs/LeafLogo.png", caption='Our company logo')
	
	# Creates a main title and subheader on your page -
	# these are static across all pages

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	#st.sidebar.markdown('<div style="text-align: center; color:White; font-size: 20px;">SELECT A PAGE BELOW</div>', unsafe_allow_html=True)
	#options = ["🏠Home ", "❔❓About Us", "📈Prediction", "ℹ️Information", "📧☎️Contact Us"]
	#selection = st.sidebar.selectbox("", options)
	#st.sidebar.info("General Information")
	with st.sidebar:
		selection = option_menu("Main Menu", ["Home", "About Us","Prediction", "Information", "Contact Us"], 
        icons=['house', 'people-group','chart-line','info','address-book'], menu_icon="cast", default_index=1)
    	
	# Building out the "About Us" page
	if selection == "About Us":

		# You can read a markdown file from supporting resources folder
		st.title("Who Are We?")
		st.subheader("Enviro")
		st.markdown("Meet The Team")
		
		col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
		
		with col1:
			st.header("A cat")
			st.image("resources/imgs/Caron_Sathekge2.jpg")

		with col2:
			st.header("A dog")
			st.image("resources/imgs/Hlengiwe.jpg")

		with col3:
			st.header("An owl")
			st.image("resources/imgs/Jade.jpg")

		with col4:
			st.header("A cat")
			st.image("resources/imgs/Palesa.jpg")

		with col5:
			st.header("A dog")
			st.image("resources/imgs/Kgotlelelo.jpg")

		with col6:
			st.header("An owl")
			st.image("https://static.streamlit.io/examples/owl.jpg")
		
		
		#with col7:
		#	st.image('resources/imgs/Caron_Sathekge.jpg', width=100)
		#with col8:
		#	st.markdown('<h5 style="color: purple; padding:20px">CEO</h5>', unsafe_allow_html=True)

		#with col9:
		#	st.image('resources/imgs/Caron_Sathekge.jpg', width=100)
		#with col10:
		#	st.markdown('<h5 style="color: purple; padding:25px">CEO</h5>', unsafe_allow_html=True)


		# Using Tabs
		tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

		with tab1:
			st.header("A Lion")
			st.image("resources/imgs/Caron_Sathekge.jpg", width=200)
			st.markdown("cd jcad cdk cd cdkv dvk dkv dv dvk dvkd vkdv dvk ")

		with tab2:
			st.header("A dog")
			st.image("https://static.streamlit.io/examples/dog.jpg", width=200)		

		st.image("resources/imgs/LeafLogo.png")
		st.image("resources/imgs/LeafLogo.png")
		st.image("resources/imgs/LeafLogo.png")
		st.image("resources/imgs/LeafLogo.png")
		st.image("resources/imgs/LeafLogo.png")
		st.image("resources/imgs/LeafLogo.png")



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
		tweet_text = st.text_area(label="Enter Text", height= 250, help="Enter a text, then click on 'Classify' below", placeholder="Enter any text here")

		#if st.button("Click Me To Classify 👈"):
		if st.button("Classify 👈"):
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
						- Category **-1** = Anti-climate change
						- Category **0** = Neutral
						- Category **1** = Pro climate change
						- Category **2** = Factual News
						""")
    #A form for the Contact Us page
	if selection == "Contact Us":
		st.subheader("Contact")
		with st.form(key='form1'):
			firstname = st.text_input("Username")
			lastname = st.text_input("Lastname")
			email = st.text_input("Firstname")
			message = st.text_area("Insert text here")

			submitted = st.form_submit_button()
		if submitted:
			st.success("Hello {} your infomation has been captured ".format(firstname))

	

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
