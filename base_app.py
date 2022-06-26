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
from PIL import Image

# Data dependencies
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import contractions

# Vectorizer
news_vectorizer = open("resources/vectoriser-ngram-(1,2).pickle","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
def preprocess(textdata):
    processedTweet = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"http\S+"
    punctuations      = "[^a-zA-Z#@_]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern,'newsurl',tweet)
        #expand contractions
        tweet= contractions.fix(tweet)              
        # Replace all punctuation.
        tweet = re.sub(punctuations, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in textdata.split():
            # Checking if the word is a stopword.
            #if word not in stopwordlist:
            if word not in stopwords.words('english'):
                if len(word)>3:
                # Lemmatizing the word.
                    word = wordLemm.lemmatize(word)
                    tweetwords += (word+' ')
            
        processedTweet.append(tweetwords)

		
        return ' '.join(word for word in processedTweet)

logo = Image.open("resources/imgs/classification_logo-removebg-preview.png")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title ("DPD AI model Co.")
	st.image(logo)
	st.text("Welcome, we are glad to have you here. Kindly use the \
		Navigation on the side to find your way aroud... Enjoy your stay")
	

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["About Us", "Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	#Building the "About us" page
	if selection== "About Us":
		st.button("Go to classifier")
		st.info("Meet the Team")
		st.text("Oluyemi Alabi")
		st.text("Joshua Umukoro")
		st.text("Stephen Tshiani")
		st.text("Lawson Iduku")
		st.text("Abiola Akinwale")
		st.text("Ifeoluwa Adeoti")
		#if st.button("Go to classifier"):
			#open sidebar.selectbox("Make Prediction")

	# Building out the "Information" page
	if selection == "Information":
		st.info("Classifier Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("The classifier used is Logistic Regression model which")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification")
		st.info("Classification Description")
		st.text('class 0- Neutral tweets')
		st.text('class 1- Pro-Climate change tweets')
		st.text('class 2- Climate change News tweets')
		st.text('class -1 - Anti-climate change tweets')
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			#tweet_text= preprocess(tweet_text)
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/gridmnb.pickle"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
