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

from pathlib import Path #for reading .md file
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Vectorizer
news_vectorizer = open("resources/countVectr1.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# Clean data
cleaning_df = raw.copy()
@st.cache
def clean(train_df):
    def replace_urls(tweet_df):
        pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
        subs_url = r'url'
        tweet_df['message'] = tweet_df['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)
        return tweet_df

    def remove_handles(tweet):
        new_tweet_list = [i for i in tweet.split() if '@' not in i]
        return ' '.join(new_tweet_list)

    def remove_rt(tweet):
        new_tweet_list = [i for i in tweet.split() if 'rt' != i.lower()]
        return ' '.join(new_tweet_list)

    def remove_punctuation(tweet):
        return ''.join([i for i in tweet if i not in string.punctuation])

    def extract_only_letters(tweet):
        tweet=re.sub('[^a-zA-Z\']',' ',tweet)
        return tweet

    lemmatizer = WordNetLemmatizer()
    def tweet_lemma(tweet, lemmatizer):
        return ' '.join([lemmatizer.lemmatize(word) for word in tweet.split()])


    train_df = replace_urls(train_df)
    train_df['message'] = train_df['message'].str.lower()  #lowercase
    train_df['message'] = train_df['message'].apply(remove_handles)
    train_df['message'] = train_df['message'].apply(remove_rt)
    train_df['message'] = train_df['message'].apply(remove_punctuation)
    train_df['message'] = train_df['message'].apply(extract_only_letters)
    train_df['message'] = train_df['message'].apply(tweet_lemma, args=(lemmatizer,))

    return train_df

cleaned_df = clean(cleaning_df)

# Read .md info file
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()
info_markdown = read_markdown_file("./resources/info.md")

# Prepare Wordclouds
@st.cache
def create_wordclouds(train_df):
    accept_tweets = train_df[train_df['sentiment'] == 1]
    accept_words = ' '.join([tweet for tweet in accept_tweets['message']])

    deny_tweets = train_df[train_df['sentiment'] == -1]
    deny_words = ' '.join([tweet for tweet in deny_tweets['message']])

    neutral_tweets = train_df[train_df['sentiment'] == 0]
    neutral_words = ' '.join([tweet for tweet in neutral_tweets['message']])

    info_tweets = train_df[train_df['sentiment'] == 2]
    info_words = ' '.join([tweet for tweet in info_tweets['message']])

    accept_wordcloud = WordCloud(width = 6000, height = 2000, random_state=1, 
                       background_color='black', collocation_threshold=3, stopwords = STOPWORDS, 
                       max_words=40).generate(accept_words)

    deny_wordcloud = WordCloud(width = 6000, height = 2000, random_state=1, 
                     background_color='black', collocation_threshold=3, stopwords = STOPWORDS, 
                     max_words=40).generate(deny_words)

    neutral_wordcloud = WordCloud(width = 6000, height = 2000, random_state=1, 
                        background_color='black', collocation_threshold=2, stopwords = STOPWORDS, 
                        max_words=40).generate(neutral_words)

    info_wordcloud = WordCloud(width = 6000, height = 2000, random_state=1, 
                     background_color='black', stopwords = STOPWORDS, 
                     max_words=40).generate(info_words)

    return accept_wordcloud, deny_wordcloud, neutral_wordcloud, info_wordcloud

accept_wordcloud, deny_wordcloud, neutral_wordcloud, info_wordcloud = create_wordclouds(cleaned_df)

f, axarr = plt.subplots(4,1, figsize=(45,35))
axarr[0].imshow(accept_wordcloud)
axarr[1].imshow(deny_wordcloud)
axarr[2].imshow(neutral_wordcloud)
axarr[3].imshow(info_wordcloud)

axarr[0].set_title('Accept', fontsize=30)
axarr[1].set_title('Deny', fontsize=30)
axarr[2].set_title('Neutral', fontsize=30)
axarr[3].set_title('Info', fontsize=30)

for ax in f.axes:
    plt.sca(ax)
    plt.axis('off')

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "Data Cleaning and Analysis", "About us", "Contact us"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown(info_markdown) #, unsafe_allow_html=True)

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

    # Building out the "About us" page
	if selection == "About us":
		st.subheader("About us")
		st.info("About us")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

	# Building out the "Contact us" page
	if selection == "Contact us":
		st.info("Contact us")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

    #models = ["Prediction", "Information", "Data Cleaning and Analysis", "About us", "Contact us"]
	    #select_mod = st.selectbox("Choose Option", models)

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

        #models = ["Prediction", "Information", "Data Cleaning and Analysis", "About us", "Contact us"]
	    #select_mod = st.selectbox("Choose Option", models)

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/logistic_model1.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

            

	# Building out the "Data Cleaning and Analysis" page
	if selection == "Data Cleaning and Analysis":
		st.info("Twitter data after cleaning")

		st.subheader("Cleaned Twitter data and label")
		st.markdown("Select checkbox to view data")
		if st.checkbox('Show cleaned data'): # data is hidden if box is unchecked
			st.write(cleaned_df[['sentiment', 'message']]) # will write the df to the page

		st.info("Word Clouds")
		st.markdown("Select checkbox to view wordclouds")
		if st.checkbox('Show Wordclouds'): # data is hidden if box is unchecked
			st.pyplot(f) # will write the df to the page

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
