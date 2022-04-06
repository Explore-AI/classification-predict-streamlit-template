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
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
import string
import re  
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
from sklearn.utils import resample

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

vectorizer = open("resources/tfidf_vectorizer.pkl","rb")
vectors = joblib.load(vectorizer)



# Load your raw data
raw = pd.read_csv("resources/train.csv")
raw1 = raw.copy()
raw1['sentiment'].replace({2:'News',1:'Pros',0:'Neutral',-1:'Anti'},inplace = True)
raw_one = raw1[raw1['sentiment']== 'Pros']
raw_two= raw1[raw1['sentiment']== 'News']
raw_zero = raw1[raw1['sentiment']== 'Neutral']
raw_neg = raw1[raw1['sentiment']== 'Anti']

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.header("Juju  Classifier")
	#st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information","EDA","About App"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here") 

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

    #Building out the EDA page
	if selection == 'EDA':
		st.subheader("**This dashboard is used to analyze sentiments of tweets** 📊 ")
		st.sidebar.subheader("Show random tweet")
		# my_dict = {'News':2,'Pros':1,'Neutral':0,'Anti':-1}
		random_tweet = st.sidebar.radio('Sentiment',("News",'Neutral','Pros','Anti') )
		
		if random_tweet == 'News':
			st.sidebar.write(raw_two['message'].sample(n=1).iloc[0])

		elif random_tweet == 'Pros':

			st.sidebar.write(raw_one['message'].sample(n=1).iloc[0])

		elif random_tweet == 'Neutral':

			st.sidebar.write(raw_zero['message'].sample(n=1).iloc[0])

		elif random_tweet == 'Anti':

			st.sidebar.write(raw_neg['message'].sample(n=1).iloc[0])

		st.sidebar.markdown("### Number of tweets by sentiment")
		select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
		sentiment_count = raw1['sentiment'].value_counts()
		sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})
		if not st.sidebar.checkbox("Hide", True):
		    st.markdown("### Number of tweets by sentiment")
		    if select == 'Bar plot':
		        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
		        st.plotly_chart(fig)
		    else:
		        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
		        st.plotly_chart(fig)

		st.sidebar.header("Word Cloud")
		word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ("News",'Neutral','Pros','Anti'))
		st.set_option('deprecation.showPyplotGlobalUse', False)
		if not st.sidebar.checkbox("Close", True, key='3'):
		    st.subheader('Word cloud for %s sentiment' % (word_sentiment))
		    df = raw1[raw1['sentiment']==word_sentiment]
		    words = ' '.join(df['message'])
		    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
		    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
		    plt.imshow(wordcloud)
		    plt.xticks([])
		    plt.yticks([])
		    st.pyplot()

	#Building About us page
	if selection == 'About App':
		with st.container():
		    st.write("---")
		    # left_column, right_column = st.columns(2)
		    # with left_column:
		    st.header("About App")
		    st.write("##")
		    st.write(
		            """
		            Here in Africa, Juju is a dreaded name for things that are perceived to be with Suprenatural or magical abilities.


		            Diving into the roots of Mama Africa and tapping from her abundance of resources , we present to the world JUJU Classifier

		            Our JUJU Classifier has a supernatural and magical ability of making predictions from tweet sentiments. It uses the power of Machine Learning and Natural Language Processing to categorize tweets into any of 4 different labels (-1, 0, 1,2).

		            JUJU Classifier is one of the many prediction tools developed by TEAM NATURE, an indigenous data science company whose mission is helping clients make data-driven decisions to boost their business growth.

		            """
		        )
		    # with right_column:

		    	 
	        	 # st.image(logo)


		

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		models = ['Logistic_regression','Naive Bayes','Bagging_Classifier','Random_Forest']
		clf = st.sidebar.selectbox("Select a Classifer", models)

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

	

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()

			X = vectors.transform([tweet_text])

			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			if clf == 'Logistic_regression':
					predictor = joblib.load(open(os.path.join("resources/twitter_simple_lg_model_2.pkl"),"rb"))
					prediction = predictor.predict(X)

					st.success("Text Categorized as: {}".format(prediction))

			elif clf == 'Naive Bayes':
					predictor = joblib.load(open(os.path.join("resources/twitter_simple_naive_bayes_model_2.pkl"),"rb"))
					prediction = predictor.predict(X)
					st.success("Text Categorized as: {}".format(prediction))

			elif clf == 'Bagging_Classifier':
					predictor = joblib.load(open(os.path.join("resources/twitter_simple_bag_model_2.pkl"),"rb"))
					prediction = predictor.predict(X)

					st.success("Text Categorized as: {}".format(prediction))

			elif clf == 'Random_Forest':
					predictor = joblib.load(open(os.path.join("resources/twitter_simple_rf_model.pkl"),"rb"))
					prediction = predictor.predict(X)

					st.success("Text Categorized as: {}".format(prediction))




			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
