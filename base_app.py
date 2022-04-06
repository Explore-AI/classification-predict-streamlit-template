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

#Images
logo = Image.open("resources/imgs/juju.jpg")
kanu = Image.open("resources/imgs/kanu.jpeg")
mburugu = Image.open("resources/imgs/mburugu.jpeg")
mercy = Image.open("resources/imgs/mercy.jpeg")
ogaga = Image.open("resources/imgs/ogaga.jpeg")
faith = Image.open("resources/imgs/faith.jpeg")

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
	options = ["Prediction", "Information","EDA","About us"]
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
	if selection == 'About us':
		with st.container():
		    st.write("---")
		    left_column, right_column = st.columns(2)
		    with left_column:
		        st.header("About us")
		        st.write("##")
		        st.write(
		            """
		            - The company was founded in 2015

		            - The driving force of the company is to help start ups build data models to solve complex problems

		            - We have helped hundreds of different companies make great business decisions using excellent data models
		            """
		        )
		    with right_column:

		    	 
	        	 st.image(logo)


		with st.container():
		    st.write("---")
		    st.header("Company Owners")
		    st.write("##")
		    image_column, text_column = st.columns((1, 5))
		    with image_column:
		        st.image(mburugu)
		    with text_column:
		        st.subheader("Mburugu - Chief Executive Officer")
		        st.write(
		            """
		            Incharge of all operations the company.
		            He is also the biggest share holder
		            """
		        )
		        
		with st.container():
		    image_column, text_column = st.columns((1, 5))
		    with image_column:
		        st.image(kanu)
		    with text_column:
		        st.subheader("Ogaga - Director")
		        st.write(
		            """
		            -Monitoring progress towards achieving the objectives and policies.

		            - Appointing senior management.
		            """
		        )


		with st.container():
		    image_column, text_column = st.columns((1, 5))
		    with image_column:
		        st.image(kanu)
		    with text_column:
		        st.subheader("Mercy  - Manager")
		        st.write(
		            """
		            - Supervises company's progress.

		            - Planning for the company.
		            """
		        ) 

		with st.container():
		    image_column, text_column = st.columns((1, 5))
		    with image_column:
		        st.image(kanu)
		    with text_column:
		        st.subheader("Kanu  - Engineer")
		        st.write(
		            """
		            - Ensures the model is in check and that it generalizes

		            -Performs juju if the model doesn't perform

		            
		            """
		        )   


		with st.container():
		    image_column, text_column = st.columns((1, 5))
		    with image_column:
		        st.image(kanu)
		    with text_column:
		        st.subheader("Faith  - RelationShip Manager")
		        st.write(
		            """
		            - Ensures people relate well in their respective work stations
		            

		            
		            """
		        )   



		        


	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		models = ['Logistic_regression','Naive Bayes','Bagging_Classifier']
		clf = st.sidebar.selectbox("Select a Classifer", models)

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

	

		if st.button("Classify"):
			# Transforming user input with vectorizer
			# vect_text = tweet_cv.transform([tweet_text]).toarray()

			## Remove urls

			pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
			subs_url = r'url-link'
			raw['clean'] = raw['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)

			# Make lower case

			raw['clean'] = raw['clean'].str.lower()

			#Punctuations removal:

			def remove_punctuation_numbers(post):
			    punc_numbers = string.punctuation + '0123456789'
			    return ''.join([l for l in post if l not in punc_numbers])
			raw['clean'] = raw['clean'].apply(remove_punctuation_numbers)

			#Removed NonAscii

			def _removeNonAscii(s): 
			    return "".join(i for i in s if ord(i)<128)

			raw['clean'] = raw['clean'].apply(_removeNonAscii)

			# #Removes retweets and tweeted at
			# def remove_users(col):

			#     col = re.sub('(rt\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', col) # remove retweet
			#     col = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', col) # remove tweeted at
			#     return col

			# raw['clean'] = raw['clean'].apply(remove_users) 



			# Clean the word using stemming
			stema = PorterStemmer()
			raw['clean'] = [stema.stem(word) for word in raw['clean']]

			# #Stopword removal:

			raw['clean'] = [word for word in raw['clean'] if word not in ENGLISH_STOP_WORDS]


			# raw_1 = raw[raw['sentiment']== 1]
			# raw_2= raw[raw['sentiment']== 2]
			# raw_0 = raw[raw['sentiment']== 0]
			# raw_n = raw[raw['sentiment']== -1]

			# base = 5000


   # #          # UPSAMPLE SENTIMENTS
			# neutral_sampled3 = resample(raw_0,
			#                           replace=True, # sample with replacement (we need to duplicate observations)
			#                           n_samples= base, # match number in minority class
			#                           random_state=27) # reproducible results

			# pro_sampled3 = resample(raw_1,
			#                           replace=False, # sample with replacement (we dont need to duplicate observations)
			#                           n_samples= base, # match number in minority class
			#                           random_state=27) # reproducible results

			# news_sampled3 = resample(raw_2,
			#                           replace=True, # sample with replacement (we need to duplicate observations)
			#                           n_samples= base, # match number in minority class
			#                           random_state=27)

			# anti_sampled3 = resample(raw_n,
			#                           replace=True, # sample with replacement (we need to duplicate observations)
			#                           n_samples= base, # match number in minority class
			#                           random_state=27)

			# train3 = pd.concat([neutral_sampled3, pro_sampled3,news_sampled3,anti_sampled3])

			features = raw['clean']

			X = vectors.transform(features)

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
					predictor = joblib.load(open(os.path.join("resources/twitter_simple_rf_model.pkl"),"rb"))
					prediction = predictor.predict(X)

					st.success("Text Categorized as: {}".format(X))



			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
