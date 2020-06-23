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
import re
import seaborn as sns
import matplotlib.pyplot as plt

# NLP packages
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk import ngrams
import collections

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

	# Building out the "Overview" page
	if selection == "Overview":
		st.write("## Comparison of categories")
		st.write("The model predicts the text to be classed into one of four categories:")
		st.write("* Denies climate change (-1)")
		st.write("* Is neutral about climate change (0)")
		st.write("* Believes in climate change (1)")
		st.write("* Provided a factual link to a news site (2)")
		st.write("You can view the raw data used to train the models at the bottom of the page.")

		# Count of each category
		st.write("### Count of each category")
		fig = sns.countplot(x='sentiment', data = raw, palette='rainbow')
		st.pyplot()
		st.write(raw['sentiment'].value_counts())
		st.write("The training data contained more tweets from climate change believers"+
				 " than any other group which implies that there may be more information"+
				 " available about this group than others.")

		st.subheader("Raw Twitter data")
		st.write("The raw Twitter data that was used to train the models.")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
		

	# Building out the prediction page
	if selection == "Prediction":
		st.info("1. Enter a sample text of your audience in the box below\n " +
				"2. Select the algorithm used to classify your text\n"+
				"3. Click on 'Predict' to get your prediction\n\n"+
				"To learn more about each group, please explore the options in the sidebar.")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter text","Type Here")

		def _preprocess(tweet):
			lowercase = tweet.lower()
			without_handles = re.sub(r'@', r'', lowercase)
			without_hashtags = re.sub(r'#', '', without_handles)
			without_URL = re.sub(r'http[^ ]+', '', without_hashtags)
			without_URL1 = re.sub(r'www.[^ ]+', '', without_URL)    
			return without_URL1
		
		tweet_text = _preprocess(tweet_text)
		
		# Allow user to select algorithm
		algorithm = st.selectbox("Select an algorithm to make the prediction",
							['Support Vector Classifier', 'Random Forest',
							'Logistic Regression'])
		
		# Classify using SVC
		if algorithm=='Support Vector Classifier':
			if st.button("Predict using Support Vector Classifier"):
				# Transforming user input with vectorizer
				#vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/support_vector.pkl"),"rb"))
				tweet_text = [tweet_text]
				prediction = predictor.predict(tweet_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == 0:
					st.success('Neutral. Select "Neutrals" in the sidebar for more informatio about this category'+
							   ' or select "Overview" for a comparison of each category.')
				if prediction == -1:
					st.success('Climate change denier. Select "Deniers" in the sidebar for more information about'+
							   ' this category or select "Overview" for a comparison of each category.')
				if prediction == 2:
					st.success('Provides link to factual news source. Select "Factuals" in the sidebar for more'+
							   ' information about this category or select "Overview" for a comparison of each'+
							   ' category.')
				if prediction == 1:
					st.success('Climate change believer. Select "Believers" in the sidebar for more information'+
							   ' about this category or select "Overview" for a comparison of each category.')

		# Classify using Random Forest
		if algorithm=='Random Forest':
			if st.button("Predict using Random Forest"):
				# Transforming user input with vectorizer
				#vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/rf_model1.pkl"),"rb"))
				tweet_text = [tweet_text]
				prediction = predictor.predict(tweet_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == 0:
					st.success('Neutral. Select "Neutrals" in the sidebar for more informatio about this category'+
							   ' or select "Overview" for a comparison of each category.')
				if prediction == -1:
					st.success('Climate change denier. Select "Deniers" in the sidebar for more information about'+
							   ' this category or select "Overview" for a comparison of each category.')
				if prediction == 2:
					st.success('Provides link to factual news source. Select "Factuals" in the sidebar for more'+
							   ' information about this category or select "Overview" for a comparison of each'+
							   ' category.')
				if prediction == 1:
					st.success('Climate change believer. Select "Believers" in the sidebar for more information'+
							   ' about this category or select "Overview" for a comparison of each category.')

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
					st.success('Neutral. Select "Neutrals" in the sidebar for more informatio about this category'+
							   ' or select "Overview" for a comparison of each category.')
				st.success('Climate change denier. Select "Deniers" in the sidebar for more information about'+
							   ' this category or select "Overview" for a comparison of each category.')
				if prediction == 2:
					st.success('Provides link to factual news source. Select "Factuals" in the sidebar for more'+
							   ' information about this category or select "Overview" for a comparison of each'+
							   ' category.')
				if prediction == 1:
					st.success('Climate change believer. Select "Believers" in the sidebar for more information'+
							   ' about this category or select "Overview" for a comparison of each category.')

	# N-grams
	raw_analysis = raw.copy()
	lem = WordNetLemmatizer()

	# Normalization
	def normalizer(tweet):
		tweet_no_url = re.sub(r'http[^ ]+', '', tweet) # Remove URLs beginning with http
		tweet_no_url1 = re.sub(r'www.[^ ]+', '', tweet_no_url) # Remove URLs beginning with http
		only_letters = re.sub("[^a-zA-Z]", " ",tweet_no_url1)  # Remove punctuation
		tokens = nltk.word_tokenize(only_letters) # Tokenization
		lower_case = [l.lower() for l in tokens] # Lowercase
		filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
		lemmas = [lem.lemmatize(t) for t in filtered_result] 
		return lemmas
	raw_analysis['normalized'] = raw_analysis['message'].apply(normalizer)

	# FUNCTION TO RETURN BIGRAMS AND TRIGRAMS
	def ngrams(input_list):
		bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
		trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
		return bigrams+trigrams
	raw_analysis['grams'] = raw_analysis['normalized'].apply(ngrams)

	# COUNT BIGRAMS AND TRIGRAMS 
	def count_words(input):
		cnt = collections.Counter()
		for row in input:
			for word in row:
				cnt[word] += 1
		return cnt
	
	dictionary = {}
	ngrams_deniers_tup = raw_analysis[(raw_analysis.sentiment == -1)][['grams']].apply(count_words)['grams'].most_common(20)
	ngrams_believers_tup = raw_analysis[(raw_analysis.sentiment == 1)][['grams']].apply(count_words)['grams'].most_common(20)
	ngrams_neutrals_tup = raw_analysis[(raw_analysis.sentiment == 0)][['grams']].apply(count_words)['grams'].most_common(20)
	ngrams_factuals_tup = raw_analysis[(raw_analysis.sentiment == 2)][['grams']].apply(count_words)['grams'].most_common(20)

	def tuples_to_dict(tup, di): 
		"""
		Convert a list of tuples into a dictionary
		"""
		di = dict(tup) 
		return di 
	
	# Create dictionary of ngrams and then convert to dataframe
	ngrams_deniers = tuples_to_dict(ngrams_deniers_tup, dictionary)
	ngrams_deniers = pd.DataFrame(ngrams_deniers.items(), columns = ['Ngram', 'Count'])

	ngrams_believers = tuples_to_dict(ngrams_believers_tup, dictionary)
	ngrams_believers = pd.DataFrame(ngrams_believers.items(), columns = ['Ngram', 'Count'])

	ngrams_neutrals = tuples_to_dict(ngrams_neutrals_tup, dictionary)
	ngrams_neutrals = pd.DataFrame(ngrams_neutrals.items(), columns = ['Ngram', 'Count'])

	ngrams_factuals = tuples_to_dict(ngrams_factuals_tup, dictionary)
	ngrams_factuals = pd.DataFrame(ngrams_factuals.items(), columns = ['Ngram', 'Count'])
	


	# Building out the deniers page
	if selection == "Deniers":
		st.write("## Climate Change Deniers")
		st.write(ngrams_deniers)
	
	# Building out the neutrals page
	if selection == "Neutrals":
		st.write("## Neutral About Climate Change")
		st.write(ngrams_neutrals)
	
	# Building out the believers page
	if selection == "Believers":
		st.write("## Climate Change Believers")
		st.write(ngrams_believers)
	
	# Building out the factuals page
	if selection == "Factuals":
		st.write("## Provided Link to Factual News Site")
		st.write(ngrams_factuals)
		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()