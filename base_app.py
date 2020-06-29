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

#----------------------------------------#
# Functions for analysis of twitter data
#----------------------------------------#
# 1) N-grams count
# 2) Word count
# 3) Length of tweet
# 4) Average word length
# 5) Table for length metrics
# 6) Most common hashtags

# 1) N-GRAMS COUNT
raw_analysis = raw.copy()
lem = WordNetLemmatizer()

# Normalization
def normalizer(tweet):
	"""
    Normalises a tweet string by removing URLs, punctuation, converting to
    lowercase, tokenisation and lemmatization.
    
    parameters:
            tweet: (string) A tweet that will be normalised
    Returns:
            lemmas: A list of the preprocessed strings

    """
	tweet_no_url = re.sub(r'http[^ ]+', '', tweet) # Remove URLs beginning with http
	tweet_no_url1 = re.sub(r'www.[^ ]+', '', tweet_no_url) # Remove URLs beginning with http
	only_letters = re.sub("[^a-zA-Z]", " ",tweet_no_url1)  # Remove punctuation
	tokens = nltk.word_tokenize(only_letters) # Tokenization
	lower_case = [l.lower() for l in tokens] # Lowercase
	filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
	lemmas = [lem.lemmatize(t) for t in filtered_result] 
	return lemmas
raw_analysis['normalized'] = raw_analysis['message'].apply(normalizer)

# Return bigrams and trigrams
def ngrams(input_list):
	"""
    Creates a list of 2 and 3 consecutive words within the input list.
    
    Parameters:
            input_list: A list of strings that come from a normalized tweet
    Returns:
            bigrams+trigrams: A list of the bigrams and trigrams for the input list
    """
	bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
	trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
	return bigrams+trigrams
raw_analysis['grams'] = raw_analysis['normalized'].apply(ngrams)

# Count bigrams and trigrams
def count_words(input):
	"""
    Counts the number of occurences of strings within the input list.
    
    Parameters:
        input_list: A list of strings
        
    Return:
        A list of tuples containing n-grams and a count of occurences for the n-gram
    """
	cnt = collections.Counter()
	for row in input:
		for word in row:
			cnt[word] += 1
	return cnt

dictionary = {}

def tuples_to_dict(tup, di): 
	"""
	Convert a list of tuples into a dictionary
	"""
	di = dict(tup) 
	return di 

def show_ngrams(category, amount):
	"""
	Finds a specified amount of top ngrams for a category

	Parameters:
		category: (int) training data label (-1, 0, 1, 2)
		amount: (int) number of ngrams to return
	Output:
		ngrams_df: A dataframe containing a specified amount of ngrams for
		a category.
	"""
	ngrams_tup = raw_analysis[(raw_analysis.sentiment == category)][['grams']].apply(count_words)['grams'].most_common(amount+1)

	ngrams_dict = tuples_to_dict(ngrams_tup, dictionary)
	ngrams_df = pd.DataFrame(ngrams_dict.items(), columns = ['Ngram', 'Count'])
	return ngrams_df

def show_words(category, amount):
	"""
	Finds a specified amount of top words for a category

	Parameters:
		category: (int) training data label (-1, 0, 1, 2)
		amount: (int) number of words to return
	Output:
		ngrams_df: A dataframe containing a specified amount of words for
		a category.
	"""
	words_tup = raw_analysis[(raw_analysis.sentiment == category)][['normalized']].apply(count_words)['normalized'].most_common(amount+1)

	words_dict = tuples_to_dict(words_tup, dictionary)
	words_df = pd.DataFrame(words_dict.items(), columns = ['Ngram', 'Count'])
	return words_df

# 2) WORD COUNT
def word_count(tweet):
	"""
	Returns the number of words in a string.
  
    Parameters:
            A pandas series (str)
    Returns:
            An length of the tweet string (int).
	"""
	return len(tweet.split())

raw_analysis['word_count'] = raw_analysis['message'].apply(word_count)
word_count_believers = raw_analysis[raw_analysis['sentiment'] == 1]['word_count']
avg_word_count_believers = word_count_believers.mean()

word_count_deniers = raw_analysis[raw_analysis['sentiment'] == -1]['word_count']
avg_word_count_deniers = word_count_deniers.mean()

word_count_neutrals = raw_analysis[raw_analysis['sentiment'] == 0]['word_count']
avg_word_count_neutrals = word_count_neutrals.mean()

word_count_factuals = raw_analysis[raw_analysis['sentiment'] == 2]['word_count']
avg_word_count_factuals = word_count_factuals.mean()

# 3) LENGTH OF TWEET
def length_of_tweet(tweet):
	"""
    Returns the number of characters in each tweet.
    
    parameters: 
            A pandas series (str)
    Returns:
            The number of characters in each tweet (int).
	"""
	return len(tweet)
raw_analysis['tweet_length'] = raw_analysis['message'].apply(length_of_tweet)

t_length_believers = raw_analysis[raw_analysis['sentiment'] == 1]['tweet_length']
avg_t_length_believers = t_length_believers.mean()

t_length_deniers = raw_analysis[raw_analysis['sentiment'] == -1]['tweet_length']
avg_t_length_deniers = t_length_deniers.mean()

t_length_neutrals = raw_analysis[raw_analysis['sentiment'] == 0]['tweet_length']
avg_t_length_neutrals = t_length_neutrals.mean()

t_length_factuals = raw_analysis[raw_analysis['sentiment'] == 2]['tweet_length']
avg_t_length_factuals = t_length_factuals.mean()

# 4) AVERAGE WORD LENGTH
def average_word_length(tweet):
	"""
    Returns the avarage length of words withing each tweet.
    
    parameters: 
            A pandas series(str)
    Returns:
            The average length of words within each tweet (float).
	"""
	words = tweet.split()
	average = sum(len(word) for word in words) / len(words)
	return round(average, 2)
raw_analysis['avg_word_length'] = raw_analysis['message'].apply(average_word_length)

w_length_believers = raw_analysis[raw_analysis['sentiment'] == 1]['avg_word_length']
avg_w_length_believers = w_length_believers.mean()

w_length_deniers = raw_analysis[raw_analysis['sentiment'] == -1]['avg_word_length']
avg_w_length_deniers = w_length_deniers.mean()

w_length_neutrals = raw_analysis[raw_analysis['sentiment'] == 0]['avg_word_length']
avg_w_length_neutrals = w_length_neutrals.mean()

w_length_factuals = raw_analysis[raw_analysis['sentiment'] == 2]['avg_word_length']
avg_w_length_factuals = w_length_factuals.mean()

# 5) TABLE FOR LENGTH METRICS
tweet_metrics = {'Average word count': [avg_word_count_deniers,
										avg_word_count_neutrals,
										avg_word_count_believers,
										avg_word_count_factuals],
				 'Average tweet length': [avg_t_length_deniers,
				 						  avg_t_length_neutrals,
										  avg_t_length_believers,
										  avg_t_length_factuals],
				 'Average word length' : [avg_w_length_deniers,
				 						  avg_w_length_neutrals,
										  avg_w_length_believers,
										  avg_w_length_factuals]}

# Convert dictionary to dataframe
tweet_metrics = pd.DataFrame.from_dict(tweet_metrics, orient='index',
                                       columns=['Deniers', 'Neutrals',
                                                'Believers', 'Factuals'])

# Divide "Average tweet length" by 10 so that it visualises nicely
tweet_metrics.iloc[1,:] = tweet_metrics.iloc[1,:].apply(lambda x: x / 10)

# Scale down "average tweet length" for visualisation
tweet_metrics = tweet_metrics.reset_index()
tweet_metrics_melted = pd.melt(tweet_metrics, id_vars=['index'],
                               value_vars=['Deniers', 'Neutrals',
                               'Believers', 'Factuals'])

# 6) MOST COMMON HASHTAGS
def find_hashtags(tweet):
	"""
	Create a list of all the hashtags in a string

	Parameters:
	tweet: String 
	Outputs:
	hashtags: List of strings containing hashtags in input string

	"""
	hashtags = []         
	for word in tweet.lower().split(' '): 
		#Appending the hashtag into the list hashtags
		if word.startswith('#'):        
			hashtags.append(word)	
	return hashtags
  

# Create new column for hashtags
raw_analysis['hashtags'] = raw_analysis['message'].apply(find_hashtags)

def show_hashtags(category, amount):
	"""
	Finds a specified amount of top hashtags for a category

	Parameters:
	category: (int) training data label (-1, 0, 1, 2)
	amount: (int) number of hashtags to return
	"""
	hashtags_tup = raw_analysis[(raw_analysis.sentiment == category)][['hashtags']].apply(count_words)['hashtags'].most_common(amount+1)

	hashtags_dict = tuples_to_dict(hashtags_tup, dictionary)
	hashtags_df = pd.DataFrame(hashtags_dict.items(), columns=['Ngram', 'Count'])
	return hashtags_df

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.write("# Climate Change Tweet Classifer")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home page", "Prediction", "Comparison of categories", "Analysis of each category"]
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


	# Building out the "Comparison of categories" page
	if selection == "Comparison of categories":
		st.write("## Comparison of categories")
		st.write("The model predicts the text to be classed into one of four categories:")
		st.write("* Denies climate change (-1)")
		st.write("* Is neutral about climate change (0)")
		st.write("* Believes in climate change (1)")
		st.write("* Provided a factual link to a news site (2)")
		st.write("View the raw data used to train the models at the bottom of the page.")
		st.write("More information relating to the most commonly used words for each category can"+
				 " be found in the 'Analysis of each category' page in the sidebar.")

		# Count of each category
		st.write("### Count of each category")
		st.write("Number of tweets available in each category of the training data.")
		if st.checkbox('Show count of each category'):
			fig1 = sns.countplot(x='sentiment', data=raw, palette='rainbow')
			st.pyplot()
			st.write(raw['sentiment'].value_counts())
			st.write("The training data contained more tweets from climate change believers"+
					" than any other group which implies that there may be more information"+
					" available about this group than others. When building the models, the categories"+
					" were resampled so that they each were of the same size.")
		
		# Length metrics
		st.write("### Length metrics")
		st.write("Comparison of categories' average tweet length, word count, and average word length.")
		if st.checkbox('Show length metrics'):
			st.write(tweet_metrics)
			fig2 = sns.barplot(x='variable', y='value', hue='index', data=tweet_metrics_melted,
            				   palette='rainbow')
			st.pyplot()
			st.write("The 'Average tweet length' metric was divided by 10 so that it could "+
					 "be visualised on the graph.")
				

		# Raw Twitter data
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
			st.write("This model is good at predicting believers and factuals. For all other groups, it may falsely identify"+
					 " the tweet as being part of that category when in fact is is part of a different category.")
			if st.button("Predict using Support Vector Classifier"):

				# Load your .pkl file with the model of your choice + make predictions
				predictor = joblib.load(open(os.path.join("resources/svc_model_resampled.pkl"),"rb"))
				tweet_text = [tweet_text]
				prediction = predictor.predict(tweet_text)


				if prediction == 0:
					st.success('Neutral. Select "Analysis of each category" in the sidebar for more informatio about this category'+
							   ' or select "Comparison of categories.".')
				if prediction == -1:
					st.success('Climate change denier. Select "Analysis of each category" in the sidebar for more information about'+
							   ' this category or select "Comparison of categories."')
				if prediction == 2:
					st.success('Provides link to factual news source. Select "Analysis of each category" in the sidebar for more'+
							   ' information about this category or select "Comparison of categories."')
				if prediction == 1:
					st.success('Climate change believer. Select "Analysis of each category" in the sidebar for more information'+
							   ' about this category or select "Comparison of categories."')

		# Classify using Random Forest
		if algorithm=='Random Forest':
			st.write("For climate change believers and deniers, This model may identify falsely identfy a tweet as part of"+
					 "a different group. It may incorrectly classify as part of the 'neutral' category.")
			if st.button("Predict using Random Forest"):
				# Load your .pkl file with the model of your choice + make predictions
				predictor = joblib.load(open(os.path.join("resources/rf_model_resampled.pkl"),"rb"))
				tweet_text = [tweet_text]
				prediction = predictor.predict(tweet_text)

				if prediction == 0:
					st.success('Neutral. Select "Analysis of each category" in the sidebar for more informatio about this category'+
							   ' or select "Comparison of categories."')
				if prediction == -1:
					st.success('Climate change denier. Select "Analysis of each category" in the sidebar for more information about'+
							   ' this category or select "Comparison of categories."')
				if prediction == 2:
					st.success('Provides link to factual news source. Select "Analysis of each category" in the sidebar for more'+
							   ' information about this category or select "Comparison of categories."')
				if prediction == 1:
					st.success('Climate change believer. Select "Analysis of each category" in the sidebar for more information'+
							   ' about this category or select "Comparison of categories."')

		# Classify using Logistic Regression
		if algorithm=='Logistic Regression':
			if st.button("Predict using Logistic Regression"):
				st.write("This model is better at predicting believers and factuals, but may incorrectly classify as"+
						 " being part of a different class when it is in fact a believer.")
				predictor = joblib.load(open(os.path.join("resources/log_model.pkl"),"rb"))
				tweet_text = [tweet_text]
				prediction = predictor.predict(tweet_text)

				if prediction == 0:
					st.success('Neutral. Select "Analysis of each category" in the sidebar for more informatio about this category'+
							   ' or select "Comparison of categories."')
				st.success('Climate change denier. Select "Analysis of each category" in the sidebar for more information about'+
							   ' this category or select "Comparison of categories."')
				if prediction == 2:
					st.success('Provides link to factual news source. Select "Analysis of each category" in the sidebar for more'+
							   ' information about this category or select "Comparison of categories."')
				if prediction == 1:
					st.success('Climate change believer. Select "Analysis of each category" in the sidebar for more information'+
							   ' about this category or select "Comparison of categories."')

	
	# Building out the Analysis of each category page
	if selection == "Analysis of each category":
		st.write("## Analysis of Individual Categories")
		st.info("1. Select a category from the dropdown menu for a more detailed analysis.\n\n"+
				"2. Select which analysis you would like to see")

		# Select category of which user would like to view data
		category = st.selectbox("Select category",
							['Deniers', 'Neutrals',
							'Believers', 'Factuals'])

		# Most frequent individual words
		st.write("### Most frequently used words")
		st.write("Most commonly-used words for this category.")
		if st.checkbox('Show most frequent words'):
			st.write("### Most frequent individual words (lemmas)")
			st.write("What you are seeing are the words' lemmas. One of the text preprocessing "+
					"steps involved lemmatization, which is simplifying a word to its most basic "+
					"form. This groups related words together, e.g. 'walked', 'walks', 'walking' "+
					"would be simplified to just 'walk.'")
			if category == 'Deniers':
				cat_slider_w = -1
			if category == 'Neutrals':
				cat_slider_w = 0
			if category == 'Believers':
				cat_slider_w = 1
			if category == 'Factuals':
				cat_slider_w = 2
			number_to_show_w = st.slider('Amount of entries to show', 1, 50, 10)
			st.write(show_words(cat_slider_w, number_to_show_w))

		# N-grams analysis
		#show_ngrams(category, amount)
		st.write("### Most frequently used phrases (n-grams)")
		st.write("Most commonly-used phrases for this category.")
		if st.checkbox('Show most frequent phrases'):
			st.write("n-grams refer to a sequence of n consecutive items. In this case, it"+
					" refers to a n consecutive words in a text. The most common bigrams (two words) and"+
					" trigrams (three words) were counted in the training data. These show the most frequently"+
					" used sets of two and three words by each category.")
			if category == 'Deniers':
				cat_slider_ngram = -1
			if category == 'Neutrals':
				cat_slider_ngram = 0
			if category == 'Believers':
				cat_slider_ngram = 1
			if category == 'Factuals':
				cat_slider_ngram = 2
			number_to_show_ngram = st.slider('Amount of entries to show', 1, 50, 10)
			st.write(show_ngrams(cat_slider_ngram, number_to_show_ngram))

		
		# Length-related metrics checkbox
		st.write("### Length-related metrics")
		st.write("Average word count per tweet, average tweet length, average word length.")
		if st.checkbox('Show length-related metrics'):
			if category == 'Deniers':
				st.write('### Metrics')
				st.write("Average word count per tweet: ", round(avg_word_count_deniers, 2))
				st.write("Average tweet length: ", round(avg_t_length_deniers, 2))
				st.write("Average word length: ", round(avg_w_length_deniers, 2))

			if category == 'Neutrals':
				st.write('### Metrics')
				st.write("Average word count per tweet: ", round(avg_word_count_neutrals, 2))
				st.write("Average tweet length: ", round(avg_t_length_neutrals, 2))
				st.write("Average word length: ", round(avg_w_length_neutrals, 2))

			if category == 'Believers':
				st.write('### Metrics')
				st.write("Average word count per tweet: ", round(avg_word_count_believers, 2))
				st.write("Average tweet length: ", round(avg_t_length_believers, 2))
				st.write("Average word length: ", round(avg_w_length_believers, 2))

			if category == 'Factuals':
				st.write('### Metrics')
				st.write("Average word count per tweet: ", round(avg_word_count_factuals, 2))
				st.write("Average tweet length: ", round(avg_t_length_factuals, 2))
				st.write("Average word length: ", round(avg_w_length_factuals, 2))
		
		# Top hashtags
		st.write("### Top hashtags")
		st.write("Most frequently-used hashtags for this category.")
		if st.checkbox("Show top hashtags"):
			if category == 'Deniers':
				cat_slider = -1
			if category == 'Neutrals':
				cat_slider = 0
			if category == 'Believers':
				cat_slider = 1
			if category == 'Factuals':
				cat_slider = 2
			number_to_show = st.slider('Amount of entries to show', 1, 50, 10)
			st.write(show_hashtags(cat_slider, number_to_show))
		
		# Analysis
		if category == "Deniers":
			st.write("From this data, there seems to be a political divide between climate change supporters and deniers, "+
					 "where climate change deniers tend to be supporters of Republican politics, whereas climate change"+
					 " believers may either be anti-Donald Trump or may be aligned towards Democratic politics.")
			st.write("In this dataset, climate change deniers seem to tend to retweet Donald Trump and Twitter user"+
			 		 " @SteveSGoddard, who has since changed his username to [@Tony__Heller](https://twitter.com/Tony__Heller)."+
					 " US President Donald Trump is generally known to not believe in climate change, and has, in the past, suggested"+
					 " that is a hoax invented by China. This ties in with one of the other common bigrams, which is"+
					 " 'created chinese.' Tony Heller is a conservative anti climate change activist. This data may suggest that"+
					 " those who don't believe in climate change may be aligned towards right-wing politics.")

		if category == "Neutrals":
			st.write("From this data, there seems to be a political divide between climate change supporters and deniers, "+
					 "where climate change deniers tend to be supporters of Republican politics, whereas climate change"+
					 " believers may either be anti-Donald Trump or may be aligned towards Democratic politics.")

		if category == "Believers":
			st.write("From this data, there seems to be a political divide between climate change supporters and deniers, "+
					 "where climate change deniers tend to be supporters of Republican politics, whereas climate change"+
					 " believers may either be anti-Donald Trump or may be aligned towards Democratic politics.")
			st.write("Tweets in this category frequently mention the idea of dying as a result of climate change."+
			         " The tweets that are frequently mentioning Twitter user @StephenSchlegel are retweets that"
					 " are responding to a tweet by Melania Trump. Melania posted a picture of a sea creature with "
					 "the caption, 'What is she thinking?' and many people responded with, 'She's thinking about"
					 " how she's going to die because your husband doesn't believe in climate change.' This may "
					 "indicate that those who believe in climate change tend to not follow Donald Trump.")
					 
		if category == "Factuals":
			st.write("From this data, there seems to be a political divide between climate change supporters and deniers, "+
					 "where climate change deniers tend to be supporters of Republican politics, whereas climate change"+
					 " believers may either be anti-Donald Trump or may be aligned towards Democratic politics.")
			st.write("These tweets seem to be centered around issues relating to policy and Donald Trump and "
					 "Scott Pruitt's (former Administrator of the U.S. Environmental Protection Agency) views on climate change.")
			st.write("There is also mention of the "
					 "[Paris Agreement](https://unfccc.int/process-and-meetings/the-paris-agreement/the-paris-agreement), which"
					 " is an agreement with the United Nations Framework Convention on Climate Change which deals with"
					 " the reduction of the impact of climate change. In 2017, President Donald Trump chose to withdraw"
					 " the U.S.'s participation from this agreement.")

	
		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()