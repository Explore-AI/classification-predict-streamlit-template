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
from itertools import count
from symbol import return_stmt
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
from PIL import Image

# Text Processing Libraries
import contractions  # Contractions is used to handle English contractions, converting them into their longer forms.
import emoji  # Emoji allows easy manipulation and analysis of emojis in the text.
from nltk.corpus import stopwords  # Stopwords module provides a list of common words to be removed from the text.
from nltk.stem import WordNetLemmatizer  # WordNetLemmatizer is used for lemmatizing words, bringing them to their root form.
from nltk import download as nltk_download  # For downloading nltk packages, here 'wordnet'.
import regex  # Regex is used for regular expression matching and manipulation.
import string  # Provides constants and classes for string manipulation.
import unicodedata  # Provides access to the Unicode Character Database for processing Unicode characters.
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack  # Used for stacking sparse matrices horizontally.

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

#with open('resources/TFIDF_Vec.pkl', 'rb') as file:
        #tf_vect = pickle.load(file)	
#with open('resources/TFIDF_Vec.pkl', 'rb') as file:
        #tf_vect = pickle.load(file)
		# 		
#new vectorizer
new_count_vec = open("resources/Count_vec.pkl","rb")
count_vec = joblib.load(new_count_vec) # loading your vectorizer from the pkl file
# Load your raw data
raw = pd.read_csv("resources/train.csv")

#load training data
df_train = pd.read_csv('resources/train.csv')

#preprocess function
def preprocess_tweet(tweets):
	# function to determine if there is a retweet within the tweet
	def is_retweet(tweet):
		word_list = tweet.split()
		if "RT" in word_list:
			return 1
		else:
			return 0
	tweets["is_retweet"] = tweets["message"].apply(is_retweet, 1)

	# function to extract retween handles from tweet	
	def get_retweet(tweet):
		word_list = tweet.split()
		if word_list[0] == 'RT':
			handle = word_list[1]
		else:
			handle = ''
		handle = handle.replace(':', "")

		return handle
	tweets['retweet_handle'] = tweets['message'].apply(get_retweet,1)

	# function to count the number of hashtags within the tweet
	def count_hashtag(tweet):
		count = 0

		word_list = tweet.split()
		for word in word_list:
			if word[0] == "#":
				count += 1

		return count
	tweets["hashtag_count"] = tweets["message"].apply(count_hashtag, 1)

	# function to extract the hashtags within the tweet
	def get_hashtag(tweet):
		hashtags = []
		word_list = tweet.split()
		for word in word_list:
			if word[0] == "#":
				hashtags.append(word)

		returnstr = ""

		for tag in hashtags:
			returnstr + " " + tag

		return returnstr
	tweets["hashtags"] = tweets["message"].apply(get_hashtag, 1)

	# function to count the number of mentions within the tweet
	def count_mentions(tweet):
		count = 0
		word_list = tweet.split()
		if "RT" in word_list:
			count += -1 # remove mentions contained in retweet from consideration
		
		for word in word_list:
			if word[0] == '@':
				count += 1
		if count == -1:
			count = 0
		return count
	tweets["mention_count"] = tweets["message"].apply(count_mentions, 1)

	def get_mentions(tweet):
		mentions = []
		word_list = tweet.split()

		if "RT" in word_list:
			word_list.pop(1) # Retweets don't count as mentions, so we remove the retweet handle from consideration

		for word in word_list:
			if word[0] == '@':
				mentions.append(word)

		returnstr = ""

		for handle in mentions:
			returnstr + " " + handle

		return returnstr
	tweets["mentions"] =  tweets["message"].apply(get_mentions, 1)

	# function to count the number of web links within tweet
	def count_links(tweet):
		count = tweet.count("https:")
		return count 
	tweets["link_count"] = tweets["message"].apply(count_links, 1)

	# function to replace URLs within the tweet
	pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
	subs_url = r'url-web'
	tweets['message'] = tweets['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)
	
	# function count number of newlines within tweet
	def enter_count(tweet):
		count = tweet.count('\n')
		return count
	tweets["newline_count"] = tweets["message"].apply(enter_count, 1)

	# function to count number of exclaimation marks within tweet
	def exclamation_count(tweet):
		count = tweet.count('!')
		return count 
	tweets["exclamation_count"] =  tweets["message"].apply(exclamation_count, 1)
	
	# Remove handles from tweet
	def remove_handles(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word[0] == "@":
				wordlist.remove(word)
		returnstr = ''

		for word in wordlist:
			returnstr += word + " "

		return returnstr
	tweets['message'] = tweets['message'].apply(remove_handles)

	# Remove hashtags from tweet
	def remove_hashtags(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word[0] == '#':
				wordlist.remove(word)
		returnstr = ''
		for word in wordlist:
			returnstr += word + " "

		return returnstr
	
	tweets["message"] = tweets["message"].apply(remove_hashtags)

	# Remove RT from tweet
	def remove_rt(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word == 'rt' or word == 'RT':
				wordlist.remove(word)
		returnstr = ''

		for word in wordlist:
			returnstr += word + ' '

		return returnstr
	
	tweets['message'] = tweets['message'].apply(remove_rt)

	# function to translate emojis and emoticons
	def fix_emojis(tweet):
		newtweet = emoji.demojize(tweet)  # Translates 👍 emoji into a form like :thumbs_up: for example
		newtweet = newtweet.replace("_", " ") # Beneficial to split emoji text into multiple words
		newtweet = newtweet.replace(":", " ") # Separate emoji from rest of the words
		returntweet = newtweet.lower() # make sure no capitalisation sneaks in

		return returntweet
	tweets["message"] = tweets['message'].apply(fix_emojis)

	# function to remove punctuation from the tweet
	def remove_punctuation(tweet):
		return ''.join([l for l in tweet if l not in string.punctuation])
	
	tweets['message'] = tweets['message'].apply(remove_punctuation)
	
	#transform tweets into lowercase version of tweets
	def lowercase(tweet):
		return tweet.lower()
	tweets["message"] = tweets["message"].apply(lowercase)

	# remove stop words from the tweet
	def remove_stop_words(tweet):
		words = tweet.split()
		return " " .join([t for t in words if t not in stopwords.words('english')])
	tweets["message"] = tweets['message'].apply(remove_stop_words)

	# function to replace contractions
	def fix_contractions(tweet):
		expanded_words = []
		for word in tweet.split():
			expanded_words.append(contractions.fix(word))

		returnstr = " ".join(expanded_words)
		return returnstr
	tweets["message"] = tweets['message'].apply(fix_contractions)

	# function to replace strange characters in tweet with closest ascii equivalent
	def clean_tweet(tweet):
		normalized_tweet = unicodedata.normalize('NFKD',tweet)

		cleaned_tweet = normalized_tweet.encode('ascii', 'ignore').decode('utf-8')

		return cleaned_tweet.lower()
	tweets["message"] = tweets['message'].apply(clean_tweet)

	#function to remove numbers from tweet
	def remove_numbers(tweet):
		return ''.join(char for char in tweet if not char.isdigit())
	
	tweets["message"] = tweets['message'].apply(remove_numbers)

	# Create a lemmatizer object
	lemmatizer = WordNetLemmatizer()

	# Create function to lemmatize tweet content
	def tweet_lemma(tweet,lemmatizer):
		list_of_lemmas = [lemmatizer.lemmatize(word) for word in tweet.split()]
		return " ".join(list_of_lemmas)
	tweets["message"] = tweets["message"].apply(tweet_lemma, args=(lemmatizer, ))

	# Make dataframe of all word counts in the data
	twt_wordcounts = pd.DataFrame(tweets['message'].str.split(expand=True).stack().value_counts())
	twt_wordcounts.reset_index(inplace=True)
	twt_wordcounts.rename(columns={"index": "word", 0:"count"}, inplace=True)
	
	# Extract unique words from data
	twt_unique_words = twt_wordcounts[twt_wordcounts["count"]==1]

	# make a list of unique words
	unique_wordlist = list(twt_unique_words["word"])

	# Function to remove unique words from data
	def remove_unique_words(tweet):
		words = tweet.split()
		return ' '.join([t for t in words if t not in unique_wordlist])
	tweets['message'] = tweets['message'].apply(remove_unique_words)

	#function to add retweets to message 
	def add_rt_handle(row):
		if row["retweet_handle"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " rt " + row["retweet_handle"]
		return ret
	tweets["message"] = tweets.apply(add_rt_handle, axis=1)

	# Function to add retweets to message
	def add_hashtag(row):
		if row["hashtags"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " " + row["hashtags"]
		return ret
	tweets["message"] = tweets.apply(add_hashtag, axis = 1)

	# Function to add mentions to message
	def add_rt_handle(row):
		if row["mentions"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " " + row["mentions"]
		return ret
	tweets["message"] = tweets.apply(add_rt_handle, axis = 1)

	# drop redundant columns
	tweets = tweets.drop(["retweet_handle", "hashtags", "mentions"], axis=1)

	return tweets

def preprocess_csv(tweets):
	# function to determine if there is a retweet within the tweet
	def is_retweet(tweet):
		word_list = tweet.split()
		if "RT" in word_list:
			return 1
		else:
			return 0
	tweets["is_retweet"] = tweets["message"].apply(is_retweet, 1)

	# function to extract retween handles from tweet	
	def get_retweet(tweet):
		word_list = tweet.split()
		if word_list[0] == 'RT':
			handle = word_list[1]
		else:
			handle = ''
		handle = handle.replace(':', "")

		return handle
	tweets['retweet_handle'] = tweets['message'].apply(get_retweet,1)

	# function to count the number of hashtags within the tweet
	def count_hashtag(tweet):
		count = 0
		word_list = tweet.split()
		for word in word_list:
			if word[0] == "#":
				count += 1

		return count
	tweets["hashtag_count"] = tweets["message"].apply(count_hashtag, 1)

	# function to extract the hashtags within the tweet
	def get_hashtag(tweet):
		hashtags = []
		word_list = tweet.split()
		for word in word_list:
			if word[0] == "#":
				hashtags.append(word)

		returnstr = ""

		for tag in hashtags:
			returnstr + " " + tag

		return returnstr
	tweets["hashtags"] = tweets["message"].apply(get_hashtag, 1)

	# function to count the number of mentions within the tweet
	def count_mentions(tweet):
		count = 0
		word_list = tweet.split()
		if "RT" in word_list:
			count += -1 # remove mentions contained in retweet from consideration
		
		for word in word_list:
			if word[0] == '@':
				count += 1
		if count == -1:
			count = 0
		return count
	tweets["mention_count"] = tweets["message"].apply(count_mentions, 1)

	def get_mentions(tweet):
		mentions = []
		word_list = tweet.split()

		if "RT" in word_list:
			word_list.pop(1) # Retweets don't count as mentions, so we remove the retweet handle from consideration

		for word in word_list:
			if word[0] == '@':
				mentions.append(word)

		returnstr = ""

		for handle in mentions:
			returnstr + " " + handle

		return returnstr
	tweets["mentions"] =  tweets["message"].apply(get_mentions, 1)

	# function to count the number of web links within tweet
	def count_links(tweet):
		count = tweet.count("https:")
		return count 
	tweets["link_count"] = tweets["message"].apply(count_links, 1)

	# function to replace URLs within the tweet
	pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
	subs_url = r'url-web'
	tweets['message'] = tweets['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)
	
	# function count number of newlines within tweet
	def enter_count(tweet):
		count = tweet.count('\n')
		return count
	tweets["newline_count"] = tweets["message"].apply(enter_count, 1)

	# function to count number of exclaimation marks within tweet
	def exclamation_count(tweet):
		count = tweet.count('!')
		return count 
	tweets["exclamation_count"] =  tweets["message"].apply(exclamation_count, 1)
	
	# Remove handles from tweet
	def remove_handles(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word[0] == "@":
				wordlist.remove(word)
		returnstr = ''

		for word in wordlist:
			returnstr += word + " "

		return returnstr
	tweets['message'] = tweets['message'].apply(remove_handles)

	# Remove hashtags from tweet
	def remove_hashtags(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word[0] == '#':
				wordlist.remove(word)
		returnstr = ''
		for word in wordlist:
			returnstr += word + " "

		return returnstr
	
	tweets["message"] = tweets["message"].apply(remove_hashtags)

	# Remove RT from tweet
	def remove_rt(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word == 'rt' or word == 'RT':
				wordlist.remove(word)
		returnstr = ''

		for word in wordlist:
			returnstr += word + ' '

		return returnstr
	
	tweets['message'] = tweets['message'].apply(remove_rt)

	# function to translate emojis and emoticons
	def fix_emojis(tweet):
		newtweet = emoji.demojize(tweet)  # Translates 👍 emoji into a form like :thumbs_up: for example
		newtweet = newtweet.replace("_", " ") # Beneficial to split emoji text into multiple words
		newtweet = newtweet.replace(":", " ") # Separate emoji from rest of the words
		returntweet = newtweet.lower() # make sure no capitalisation sneaks in

		return returntweet
	tweets["message"] = tweets['message'].apply(fix_emojis)

	# function to remove punctuation from the tweet
	def remove_punctuation(tweet):
		return ''.join([l for l in tweet if l not in string.punctuation])
	
	tweets['message'] = tweets['message'].apply(remove_punctuation)
	
	#transform tweets into lowercase version of tweets
	def lowercase(tweet):
		return tweet.lower()
	tweets["message"] = tweets["message"].apply(lowercase)

	# remove stop words from the tweet
	def remove_stop_words(tweet):
		words = tweet.split()
		return " " .join([t for t in words if t not in stopwords.words('english')])
	tweets["message"] = tweets['message'].apply(remove_stop_words)

	# function to replace contractions
	def fix_contractions(tweet):
		expanded_words = []
		for word in tweet.split():
			expanded_words.append(contractions.fix(word))

		returnstr = " ".join(expanded_words)
		return returnstr
	tweets["message"] = tweets['message'].apply(fix_contractions)

	# function to replace strange characters in tweet with closest ascii equivalent
	def clean_tweet(tweet):
		normalized_tweet = unicodedata.normalize('NFKD',tweet)

		cleaned_tweet = normalized_tweet.encode('ascii', 'ignore').decode('utf-8')

		return cleaned_tweet.lower()
	tweets["message"] = tweets['message'].apply(clean_tweet)

	#function to remove numbers from tweet
	def remove_numbers(tweet):
		return ''.join(char for char in tweet if not char.isdigit())
	
	tweets["message"] = tweets['message'].apply(remove_numbers)

	# Create a lemmatizer object
	lemmatizer = WordNetLemmatizer()

	# Create function to lemmatize tweet content
	def tweet_lemma(tweet,lemmatizer):
		list_of_lemmas = [lemmatizer.lemmatize(word) for word in tweet.split()]
		return " ".join(list_of_lemmas)
	tweets["message"] = tweets["message"].apply(tweet_lemma, args=(lemmatizer, ))

	# Make dataframe of all word counts in the data
	twt_wordcounts = pd.DataFrame(tweets['message'].str.split(expand=True).stack().value_counts())
	twt_wordcounts.reset_index(inplace=True)
	twt_wordcounts.rename(columns={"index": "word", 0:"count"}, inplace=True)
	
	# Extract unique words from data
	twt_unique_words = twt_wordcounts[twt_wordcounts["count"]==1]

	# make a list of unique words
	unique_wordlist = list(twt_unique_words["word"])

	# Function to remove unique words from data
	def remove_unique_words(tweet):
		words = tweet.split()
		return ' '.join([t for t in words if t not in unique_wordlist])
	tweets['message'] = tweets['message'].apply(remove_unique_words)

	#function to add retweets to message 
	def add_rt_handle(row):
		if row["retweet_handle"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " rt " + row["retweet_handle"]
		return ret
	tweets["message"] = tweets.apply(add_rt_handle, axis=1)

	# Function to add retweets to message
	def add_hashtag(row):
		if row["hashtags"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " " + row["hashtags"]
		return ret
	tweets["message"] = tweets.apply(add_hashtag, axis = 1)

	# Function to add mentions to message
	def add_rt_handle(row):
		if row["mentions"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " " + row["mentions"]
		return ret
	tweets["message"] = tweets.apply(add_rt_handle, axis = 1)

	# drop redundant columns
	tweets = tweets.drop(["retweet_handle", "hashtags", "mentions"], axis=1)

	return tweets
# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages

	image = Image.open('resources/imgs/logo.jpg')

	col1, col2 = st.columns([3, 3])
	with col1:
		st.image(image, use_column_width=True)
	with col2:
		st.title("Twitter Sentiment Classifier App")
	#add more text

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	st.sidebar.title('App Navigation')

	options = ["Prediction", "Model Explainations","Explore the data", "About Us"]
	selection = st.sidebar.radio("Choose Option", options)

	#build out the "home" company page
	if selection == "About Us":
		st.info('Welcome to That\'s Classified Data Solutions (PTY) LTD ')
		st.markdown("At That\'s Classified Data Solutions we consider ourselves your partners, and we take care of your data so that you can focus on your customers’ needs. The goal of every member of our team is to maximise your productivity, increase your profits, and most of all, future-proof your business. ")
		st.write('To access the codebase for this application, please visit the following GitHub repository:https://github.com/Kabous0017/Advanced_Classification_-_Team_GM3_2301FTDS')

		st.subheader('Meet the team')

		#director 
		col1, col2 = st.columns([1, 6])
		with col1:
			image_k = Image.open('resources/imgs/Kobus.png')
			st.image(image_k, use_column_width=True,caption = 'Director: Kobus Le Roux')

		# assistant director
		col1, col2 = st.columns([1, 6])
		with col1:
			image_m = Image.open('resources/imgs/Mkhanyisi.png')
			st.image(image_m, use_column_width=True, caption = 'Assistant Director: Mkhanyisi Mlombile')

		# data scientist 1
		col1, col2 = st.columns([1, 6])
		with col1:
			image_h = Image.open('resources/imgs/Hilda.png')
			st.image(image_h, use_column_width=True, caption = 'Data Scientist: Hilda Sinclair')
		
		# data scientist 2
		col1, col2 = st.columns([1, 6])
		with col1:
			image_t = Image.open('resources/imgs/temishka.png')
			st.image(image_t, use_column_width=True, caption = 'Data Scientist: Temishka Robyn Pillay')

		# data scientist 3
		col1, col2 = st.columns([1, 6])
		with col1:
			image_kg = Image.open('resources/imgs/Kgomotso.png')
			st.image(image_kg, use_column_width=True, caption = 'Data Scientist: Kgomotso Modihlaba')

		# data scientist 4
		col1, col2 = st.columns([1, 6])
		with col1:
			image_i = Image.open('resources/imgs/Isaac.png')
			st.image(image_i, use_column_width=True, caption = 'Data Scientist: Isaac Sihlangu')

		# data scientist 5
		col1, col2 = st.columns([1, 6])
		with col1:
			image_b= Image.open('resources/imgs/Bongokuhle.png')
			st.image(image_b, use_column_width=True, caption = 'Data Scientist: Bongokuhle Dladla')

	# Building out the "Model Explaination" page
	if selection == "Model Explainations":
		options = ['Logistic Regression','Linear Support Vector Classifier','Random Forest Classifier','XGBoost Classifier','CatBoost Classifer', 'Neural Networks Classifier','Multinomial Naives Bayes Classifier','KNN Classifier']
		selection = st.selectbox('Which model would you like to learn more about?',options)

		if selection == "Logistic Regression":
			#st.info('Explain the inner workings of Logistic Regression model')
			st.markdown('Logistic regression is a classification algorithm used to predict the probability of a binary outcome based on one or more input features. It models the relationship between the input variables and the probability of the outcome belonging to a particular class. Logistic regression uses the logistic function (also known as the sigmoid function) to map the output of a linear combination of the input features to a value between 0 and 1, representing the probability of belonging to the positive class.')
			st.markdown('In simpler terms, logistic regression aims to find the best-fitting S-shaped curve that separates the two classes. It estimates the coefficients (weights) of the input features through a process called maximum likelihood estimation, optimizing the parameters to maximize the likelihood of the observed data.')
			st.markdown('Once trained, logistic regression can make predictions by calculating the probability of the positive class based on the input features. A threshold is then applied to determine the final predicted class.')
			st.markdown('Logistic regression models are known for their simplicity and interpetability. Since they are more simplistic models, they are relatively quick to train and computationally efficient. They can also be expanded to handle multiclass classification as is the case for our data. This model does assume a linear relationship between the features and the log-odds of the outcome, however, which does not necessarily hold true in many cases. It is also sensitive to outliers and irrelevant features. ')
			
		if selection == "Linear Support Vector Classifier":
			#st.info('Explain the inner workings of Support Vector Machines model')
			st.markdown('Linear Support Vector Classification (LinearSVC) is a variant of Support Vector Machines (SVMs) used for classification tasks. It employs a linear kernel to create a hyperplane in the high-dimensional feature space to separate different classes of data.')
			st.markdown('Unlike other SVMs that might use various kernels, LinearSVC assumes a linear relationship between the features and the target variable. This model is robust and efficient in numerous applications, offering good performance even with less training data. However, its effectiveness can be sensitive to feature selection and requires careful preprocessing of the data.')
			st.markdown('Key parameters include the C parameter, controlling the trade-off between a smooth decision boundary and classifying training points correctly. Despite its underlying assumption of a linear relationship, LinearSVC has proven to be versatile in its performance, with careful tuning and preprocessing.')
		if selection == "Random Forest Classifier":
			#st.info('Explain the inner workings of Random Forest model')
			st.markdown('Random forest consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction. The reason behind is that a large number of relatively uncorrelated models (decision trees) operating as a group will outperform any of the individual constituent models. The low correlation between models is the key, with trees protecting each other from their individual errors(as long as they do not deviate in the same direction). While some trees may be wrong, many other trees will be right, so as a group the trees are able to move in the correct direction.')
			st.markdown('Random forest ensures that the behavior of each individual is not too correlated with the behavior of any other tree in the model by using bagging or bootstrap aggregation which allows each individual tree to randomly sample from the dataset with replacement resulting in different trees. Random forest also make use of feature randomness, in decision trees when it is time to split at each mode(a splitting point), we consider every possible feature and pick the one that produces the most separation between the observation in the left node vs those in the right node')
			st.markdown('In contrast, each tree in the random forest can only pick from a random subset of features, forcing even more variation amongst the trees resulting in lower correlation and more diversification.Son in random forest, we end up with trees that are not only trained on different sets of data through bagging but also use different features to make decisions(compared to individual trees which consider every feature to make a decision).')
		if selection == "XGBoost Classifier":
			#st.info('Explain the inner workings of XGBoost model')
			st.markdown('XGBoost stands for Extreme Gradient Boosting and is a gradient boosting algorithm known for its high performance and accuracy in various machine learning tasks, including classification. It is an ensemble method that combines the predictions of multiple weak predictive models, usually decision trees, to create a strong predictive model. XGBoost builds an ensemble of decision trees sequentially, where each new tree is trained to correct the mistakes made by the previous trees. It uses a gradient-based optimization technique to minimize a specific loss function, such as logistic loss for classification tasks. The algorithm calculates gradients and hessians to update the model parameters, ensuring that each subsequent tree focuses on the areas where the previous trees performed poorly.')
			st.markdown('Additionally, XGBoost incorporates several regularization techniques, such as shrinkage (learning rate) and tree pruning, to prevent overfitting and improve generalization. It also supports parallelization and distributed computing, making it efficient for training on large datasets.')
			st.markdown('XGBoost boasts exceptional predictive performance and accuracy, and is robust to outliers in the data. It also supports feature importance estimation, allowing for better understanding of feature contributions. It does however require careful hyperparameter tuning for optimal performance. It is also computationally intensive, especially with a large number of trees and complex datasets.')
		if selection == "CatBoost Classifier":
			st.info('Explain the inner workings of CatBoost model')
			#st.markdown('Explain the inner workings of this model')
		if selection == "Neural Networks Classifier":
			st.info('Explain the inner workings of the Neural Networks model')
			#st.markdown('Explain the inner workings of this model')
		if selection == "Multinomial Naives Bayes Classifier":
			#st.info('Explain the inner workings of Naives Bayes model')
			st.markdown('Naive Bayes is a probabilistic classification model based on Bayes theorem, which calculates the probability of a class given the input features. It assumes that the features are conditionally independent, meaning that the presence of one feature does not affect the presence of another feature. Despite this simplifying assumption, Naive Bayes can be surprisingly effective in many real-world scenarios.')
			st.markdown('Multinomial Naive Bayes is a variant of Naive Bayes that is specifically designed for classification tasks with discrete features. It is commonly used for text classification, where the input features are typically word frequencies or counts. Unlike Gaussian Naive Bayes, which assumes a Gaussian distribution for continuous features, Multinomial Naive Bayes assumes a multinomial distribution for discrete features.')
			st.markdown('In Multinomial Naive Bayes, the model learns the probability distribution of each feature given the class label. It estimates the probabilities using the training data, where the feature values represent the frequencies or counts of each feature in the documents of each class. To predict the class of a new instance, the model calculates the likelihood of observing the given feature values for each class and combines it with the prior probability of the class using Bayes theorem. The class with the highest probability is chosen as the predicted class.')
			st.markdown('The key difference between this model, and the Gaussian Naive Bayes is in the assumptions made about the data. Multinomial Naive Bayes assumes a multinomial distribution for discrete features, whereas Gaussian Naive Bayes assumes a Gaussian distribution for continuous features. Multinomial Naive Bayes is appropriate for discrete features, such as word counts, while Gaussian Naive Bayes is suitable for continuous or normally distributed features.')
			st.markdown('This model is generally known to be efficient, even in large feature spaces. It also works well with unbalanced data, which is handy in our case. It is also able to handle multiclass classification problems, which our classification falls into.')
		if selection == "KNN Classifier":
			st.info('Explain the inner workings of KNN model')
			#st.markdown('Explain the inner workings of this model')

		
	if selection == "Explore the data":
		options =  ['Dataset','Distribution of data per sentiment class','Proportion of retweets','Popular retweet handles per sentiment group in a word cloud', 'Popular hashtags in per sentiments group','Popular mentions per sentiment group']
		selection = st.selectbox('What would like to explore?', options)

		if selection == 'Dataset':
			st.subheader('Overview of dataset:')
			st.write(df_train.head(10))
			st.info("The dataset consists of three columns namely: Sentiment, Tweet and Tweet ID")
			st.markdown("The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43,943 tweets were collected. Each tweet is labelled as one of 4 classes.")
		if selection == 'Distribution of data per sentiment class':
			st.subheader('Distribution of data per sentiment class')
			st.image('resources/imgs/distribution_of_data_in_each_class.png')
			st.markdown('From the figures above, we see that the dataset we are working with is very unbalanced. More than half of our dataset is people having pro-climate change sentiments, while only  8% of our data represents people with anti-climate change opinions. This might lead our models to become far better at identifying pro-climate change sentiment than anti-climate change sentiment, and we might need to consider balancing the data by resampling it.')

		if selection == 'Proportion of retweets':
			st.subheader('Proportion of retweets')
			st.image('resources/imgs/proportion_of_retweets_hashtags_and_original_mentions.png')
			st.markdown('We see that a staggering  60% of all our data is not original tweets, but retweets! This indicates that extracting more information from the retweets could prove integral to optimizing our model\'s predictive capabilities.')

		if selection == 'Popular retweet handles per sentiment group in a word cloud':
			st.subheader('Popular retweet handles per sentiment group in a word cloud')
			st.image('resources/imgs/wordcloud_of_popular_retweet_handles_per_sentiment_group.png')
			st.markdown('From the above, we see a clear difference between every sentiment with regards to who they are retweeting. This is great news, since it will provide an excellent feature within our model. Little overlap between categories is visible, which points to the fact that this feature could be a very strong predictor.')
			st.markdown('We see that people with anti-climate change sentiments retweets from users like @realDonaldTrump and @SteveSGoddard the most. Overall retweets associated with anti-climate science opinions are frequently sourced from prominent Republican figures such as Donald Trump, along with individuals who identify as climate change deniers, like Steve Goddard.')
			st.markdown('In contrast to this, people with pro-climate change views often retweet Democratic political figures such as @SenSanders and @KamalaHarris. Along with this, we see a trend to retweet comedians like @SethMacFarlane. The most retweeted individual for this category, is @StephenSchlegel.')
			st.markdown('Retweets in the factual news category mostly contains handles of media news organizations, like @thehill, @CNN, @wasgingtonpost etc...')
			st.markdown('People with neutral sentiments regarding climate change seems to not retweet overtly political figures. Instead, they retweet handles unknown to the writer like @CivilJustUs and @ULTRAVIOLENCE which no longer currently exist on twitter. The comedian @jay_zimmer is also a common retweeted incividual within this category.')

		if selection == 'Popular hashtags in per sentiments group':
			st.subheader('Popular hashtags in per sentiments group')
			st.image('resources/imgs/popular_hashtags_per_sentiment_group_wordcloud.png')
			st.markdown('From the visual above, we notice a few things:')
			st.markdown('We see that a lot of hashtags are common in every sentiment category. Hashtags like #climatechange, #cllimate and #Trump is abundant regardless of which category is considered, and can therefore be removed from the list of hashtags since they won\'t contribute any meaningful insight to our models.')
			st.markdown('Finally there is some hashtags that are more prominent within certain sentiment groups. Take #MAGA and #fakenews in the anti-climate change category, or #ImVotingBecause in the pro-climate change category. This indicates that some useful information can be extracted from this feature, and should remain within the model.')

		if selection == 'Popular mentions per sentiment group':
			st.subheader('Popular mentions per sentiment group')
			st.image('resources/imgs/popular_hashtags_per_sentiment_group_wordcloud.png')
			st.markdown('As was the case when we considered hashtags, we see that some handles get mentioned regardless of sentiment class. An example of this is @realDonaldTrump, which is prominent in every sentiment category, and as such should be removed before training our models, since it adds no value towards our data.')
			st.markdown('Furthermore, there is some mentions that are more prominent in certain classes than others. Take @LeoDiCaprio for example, which features heavily in both pro-climate change as well as neutral towards climate change sentiment, but is not represented in the other two categories. This indicates that this feature could be beneficial for categorizing our data, and should remain within the dataset')

		
	# Building out the predication page
	if selection == 'Prediction':
		st.write('Predict the sentiment of each twitter using various models with each tweet falling into one of 4 categories: anti-man made climate change, neutral, pro-man made climate change and lastly, whether a tweet represent factual news!')
	
		pred_type = st.sidebar.selectbox("Predict sentiment of a single tweet or submit a csv for multiple tweets", ('Single Tweet', 'Multiple Tweets'))

		if pred_type == "Single Tweet":
			st.info("Prediction with ML Models")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter tweet here","Type Here")

			options = [" Multinomial Naive Bayes Classifier","Logistic Regression Classifier", "Linear Support Vector Classifier", "Gaussian Naives Bayes Classifier"] "XGBoost Classifier", "CatBoost Classfier"
			selection = st.selectbox("Choose Your Model", options)

			if st.button("Classify Tweet"):
				#process single tweet using our preprocess_tweet() function

				# create dataframe for tweet
				text = [tweet_text]
				df_tweet = pd.DataFrame(text, columns=['message'])

				processed_tweet = preprocess_tweet(df_tweet)
				
				# Create a dictionary for tweet prediction outputs
				dictionary_tweets = {'[-1]': "A tweet refuting man-made climate change",
                     				  '[0]': "A tweet neither supporting nor refuting the belief of man-made climate change",
                     				  '[1]': "A pro climate change tweet",
                     				  '[2]': "This tweet refers to factual news about climate change"}

				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = None
				X_pred = None
				if selection == "Multinomial Naive Bayes Classifier":
					predictor = joblib.load(open(os.path.join("resources/MultinomialNaiveBeyes.pkl"),"rb"))
					#mnb = pickle.load(open('/resources/MultinomialNaiveBeyes.pkl','rb'))
					#predictor = mnb	
				elif selection == "Logistic Regression Classifier":
					#lr = pickle.load(open('\resources\LogisticRegression.pkl','rb'))
					predictor = joblib.load(open(os.path.join("resources/LogisticRegression.pkl"),"rb"))
				elif selection == "Linear Support Vector Classifier":
					#lsvc = pickle.load(open('/resources/LinearSVC.pkl','rb'))
					predictor = joblib.load(open(os.path.join("resources/LinearSVC.pkl"),"rb"))
					#predictor = lsvc
				# elif selection == "XGBoost Classifier":
				# 	predictor = joblib.load(open(os.path.join("resources/XGBoost.pkl"),"rb"))
				elif selection == "Gaussian Naives Bayes Classifier":
					predictor = joblib.load(open(os.path.join("resources/GaussianNaiveBeyes.pkl"),"rb"))
				# elif selection == "CatBoost Classfier":
				# 	predictor = joblib.load(open(os.path.join("resources/CatBoost.pkl"),"rb"))

				#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))

				
				# Transforming user input with vectorizer
				X_pred = processed_tweet['message']
				vect_text = count_vec.transform(X_pred)
				
				sparse_vec_msg_df = pd.DataFrame.sparse.from_spmatrix(vect_text, columns = count_vec.get_feature_names_out())
				df_vectorized_combined = pd.concat([processed_tweet.reset_index(drop=True), sparse_vec_msg_df.reset_index(drop=True)], axis=1)
				df_vectorized_combined = df_vectorized_combined.drop("message", axis='columns')

				prediction = predictor.predict(df_vectorized_combined)
				prediction_str = np.array_str(prediction)

				prediction_str = prediction_str.replace(".","")
				
				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(dictionary_tweets[prediction_str]))

		if pred_type == "Multiple Tweets":
			st.markdown('This section coming soon! Watch this space!')
			# tweets_csv = st.file_uploader('Upload a CSV file here', type='csv', accept_multiple_files=False, key=None, help='Only CSV files are accepted', on_change=None, args=None, kwargs=None)
			# df_uploaded = None
			# X_pred = None
			# if tweets_csv is not None:
			# 	df_uploaded = pd.read_csv(tweets_csv)
			# 	processed_df = preprocess_csv(df_uploaded)
			# 	X_pred = processed_df['message']
			
			# options = [" Multinomial Naive Bayes Classifier","Logistic Regression Classifier", "Linear Support Vector Classifier"]
			# selection = st.selectbox("Choose Your Model", options)

			# if st.button("Classify CSV"):
			# 	# Transforming user input with vectorizer
			# 	#vect_text = tweet_cv.transform([tweet_text]).toarray()
			# 	# Load your .pkl file with the model of your choice + make predictions
			# 	# Try loading in multiple models to give the user a choice
			# 	predictor = None
			# 	if selection == "Multinomial Naive Bayes Classifier":
			# 		mnb = pickle.load(open('/resources/MultinomialNaiveBeyes.pkl','rb'))
			# 		predictor = mnb	
			# 	elif selection == "Logistic Regression Classifier":
			# 		#lr = pickle.load(open('/resources/LogisticRegression.pkl','rb'))
			# 		predictor = joblib.load(open(os.path.join("resources/LogisticRegression.pkl"),"rb"))
			# 	elif selection == "Linear Support Vector Classifier":
			# 		lsvc = pickle.load(open('resources/LinearSVC.pkl','rb'))
			# 		predictor = lsvc

			# 	#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			# 	#processed_df = preprocess_csv(df_uploaded)
			# 	vect_text = count_vec.transform(X_pred)
			# 	sparse_vec_msg_df = pd.DataFrame.sparse.from_spmatrix(vect_text, columns = count_vec.get_feature_names_out())
			# 	df_vectorized_combined = pd.concat([processed_df.reset_index(drop=True), sparse_vec_msg_df.reset_index(drop=True)], axis=1)

			# 	df_vectorized_combined = df_vectorized_combined.drop(["tweetid","message"], axis='columns')


			# 	prediction = predictor.predict(df_vectorized_combined)
			# 	df_download = df_uploaded.copy()
			# 	df_download['sentiment'] = prediction

			# 	# When model has successfully run, will print prediction
			# 	# You can use a dictionary or similar structure to make this output
			# 	# more human interpretable.

			# 	#st.success("Text Categorized as: {}".format(prediction))
			# 	st.success("Tweets succesfully classified")
			# 	st.dataframe(data=df_download, width=None, height=None)
			# 	st.download_button(label='Download CSV with sentiment predictions', data=df_download.to_csv(),file_name='sentiment_predictions.csv',mime='text/csv')

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
