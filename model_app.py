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

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st 
import csv
import warnings
warnings.filterwarnings("ignore")

# Libraries for data preparation and model building
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas_profiling import ProfileReport
import spellchecker
import autocorrect

import nltk
from nltk import TreebankWordTokenizer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string
import urllib
from wordcloud import WordCloud, STOPWORDS

STOPWORDS = set(stopwords.words('english'))

# Reads 'train.csv.csv' file
def classify_desc (description):
	if description == '[-1]':
		return "The tweet does not believe in man-made climate change (Anti)"
	elif description == '[0]':
		return "The tweet neither supports nor refutes the belief of man-made climate change (Neutral)"
	elif description == '[1]':
		return "The tweet supports the belief of man-made climate change (Pro)"
	elif description == '[2]':
		return "The tweet links to factual news about climate change (News)"

def upload_file():
	upload_file = st.file_uploader("Upload a .csv file that contains tweets",'csv')
	if upload_file is not None:
		return(upload_file)

def to_lower(text):
	text = text.str.lower()
	return(text)

def remove_url(text):
	text = re.sub(r"http\S+", "", text)
	return(text)

def remove_punctuation(text):
	text = re.sub('[^a-zA-z0-9\s]', '', text)
	return(text)

def remove_special_char(text):
	text = ''.join([x for x in text if x not in string.punctuation])
	return(text)

def remove_digits(text):
	text = "".join(filter(lambda x: not x.isdigit(), text))
	return(text)

def remove_stop_words(text):
	text = " ".join([word for word in str(text).split() if word not in STOPWORDS])
	return(text)

def cleaning_text(tweet):
	tweet = tweet.lower() #Change everything to lower case
	tweet = re.sub(r"http\S+", "", tweet) # remove urls
	tweet = re.sub('[^a-zA-z0-9\s]', '', tweet) # remove all puncuation
	tweet = ''.join([x for x in tweet if x not in string.punctuation]) # remove all special characters
	tweet = "".join(filter(lambda x: not x.isdigit(), tweet)) #remove all digits
	tweet = "".join(filter(lambda x: not x.isdigit(), tweet)) # remove all stop words
	return(tweet)

	

		
	
def word_map(file):
	if file is not None:	
		comment_words = ''
		stopwords = set(STOPWORDS)
		# iterate through the csv file
		for val in file.message:
			# typecaste each val to string
			val = str(val)
			# split the value
			tokens = val.split()
	
			# Converts each token into lowercase
			for i in range(len(tokens)):
				tokens[i] = tokens[i].lower()
	
			comment_words += " ".join(tokens)+" "

		wordcloud = WordCloud(width = 800, height = 800,
					background_color ='white',
					stopwords = stopwords,
					min_font_size = 10).generate(comment_words)

		# plot the WordCloud image					
		fig = plt.figure(figsize = (8, 8), facecolor = None)
		fig = plt.imshow(wordcloud)
		fig = plt.axis("off")
		fig = plt.tight_layout(pad = 0)
		st.pyplot(fig)
		

