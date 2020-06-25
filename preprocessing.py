"""

    Helper Functions.

    Author: JHB-EN2.

    Description: These helper functions are to be used to clean data for predictions
    and EDA 

"""
######################################################################################################
##################################----------TITUS-&-BULELANI------------##############################
######################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Data cleaning tools
import nltk
from nltk.tokenize import TweetTokenizer
from collections import Counter
import re
from nltk.corpus import stopwords


# Data analysis libraries
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
nltk.download('vader_lexicon')
nltk.download('stopwords')

# !python -m spacy download en_core_web_md

# Feature creation functions
def findURLs(tweet):
    """Function that finds urls and replaces """
    pattern = r'ht[t]?[p]?[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

    all_ = " ".join(re.findall(pattern,tweet))
    return all_




## INCLUDE HELPER FUNCTIONS

######################################################################################################
##################################----------END------------##############################
######################################################################################################


# class TweetPreprocessing:
    
#     # def __init__(self):
#     #     self.tweet = tweet
    
#     def lemmatizeTweet(self, tweet):
#         lemmatized_tweet = list()
#         doc = nlp(tweet)
#         for token in doc:
#             lemmatized_tweet.append(token.lemma_)
#         lemmatized_tweet = ' '.join(lemmatized_tweet)
#         return lemmatized_tweet
    
#     def tweetPreprocessor(self, tweet):
#         clean_tweat = tweet.replace('\n', ' ')
#         clean_tweat = re.sub('\w*\d\w*', ' ', clean_tweat)
#         clean_tweat = re.sub(r'[::.,_()/\{}"?!¬¦ãÃâÂ¢\d]', '', clean_tweat)
#         return clean_tweat
    
#     def tweetTokenizer(self, tweet):
#         tokenizer = TweetTokenizer(preserve_case = False, strip_handles = True)
#         tokenized_tweet = tokenizer.tokenize(tweet)
#         return tokenized_tweet
    
#     def removeStopWords(self, tweet):
#         clean_tweet = list()
#         for token in tweet:
#             if token not in stopwords.words('english'):
#                 if token.startswith('#') == False:
#                     if len(token) > 1:
#                         clean_tweet.append(token)
#         return clean_tweet
    
# # tweet = train_data.loc[0, 'message']
# # tweet = tweet_preproc.lemmatizeTweet(tweet)
# # tweet = tweet_preproc.tweetPreprocessor(tweet)
# # tweet = tweet_preproc.tweetTokenizer(tweet)
# # tweet = tweet_preproc.removeStopWords(tweet)
