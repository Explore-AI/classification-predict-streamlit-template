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

# Data Cleaning
def findURLs(tweet):
    """
    return a string of all urls in a text
    Parameters
    ----------
        tweet (str): tweet containing urls
    Returns
    -------
        all_ (str): all urls in text
    Examples
    --------
        >>> findURLs("you can view this at https://github.com/mrmamadi/classification-predict-streamlit-template")
        "https://github.com/mrmamadi/classification-predict-streamlit-template"
    """
    pattern = r'ht[t]?[p]?[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

    all_ = " ".join(re.findall(pattern,tweet))
    return all_


def strip_url(df):
    """
    removes all urls from a the DataFrame raw message and replaces them with "urlweb"
    Parameters
    ----------
        df (DataFrame): input dataframe
    Returns
    -------
        clean_df (DataFrame): output dataframe
    Examples
    --------
            |index|sentiment|message|tweetid
            |0    |-1       |https..|13442
        >>> strip_url(train)
            |index|sentiment|message|tweetid
            |0    |-1       |urlweb |13442
    """
    clean_df = df.copy()
    pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-@.&+]|[!*(),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    subs_url = r'urlweb'
    clean_df['message'] = clean_df['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)
    return clean_df

def findHandles(tweet):
    """
    returns a list of all handles in a tweet
    Parameters
    ----------
        tweet (str): text containing handles
    Returns
    -------
        handles (list): list of all handles
    Examples
    --------
        >>> findHandles("hi @SenBernieSanders, you will beat @realDonaldTrump")
            ['@SenBernieSanders','@realDonaldTrump']
    """
    handles = list()
    for token in tweet.split():
        if token.startswith('@'):
            handles.append(token)#.replace('@', '')
    return handles

def findHashTags(tweet):
    """
    returns a list of all hashtags in a tweet
    Parameters
    ----------
        tweet (str): text containing hashtags
    Returns
    -------
        hash_tags (list): list of all hashtags
    Examples
    --------
        >>> findHashTags("Oil is killing the world renewables and EVS are the way the go! #EVs #GlobalWarming #Fossilfuels")
            ['#EVs', '#GlobalWarming', '#Fossilfuels']
    """
    hash_tags = list()
    for token in tweet.split():
        if token.startswith('#'):
            hash_tags.append(token)
    return hash_tags

# Feature Creation and extration
def removePunctuation(tweet):
    """
    akes as input a single tweet and removes punctuation and other uncommon characters in the tweet.
    See example implementation below and how the punctuation has been removed.
    Parameters
    ----------
        tweet (str): string containing punctuation to be removed
    Returns
    -------
        clean_tweet (str): string without punctuation
    Examples
    --------
        >>> removePunctuation("Hey! Check out this story: urlweb. He doesn't seem impressed. :)")
            "Hey Check out this story urlweb He doesn't seem impressed"
    """
    clean_tweet = tweet.replace('\n', ' ') # first remove line spaces
    clean_tweet = re.sub('\w*\d\w*', ' ', clean_tweet) # substitute digits within text with an empty string
    clean_tweet = re.sub(r'[:;.,_()/\{}"?\!&¬¦ãÃâÂ¢\d]', '', clean_tweet) # remove punctuation
    # some of the character removed here were determined by visually inspecting the text
    return clean_tweet


def tweetTokenizer(tweet):
    """
    tokenizes and strips handles from twitter data
    Parameters
    ----------
        tweet (str): string to be tokenized
    Returns
    -------
        tokenized_tweet (list): list of tokens in tweet
    Examples
    --------
        >>> tweetTokenizer("Read @swrightwestoz's latest on climate change insurance amp lending featuring APRA speech and @CentrePolicyDev work urlweb")
            ['read',
             'latest',
             'on',
             'climate',
             'change',
             'insurance',
             'amp',
             'lending',
             'featuring',
             'apra',
             'speech',
             'and',
             'work',
             'urlweb']
    """
    tokenizer = TweetTokenizer(preserve_case = False, strip_handles = False)
    tokenized_tweet = tokenizer.tokenize(tweet)
    return tokenized_tweet

def removeStopWords(tokenized_tweet):
    """
    removes stop words and punctation relics
    Parameters
    ----------
        tokenized_tweet (list): list of tokens to be cleaned
    Returns
    -------
        clean_tweet (list): list of tokens without stopwords
    Examples
    --------
        >>> removeStopWords(['read',
                            'latest',
                            'on',
                            'climate',
                            'change',
                            'insurance',
                            'amp',
                            'lending',
                            'featuring',
                            'apra',
                            'speech',
                            'and',
                            'work',
                            'urlweb'])
            ['read',
             'latest',
             'on',
             'climate',
             'change',
             'insurance',
             'amp',
             'lending',
             'featuring',
             'apra',
             'speech',
             'and',
             'work',
             'urlweb']
    """
    clean_tweet = list() # initialising an empty list as container for the cleaned tweet
    for token in tokenized_tweet:  # iterating through all words in a list
        # checking if current word is not a stopword
        if token not in stopwords.words('english') + ['amp','rt','urlweb']:
            # also checking if the current word is a hash_tag
            if token.startswith('#') == False:
                # also checking if the current word has more than one character
                if len(token) > 1:
                    # if all condition are satisfied, keep the word
                    clean_tweet.append(token)
                    
    # return the cleaner tweet
    return clean_tweet

def lemmatizeTweet(tweet):
    """
    tweet lemmatizer
    Parameters
    ----------
        tweet (list): tokens to be lemmatized
    Returns
    -------
        lemmatized_tweet (list): lemmatized list of tokens 
    Examples
        >>> lemmatizeTweet(['read',
                            'latest',
                            'on',
                            'climate',
                            'change',
                            'insurance',
                            'amp',
                            'lending',
                            'featuring',
                            'apra',
                            'speech',
                            'and',
                            'work',
                            'urlweb'])
            ['read',
             'latest',
             'climate',
             'change',
             'insurance',
             'lending',
             'featuring',
             'apra',
             'speech',
             'work',
             'urlweb']
    """
    lemmatized_tweet = list()
    lmtzr = WordNetLemmatizer()
    for token in tweet:
        lemmatized_tweet.append(lmtzr.lemmatize(token))
        
    return lemmatized_tweet = list()
    lmtzr = WordNetLemmatizer()
    for token in tweet:
        lemmatized_tweet.append(lmtzr.lemmatize(token))
        
    return lemmatized_tweet

def removeInfrequentWords(tweet, top_n_words):
    """
    Function that goes through the words in a tweet,
    determines if there are any words that are not in
    the top n words and removes them from the tweet
    and return the filtered tweet.
    Parameters
    ----------
        tweet (list): list tokens to be flitered
        top_n_words (int): number of tweets to keep
    Returns
    -------
        filt_tweet (list): list of top n words
    Examples
    --------
        >>> bag_of_words = [('change', 12634),
                            ('climate', 12609),
                            ('rt', 9720),
                            ('urlweb', 9656),
                            ('global', 3773)],
        >>> removeInfrequentWords(['rt', 'climate', 'change', 'equation', 'screenshots', 'urlweb'],2)
            ['change', 'climate']    

    """
    filt_tweet = list()
    for token in tweet:
        if token in top_n_words:
            filt_tweet.append(token)
    return filt_tweet

def removeCommonWords(tweet):
    """
    removes the most common words from a list of given words
    Parameters
    ----------
        tweet (list): list of words to be cleaned
    Returns
    -------
        filt_tweet (list): list of cleaned words
    Examples
    --------
        >>> very_common_words = ['change', 'climate', 'rt', 'urlweb', 'global']
        >>> removeCommonWords(['rt', 'climate', 'change', 'equation', 'screenshots', 'urlweb'])
            ['equation']
    """
    filt_tweet = list()
    for token in tweet:
        if token not in very_common_words:
            filt_tweet.append(token)
    return filt_tweet

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
