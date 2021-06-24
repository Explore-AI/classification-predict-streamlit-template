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
from pandas.core.frame import DataFrame
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd
import re
import string
import inflect
import unicodedata
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

# Vectorizer
news_vectorizer = open("resources/vectoriser.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# preprocessing the raw data


def pre_process(text):
    # remove stpo_words
    text = ' '.join([word for word in str(text).split()
                    if word not in stopwords.words('english')])

    # This function take a string as an input and removes any url that are present in that string
    text = re.sub(r"http\S+", "", text)

    # remove punctuations
    words = str.maketrans('', '', string.punctuation)
    text = text.translate(words)

    # replace slang
    text = ' '.join([word.replace('rt', '') for word in text.split()])

    # remove ascii
    text = ''.join([unicodedata.normalize('NFKD', word).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore') for word in text])

    # Convert all characters to lowercase from list of tokenized words
    text = ''.join([word.lower() for word in text])

    # Replace all intergers
    inflector = inflect.engine()
    text = ''.join([inflector.number_to_words(
        word) if word.isdigit() else word for word in text])

    # other
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    #
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"
                u"\U0001F300-\U0001F5FF"
                u"\U0001F680-\U0001F6FF"
                u"\U0001F1E0-\U0001F1FF"
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
     "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    return text


# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page
    # these are static across all pages

    
    #st.subheader("You tweet, we classify!")
    st.sidebar.image("c2fea606b12a4a2ebdc4dd18e5cc9b54.png", use_column_width=True)

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ("Introduction", "Analysis", "Prediction", "Team")
    selection = st.sidebar.radio("Choose Option", options)
    
    # Building out the "Information" page
    if selection == "Introduction":
        st.title("Introduction")

        # You can read a markdown file from supporting resources folder
        st.markdown("![climate](https://images.unsplash.com/photo-1580868636775-b8a1818ca086?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=700&h=300&q=80)  \n\n"
                    "Hey there! Welcome to Clamassifer App. An app that classifies climate change tweets.  \n\n"
                    "Are you a company planning to launch a product or offer a service that is environmentally friendly and sustainable? But you want to know how your customers perceive climate change?   \n\n"
                    "We got you! Clamassifer helps you understand how your products/service may be received so that you can come up with better marketing strategies and potentially increase your revenues.")

    # Building out the "Information" page
    if selection == "Analysis":
        raw_copy = raw.copy()
        st.title("Part of our EDA analysis")
        st.info("Data Visiualization")
        sns.set()
        raw['message'] = pre_process(raw)
        raw['sentiment_labels']  = raw['sentiment'].map({-1:'Negative', 0:'Neutral', 1:'Positive', 2:'News'})

        fig, axes = plt.subplots(ncols = 2, nrows = 1, figsize = (15, 10), dpi = 100)
        
        sns.countplot(raw['sentiment_labels'], ax = axes[0]).set_ylabel('Number of Tweets')
        Sentiments_ = ['Positive', 'News', 'Neutral', 'Negative']
        axes[1].pie(raw['sentiment_labels'].value_counts(),
            labels = Sentiments_,
            autopct = '%1.0f%%',
            startangle = 90,
            explode = (0.1, 0.1, 0.1, 0.1))
        fig.suptitle('Count for each sentiment class', fontsize=20)
        st.write(fig)

         # Distribution plots for the labels
        fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows = 1, ncols = 4, figsize = (16, 8), dpi = 100)

        #For Positive 
        sns.distplot(raw[raw['sentiment'] == 1]['message'].str.len(), hist=True, kde=True, bins = int(200/25),
                    color = 'blue', ax = ax1, hist_kws = {'edgecolor':'black'}, kde_kws = {'linewidth': 4})
        ax1.set_title('Positive')
        ax1.set_xlabel('message_Length')
        ax1.set_ylabel('Density')

        #For Negative 
        sns.distplot(raw[raw['sentiment'] == -1]['message'].str.len(), hist=True, kde=True, bins = int(200/25),
                    color = 'lightblue', ax = ax2, hist_kws = {'edgecolor':'black'}, kde_kws = {'linewidth': 4})
        ax2.set_title('Negative ')
        ax2.set_xlabel('message_Length')
        ax2.set_ylabel('Density')

        #For Neutral 
        sns.distplot(raw[raw['sentiment'] == 0]['message'].str.len(), hist=True, kde=True, bins = int(200/25),
                    color = 'purple', ax = ax3, hist_kws = {'edgecolor':'black'}, kde_kws = {'linewidth': 4})
        ax3.set_title('Neutral ')
        ax3.set_xlabel('message_Length')
        ax3.set_ylabel('Density')

        #For News
        sns.distplot(raw[raw['sentiment'] == 2]['message'].str.len(), hist=True, kde=True, bins=int(200/25),
                    color = 'green', ax = ax4, hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 4})
        ax4.set_title('News')
        ax4.set_xlabel('message_Length')
        ax4.set_ylabel('Density')
        st.write(fig)

        # Visualizing text lengths for each sentiment
        fig = plt.figure(figsize = (15, 10))
        labels = ['Negative', 'Neutral', 'Positive', 'News']
        sns.barplot(x = 'sentiment', y = raw_copy['message'].apply(len), data = raw_copy, palette='husl').set_xticklabels(labels)
        plt.ylabel('Length')
        plt.xlabel('Sentiment')
        plt.title('Average Length of Message by Sentiment')
        st.write(fig)


        # Creating a column of hastags users
        raw_copy['users'] = [''.join(re.findall(r"@\w+", sentence.lower())) if '@' in sentence else np.nan for sentence in raw_copy.message]

        # Creating a column of hastags users
        raw_copy['hashtags'] = [''.join(re.findall(r"#\w+", sentence.lower())) if '#' in sentence else np.nan for sentence in raw_copy.message]

        hashtag = raw_copy[['message', 'hashtags']].groupby('hashtags', as_index=False).count().sort_values(by='message', ascending = False)

        fig = plt.figure(figsize=(12, 6))
        sns.barplot(data = hashtag.iloc[:20], y = 'hashtags', x = 'message', orient='h', color = 'blue')
        plt.title("Top 20 most used hashtags", fontsize = 14)
        plt.xlabel('Hashtag Count')
        plt.ylabel('Hashtags')
        st.write(fig)

        # Number of messages per users
        users = raw_copy[['message', 'users']].groupby('users', as_index=False).count().sort_values(by='message', ascending = False)

        fig = plt.figure(figsize=(12, 6))
        sns.countplot(data = raw_copy, y = 'users', order = raw_copy.users.value_counts().iloc[:20].index, orient='h')
        plt.title('Top 20 Most Popular Tags', fontsize = 14)
        plt.xlabel('Number of Tags')
        plt.ylabel('User')
        st.write(fig)

        # Separate minority and majority classes
        News = raw_copy[raw_copy['sentiment'] == 2]
        Pro = raw_copy[raw_copy['sentiment'] == 1]
        Neutral = raw_copy[raw_copy['sentiment'] == 0]
        Anti = raw_copy[raw_copy['sentiment'] == -1]

        
        fig = plt.figure(figsize = (15,10))
        wc = WordCloud(max_words = 200 , width = 1600 , height = 800, collocations = False).generate(" ".join(Anti['message']))
        st.image(wc.to_array())

        
        fig = plt.figure(figsize = (15,10))
        wc = WordCloud(max_words = 200 , width = 1600 , height = 800, collocations = False).generate(" ".join(Pro['message']))
        st.image(wc.to_array())

        fig = plt.figure(figsize = (15,10))
        wc = WordCloud(max_words = 200 , width = 1600 , height = 800, collocations = False).generate(" ".join(Neutral['message']))
        st.image(wc.to_array())

        fig = plt.figure(figsize = (15,10))
        wc = WordCloud(max_words = 200 , width = 1600 , height = 800, collocations = False).generate(" ".join(News['message']))
        st.image(wc.to_array())

    # Building out the predication page
    if selection == "Prediction":
        st.title("Analysing Your Tweet")
        option = ["MultinomialNB Model", "LogisticRegresion Model"]
        select = st.selectbox("Select Model To Use", option)
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text To Classify",str("Type Here"))

        if select == "MultinomialNB Model":
           predictor = joblib.load(open(os.path.join("resources/multinb.pkl"),"rb"))
           if st.button("Classify"): 
              # Transforming user input with vectorizer
              vect_text = tweet_cv.transform([pre_process(tweet_text)]).toarray()
              # Load your .pkl file with the model of your choice + make predictions
              # Try loading in multiple models to give the user a choice
              prediction = predictor.predict(vect_text)

              # When model has successfully run, will print prediction
              # You can use a dictionary or similar structure to make this output
              # more human interpretable.
              st.success("Text Categorized as: {}".format(prediction))

        if select == "LogisticRegresion Model":
           predictor = joblib.load(open(os.path.join("resources/logreg.pkl"),"rb"))
           if st.button("Classify"):
              # Transforming user input with vectorizer
              vect_text = tweet_cv.transform([pre_process(tweet_text)]).toarray()
              # Load your .pkl file with the model of your choice + make predictions
              # Try loading in multiple models to give the user a choice
              prediction = predictor.predict(vect_text)

              # When model has successfully run, will print prediction
              # You can use a dictionary or similar structure to make this output
              # more human interpretable.
              st.success("Text Categorized as: {}".format(prediction))
    #Building Team page     
    if selection == "Team":
        st.title("Team Members")
        st.markdown("Wisley Ramukhuba (Team Coodinator)\n\n"
                    "Mukovhe Lugisani (Team Member) \n\n"
                    "Boitumelo Magakwe (Team Member) \n\n"
                    "Onkarabile Tshele (Team Member) ")
        
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
        main()
