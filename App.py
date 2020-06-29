#!/usr/bin/env python
# coding: utf-8

# **Imports**

# In[4]:


import streamlit as st
import joblib
import os
import markdown
import pandas as pd
import re
import en_core_web_sm
import spacy
from wordcloud import WordCloud
from spacy import displacy
from textblob import TextBlob
import matplotlib.pyplot as plt
nlp = en_core_web_sm.load()

# Vectorizer
lsvc_vectorizer = open("resources/tfidfvect.pkl", "rb")
# loading your vectorizer from the pkl file
lsvc_vectorizer = joblib.load(lsvc_vectorizer)
svc_vectorizer = open("resources/tf_vect1.pkl", "rb")
svc_vectorizer = joblib.load(svc_vectorizer)
sgd_vectorizer = open("resources/tf_vect2.pkl", "rb")
sgd_vectorizer = joblib.load(sgd_vectorizer)



# In[ ]:


# Load your raw data
raw = pd.read_csv("resources/train.csv")


# In[ ]:


# Load lvc model
def load_prediction_models(model_file):
    model = joblib.load(open('resources/lvc.pkl', "rb"))
    return model


# In[ ]:


def get_keys(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

# Cleaning twitter data


def clean_text(text):
    string = re.sub(r'http\S+', 'LINK', text)
    string = re.sub(r'[^\w\s]', '', string)
    string = string.lstrip()
    string = string.rstrip()
    string = string.replace('  ', ' ')
    string = string.lower()
    return string


def extract_entity(text):
    docx = nlp(text)
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    return entities


def extract_username(text):
    text = text.lower()
    text = re.findall("@([a-z0-9_]+)", text)
    return ' '.join(text)


def extract_tags(text):
    text = text.lower()
    text = re.findall("#([a-z0-9_]+)", text)
    return ' '.join(text)


# In[4]:


# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Set page title
    st.title("Tweet Classifer")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["INTRODUCTION", "EDA", "Clean Data",
               "PREDICTION", "NLP", "POLARITY", "NAMED ENTITY",
               "CONCLUSION"]
    st.sidebar.title("Pages")
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "General" page
    if selection == "INTRODUCTION":
        st.info("Project Overview")
    # You can read a markdown file from supporting resources folder
        from PIL import Image
        image = Image.open('resources/Capture.JPG')
        st.image(image, use_column_width=True)

    # Building out the "EDA" page
    if selection == "EDA":
        st.info("Exploratory Data Analysis")
    # You can read a markdown file from supporting resources folder
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            # will write the df to the page
            st.write(raw[['sentiment', 'message']])
        if st.checkbox("Show Tweets Per Sentiment"):
            sentiment = st.multiselect('Show Tweets per Sentiment',
                                       raw['sentiment'].unique())
            new_df = raw[(raw['sentiment'].isin(sentiment))]
            st.write(new_df)
        if st.checkbox('Named Entity Recognition Per Sentiment'):
            j = 0
            for message in raw['message']:
                message = message.lower()
                message = re.sub(r'http\S+', 'LINK', message)
                message = re.sub(r'[^\w\s]', '', message)
                message = message.lstrip()
                message = message.rstrip()
                message = message.replace('  ', ' ')
                raw.loc[j, 'message'] = message
                j += 1
            sentiment = st.multiselect('Show Tweets per Sentiment',
                                   raw['sentiment'].unique())
            new_df = raw[(raw['sentiment'].isin(sentiment))]
            top200 = new_df.head(200)
            string = ' '.join(top200['message'])
            result = extract_entity(string)
            st.json(result)
        if st.checkbox('WordCloud per Sentiment'):
            sentiment = st.multiselect('Show Tweets per Sentiment',
                                        raw['sentiment'].unique())
            new_df = raw[(raw['sentiment'].isin(sentiment))]
            j = 0
            for message in new_df['message']:
                message = message.lower()
                message = re.sub(r'http\S+', 'LINK', message)
                message = re.sub(r'[^\w\s]', '', message)
                message = message.lstrip()
                message = message.rstrip()
                message = message.replace('  ', ' ')
                new_df.loc[j, 'message'] = message
                j += 1
            try:
                wordcloud = WordCloud().generate(' '.join(new_df['message']))
                plt.figure(figsize=(10, 7))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis('off')
                plt.show()
                st.pyplot()
            except ValueError:
                raise ValueError("Please Select Atleast one Sentiment Option From Dropbox Above")
        if st.checkbox("Display Top 50 Usernames"):
            raw['Usernames'] = raw['message'].apply(lambda x: extract_username(x))
            counts = raw['Usernames'].value_counts().rename_axis('Usernames').reset_index(name = 'Counts')
            counts.drop(0, axis = 0, inplace = True)
            names = counts['Usernames'].head(50)
            values = counts['Counts'].head(50)
            fig = plt.figure(figsize=(20,16))
            ax = fig.add_subplot(111)
            yvals = range(len(names))
            ax.barh(yvals, values, align='center', alpha=1)
            plt.yticks(yvals,names, fontsize = 12)
            plt.title("Top 50 Usernames Tweeting About Climate Change")
            plt.ylabel("Usernames")
            plt.xlabel("Count of Username")
            plt.tight_layout()
            plt.show()
            st.pyplot()
        if st.checkbox("Display Top 50 Hash Tags"):
            raw['Tags'] = raw['message'].apply(lambda x: extract_tags(x))
            counts = raw['Tags'].value_counts().rename_axis('Tags').reset_index(name = 'Counts')
            counts.drop(0, axis = 0, inplace = True)
            names = counts['Tags'].head(50)
            values = counts['Counts'].head(50)
            fig = plt.figure(figsize=(20,16))
            ax = fig.add_subplot(111)
            yvals = range(len(names))
            ax.barh(yvals, values, align='center', alpha=1)
            plt.yticks(yvals,names, fontsize = 12)
            plt.title("Top 50 Hash Tags Used While Tweeting About Climate Change")
            plt.ylabel("Hash Tags")
            plt.xlabel("Count of Hash Tags")
            plt.tight_layout()
            plt.show()
            st.pyplot()
        if st.checkbox("Show Top 50 Usernames Per Sentiment"):
            sentiment = st.multiselect('Show Tweets per Sentiment',
                                       raw['sentiment'].unique())
            new_df = raw[(raw['sentiment'].isin(sentiment))]
            try:
                new_df['Usernames'] = new_df['message'].apply(lambda x: extract_username(x))
                counts = new_df['Usernames'].value_counts().rename_axis('Usernames').reset_index(name = 'Counts')
                counts.drop(0, axis = 0, inplace = True)
                names = counts['Usernames'].head(50)
                values = counts['Counts'].head(50)
                fig = plt.figure(figsize=(20,16))
                ax = fig.add_subplot(111)
                yvals = range(len(names))
                ax.barh(yvals, values, align='center', alpha=1)
                plt.yticks(yvals,names, fontsize = 12)
                plt.title("Top 50 Usernames Tweeting About Climate Change")
                plt.ylabel("Usernames")
                plt.xlabel("Count of Username")
                plt.tight_layout()
                plt.show()
                st.pyplot()
            except KeyError:
                raise ValueError("Please Select Atleast one Sentiment Option From Dropbox Above")
        if st.checkbox("Show Top 50 Hash Tags Per Sentiment"):
            sentiment = st.multiselect('Show Tweets per Sentiment',
                                       raw['sentiment'].unique())
            new_df = raw[(raw['sentiment'].isin(sentiment))]
            try:
                new_df['Usernames'] = new_df['message'].apply(lambda x: extract_tags(x))
                counts = new_df['Usernames'].value_counts().rename_axis('Usernames').reset_index(name = 'Counts')
                counts.drop(0, axis = 0, inplace = True)
                names = counts['Usernames'].head(50)
                values = counts['Counts'].head(50)
                fig = plt.figure(figsize=(20,16))
                ax = fig.add_subplot(111)
                yvals = range(len(names))
                ax.barh(yvals, values, align='center', alpha=1)
                plt.yticks(yvals,names, fontsize = 12)
                plt.title("Top 50 Usernames Tweeting About Climate Change")
                plt.ylabel("Usernames")
                plt.xlabel("Count of Username")
                plt.tight_layout()
                plt.show()
                st.pyplot()
            except KeyError:
                raise ValueError("Please Select Atleast one Sentiment Option From Dropbox Above")


    # Building Clean Data Page
    if selection == "CLEAN DATA":
        st.info("Clean Your Text")
        tweet_text = st.text_area("Enter Text", "Type Here")
        if st.button("Clean"):
            result = clean_text(tweet_text)
            st.text(result)

    # Building out the predication page
    if selection == "PREDICTION":
        st.info("Prediction with ML Models")
        st.subheader("For best results, first clean text data using Clean Data option in drop box.")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")
        prediction_labels = {'Anti': -1, 'Neutral': 0, 'Pro': 1, 'News': 2}
        model_type = ["Linear SVC", "SVC", "SGD Classifier"]
        task_choice = st.selectbox("Choose Model", model_type)

        if st.button("Classify"):
            st.text("Original test :\n{}".format(tweet_text))

            tweet_text = clean_text(tweet_text)
            # Transforming user input with vectorizer
            vect_text = lsvc_vectorizer.transform([tweet_text]).toarray()
            vect_text1 = svc_vectorizer.transform([tweet_text]).toarray()
            vect_text2 = sgd_vectorizer.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            if task_choice == "Linear SVC":
                predictor = joblib.load(open(os.path.join("resources/linear_svc.pkl"), "rb"))
                prediction = predictor.predict(vect_text)

            elif task_choice == "SVC":
                predictor = joblib.load(open(os.path.join("resources/SVC.pkl"), "rb"))
                prediction = predictor.predict(vect_text1)

            elif task_choice == "SGD Classifier":
                predictor = joblib.load(open(os.path.join("resources/SGD.pkl"), "rb"))
                prediction = predictor.predict(vect_text2)


            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

    # Building the NLP Page
    if selection == 'NLP':
        st.info("Natural Language Processing")
        tweet_text = st.text_area("Enter Text", "Type Here")
        nlp_task = ["Tokenization", "Lemmatization", "POS Tags"]
        task_choice = st.selectbox("Choose NLP Task", nlp_task)
        if st.button("Analyze"):

            messagee = nlp(tweet_text)
            if task_choice == 'Tokenization':
                result = [token.text for token in messagee]
                st.json(result)
            elif task_choice == 'Lemmatization':
                result = ["Token: {},Lemma: {}".format(token.text, token.lemma_) for token in messagee]
                st.json(result)
            elif task_choice == 'POS Tags':
                result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text, word.tag_, word.dep_) for word in messagee]
                st.json(result)

    # Building the polarity page
    if selection == "POLARITY":
        st.info("Sentiment Analysis")
        tweet = st.text_area("Enter Text", "Type Here")
        st.subheader("For best results, first clean text data using Clean Data option in drop box.")

        if st.button("Analysis"):
            b = TextBlob(tweet)
            result = b.sentiment
            st.success(result)

    # Building the named entity page
    if selection == "NAMED ENTITY":
        st.info("Extract Entities from Text")
        tweet_text = st.text_area("Enter Text", "Type Here")
        if st.button("Extract"):
            result = extract_entity(tweet_text)
            st.success(result)
    
    # Building the conclusion page
    if selection == "CONCLUSION":
        st.subheader("CONCLUSION")
        st.markdown("The exploratory data analysis performed during this sprint has highlighted many key insights into understanding who was tweeting about climate change, how often these individuals tweeted about climate change and whether these tweets were of a positive or negative nature. We can see from the charts provided, that there are a larger number of pro-climate change tweets than any of the other types, meaning that a larger number of individuals believe in climate change than those who do not. Through the classification techniques provided by this app, companies can access a broad base of tweet sentiments, classifying these tweets as either pro-climate change, news, neutral or anti-climate change. This will increase their level of insights on the community at large and how they feel about our current environmental state. With many individuals believing in climate change, we believe that companies can plan their market strategies around these individuals as they may be potential customers.")

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()

# %%
