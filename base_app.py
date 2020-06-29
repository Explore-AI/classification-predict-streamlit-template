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
import joblib
import os

# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud

import nltk

from PIL import Image

from model import select_model
from preprocessing import clean_text
from visualisation import visualize_data, common_words
from hashtags import extract_hash

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)
# Load your raw data
raw = pd.read_csv("resources/train.csv")

def display_image(name, caption):
    image = Image.open(f"resources/imgs/undraw/{name}.png")
    st.image(image, caption=caption, use_column_width=True)

def information_view(df, other):

    st.info("How to use this platform")
    # You can read a markdown file from supporting resources folder
    st.markdown("""
        ### **So you want to find out what your customers think about Climate Change?**
        ### Instructions
        1. Go to the sidebar and upload your dataset or continue with existing one
        2. Go to Insights to see data visualisations about what users think about climate change.
        3. Finally, go to the Predictions page to use machine learning models find out sentiments of tweets.
    """)

    display_image('undraw_data_reports_706v','')

    st.markdown("""
        ## Pre-existing dataset
    """)

    st.subheader("Raw Twitter data and label")
    if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
        # will write the df to the page
        st.table(df[['sentiment', 'message']].head())

    st.subheader("Clean Twitter data and label")
    if st.checkbox('Show clean data'):
        st.table(other[['sentiment', 'message']].head())

def model_prediction_view():

    display_image('undraw_viral_tweet_gndb','')
    st.info("Classification with Machine Learning Models")
    st.markdown("""
        ### Use some of the built-in models to find out the sentiment of your tweet.
        #### **Instructions:**
        1. Simply pick a model you would like used to classify your tweet from the dropdown menu
        2. Type in your tweet on the text area
        3. Click the 'Classify' button and see your results below.
    """)

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    models = ("Logistic_regression", "Linear SVC", "SVM")
    st.subheader('Pick a model to classify your text:')
    chosen = st.selectbox('', models)

    if chosen in models:
        select_model(chosen)

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    display_image('undraw_welcome_cats_thqn','')
    st.title("Sentiment Analysis on Climate Change")

    st.subheader("Should your business be Eco-friendly?")
    st.markdown("""
        This platform helps you make data-driven decisions. Find out how your customers feel about climate change.
    """)

    df = clean_text(raw)

    data = None
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")


    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Information", "Insights", "Predictions"]
    selection = st.sidebar.selectbox("Menu", options)

    # Building out the "Insights" page
    if selection == "Insights":
        title = 'Below are some data visualisations and Insights extracted from the tweets'
        display_image('undraw_google_analytics_a57d',title)
        st.write("## **Wordcloud Visualisations**")
        visualize_data(df)

        st.write("### **The barplots below shows the most common words per category**")
        options = st.multiselect('Select tweet category to visualize with BarPlot:', ['Pro', 'Anti', 'Neutral', 'News'], ['Pro'])
        for sentiment in options:
            common_words(df, sentiment, f'{sentiment} Tweets')

        st.subheader("Observations")
        st.write("""
            * Climate Change and Global warming appear to be the most popular words amongst these tweets.
                """)

        extract_hash(df)
        #plot_pie(df_pol, 'Political View')
        #plot_pie(df_pol, 'Political View')

        st.subheader("Observations")
        st.write("""
            * Investigating individual words still shows that there is an overlap of most used words between the classes.
            * However, it does become very apparent that there are themes that help formulate or form tweeters opinions on twitter.
            * Seeing words such as Trump, Obama would lead one to believe that there is a political connection to what people tweet about climate change.
            * We can also see the word 'husband' appearing as most common under the pro tweets, this shows that the climate change topic is being discussed amongst families as well, or that people do think about climate change in relation to people close to them.
            * We can then also assume that there is perhaps a social aspect to how people form their opinion on climate change.
            * Hashtags provide more context, as people will most usually tweet under a certain hashtag as a means of making it easier to find information with a theme or specific context.
            """)

    # Building out the "Information" page
    if selection == "Information":
        display_image('undraw_my_code_snippets_lynx','Find out what users say about your business')

        information_view(pd.read_csv("resources/train.csv"), df)

        if uploaded_file is not None:
            st.markdown("""
                ## Your new dataset.
            """)
            data = pd.read_csv(uploaded_file)
            st.table(data.message.head())

    # Building out the predication page
    if selection == "Predictions":
        model_prediction_view()

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
