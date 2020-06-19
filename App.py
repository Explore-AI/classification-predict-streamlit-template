#!/usr/bin/env python
# coding: utf-8

# **Imports**

# In[4]:


import streamlit as st
import joblib,os
import pandas as pd
import re


# In[ ]:


# Vectorizer
news_vectorizer = open("Downloads/train1.csv/classification-predict-streamlit-template-master/resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file


# In[ ]:


# Load your raw data
raw = pd.read_csv("resources/train.csv")


# In[ ]:


# Load lvc model
def load_prediction_models(model_file):
    model = joblib.load(open('lvc.pkl',"rb"))
    return model


# In[ ]:


def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key


# In[4]:


# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
    
    #Set page title
    st.title("Tweet Classifer")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information", "NLP"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page
            
    # Building out the predication page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here")
        prediction_labels = {'Anti':-1,'Neutral':0,'Pro':1,'News':2}


        if st.button("Classify"):
            st.text("Original test ::\n{}".format(tweet_text))
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))
            
    if choice == 'NLP':
        st.info("Natural Language Processing")
        tweet_text = st.text_area("Enter Text","Type Here")
        nlp_task = ["Tokenization","NER","Lemmatization","POS Tags"]
        task_choice = st.selectbox("Choose NLP Task",nlp_task)
        if st.button("Analyze"):
            st.info("Original Text {}".format(news_text))

            messagee = nlp(tweet_text)
            if task_choice == 'Tokenization':
                result = [ token.text for token in messagee ]

            elif task_choice == 'Lemmatization':
                result = ["'Token':{},'Lemma':{}".format(token.text,token.lemma_) for token in messagee]
            elif task_choice == 'NER':
                result = [(entity.text,entity.label_)for entity in messagee.ents]
            elif task_choice == 'POS Tags':
                result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in messagee]

                         

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

