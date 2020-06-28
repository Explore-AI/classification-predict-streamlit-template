#!/usr/bin/env python
# coding: utf-8

# **Imports**

# In[4]:


import streamlit as st
import joblib,os
import pandas as pd
import re
import en_core_web_sm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
nlp = en_core_web_sm.load()
import spacy
from spacy import displacy
from textblob import TextBlob


# In[ ]:


# Vectorizer
lsvc_vectorizer = open("tfidfvect.pkl","rb")
lsvc_vectorizer = joblib.load(lsvc_vectorizer) # loading your vectorizer from the pkl file
svc_vectorizer = open("tf_vect1.pkl", "rb")
svc_vectorizer = joblib.load(svc_vectorizer) 
sgd_vectorizer = open("tf_vect2.pkl", "rb")
sgd_vectorizer = joblib.load(sgd_vectorizer)
rf_vectorizer = open("tf_vect3.pkl", "rb")
rf_vectorizer = joblib.load(rf_vectorizer)


# In[ ]:


# Load your raw data
raw = pd.read_csv("train.csv")


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

  #Cleaning twitter data
def clean_text(text):
    string = re.sub(r'http\S+', 'LINK', text)
    string = re.sub(r'[^\w\s]', '', string)
    string = string.lstrip()
    string = string.rstrip()
    string = string.replace('  ', ' ')
    string = string.lower()
    return string

def extract_entity(text):
    #nlp = spacy.load('en')
    docx = nlp(text)
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    return entities


# In[4]:


# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
    
    #Set page title
    st.title("Tweet Classifer")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Information","General","EDA","Clean Data","Prediction","NLP","Polarity", "Named Entity"]
    st.sidebar.title("Pages")
    selection = st.sidebar.selectbox("Choose Option", options)



    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page
        if st.checkbox('Show WordCloud'):
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


            wordcloud = WordCloud().generate(' '.join(raw['message']))
            plt.figure(figsize=(10, 7))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis('off')
            plt.show()
            st.pyplot()


        if st.checkbox('Named Entity Recognition'):
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


            result = extract_entity(raw['message'])

            wordcloud = WordCloud().generate(result)
            plt.figure(figsize=(10, 7))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis('off')
            plt.show()
            st.pyplot()


    # Building out the "General" page
    if selection == "General":
        st.info("Project Overview")
	# You can read a markdown file from supporting resources folder
        st.markdown(open("generaloverview.md","r").read())     


    # Building out the "EDA" page
	#if selection == "EDA":
	    #st.info("Exploratory Data Analysis")
	# You can read a markdown file from supporting resources folder--------
		#st.markdown(open("eda.md","r").read())




    # Building Clean Data Page
    if selection == "Clean Data":
        st.info("Clean Your Text")
        tweet_text = st.text_area("Enter Text", "Type Here")
        if st.button("Clean"):
            result = clean_text(tweet_text)
            st.text(result)

    # Building out the predication page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        st.subheader("For best results, first clean text data using Clean Data option in drop box.")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here")
        prediction_labels = {'Anti':-1,'Neutral':0,'Pro':1,'News':2}
        model_type = ["Linear SVC","SVC","SGD Classifier","Random Forest Classifier"]
        task_choice = st.selectbox("Choose Model",model_type)
       
        if st.button("Classify"):
            st.text("Original test :\n{}".format(tweet_text))
          
            tweet_text = clean_text(tweet_text)
            # Transforming user input with vectorizer
            vect_text = lsvc_vectorizer.transform([tweet_text]).toarray()
            vect_text1 = svc_vectorizer.transform([tweet_text]).toarray()
            vect_text2 = sgd_vectorizer.transform([tweet_text]).toarray()
            vect_text3 = rf_vectorizer.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            if task_choice == "Linear SVC":
                predictor = joblib.load(open(os.path.join("linear_svc.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
               

            elif task_choice == "SVC":
                predictor = joblib.load(open(os.path.join("SVC.pkl"), "rb"))
                prediction = predictor.predict(vect_text1)
                
           
            elif task_choice == "SGD Classifier":
                predictor = joblib.load(open(os.path.join("SGD.pkl"), "rb"))
                prediction = predictor.predict(vect_text2)
                

            elif task_choice == "Random Forest Classifier":
                predictor = joblib.load(open(os.path.join("RFC.pkl"), "rb"))
                prediction = predictor.predict(vect_text3)
                

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

    # Building the NLP Page        
    if selection == 'NLP':
        st.info("Natural Language Processing")
        tweet_text = st.text_area("Enter Text","Type Here")
        nlp_task = ["Tokenization","Lemmatization","POS Tags"]
        task_choice = st.selectbox("Choose NLP Task",nlp_task)
        if st.button("Analyze"):
            

            messagee = nlp(tweet_text)
            if task_choice == 'Tokenization':
                result = [ token.text for token in messagee]
                st.json(result)
            elif task_choice == 'Lemmatization':
                result = ["Token: {},Lemma: {}".format(token.text,token.lemma_) for token in messagee]
                st.json(result)
            elif task_choice == 'POS Tags':
                result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in messagee]
                st.json(result)

    # Building the polarity page
    if selection == "Polarity":
        st.info("Sentiment Analysis")
        tweet = st.text_area("Enter Text","Type Here")
        st.subheader("For best results, first clean text data using Clean Data option in drop box.")

        if st.button("Analysis"):
            b = TextBlob(tweet)
            result = b.sentiment
            st.success(result)


    # Building the named entity page
    if selection == "Named Entity":
        st.info("Extract Entities from Text")
        tweet_text = st.text_area("Enter Text","Type Here")
        if st.button("Extract"):
            result = extract_entity(tweet_text)
            st.success(result)
    
    # Show Plots
    if st.checkbox("Simple Bar Plot with Matplotlib "):
        train.plot(kind='bar')
        st.pyplot()


    # Show Correlation Plots
    if st.checkbox("Simple Correlation Plot with Matplotlib "):
        train.matshow(train.corr())
        st.pyplot()

    # Show Correlation Plots with Sns
    if st.checkbox("Simple Correlation Plot with Seaborn "):
        st.write(sns.heatmap(train.corr(),annot=True))
        st.pyplot()

    # Show Plots
    if st.checkbox("Bar Plot of Groups or Counts"):
        v_counts = train.groupby('species')
        st.bar_chart(v_counts)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

