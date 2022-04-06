#core packages
import streamlit as st
import altair as alt

#EDA PKGs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer

import re


#utils
import joblib

# Load your raw data
data = pd.read_csv("data/train.csv")

model_lr = joblib.load(open('models\sentiment analysis pipe_lr1.pkl', 'rb'))
model_mnb =joblib.load(open('models\sentiment analysis pipe_mnb.pkl', 'rb'))
#model_pass = joblib.load(open('models\sentiment analysis pipe_pac.pkl', 'rb'))

#fnx
def predict_sentiment(docx):
    result = model_lr.predict([docx])
    return result[0]

def get_predict_proba(docx):
    results = model_lr.predict_proba([docx])
    return results

#fnx
def predict_sentiment(docx):
    result = model_mnb.predict([docx])
    return result[0]

def get_predict_proba(docx):
    results = model_mnb.predict_proba([docx])
    return results 


sentiment_name_dict = {-1 : 'Anti', 0 : 'Neutral', 1 : 'Pro', 2 : 'News'}


def main():
    st.title('Sentiment classifier app')
    menu =['About','Visualization','Text Sentiment Predictions']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Sentiment Text Prediction')
        st.sidebar.subheader('Tweets')
	    
        model_type = st.sidebar.selectbox("Model Type",('Naive Bayes', 'Logistic Regression'))

        if model_type == 'Naive Bayes':
            with st.form(key='sentiment_clf_form'):
                raw_text = st.text_area("Type here")
                submit_text = st.form_submit_button(label='Submit')
        
            if submit_text:
                col1,col2 = st.columns(2)
            
            # apply function here
                prediction = predict_sentiment(raw_text)
                probability = get_predict_proba(raw_text)

                with col1:
                    st.success('Original text')
                    st.write(raw_text)

                    st.success("Prediction")
                    sentiment_name = sentiment_name_dict[prediction]

                    st.write('{}:{}'.format(prediction,sentiment_name))

                    #get the confidence of the prediction
                    st.write('Confidence: {}'.format(np.max(probability)))



                with col2:
                    st.success("Prediction Probability")
                    st.write(probability)
                    #convert the entire probability into a adataframe
                    proba_df = pd.DataFrame(probability, columns=model_mnb.classes_)
                    st.write(proba_df.T)

                    #modify to plot it right
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ['sentiments', 'probability']

                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='sentiments', y ='probability', color = 'sentiments')
                    st.altair_chart(fig, use_container_width=True)

        else:
            with st.form(key='sentiment_clf_form'):
                raw_text = st.text_area("Type here")
                submit_text = st.form_submit_button(label='Submit')
        
            if submit_text:
                col1,col2 = st.columns(2)
            
            # apply function here
                prediction = predict_sentiment(raw_text)
                probability = get_predict_proba(raw_text)

                with col1:
                    st.success('Original text')
                    st.write(raw_text)

                    st.success("Prediction")
                    sentiment_name = sentiment_name_dict[prediction]

                    st.write('{}:{}'.format(prediction,sentiment_name))

                    #get the confidence of the prediction
                    st.write('Confidence: {}'.format(np.max(probability)))



                with col2:
                    st.success("Prediction Probability")
                    st.write(probability)
                    #convert the entire probability into a adataframe
                    proba_df = pd.DataFrame(probability, columns=model_lr.classes_)
                    st.write(proba_df.T)

                    #modify to plot it right
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ['sentiments', 'probability']

                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='sentiments', y ='probability', color = 'sentiments')
                    st.altair_chart(fig, use_container_width=True)

    
    elif choice == 'About':

            st.info("General Information")
            # You can read a markdown file from supporting resources folder
            st.markdown("A Machine Learning Model that is able to classify whether or not a person believes in climate change, based on their novel tweet data")

            st.subheader("Raw Twitter data and label")
            if st.checkbox('Show raw data'): # data is hidden if box is unchecked
                st.write(data[['sentiment', 'message']]) # will write the df to the page


                


    else:
        select = st.sidebar.selectbox('Visualization of Tweets',['Bar graph', 'Pie Chart'], key=1)

        sentiment = data['sentiment'].value_counts()
        sentiment = pd.DataFrame({'Sentiment': sentiment.index, 'Tweets': sentiment.values})
        st.markdown('Sentiment count')
        if select == 'Bar graph':
            fig = px.bar(sentiment, x='Sentiment',y='Tweets', color = 'Tweets', height = 500)
            st.plotly_chart(fig)
        else:
            fig = px.pie(sentiment, values='Tweets', names = 'Sentiment')
            st.plotly_chart(fig)
            
        

        

        with model_lr or model_mnb:
            options = ['Naive Bayes', 'Logistic Regression']
            model_choice = st.selectbox('Choose Model', options)
            if model_choice == 'Naive Bayes':
                col, cols = st.columns(2)
                col.subheader('Multimorminal_NB Accurac:')
                #col.write(mnb_score)

        
    

if __name__ == '__main__':
    main()
