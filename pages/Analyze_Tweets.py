#streamlit dependencies
from matplotlib.pyplot import title
from nltk.corpus.reader.pl196x import ANA
import streamlit as st
from nltk.tokenize import TreebankWordTokenizer
tbt = TreebankWordTokenizer()
import pandas as pd
from nlppreprocess import NLP
nlp = NLP()
from PIL import Image

df_train = pd.read_csv('https://raw.githubusercontent.com/TEAM-CW3/classification-predict-streamlit-data/main/train.csv')
st.info('The following are some of the charts that we have created from the raw data. Some of the text is too long and may cut off, feel free to right click on the chart and either save it or open it in a new window to see it properly.')

# Number of Messages Per Sentiment
st.write('Distribution of the sentiments')
all_analyzers = ['Choose Analyzer','Line & Bar Graphs', 'Pie Chart', 'Word Cloud', 'Missing Values']
analyzer = st.selectbox('Select Analyzer', all_analyzers)

#delete above

# Inserting our visuals for analysing data
if analyzer == 'Line & Bar Graphs':
        #import functions.line_and_bar as graph
        #graph.plot_line_and_bar()
        image = Image.open('resources/imgs/Bargraph.png')
        st.image(image)

elif analyzer == 'Pie Chart':
        import functions.pie_chart as pie
        pie.plot_pie_chart()
       

elif analyzer == 'Word Cloud':
        #import functions.plot_word_cloud as pwd
        #pwd.gen_wordcloud(title)
        image = Image.open('resources/imgs/word_cloud.png')
        st.image(image)

elif analyzer == 'Missing Values':
        #import functions.missing_values as missing
        #missing.missing_vals()
        image = Image.open('resources/imgs/missingwords.png')
        st.image(image)