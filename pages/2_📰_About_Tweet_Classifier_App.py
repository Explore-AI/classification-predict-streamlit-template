#streamlit dependencies
import streamlit as st
import pandas as pd
from nlppreprocess import NLP
nlp = NLP()

st.info('About the Tweet Classifier App')
st.write('')
st.snow()
st.markdown(""" Many companies are built around lessening one’s environmental impact or
            carbon footprint. They offer products and services that are environmentally friendly
            and sustainable, in line with their values and ideals. We have developed models to
            determine how people perceive climate change and whether or not they believe it is a
            real threat or not. This would add to their market research efforts in gauging how their
            product/service may be received. The messages are classified by their respective status.
            """)

link = 'https://github.com/TEAM-CW3/climate-change-belief-analysis-2022#readme'
about ='For more information CW3\'s ML Models please ' + f'<a style="color:white" target="_blank" href="{link}">click here</a>'
st.write(about + ' 👈', unsafe_allow_html=True)
raw = st.checkbox('See raw data')
if raw:
    data = pd.read_csv('https://raw.githubusercontent.com/TEAM-CW3/classification-predict-streamlit-data/main/train.csv')
    st.dataframe(data.head(25))