#streamlit dependencies
import streamlit as st
import joblib, os
import functions.classifier as classified
from nlppreprocess import NLP
nlp = NLP()

vectorizer = open('./resources/tfidfvect.pkl','rb')
tweet_cv = joblib.load(vectorizer)
data_source = ['Select option', 'Tweet', 'Dataset']

st.subheader('Tweet Classification')

source_selection = st.selectbox('Select data source', data_source)
# st.rain()
if source_selection == 'Tweet':
    # Load Our Models
    def load_prediction_models(model_file):
        loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
        return loaded_models

    # getting the predictions
    def get_keys(val,my_dict):
        for key,value in my_dict.items():
            if val == value:
                return key

    input_text = st.text_area('Enter Climate Belief Tweet (max. 280 characters)')
    all_ml_models = ['Logistic Regression', 'Random Forest','Decision Tree']
    model_choice = st.selectbox('Select Classification ML Model', all_ml_models)
    
    prediction_labels = {'This tweet supports the belief of man-made climate change 🙂':1,
                        'This tweet does not believe in man-made climate change 🙃':-1,
                        'This tweet neither supports nor refutes the belief of man-made climate change 😐':0,
                        'This tweet links to factual news about climate change 📰':2}
    
    m = st.markdown("""<style>
                            div.stButton > button:first-child {
                                background-color: #262730;
                                color:#ffffff;
                            }
                            div.stButton > button:hover {
                                background-color: #0099ff;
                                color:#ffffff;
                            }
                        </style>""", unsafe_allow_html=True)
    b = st.button('Classify')

    if b:
        result = classified.classify(model_choice, input_text, prediction_labels)
        st.info(result)