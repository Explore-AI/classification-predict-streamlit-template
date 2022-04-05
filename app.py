#core packages
import streamlit as st

#EDA PKGs
import pandas as pd
import numpy as np

#utils
import joblib

model = joblib.load(open('sentiment analysis pipe_lr.pkl', 'rb'))

#fnx
def predict_sentiment(docx):
    result = model.predict([docx])
    return result[0]

def get_predict_proba(docx):
    results = model.predict_proba([docx])
    return results

def main():
    st.title('Sentiment classifier app')
    menu =['Home','Monitor','About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Home-Sentiment in text')

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
                st.write(prediction)

            with col2:
                st.success("Prediction Probability")
                st.write(probability)

                


    elif choice=='Monitor':
        st.subheader('Monitor App')
    
    else:
        st.subheader('About')

if __name__ == '__main__':
    main()
