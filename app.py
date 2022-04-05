#core packages
import streamlit as st
import altair as alt

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

sentiment_name_dict = {-1 : 'Anti', 0 : 'Neutral', 1 : 'Pro', 2 : 'News'}


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
                sentiment_name = sentiment_name_dict[prediction]

                st.write('{}:{}'.format(prediction,sentiment_name))

                #get the confidence of the prediction
                st.write('Confidence: {}'.format(np.max(probability)))



            with col2:
                st.success("Prediction Probability")
                #st.write(probability)
                #convert the entire probability into a adataframe
                proba_df = pd.DataFrame(probability, columns=model.classes_)
                #st.write(proba_df.T)

                #modify to plot it right
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ['sentiments', 'probability']

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='sentiments', y ='probability', color = 'sentiments')
                st.altair_chart(fig, use_container_width=True)

                


    elif choice=='Monitor':
        st.subheader('Monitor App')
    
    else:
        st.subheader('About')

if __name__ == '__main__':
    main()
