"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
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
# Streamlit dependencies
import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import os
from PIL import Image

# Data dependencies
import pandas as pd
import time

# Vectorizer
news_vectorizer = open("resources/Linear_SVC_vect.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file
news_vectorizer_1 = open("resources/Linear_SVC_vect.pkl", "rb")
tweet_cv_1 = joblib.load(news_vectorizer_1)

# Load your raw data
raw = pd.read_csv("resources/train.csv")
raw2 = pd.read_csv("resources/training_data.csv")

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit"""

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Elites  \n _____________________________________________________")
    st.subheader("Climate change tweet classification")

    # Creating sidebar using streamlit-option-menu
    with st.sidebar:
        selected = option_menu(
            "Menu",
            ["Home", "Raw Data", "Predictions", "About Us", "Contact Us"],
            icons=["house", "table", "graph-up-arrow", "info-circle", "telephone"],
            menu_icon="menu-button",
            default_index=0,
        )

    # Building out the "Home" page
    if selected == "Home":
        image = Image.open("resources/imgs/KB.png")
        st.image(image)

        st.subheader("Tweet Classifier")
        st.markdown("Consumers gravitate toward companies that are built around lessening one’s environmental impact. Elites provides an accurate and robust solution that gives companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories, thus increasing their insights and informing future marketing strategies.")
        st.markdown("Choose Elites and walk a greener path.")

    # Building out the raw data page
    if selected == "Raw Data":
        tab1, tab2 = st.tabs(["Data description", "Data Visualizations"])
        with tab1:
            st.markdown(
                "The collection of the raw data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo."
            )
            st.write(
                """
                This dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were annotated. Each tweet is labelled independently by 3 reviewers. This dataset only contains tweets that all 3 reviewers agreed on (the rest were discarded). \n
                Each tweet is labelled as one of the following classes: \n
                * 2(News): the tweet links to factual news about climate change \n
                * 1(Pro): the tweet supports the belief of man-made climate change \n
                * 0(Neutral): the tweet neither supports nor refutes the belief of man-made climate change \n
                * -1(Anti): the tweet does not believe in man-made climate change
                """
            )
            st.write("")

        with tab2:
            if st.checkbox("Show sentiment value count"):
                st.bar_chart(data=raw2["sentiment"].value_counts(), x=None, y=None, width=220, height=320, use_container_width=True)
            
            if st.checkbox("Show raw data"):
                job_filter = st.selectbox("Select sentiment", pd.unique(raw['sentiment']))
           
           
                # creating a single-element container.
                placeholder = st.empty()
           
                # dataframe filter 
           
                df = raw[raw['sentiment']==job_filter]
            
                for seconds in range(100):
                #while True: 
                                   
                   with placeholder.container():       
            
                  
                       st.markdown("### Raw data")
                       st.dataframe(df)
                       time.sleep(1)

            if st.checkbox("Show raw data word cloud"):
                image5 = Image.open("resources/50commonWords.png")
                st.image(image5)
                
    # Building out the predications page
    if selected == "Predictions":
        st.subheader("How It Works")
        st.markdown("Click on a tab to choose your desired classifier then enter a tweet relating to climate change and it will be classified according to its sentiment. \n See below on how to interpret results.")
        #using tabs for different predictors
        tab1 = st.tabs(["Linear SVC"])
        with tab1:
                        
            st.markdown("To test classifier accuracy, copy and past one of the tweets in the list below into the classifier and check the corresponding sentiment that the model outputs.")
                    
            with st.expander("🐤 Tweets", expanded=False):
                            st.write(
                            """
                            * The biggest threat to mankind is NOT global warming but liberal idiocy👊🏻🖕🏻\n
                            Expected output = -1 \n
                            * Polar bears for global warming. Fish for water pollution.\n
                            Expected output = 0 \n
                            * RT Leading the charge in the climate change fight - Portland Tribune  https://t.co/DZPzRkcVi2 \n
                            Expected output = 1 \n
                            * G20 to focus on climate change despite Trump’s resistance \n
                            Expected output = 2
                    
                            """
                        )
            st.write("")
                        
            # Creating a text box for user input
            tweet_text = st.text_area("Enter Text Below")            
                        
            if st.button("Predict Tweet Sentiment"):
            # Transforming user input with vectorizer     
                vect_text = tweet_cv.transform([tweet_text]).toarray()

            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
                predictor = joblib.load(open(os.path.join("resources/LinearSVC.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                
                # When model has successfully run, will print prediction
                # You can use a dictionary or similar structure to make this output
                # more human interpretable.
                st.success("Sentiment: {}".format(prediction))
                
                with st.expander("ℹ️ How to interpret the results", expanded=False):
                    st.write(
                 """
                 Sentiment is categorized into 4 classes:\n
                 [-1] = **Anti**: the tweet does not believe in man-made climate change \n
                 [0] = **Neutral**: the tweet neither supports nor refutes the belief of man-made climate change \n
                 [1] = **Pro**: the tweet supports the belief of man-made climate change \n
                 [2] = **News**: the tweet links to factual news about climate change \n
         
                 """
             )
                st.write("")
            
            with tab2:
                
                st.markdown("To test classifier accuracy, copy and past one of the tweets in the list below into the classifier and check the corresponding sentiment that the model outputs.")
                
                with st.expander("🐤 Tweets", expanded=False):
                    st.write(
                    """
                    * The biggest threat to mankind is NOT global warming but liberal idiocy👊🏻🖕🏻\n
                    Expected output = -1 \n
                    * Polar bears for global warming. Fish for water pollution.\n
                    Expected output = 0 \n
                    * RT Leading the charge in the climate change fight - Portland Tribune  https://t.co/DZPzRkcVi2 \n
                    Expected output = 1 \n
                    * G20 to focus on climate change despite Trump’s resistance \n
                    Expected output = 2
            
                    """
                )
                st.write("")
                
            # Building out the contact page
            if selected == "Contact Us":
                with st.form("form1", clear_on_submit=True):
                    st.subheader("Get in touch with us")
                    name = st.text_input("Enter full name")
                    email = st.text_input("Enter email")
                    message = st.text_area("Message")
                    
                    submit = st.form_submit_button("Submit Form")
                    if submit:
                        st.write("Your form has been successfully submitted and we will be in touch")
                
                job_filter = st.selectbox(
                    "Select sentiment", pd.unique(raw['sentiment'])
                )

                # creating a single-element data frame with the selected sentiment
                filtered_data = raw[raw["sentiment"] == job_filter]

                # displaying the filtered data
                st.write(filtered_data)

# Run the application
if __name__ == "__main__":
    main()
