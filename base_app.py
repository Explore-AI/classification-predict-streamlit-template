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
# pip install spacy -q  3
# python -m spacy download en_core_web_sm -q
import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import os
from PIL import Image

# Data dependencies
import pandas as pd
import time
import unicodedata
from num2words import num2words
import spacy
import re

# Change Streamlit interface colors
primary_color = "#A4A69C"
text_color = "#0C0C0C"
background_color = "#FFFFFF"
secondary_background_color = "#3A9664"
# Custom CSS style
custom_css = f"""
<style>
body {{
    color: {text_color};
    background-color: {background_color};
}}
.sidebar .sidebar-content {{
    background-color: {secondary_background_color};
}}
.primary-button {{
    background-color: {primary_color} !important;
}}
</style>
"""

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

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
    

    # Creating sidebar using streamlit-option-menu
    with st.sidebar:
        selected = option_menu(
            "Menu",
            ["Home", "Predictions", "Raw Data", "How it Works", "About Us", "Contact Us"],
            icons=["house", "table", "graph-up-arrow", "info-circle", "telephone"],
            menu_icon="menu-button",
            default_index=0,
        )

    # Building out the "Home" page
    if selected == "Home":
        image = Image.open("resources/Elites.pptx (3).png")

        st.title("Elites Data Science")

        st.subheader("Climate change tweet classification")

        st.image(image)

        st.subheader("Tweet Classifier")
        st.markdown("Consumers gravitate toward companies that are built around lessening one’s environmental impact. Elites provides an accurate and robust solution that gives companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories, thus increasing their insights and informing future marketing strategies.")
        st.markdown("Choose Elites and walk a greener path.")

     # Building out the "About Us" page
    if selected == "About Us":
        # Using Tabs   

        st.title("About Us")   

        tab1, tab2, tab3, tab4 = st.tabs(["About Elites", "Our Classifier", "Meet the Team", "Contact Details"]) 
        with tab1:
            
            st.markdown("We’re proud to be an industry leader in promoting eco-friendly business practices. Striving to protect and sustain our environment is a given at every stage of our services.\n  Our green vision goes beyond helping businesses sustain their core mission as a green company. We design innovative technology to help businesses save time, reduce costs, and make better business decisions to ensure their footprint is greener.")
            st.markdown("Elites mission is to accelerate the world’s transition to sustainable energy, Earth is still preserved for generations to come. Our data science consultants deliver incredible value by evaluating the model and proving insights.")
            image4 = Image.open("resources/Elites.pptx-removebg-preview.png")
            st.image(image4)
        with tab2:
            st.write(
            """
            Our Tweet Classifier app gives you a variety of Machine Learning Models to choose from. The models selected showed high performance over the others with a sentiment claassification accuracy. \n 
            Our leading model is the Linear SVC Classifier with an impressive accuracy, ensuring our users accuracy that will inform great business decisions.
    
            """
            )
            image3 = Image.open("resources/50commonWords.png")
            st.image(image3)
        with tab3:
            image5 = Image.open("resources/MeetTheTeam.png")
            st.image(image5)
     
        with tab4:
            st.markdown("Email Address: admin@elites.com")
            st.markdown("Website: www.elites.co.za")
            st.markdown("Telephone: 012 455 7762")

    
    if selected == "Predictions":
        
        st.title("Classify tweets")   

        #Adding sidebar image
        image = Image.open("resources/Elites.pptx (3).png")
        st.sidebar.image(image, use_column_width=True)  


        st.markdown("Enter your Tweet relating to climate change below and our model will classify it based on its percieved sentiment. \n See below on how to interpret results.")
        #using tabs for different predictors
        tab1, tab2 = st.tabs(["Predict Tweet Sentiment", "Test the model"])


        with tab1:          
                                   
            # Creating a text box for user input
            tweet_text = st.text_area("Enter Text Below")            
                        
            if st.button("Predict Tweet Sentiment"):
            
            # Define function to remove symbols, punctuation, and emojis
                def remove_symbols(text):
                    text = re.sub(r'[^\w\s.,!?]', '', text)
                    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')
                    text = re.sub(r'https\S+|www\S+', '', text)
                    text = re.sub(r'\s+', ' ', text)
                    text = re.sub(r'\b(\d+)\b', lambda match: num2words(int(match.group(0))), text)
                    return text    

                # Define nlp for use in Spacy
                nlp = spacy.load('en_core_web_sm')

                # Define spacy tokenizer
                def tokenize_text_spacy(text):
                    doc = nlp(text)
                    tokens = [token.text for token in doc]
                    return ' '.join(tokens)
                
                #Apply data cleaning and feature engineering and also transform to lowercase
                processed_text = remove_symbols(tweet_text.lower())
                processed_text = tokenize_text_spacy(processed_text)
           

            # Transforming user input with vectorizer     
                vect_text = tweet_cv.transform([processed_text]).toarray()

            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
                predictor = joblib.load(open(os.path.join("resources/LinearSVC.pkl"),"rb"))
                prediction = predictor.predict(vect_text)


                #Classification dictionary:
                label_mapping = {2: "News - the tweet links to factual news about climate change",
                                 1: "Pro - the tweet supports the belief of man-made climate change",
                                 0: "Neutral - the tweet neither supports nor refutes the belief of man-made climate change",
                			     -1: "Anti - the tweet does not believe in man-made climate change"}
                
                # When model has successfully run, will print prediction
                # You can use a dictionary or similar structure to make this output
                # more human interpretable.


                #Map categories to labels
                category_label = label_mapping[prediction[0]]

                st.success("Text Categorized as: {}".format(category_label))
                
                with st.expander("ℹ️ How to interpret the results", expanded=False):
                    st.write(
                 """
                 Sentiment is categorized into 4 classes:\n
                 **Anti**: the tweet does not believe in man-made climate change \n
                 **Neutral**: the tweet neither supports nor refutes the belief of man-made climate change \n
                 **Pro**: the tweet supports the belief of man-made climate change \n
                 **News**: the tweet links to factual news about climate change \n
         
                 """
             )
                st.write("")
            
        with tab2:
            
            st.markdown("To test classifier accuracy, copy and past one of the tweets in the list below into the classifier and check the corresponding sentiment that the model outputs.")
            
            with st.expander("🐤 Tweets", expanded=False):
                st.write(
                """
                * The biggest threat to mankind is NOT global warming but liberal idiocy👊🏻🖕🏻\n
                Expected output = Anti \n
                * Polar bears for global warming. Fish for water pollution.\n
                Expected output = Neutral \n
                * RT Leading the charge in the climate change fight - Portland Tribune  https://t.co/DZPzRkcVi2 \n
                Expected output = Pro \n
                * G20 to focus on climate change despite Trump’s resistance \n
                Expected output = News
        
                """
            )
            st.write("")

    # Building out the raw data page
    if selected == "Raw Data":

        st.title("Raw Data")  

        #Adding sidebar image
        image = Image.open("resources/Elites.pptx (3).png")
        st.sidebar.image(image, use_column_width=True)     
    
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

            if st.checkbox("Show raw data sentiments"):
                image6 = Image.open("resources/sentiments.png")
                st.image(image6)
    
    #Inserting technical page
    if selected == "How it Works":

        st.title("How it Works")  

        image = Image.open("resources/Elites.pptx (3).png")
        st.sidebar.image(image, use_column_width=True)  

        #Describe intro
        st.subheader("Overview")
        st.markdown("This app aims to classify Tweets according to their sentiment towards Climate change, furthermore whether the Tweet indicates the Tweeter's stance towards man made Climate change. The below attempts to provide a simplified description of how our model works. Other inputs can also be used in the form of sentences or paragraphs.")

        #Describe data processing
        st.subheader("1) Data cleaning and feature engineering")
        st.markdown("Firstly we have to 'clean' our data, this is where we remove non-English characters, emoji's and other uninteligible words which would not be usefull. Essentially we are attempting to classify the tweets based on the occurence of certain words or groups of words and we only want to include words which convey meaning.")
        
        #Describe Model
        st.subheader("2) Modelling")
        st.markdown("We then tested various Machine Learning models and optimised the best performing model, the model was tested based on its ability to accurately classify tweets based on a existing dataset.")

        #Describe prediction
        st.subheader("3) Tweet sentiment prediction")
        st.markdown("Finally we use our model to predict the sentiment of the given tweet based on its contents.")

            
    # Building out the contact page
    if selected == "Contact Us":

        #Adding sidebar image
        image = Image.open("resources/Elites.pptx (3).png")
        st.sidebar.image(image, use_column_width=True)  

        with st.form("form1", clear_on_submit=True):
            st.subheader("Get in touch with us")
            name = st.text_input("Enter full name")
            email = st.text_input("Enter email")
            message = st.text_area("Message")
            
            submit = st.form_submit_button("Submit Form")
            if submit:
                st.write("Your form has been submitted and we will be in touch 🙂")
            
    

# Run the application
if __name__ == "__main__":
    main()
