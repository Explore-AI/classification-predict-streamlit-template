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
import streamlit as st
import joblib
import os
import altair as alt
import matplotlib.pyplot as plt


# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")
sentiment = ["1", "2", "0", "-1"]

# The main function where we will build the actual app


def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Welcome to Team Gm3")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Introduction", "Information", "EDA",
               "Model prediction", "Model evaluation", ]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("""Many companies are built around lessening one’s environmental impact or carbon footprint. They offer products and services
         that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive
         climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their
         product/service may be received.With this context, EDSA is challenging you during the Classification Sprint with the task of creating a
         Machine Learning model that is able to classify whether or not a person believes in climate change, based on their novel tweet data.Providing
         an accurate and robust solution to this task gives companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic
          categories - thus increasing their insights and informing future marketing strategies. """)

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            # will write the df to the page
            st.write(raw[['sentiment', 'message']])

    if selection == "EDA":
        st.subheader("Sentiment type vs Count bar chart")
        sentiment_type = st.multiselect(
            "Which sentiment would you like to see?", sentiment)
        st.write("  Sentiment type vs Count bar chart")
        source = pd.DataFrame({'Count': [8530, 3640, 2353, 1296],
                               'Sentiment': ["1", "2", "0", "-1"]})

        bar_chart = alt.Chart(source).mark_bar().encode(
            y='Count:Q',
            x='Sentiment:O',
        )
        st.altair_chart(bar_chart, use_container_width=True)

        st.subheader("Pie chart distribution of sentiments in percentage")
        labels = "1", "2", "0", "-1"
        sizes = [8530, 3640, 2353, 1296]
        #explode = (0.1, 0.1, 0.1, 0.1)

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False,
                startangle=90)
        ax1.axis('equal')

        st.pyplot(fig1)

        # Building out the predication page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(
                open(os.path.join("resources/Logistic_regression.pkl"), "rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
