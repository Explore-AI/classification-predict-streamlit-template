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
import joblib,os
import base64

# Data dependencies
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import plotly.express as px

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url('https://anhdepfree.com/wp-content/uploads/2019/05/50-anh-background-dep-nhat-4.jpg');
background-size: cover;
}
[data-testid="stHeader"] {
background-color: rgba(0,0,0,0);

}

[data-testid="stToolbar"] {
right: 2rem;
}

[data-testid="stSidebar"] {
background-image: url('https://images.pexels.com/photos/2088203/pexels-photo-2088203.jpeg?auto=compress&cs=tinysrgb&w=600');
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)




# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	st.info("This application is all about tweet sentiment analysis of climate change. It is able to classify whether" 
			 "or not a person believes in climate change, based on their novel tweet data.")
	#st.()
		# You can read a markdown file from supporting resources folder
		#st.markdown("")
	if st.checkbox('Show raw data'): # data is hidden if box is unchecked
		st.write(raw[['sentiment', 'message']]) # will write the df to the page


    # Creating sidebar with selection box -
    # you can create multiple pages this way
	options = ["Home", "About us", "App tour", "Tweet classifier", "Tweet analysis"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Tweet Sentitment classification " page
	if selection == "Tweet classifier":
		st.info("Prediction with ML Models")
		st.markdown("You can enter text or upload file")
        # Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		# upload a file
		upload_file = st.file_uploader("Upload file")

		if st.button("Classify"):
            # Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)


	# Building out the "Tweet Sentitment analysis " page		
	if selection == "Tweet analysis":
		st.info("This app analyses sentiments on climate change based on tweet data")
		#top level filters
		#message_filter = st.selectbox("Select the message", pd.unique(raw['sentiment']))
		# dataframe filter
		#df = raw[raw['sentiment']== message_filter] 
		st.markdown("### Tweet distribution")
		sentiment = raw['sentiment'].value_counts()
		sentiment = pd.DataFrame({'Sentiment':sentiment.index, 'Tweets':sentiment.values})
	
		# create two columns for charts
		fig_col1, fig_col2 = st.columns(2)
		
		with fig_col1:
			fig = fig = px.bar(sentiment, x='Sentiment', y='Tweets', color = 'Tweets', height= 500)
			st.plotly_chart(fig)
	
       	#
		with fig_col2:
			fig = px.pie(sentiment, values= 'Tweets', names= 'Sentiment')
			st.plotly_chart(fig)
			
		

		

	
		






	
		

			
	
		#with fig_col1:
		#	st.markdown("### First Chart")
		#	fig = px.density_heatmap(
       	#	data_frame=df, y="message", x="sentiment"
    	#	)
		#	st.write(fig)
   
		
		#	fig2 = px.histogram(data_frame=df, x="sentiment")
		#	st.write(fig2)

		


	

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
