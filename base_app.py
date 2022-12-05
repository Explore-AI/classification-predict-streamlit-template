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

# Data dependencies
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt




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

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "Visualization", "Dashboard"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

	

		st.subheader("Raw Twitter data and label")
	if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Visualisation
	if selection == "Visualization":
		st.subheader("Sentiment distribution")
		sentiment_dist = pd.DataFrame(raw['sentiment'].value_counts()).head()
		st.bar_chart(sentiment_dist)

		st.subheader("A wide column with a chart")
		st.line_chart(raw['sentiment'])

		st.subheader("A narrow column with the data")
		st.write(raw['sentiment'])


	# Dashboard experiment
	if selection == "Dashboard":
		st.subheader("Sentiment distribution")
		sentiment_dist = pd.DataFrame(raw['sentiment'].value_counts()).head()
		fig, ax = plt.subplots()
		ax.hist(sentiment_dist, bins=20)
		st.pyplot(fig)

		values = st.slider(
		'Select sentiment',
    	-1, 2, (0, 1))
		st.write('Values:', values)


		chart_data = pd.DataFrame(
   		np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
   		columns=['sentiment', 'message'])

		st.pydeck_chart(pdk.Deck(
   	 	map_style=None,
    	initial_view_state=pdk.ViewState(
        latitude=37.76,
        longitude=-122.4,
        zoom=11,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
           'HexagonLayer',
           data=chart_data,
           get_position='[message, sentiment]',
           radius=200,
           elevation_scale=4,
           elevation_range=[0, 1000],
           pickable=True,
           extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=chart_data,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=200,
        ),
    ],
))



	
	

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		st.markdown("You can enter text or upload file ")
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

			# rfc model
			#predictor = joblib.load(open(os.path.join("resources/rfc_model.pkl"),"rb"))
			#prediction = predictor.predict(vect_text)


			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
