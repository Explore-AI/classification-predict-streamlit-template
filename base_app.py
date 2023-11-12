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
from PIL import Image

#st.beta_set_page_config(page_title='AI Origins')

# Vectorizer
#news_vectorizer = open("resources/tfidfvect.pkl","rb")
news_vectorizer = open("resources/vectoriser-ngram-(1,2).pickle", "rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("AI Origins")
	
	#st.write("AI Origins")
	col_1, mid, col_2 = st.columns([1, 1, 20])
	with col_1:
		st.image('Logo_b.jpeg', width=60)
	#with col_2:
		#st.write('AI Origins')


	#st.subheader("Climate change tweet classification")

	# # Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["About Us", "Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)


	#Building out the "Introduction Page"
	if selection == "About Us":
		st.header("Meet the Team")

		
		col1, col2, col3 = st.columns(3) #create images side by side

		ife = Image.open("Ife.jpeg")
		col1.subheader("Ifeolu Adeoti")
		col1.image(ife, width=200)
		col1.caption("Ife is the founder and CEO of AI Origins. She is highly driven and communicates her vision and drive to the team and customers")

		yemi = Image.open("yemi.jpg")
		col2.subheader("Oluyemi Alabi")
		col2.image(yemi, width=200)
		col2.caption("Yemi is the Technical lead. He supervises the coding and deployment of applications")

		abiola = Image.open("Abiola.jpeg")
		col3.subheader("Abiola Akinwale")
		col3.image(abiola, width=200)
		col3.caption("Abiola is the Creative Director. He is responsible for visuals")

		col4, col5, col6 = st.columns(3) #create images side by side

		joshua = Image.open("Joshua.jpeg")
		col4.subheader("Joshua Umukoro")
		col4.image(joshua, width=200)
		col4.caption("Joshua is the Financial Head")

		lawson = Image.open("Lawson.jpeg")
		col5.subheader("Lawson Iduku")
		col5.image(lawson, width=200)
		col5.caption("Lawson is the Head of Human Resources")

		stephen = Image.open("Stephen.jpeg")
		col6.subheader("Stephen Tshiani ")
		col6.image(stephen, width=200)
		col6.caption("Stephen is the Sales Manager")



	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("### Prediction with ML Models")
		

		model_selection = st.radio("Please choose a model", ["Logistic Regression" ,"Multinomial Naive Bayes (Recommended)" ,"Linear Support Vector Classifier"], help= 'Select a model that will be used for prediction')

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()

			if model_selection == "Logistic Regression":				
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/lr.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

			elif model_selection == "Multinomial Naive Bayes (Recommended)":
				predictor = joblib.load(open(os.path.join("resources/mnb.pickle"),"rb"))
				prediction = predictor.predict(vect_text)

			elif model_selection == "Linear Support Vector Classifier":
				predictor = joblib.load(open(os.path.join("resources/svc_model.pickle"),"rb"))
				prediction = predictor.predict(vect_text)


			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			output = {-1: "an Anti-climate tweet", 0: "a Neutral tweet", 1: "a Pro-climate tweet", 2: "News"}

			f = output.get(prediction[0])	
			

			st.success("Text Categorized as {}".format(f))
			

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
