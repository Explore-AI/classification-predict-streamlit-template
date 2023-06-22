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

# Vectorizer
news_vectorizer = open("resources/tfidv.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	from PIL import Image
	logo = Image.open("resources/imgs/SC.png")
	
	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	st.sidebar.image(logo, caption = None, use_column_width = True)
	selection = st.sidebar.selectbox("Choose Option", options)

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
		model = None
		st.info("Prediction with ML Models")
		model_options = [
				"Model 1:  	Logistic Regression", 
		    	"Model 2:  	Random Forest",
				"Model 3:  	Support Vectors",
				"Model 4: 	Naive Bayes"  
				]
		model_selector = st.selectbox("Choose Classification Model", model_options)
		if model_selector == "Model 1: Logistic Regression":
			model = "resources/lr_model.pkl"
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			
			predictor = joblib.load(open(os.path.join(model),"rb"))
			prediction = predictor.predict(vect_text)

			label_dict = {2: "News", 1: "Pro", 0: "Neutral", -1:"Anti" }
			output = label_dict[prediction[0]]

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(output))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
