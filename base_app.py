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
news_vectorizer = open("resources/vect.pkl","rb")
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
		tab1, tab2, tab3 = st.tabs(["Home", "Visuals", "Cool Stuff"])
		
		with tab1:
			st.header("Home")
			
			model = None
			st.info("Prediction with ML Models")
			model_options = [
					"Model 1: Ridge Classifier", 
		    		"Model 2: Logostic Regression",
					"Model 3: Random Forest",
					"Model 4: Support Vector Classifier",
					"Model 5: Multinomial Naive Bayes"   ]
			
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
			with tab2:
				st.header("Visuals")

				with st.container():
					st.header('Class Distribution Diagram')
					st.image('resources/imgs/visuals/class_dist_bar.png')
				
				with st.container():

					col1, col2 = st.columns(2)
				
				with col1:
					st.header('Box Plot of Tweet Lengths Per Category')
					st.image('resources/imgs/visuals/len_box_cat.png')
					
				with col2:
						st.header('Distribution of Tweet Lengths')
						st.image('resources/imgs/visuals/len_dist.png')

				with st.container():

					st.header('Pie Charts of Tweets Starting With "RT" vs Tweets Without "RT"')
					st.image('resources/imgs/visuals/rt_pie.png')

				with st.container():
					st.header('Wordclouds')
					col1, col2 = st.columns(2)
				
				with col1:
					st.header('Terms Before Data Cleaning')
					st.image('resources/imgs/visuals/wordcloud.png')
					
				with col2:
						st.header('Terms After Data Cleaning')
						st.image('resources/imgs/visuals/wordcloud_clean.png')

				with st.container():
					st.header('Top 30 Most Prevalent Words')
					st.image('resources/imgs/visuals/top_30_words.png')


			with tab3:
				st.header("Data From Our Machine Learning Experiments")
				

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
