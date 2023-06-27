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
news_vectorizer = open("resources/scale_vect.pkl","rb")
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
		st.markdown("The dataset at hand comprises novel tweet data. It serves as a valuable resource in our machine learning experiments. The data set is specifically designed to categorize tweets into distinct labels.")
		st.markdown("The catgories are:")
		st.markdown("-1: Anti")
		st.markdown(" 0: Neutral")
		st.markdown(" 1: Pro")
		st.markdown(" 2: News")
		st.markdown("These labels enable the classification and analysis of the tweets based on their sentiment. These labels enable the classification and analysis of the tweets based on their sentiment. Such an extensive and well-labeled dataset significantly contributes to the advancement of sentiment analysis, opinion mining, and other natural language processing tasks.")

		st.subheader("Raw Twitter Data and Label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		tab1, tab2, tab3 = st.tabs(["Home", "Visuals", "Cool Stuff"])
		
		with tab1:

			st.header("Home")

			st.info("Prediction with ML Models")

			model = None

			model_options = [
					"Model 1: Ridge Classifier", 
		    		"Model 2: Logistic Regression",
					"Model 3: Random Forest",
					"Model 4: Linear Support Vector Classifier",
					"Model 5: Bernoulli Naive Bayes"   ]
			
			model_selector = st.selectbox("Choose Classification Model", model_options)

			if model_selector == "Model 1: Ridge Classifier":
				model = None
				model = "resources/model1.pkl"

			if model_selector == "Model 2: Logistic Regression":
				model = None
				model = "resources/model2.pkl"

			if model_selector == "Model 3: Random Forest":
				model = None
				model = "resources/model3.pkl"

			if model_selector == "Model 4: Linear Support Vector Classifier":
				model = None
				model = "resources/model4.pkl"

			if model_selector == "Model 5: Bernoulli Naive Bayes":
				model = None
				model = "resources/model5.pkl"

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

				st.info("Graphs generated from our data analysis.")

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
					st.header('Before Data Cleaning')
					st.image('resources/imgs/visuals/wordcloud.png')
					
				with col2:
						st.header('After Data Cleaning')
						st.image('resources/imgs/visuals/wordcloud_clean.png')

				with st.container():
					st.header('Top 30 Most Prevalent Words')
					st.image('resources/imgs/visuals/top_30_words.png')


			with tab3:
				st.header("Data From Our Machine Learning Experiments")

				st.info("Some metadata from our Comet experiments")
				

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
