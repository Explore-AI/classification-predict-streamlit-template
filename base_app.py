"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
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
import markdown

# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
	
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
	options = ["Information", "EDA", "Prediction"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown(open("resources/info.md").read(),unsafe_allow_html=True)
		
		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a selection box to choose different models
		models = ['Support Vector Classifier','Logistic Regression']
		classifiers = st.selectbox("Choose a classifier", models)
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			
			if classifiers == 'Support Vector Classifier':
				# Transforming user input with vectorizer
				vect_text = [tweet_text]#.toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/linear_svc.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

			elif classifiers == 'Logistic Regression':
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if prediction == -1:
				result = 'Anti'
			elif prediction == 0:
				result = 'Neutral'
			elif prediction == 1:
				result = 'Pro'
			else:
				result = 'News'
			st.success("Text Categorized as: {}".format(result))

	# Building out the EDA page
	if selection == "EDA":
		st.info("Exploratory Data Analysis")
		eda_options = ["All", "News", "Pro", "Neutral", "Anti"]
		eda_selection = st.selectbox("Choose Sentiment", eda_options)
		if eda_selection == "All":
			st.write('Introduction to EDA')
			# Plot count of sentiments
			fig, ax = plt.subplots()
			fig = sns.countplot(x='sentiment', data=raw, palette='husl')
			plt.xlabel('Sentiment')
			plt.ylabel('Number of Observations')
			plt.title('Count of Observations by Class\n')
			ax.set_xticklabels(['Anti', 'Neutral', 'Pro', 'News'])
			plt.xticks(rotation=45)
			st.pyplot()
			eda_info = md[1305:1400]
			st.markdown(eda_info,unsafe_allow_html=True)
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
