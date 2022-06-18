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
from turtle import color
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

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

	#New - added "model" option
	
	options = ["Prediction","Information"]
	m_options = ["Decision_Tree ml","KNN","Random_Forrest"]

	selection = st.sidebar.selectbox("Choose Option",options)
	m_selection = st.sidebar.selectbox("Choose Model",m_options)
	#st.sidebar.info("Model selection")
	#model1 = st.sidebar.checkbox("Decision Tree",key = 1)
	#model2 = st.sidebar.checkbox("Kth Nearest Neighbour",key = 2)
	#model3 = st.sidebar.checkbox("Random Forrest",key = 3)
	
  

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		
	# Building out the predication page 
	# Using the "Decision tree" ML Model

	if selection == "Prediction" and m_selection == "Decision_Tree ml":
	
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text Below","Type Here")
		st.write("(Prediction with ML Models)")
	
        
		if st.button("Classify"):
			
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
		 
			predictor = joblib.load(open(os.path.join("resources/CW3.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			#status = st.success("{}".format(prediction))
			status = '{}'.format(prediction)
			
			if status == '[1]':
				st.write("You believe in climate change!")
			elif status == '[-1]':
				st.write("you don't believe in climate change")
			elif status == '[0]':
				st.write("You have a neutral opinion on climate change")

	##_##_##_##_


	# Making the same predictions
	# Using the "KNN" Model

	elif selection == "Prediction" and m_selection == "KNN":
	
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text Below","Type Here")
		st.write("(Prediction with ML Models)")
	
        
		if st.button("Classify"):
			
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
		 
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			#status = st.success("{}".format(prediction))
			status = '{}'.format(prediction)
			
			if status == '[1]':
				st.write("You in climate change!")
			elif status == '[-1]':
				st.write("you don't believe in climate change")
			elif status == '[0]':
				st.write("You have a neutral opinion on climate change")
	
	##_##_##_##_
				



	# Making the same prediction
	# Using the "Random_Forrest" ML Model

	elif selection == "Prediction" and m_selection == "Random_Forrest":
	
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text Below","Type Here")
		st.write("(Prediction with ML Models)")
	
        
		if st.button("Classify"):
			
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
		 
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			#status = st.success("{}".format(prediction))
			status = '{}'.format(prediction)
			
			if status == '[1]':
				st.subheader("You believe climate change!")
			elif status == '[-1]':
				st.write("you don't believe in climate change")
			elif status == '[0]':
				st.write("You have a neutral opinion on climate change")



	

	##_##_##_##_
				
				
				
			



		
		
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
