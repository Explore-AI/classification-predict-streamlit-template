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
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
import model_app
import base64


# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/Count_Vectorizer.pkl","rb")
tweet_vect = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
#upload_file 


# Load your raw data
#raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	with open("resources/imgs/testing_bck.jpg","rb") as background_img:
		encoded_string = base64.b64encode(background_img.read())

	st.markdown(
    f"""
    <style>
    .stApp {{
       background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
		background-attachment: fixed;
       background-size: cover
    }}
   	</style>
    """,
    unsafe_allow_html=True
    )
	
	options = ["Background","Know your file","Text tweet prediction","File tweet classification"]
	selection = st.sidebar.selectbox("Lets interact", options)
# file stuffies.........
	# Building out the "Information" page
	if selection == "Background":
		st.subheader("Background")
		st.info("Bio Straw has tasked Thynk Data to create a mechine learning model that is able to classify weather or not a person belives in climate change based on their novel tweet data")

		st.info ("Bio straw like many other comapnies strive to offer products and services that are enviormentally firiendly and sustainable in line with their values and ideals. With this said Bio straw would like to know how people percive cimate change and weather or not they belive it is a real threat. This information would add to their market reserch efforts in gauging how their prducts and services may be recieved.")
		image_climate = Image.open(os.path.join("resources/imgs/Clmate change.jpg"))
		image_climate = image_climate.resize((300,300))
		st.image(image_climate)
		st.info("Bio Straw has tasked Thynk Data to create a mechine learning model that is able to classify weather or not a person belives in climate change based on their novel tweet data")
		st.info("Twitter is a Social media platform where people express their opions using tweets (a message) about anything hapennig around the world. sit tight as the team take you through on how you can collect data, process it and extract meaningful information that can be used to make future predictions about curent products and services.")
		# You can read a markdown file from supporting resources folder
		
		image_twitter = Image.open(os.path.join("resources/imgs/Twitter_tweet.jpg"))
		image_twitter = image_twitter.resize((300,450))
		st.image(image_twitter)
		# pulling to main
		
		#st.image(image, caption='Sunrise by the mountains')
		#st.info("Twitter is a Social media platform where people express their opions using tweets (a message) about anything hapennig around the world. sit tight as the team take you through on how you can collect data, process it and extract meaningful information that can be used to make future predictions about curent products and services.")
		
	if selection == "Know your file":
		st.subheader("Let us explore our data")
		upload_file = model_app.upload_file()
		if upload_file is not None:
			raw = pd.read_csv(upload_file )

			if st.checkbox('Display the data in your file'): # data is hidden if box is unchecked
				st.write(raw) # will write the df to the page
			if st.checkbox('Display the wordmap of the uploaded file'):
				testing_wordMap = model_app.word_map(raw)
			
	# Building out the predication page
	if selection == "Text tweet prediction":
		st.info("Classifying tweets using models")
		model = st.radio(
    	"Select a model to classifiy your tweet",
    		('Logistic_regression','Naive_Bayes','Linear_Support_Vector','Random Forest','K Neighbors' ))
		# Creating a text box for user input
		tweet_text = st.text_area("Type a tweet")
		tweet_text = model_app.cleaning_text(tweet_text)

		if model == 'Random Forest' :
			if st.button("Classify"):
				# Transforming user input with vectorizer	
				st.info(tweet_text)
				rfc_text = tweet_vect.transform([tweet_text]).toarray()
				# Load your randomfc_model.pkl file 
				predictor = joblib.load(open(os.path.join("resources/Random_Forest.pkl"),"rb"))
				prediction = predictor.predict(rfc_text)
				results = model_app.classify_desc(format(prediction))
				# When model has successfully run, will print prediction
				st.success("Your tweet is classified as: {} ".format(results) )
	
		if model == 'Logistic_regression':
			if st.button("Classify"):
				
				# Transforming user input with vectorizer
				vect_text = tweet_vect.transform([tweet_text]).toarray()
				# Load your Logistic_regression.pkl file 
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				results = model_app.classify_desc(format(prediction))
				# When model has successfully run, will print prediction
				st.success("Your tweet is classified as: {} ".format(results))
				
		if model == 'K Neighbors' :
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_vect.transform([tweet_text]).toarray()
				# Load your Logistic_regression.pkl file 
				predictor = joblib.load(open(os.path.join("resources/K_Neighbors.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				results = model_app.classify_desc(format(prediction))
				# When model has successfully run, will print prediction
				st.success("Your tweet is classified as: {} ".format(results))

		if model == 'Naive_Bayes' :
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_vect.transform([tweet_text]).toarray()
				# Load your Logistic_regression.pkl file 
				predictor = joblib.load(open(os.path.join("resources/Naive_Bayes.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				results = model_app.classify_desc(format(prediction))
				# When model has successfully run, will print prediction
				st.success("Your tweet is classified as: {} ".format(results))

		if model == 'Linear_Support_Vector' :
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_vect.transform([tweet_text]).toarray()
				# Load your Logistic_regression.pkl file 
				predictor = joblib.load(open(os.path.join("resources/Linear_Support_Vector.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				results = model_app.classify_desc(format(prediction))
				# When model has successfully run, will print prediction
				st.success("Your tweet is classified as: {} ".format(results))

				

	if selection == "File tweet classification":
		#code to call the file upload function
		st.info("Classifying tweets using files")
		upload_file = model_app.upload_file()

		if upload_file is not None:
			raw = pd.read_csv(upload_file)
			model = st.radio("Select a model to classifiy your tweet",
    			('Random Forest', 'Logistic_regression','K Neighbors', 'Naive_Bayes', 'Linear_Support_Vector'))
			upload_file['message'] = upload_file['message'].apply(lambda text:model_app.cleaning_text(text))
			f1_score(y_val,rfc_pred, average ='macro')
			from sklearn.metrics import f1_score

			
			#if model == 'Random Forest' :
			#	if st.button("Classify"):
					# process of cleaning the data, then 


		
		
       
		#st.info("Classifying tweets using models")
		#if upload_file is None:
		#	upload_file = st.file_uploader("Upload a .csv file that contains tweets",'csv')
		#	if upload_file is not None:
		#		raw = pd.read_csv(upload_file )
		#elif upload_file is not None:
		#	model = st.radio("Select a model to classifiy your tweet",
    	#	('Random Forest', 'Logistic regression','K Neighbors', 'Naive_Bayes', 'Linear_Support_Vector'))
		#	if model == "Random Forest":
		#		data = st.radio(
    	#		"How do you want to load data",
    	#		('Upload tweets samples', 'Type your tweet'))
#
#			if data == 'Upload tweets samples' :
#				upload_file = st.file_uploader("Upload file")



# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
