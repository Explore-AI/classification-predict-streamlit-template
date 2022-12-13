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
	

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction","Data Insights",'About the project',"Information", 'The team']
	selection = st.sidebar.selectbox("Choose Option", options)

	#Information on the project
	if selection == "About the project":
		st.title("Climate change tweet classification")
		st.write("Climate action, is the goal 13 of the sustainable development goals. It calls for urgent actions to combact climate change and its impact. To address climate change, countries adopt the paris agreement to limit global temperature rise to well below 2 degree celsius. People are experiencing the significant impacts of climate change, which include changing weather patterns, rising sea level, and more extreme weather events. The greenhouse gas emissions from human activities are driving climate change and continue to rise. ")
		image = Image.open('climate action.jpeg')
		st.image(image, caption='SDG Goal 13: Climate action')
		st.write("South Africa’s National Climate Change Response Policy (NCCRP) (DEA 2011) commits the Department of Environmental Affairs (DEA) in Section 12 to publish annual progress reports on monitoring climate change responses.The South African government has also pledged to continue contributing positively to adresssing the climate emergency and is planning on long term efforts to  change the attitude of people towards climate change. But in order to do that, there is a need to know  what people's opinion are regarding climate change")
		st.write("In this project, various machine learning models were utilised to predict people's sentiment regarding climate change. The machines were trained using messages and known sentiments from twitter. Through that, we can predict,  what sentiment an individaul has, based on their tweet. This would enable  the govermnent ascertain people's current opionion regarding climate change and how much effort is required to positively influence that" )
	#Exploratory data analysis
	if selection == "Data Insights":
		st.title("Exploratory Data Analysis")
		st.write('In this section, we will be discussing insights from the Tweet classification dataset')
		st.subheader('Tweet distribution')
		st.write('From the chart below, It is obvious that pro-climate change tweets, account for more than half of the overall number of  tweets. This is more than the combine  tweets from news,neutral and anti- climate change.') 
		image = Image.open('tweet.png')
		st.image(image, caption='Tweet distribution')
		st.subheader('Climate change buzzwords')
		st.write("The word cloud below  displays the 25 most popular words found in the tweets for each classes.The top buzzwords accross all classes are climate, change, global and retweet. The frequency of retweet means that many opinions are being shared and viewed by large audiences. This is true for all four classes.'Trump' is a also a frequently occuring word in all four classes. This is unsurprising given his controversial view on the topic. Words like denier, believe, think, fight, etc. occur frequently in pro climate change tweets. In contrast, anti climate change tweets contain words such as 'hoax', 'scam', 'tax' and 'liberal'. There is a stark difference in tone and use of emotive language in these two classes of tweets. From this data we could deduce that people who are anti climate change believe that global warming is a 'hoax' and feel negatively towards a tax–based approach to slowing global climate change. Words like 'science' and 'scientist' occur frequently as well which could imply that people are tweeting about scientific studies that support their views on climate change.")
		image = Image.open('word_cloud.png')
		st.image(image, caption='Word cloud showing most popular buzz words')

		st.subheader('Hashtag analysis')
		st.write('People use the hashtag symbol (#) before a relevant keyword or phrase in their Tweet to categorize those Tweets and help them show more easily in Twitter search. Clicking or tapping on a hashtagged word in any message shows you other Tweets that include that hashtag. Hashtags can be included anywhere in a Tweet. Hashtagged words that become very popular are often trending topics.')
		image = Image.open('hashtag analysis.png')
		st.image(image, caption='Hashtag analysis of Tweets')
	if selection == "The team":
		st.title("About the team")
		st.markdown("### Our team is made up of highly qualified and reputable inviduals in the field of data science ")

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
		st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification")
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		option = st.selectbox(
		'Select a model to use',
		['Linear SVC','Naive Bayes', 'K Nearest Neighbors', 'Logistic Regression', 'Random Forest'])

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# select option to choose 
			output_text = {
			'0': 'Neutral', 
			'-1': 'Anti climate change', 
			'1': 'Pro Climate change', 
			'2': 'News'
			}
			# more human interpretable.
			st.success("{} Model categorised text as : {}".format(option, output_text[str(prediction[0])]))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
