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
import pandas as pd

# import nltk 
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from streamlit_option_menu import option_menu   # To install: pip install streamlit-option-menu

#from nltk.stem import WordNetLemmatizer
#from nltk.corpus import stopwords

import pickle

# Vectorizer
news_vectorizer = open("resources/count_vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
df_train = pd.read_csv("resources/train_2.csv")
df_test=pd.read_csv("resources/test_with_no_labels.csv")

pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
subs_url = r'url-web'
df_train['message'] = df_train['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)
df_test['message'] = df_test['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)

df_train['message'] = df_train['message'].str.lower()
df_test['message'] = df_test['message'].str.lower()

import string
def remove_punctuation(post):
    return ''.join([l for l in post if l not in string.punctuation])

df_train['message'] = df_train['message'].apply(remove_punctuation)
df_test['message'] = df_test['message'].apply(remove_punctuation)

df_train['message'].str.replace("rt","")
df_train['message'].str.replace("@","")

df_test['message'].str.replace("rt","")
df_test['message'].str.replace("@","")

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(df_train['message'])
vect = CountVectorizer()
vect.fit(df_train['message'])
vect = CountVectorizer(stop_words='english')
vect = CountVectorizer(ngram_range=(1, 2))



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# This is our company logo
	#st.image("resources/imgs/LeafLogo.png", caption='Our company logo')
	
	# Creates a main title and subheader on your page -
	# these are static across all pages

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	#st.sidebar.markdown('<div style="text-align: center; color:White; font-size: 20px;">SELECT A PAGE BELOW</div>', unsafe_allow_html=True)
	#options = ["🏠Home ", "❔❓About Us", "📈Prediction", "ℹ️Information", "📧☎️Contact Us"]
	#selection = st.sidebar.selectbox("", options)
	#st.sidebar.info("General Information")
	with st.sidebar:
		selection = option_menu("Main Menu", ["Home", "About Us", "Prediction", "Information", "Contact Us"], 
        icons=['house', 'people','graph-up-arrow','info-circle','envelope'], menu_icon="cast", default_index=0)
																				# default_index = 0 (Home page)

	# Building out the "About Us" page
	if selection == "Home":
		# This is our company logo
		#st.image("resources/imgs/LeafLogo.png", caption='Our company logo')
		#st.image("resources/imgs/LogoBanner.png", caption='Our company logo', width= 1100)
		
		# Centering the logo image
		col1, col2, col3 = st.columns([1,6,1])

		with col1:
			st.write("")

		with col2:
			st.image("resources/imgs/LeafLogo.png")

		with col3:
			st.write("")
		st.info("Climate Change")

		st.markdown("Climate change refers to long-term shifts in temperatures and weather patterns. \
					These shifts may be natural, or may be caused by human activities.")
		#st.markdown("==================================================================================")
		st.markdown("Many companies are built around lessening one’s environmental impact or carbon footprint.\
					They offer products and services that are environmentally friendly and sustainable, in \
					line with their values and ideals. They would like to determine how people perceive\
					climate change and whether or not they believe it is a real threat.")
		st.markdown("The team was tasked in creating a Machine Learning model that is able to classify \
					whether or not a person believes in climate change, based on their novel tweet data.")
		st.markdown("The collection of this data was funded by a Canada Foundation for Innovation JELF Grant\
					 to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate\
					change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43,943 tweets were collected. \
					")
		#st.markdown("==================================================================================")
		st.markdown("")

		st.info("The Business Value of Our Product")
		st.markdown("The model we have deployed successfully classifies a users tweet as either Anti, Neutral, Pro or Factual. \
					This will help you, as a business, understand what your clients and potential clients are saying about \
					climate change, and what they believe. As a business, it is vital to understand this information, so as \
					to *customise* the type of advertisements you could show to each group of customers. This will in turn \
					ensure that you bring the *right product* to the *right customer*, \
					which is vital for the **success** of a business.\
					")
		st.markdown("")

	
	# Building out the "About Us" page
	if selection == "About Us":
		# This is our company logo
		#st.image("resources/imgs/LeafLogo.png", caption='Our company logo')

		# Centering the logo image
		col1, col2, col3 = st.columns([1,6,1])

		with col1:
			st.write("")

		with col2:
			st.image("resources/imgs/LeafLogo.png")

		with col3:
			st.write("")

		# You can read a markdown file from supporting resources folder
		#st.title("Who Are We?")
		st.markdown("")
		st.markdown("")

		st.markdown('<div style="text-align: center; color:Black; font-weight: bold; font-size: 30px;">Who Are We?</br></br></div>', unsafe_allow_html=True)

		st.subheader("Enviro Co.")
		st.markdown('We are a company that deals with recycable products and materials. \
					We are naturally concerned about `climate change` and how we can help the public lead and live a greener lifestyle. \
					Most of what we do revolves around the full Data Science Life Cycle:   \
					')
		st.markdown(f"""
				- Data Collection
				- Data Cleaning
				- Exploratory Data Analysis
				- Model Building
				- Model Deployment
				""")
		#st.subheader("Meet The Team")
		st.markdown('<div style="text-align: center; color:Black; font-weight: bold; font-size: 30px;">Meet The Team</br></br></div>', unsafe_allow_html=True)

		col1, col2, col3, col4, col5, col6 = st.columns(6)
		
		with col1:
			#st.subheader("Caron")
			st.markdown('Caron')
			st.image("resources/imgs/Caron_Sathekge2.jpg")

		with col2:
			#st.subheader("Hlengiwe")
			st.markdown('Hlengiwe')
			st.image("resources/imgs/Hlengiwe2.jpg")

		with col3:
			#st.subheader("Jade")
			st.markdown('Jade')
			st.image("resources/imgs/Jade2.jpg")

		with col4:
			#st.subheader("Palesa")
			st.markdown('Palesa')
			st.image("resources/imgs/Palesa3.jpg")

		with col5:
			#st.subheader("Kgotlelelo")
			st.markdown('Kgotlelelo')
			st.image("resources/imgs/Kgotlelelo2.jpg")

		with col6:
			#st.subheader("Nakedi")
			st.markdown('Nakedi')
			st.image("resources/imgs/Nakedi2.jpg")

	# Building out the predication page
	if selection == "Prediction":
		#st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification")
		st.info("Predicting with our best performing ML Model!")

		# Creating a text box for user input
		tweet_text = st.text_area(label="Enter Text", height= 250, help="Enter a text, then click on 'Classify' below", placeholder="Enter any text here")

		#if st.button("Click Me To Classify 👈"):
		if st.button("Classify 👈"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/logistic_regression_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Your text was categorized as: {}".format(prediction), icon="✅")
			st.balloons()
			#st.snow()

		if st.checkbox('See Category Meanings'):
			st.markdown(f"""
				**THE MEANING OF THESE CATEGORIES?**
				- Category **-1** = Anti-climate change
				- Category **0** = Neutral
				- Category **1** = Pro climate change
				- Category **2** = Factual News
				""")

	# Building out the "Information" page
	if selection == "Information":
		#st.title("Tweet Classifer")
		st.markdown('<div style="text-align: center; color:Black; font-size: 30px;">Some Information About The Data</div>', unsafe_allow_html=True)
		st.markdown("")
		st.markdown("")
		st.markdown("")

		#st.subheader("Climate change tweet classification")
		st.info("A Bar Graph showing the number of tweets per sentiment")
		# You can read a markdown file from supporting resources folder
		#st.markdown("A Bar Graph showing the number of tweets per sentiment")
		st.bar_chart(data=df_train["sentiment"].value_counts(), x=None, y=None, width=220, height=320, use_container_width=True)

		st.info("As it is observed from the bar graph above, it is well noted that many people \
					support the belief of man-made climate change.")
		st.markdown("")
		st.markdown("")
		st.markdown("")

		#st.info("Our Pricing")
		st.markdown('<div style="text-align: center; color:Black; font-size: 30px;">Our Pricing</div>', unsafe_allow_html=True)
		st.markdown("")
		st.markdown("")

		st.info("Standard - *Free*")
		st.info("Pro - *$3 p.a*")
		st.markdown("Sentiments get emailed to you")
		st.info("Premium - *$7 p.a*")
		st.markdown("Sentiments get emailed to you and your whole team")
		st.markdown("Accompanying sentiment visuals")

		st.markdown("")
		st.markdown("")
		st.markdown("")

		st.subheader("Raw Twitter data")
		if st.checkbox('View the raw data represented by the bar chart above'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page


    #A form for the Contact Us page
	if selection == "Contact Us":
		st.subheader("Contact Us")
		with st.form(key='form1'):
			firstname = st.text_input("Firstname")
			lastname = st.text_input("Lastname")
			email = st.text_input("Email")
			message = st.text_area("Insert text here")

			submitted = st.form_submit_button()
		if submitted:
			st.success("Hello {}, your infomation has been captured ".format(firstname))

	

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
