"""

    Simple Streamlit webserver application for serving developed classification
	models.
 
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
from numpy import result_type
import scipy
import streamlit as st	#pip install streamlit
import joblib,os
from PIL import Image
from streamlit_lottie import st_lottie		#pip install streamlit-lottie
import requests	#pip install requests
import json
from streamlit_option_menu import option_menu	#pip install streamlit-option-menu

# Data dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import re
from nlppreprocess import NLP
nlp = NLP()
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from scipy import sparse
#------------------------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_icon="resources/imgs/Logo.png", page_title="Lumina Datamatics")


#theme.primaryColor
primary_clr = st.get_option("theme.primaryColor")
txt_clr = st.get_option("theme.textColor")
    # I want 3 colours to graph, so this is a red that matches the theme:
second_clr = "#d87c7c"



# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
Raw_data = pd.read_csv("resources/train.csv")

#------------------------------------------------------------------------------------------------------------
#---------------USE local CSS--------------------------------------------------------------------------------
def local_css(file_name):
	with open(file_name) as f:
		st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
  
local_css("resources/style/style.css")
#------------------------------------------------------------------------------------------------------------
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
#------------------------------------------------------------------------------------------------------------

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_visuals = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_cvnua1fYWk.json")
lottie_class = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_sNB9tmihGZ.json")
#-------------------------------------------------------------------------------------------------------------
    



#-------------------------------------------------------------------------------------------------------------

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	with st.sidebar:
    		selection = option_menu(
			menu_title="Main Menu",
			options = ["Home", 'Data Visualization', "Tweets Classification", "Lumina Datamatics Team", "Contact Us"],
   			icons=["house","bar-chart-fill","basket2-fill","people-fill","envelope"],		#Optional
			styles={
			"container": {"padding": "0!important", "background-color": "#00172B"},
			"icon": {"color": "blue", "font-size": "15px"}, 
            "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "green"},
       		},
      		)
	# st.title("Tweet Classifer")
	st.title("Lumina Datamatics Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
 
	#options = ["General Information", 'Data Visualization', "Tweets Classification", "Lumina Datamatics Team", "Contact Us"]
	#selection = st.sidebar.image("resources/imgs/Logo1.png",
   	#width=300,)
	#selection = st.sidebar.selectbox("Choose an Option Here:", options)

#---------------------------------------------------------------------------------------------------------------------------
	# Building out the "General Information" page
	if selection == "Home":
		#st.image("https://imgur.com/zUvnHT1.png")
		image = Image.open('resources/imgs/Logo1.png')
		st.image(image, caption = 'Shinning some light on data to give you insights', use_column_width=True)
		st.subheader("""Climate Change Tweeter Sentiment Classification""")

		st.info("""This work creates and makes publicly available the most comprehensive solution to date regarding climate change and human 
  		opinions via Twitter, thus increasing their insights and informing future marketing strategies. 
    	This sentiment analysis can help marketers understand consumer behavior and preferences, and develop targeted advertising campaigns.""")
		
		st.info('''IMPORTANT INSTRUCTIONS: You are required to input text (ideally a tweet relating to climate change),
		and the model will classify it according to whether the believe in climate change or not.
		You can classify your tweets on the 'Tweets Classification' page as you navigate it on the sidebar.''')

		st.write("The classes can be interpreted as follows:")				
		st.write("2: News  --  The tweet links to factual news about climate change.")
		st.write("1: Pro  --  The tweet supports the belief of man-made climate change.")
		st.write("0: Neutral  --  The tweet neither supports nor refutes the belief of man-made climate change.")
		st.write("-1: Anti  --  The tweet does not believe in man-made climate change.")

		st.subheader("View Raw Data Here")
		data_display = ['Select option', 'Columns', 'View Random row from the dataset', 'Display Full Dataset']
		source_selection = st.selectbox('Select desired display:', data_display)

		if source_selection == 'Columns':
			st.write('Display of the columns in the dataset:')
			st.write(Raw_data.columns)

		if source_selection == 'View Random row from the dataset':
			st.write('Display of a random row in the dataset:')
			st.write(Raw_data.sample())
			st.write('You can re-select this same option from the dropdown to view another random row.')

		if source_selection == 'Display Full Dataset':
			st.write("Display of the full raw data that was used to train our model algorithms:")
			st.write(Raw_data)
#--------------------------------------------------------------------------------------------------------------------------------------
   	# Building out the Visuals page
	if selection == "Data Visualization":
     
		image1 = Image.open('resources/imgs/Img9.jpeg')
		st.image(image1, use_column_width=True)


		visual_options = ["Visuals (Home)", "Bar Graphs", "Word Clouds", "Model Performance"]
		visual_options_selection = st.selectbox("Which visual category would you like to choose?",
		visual_options)

		if visual_options_selection == "Visuals (Home)":
			#st.image('https://l8.nu/qQC9', width=730)
		#https://l8.nu/qQC9
			st_lottie(
				lottie_visuals,
				speed=1,
				reverse=False,
				loop=True,
				quality="low",
				#renderer="svg",
				height=None,
				width=None,
				key=None,
			)
#-----------------------------------------------------------------------------------
		if visual_options_selection == "Model Performance":
			per_listed = ['F1_measure'] #,'Fit_time']
			per_list = st.selectbox('I would like to view the...', per_listed)

			if per_list == 'F1_measure':
				st.subheader('F1 scores of the various models used')
				st.image('https://imgur.com/o1zYqC8.png', width=730)
			#if per_list == 'Fit_time':
				#st.subheader('Fit time of the various models used')
				#st.image('https://imgur.com/N4DVf2s.png', width=730)


#-------------------------------------------------------------------------------------------------

		if visual_options_selection == "Bar Graphs":
			#st.image('https://i.imgur.com/p3J5Gcw.png')

			bar_nav_list = ['Sentiment Distribution Of Raw Data', 
			'Most Common Words In Various Sentiment Classes From Raw Data', 
			'Most Common Words In Various Sentiment Classes From Cleaned Data)']

			bar_nav = st.selectbox('I would like to view the...', bar_nav_list)


			if bar_nav == 'Sentiment Distribution Of Raw Data':
				st.subheader('Sentiment Distribution Of Raw Data')
				st.image('https://i.imgur.com/JT9HzVW.png', width=700)
				st.write("This is how the raw data is distributed amongst the various sentiment classes.")
				st.write("The classes can be interpreted as follows:")				
				st.write("2: News  --  The tweet links to factual news about climate change.")
				st.write("1: Pro  --  The tweet supports the belief of man-made climate change.")
				st.write("0: Neutral  --  The tweet neither supports nor refutes the belief of man-made climate change.")
				st.write("-1: Anti  --  The tweet does not believe in man-made climate change.")
		
			if bar_nav == 'Most Common Words In Various Sentiment Classes From Raw Data':

				raw_common_words_list = ['All Tweets Class', 'Negative Tweets Class', 'Positive Tweets Class', 
				'News-related Tweets Class', 'Neutral Tweets Class']
				raw_common_words = st.radio('Raw Data Sentiment Classes:', raw_common_words_list)

				if raw_common_words == 'All Tweets Class':
					st.subheader('Common Words in All Tweets Class')
					st.image('https://i.imgur.com/HSdxDP9.png', width=700)

				if raw_common_words == 'Negative Tweets Class':
					st.subheader('Common Words in Negative Tweets Class')
					st.image('https://i.imgur.com/FqnrN1Y.png', width=700)

				if raw_common_words == 'Positive Tweets Class':
					st.subheader('Common Words in Positive Tweets Class')
					st.image('https://i.imgur.com/glF9Z0M.png', width=700)

				if raw_common_words == 'News-related Tweets Class':
					st.subheader('Common Words in News-related Tweets Class')
					st.image('https://i.imgur.com/fWDhrTL.png', width=700)

				if raw_common_words == 'Neutral Tweets Class':
					st.subheader('Common Words in Neutral Tweets Class')
					st.image('https://i.imgur.com/LEnkE9V.png', width=700)


			if bar_nav == 'Most Common Words In Various Sentiment Classes From Cleaned Data)':

				clean_common_words_list = ['All Tweets Class', 'Negative Tweets Class', 'Positive Tweets Class', 
				'News-Related Tweets Class', 'Neutral Tweets Class']
				clean_common_words = st.radio('Cleaned Data Sentiment Classes:', clean_common_words_list)

				if clean_common_words == 'All Tweets Class':
					st.subheader('Common Words in All Tweets Class')
					st.image('https://i.imgur.com/1aOr9DD.png', width=700)

				if clean_common_words == 'Negative Tweets Class':
					st.subheader('Common Words in Negative Tweets Class')
					st.image('https://i.imgur.com/7Mp2NRX.png', width=700)

				if clean_common_words == 'Positive Tweets Class':
					st.subheader('Common Words in Positive Tweets Class')
					st.image('https://i.imgur.com/3SDTh6c.png', width=700)

				if clean_common_words == 'News-Related Tweets Class':
					st.subheader('Common Words in News-Related Tweets Class')
					st.image('https://i.imgur.com/sDFn7PF.png', width=700)

				if clean_common_words == 'Neutral Tweets Class':
					st.subheader('Common Words in Neutral Tweets Class')
					st.image('https://i.imgur.com/0ur3J3C.png', width=700)

		if visual_options_selection == "Word Clouds":
			#st.image('https://i.imgur.com/QDhrJTR.png')

			wc_nav_list = ['Most Common Words From Raw Data', 
			'Most Common Words From Cleaned Data']

			wc_nav = st.selectbox('I would like to view the...', wc_nav_list)
   
#----------------------------
#most common words from raw data
			if wc_nav == 'Most Common Words From Raw Data':
				#st.subheader('Most Common Words for all Tweets in raw data')
				#st.image('https://i.imgur.com/MsGuWFv.png')

				wc_clean_list = ['All Tweets Class', 'Negative Tweets Class', 'Positive Tweets Class', 
				'News-Related Tweets Class', 'Neutral Tweets Class']
				wc_clean = st.radio('Raw Data Sentiment Classes:', wc_clean_list)

				if wc_clean == 'All Tweets Class':
					st.subheader('Common Words in All Tweets Class')
					st.image('https://i.imgur.com/MsGuWFv.png')

				if wc_clean == 'Negative Tweets Class':
					st.subheader('Common Words in Negative Tweets Class')
					st.image('https://i.imgur.com/Fqk8iIa.png')
					st.subheader('Top 10 hashtags in Sentiment -1(Does not believe in man-made climate change)')
					st.image('https://i.imgur.com/zgLDiwU.png')

				if wc_clean == 'Positive Tweets Class':
					st.subheader('Common Words in Positive Tweets Class')
					st.image('https://i.imgur.com/4bYQx2M.png')
					st.subheader('Top 10 hashtags in Sentiment 1(Supports the belief of man-made climate change)')
					st.image('https://i.imgur.com/fxop4jQ.png')

				if wc_clean == 'News-Related Tweets Class':
					st.subheader('Common Words in News-Related Tweets Class')
					st.image('https://i.imgur.com/uXdEPmc.png')
					st.subheader('Top 10 hashtags in Sentiment 2(Links to factual news about climate change)')
					st.image('https://i.imgur.com/uVzSl48.png')

				if wc_clean == 'Neutral Tweets Class':
					st.subheader('Common Words in Neutral Tweets Class')
					st.image('https://i.imgur.com/woubDx6.png')
    				#st.subheader('Top 10 hashtags in Sentiment 0(Neither supports nor refutes the belief of man-made climate change)')
					st.image('https://i.imgur.com/7MRVnzi.png')
     
#-----------------------------
#most common words from cleaned data
			if wc_nav == 'Most Common Words From Cleaned Data':

				wc_clean_list = ['All Tweets Class', 'Negative Tweets Class', 'Positive Tweets Class', 
				'News-Related Tweets Class', 'Neutral Tweets Class']
				wc_clean = st.radio('Cleaned Data Sentiment Classes:', wc_clean_list)

				if wc_clean == 'All Tweets Class':
					st.subheader('Common Words in All Tweets Class')
					st.image('https://i.imgur.com/0MeELLk.png')

				if wc_clean == 'Negative Tweets Class':
					st.subheader('Common Words in Negative Tweets Class')
					st.image('https://i.imgur.com/fc0Aa9l.png')

				if wc_clean == 'Positive Tweets Class':
					st.subheader('Common Words in Positive Tweets Class')
					st.image('https://i.imgur.com/4LSqpBm.png')

				if wc_clean == 'News-Related Tweets Class':
					st.subheader('Common Words in News-Related Tweets Class')
					st.image('https://i.imgur.com/cmrDvhk.png')

				if wc_clean == 'Neutral Tweets Class':
					st.subheader('Common Words in Neutral Tweets Class')
					st.image('https://i.imgur.com/AJMYNOu.png')

#--------------------------------------------------------------------------------------------------------------------------------
	#Build the "Classify Tweets" Page

	if selection == "Tweets Classification":
		#st.image("https://i.imgur.com/yhNAXaS.png")
		st_lottie(
			lottie_class,
			speed=1,
			reverse=False,
			loop=True,
			quality="low",
			#renderer="svg",
			height=400,
			width=700,
			key=None,
		)

		st.info("Classifying Tweets In The Classifier Section Below")
		data_source = ['Select option', 'Single Tweet'] #Defines the type of input to classify
		source_selection = st.selectbox('Select Single Tweets Section Option Below:', data_source)
		st.info('Analyze Your Tweet Using These Machine Learning Algorithms')

		all_models = ["Logistic Regression", "Multinomial Naive Bayes" ,"Linear Support Vector Machine (Linear SVC)", "Stochastic Gradient Descent (SGD) Classifier", "Support Vector Classifier(SVC)"]
  
		def clean(tweet):
			tweet = re.sub(r'@[A-za-z0-9_]+', '', tweet) # remove twitter handles (@user)
			tweet = re.sub(r'https?:\/\/[A-za-z0-9\.\/]+', '', tweet) # remove http links
			tweet = re.sub(r'RT ', '', tweet) # remove 'RT'
			tweet = re.sub(r'[^a-zA-Z0-9 ]', '', tweet) # remove special characters, numbers and punctuations
			tweet = re.sub(r'#', '', tweet) # remove hashtag sign but keep the text
			tweet = tweet.lower() # transform to lowercase 
			return tweet

		if source_selection == "Single Tweet":
			st.subheader('Tweet classification')
			tweet_text = st.text_area("Enter Tweet (max. 120 characters):")
			
			selected_model = st.selectbox("Select preferred Model to use:", all_models)

			
			if selected_model == "Logistic Regression":
				model = "resources/Lrmodel.pkl"
			elif selected_model == "Linear Support Vector Machine (Linear SVC)":
				model = "resources/Linsvcmodel.pkl"
			elif selected_model == "Multinomial Naive Bayes":
				model = "resources/multimodel.pkl"
			elif selected_model == "Stochastic Gradient Descent (SGD) Classifier":
				model = "resources/SGDmodel.pkl"
			else:
				model = "resources/SVCmodel.pkl"

			if st.button ("Classify"):
				st.text("Your inputted tweet: \n{}".format(tweet_text))
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				predictor = joblib.load(open(os.path.join(model), "rb"))
				prediction = predictor.predict(vect_text)

				result = ""
				if prediction == 0:
					result = '"**Neutral**"; it neither supports nor negates the belief of man-made climate change'
				elif prediction == 1:
					result = '"**Pro**"; it  supports the belief of man-made climate change'
				elif prediction == 2:
					result = '"**News**"; it contains factual links to climate change'
				else:
					result = '"**Negative**"; it negates the belief of man-made climate change'
				
				st.success("Categorized as {}".format(result))

					
#----------------------------------------------------------------------------------------------------------------------------
		
	if selection == "Lumina Datamatics Team":
		st.title("Meet the Lumina Datamatics Team")
		st.info("The Real People Behind All The Smoke And Mirrors")

		#Mpho = Image.open("resources/imgs/Mpho.jpg")
		#title = "Team Lead: Mpho Tjale"
		#st.image(Mpho, caption = title, width = 500)

		#Tshepo = Image.open("resources/imgs/Tshepo.JPG")
		#st.image(Tshepo, caption = "Creative Technologist: Tshepo", width = 300)

		#Nyiko = Image.open("resources/imgs/Nyiko.jpg")
		#st.image(Nyiko, caption = "Deputy Technical Lead: Nyiko", width = 300)

		#Abaidance = Image.open("resources/imgs/Abidance.jpg")
		#st.image(Abaidance, caption = "Technical Lead: Abaidance", width = 300)

		#Ayanda = Image.open("resources/imgs/Ayanda.png")
		#st.image(Ayanda, caption = "Communications Lead: Ayanda", width = 300)

		#Mathapelo = Image.open("resources/imgs/Mathapelo.jpg")
		#st.image(Mathapelo, caption = "Design Director: Mathapelo", width = 300)
		#Emmanuel = Image.open("resources/imgs/West.jpg")
		#st.image(Emmanuel, caption = "Operations & Projects: Emmanuel", width = 300)

		with st.container():
			st.write("---")
			st.header("Meet the Lumina Datamatics Team")
			st.write("##")
			image_column, text_column = st.columns((1, 2))
			with image_column:
				Mpho = Image.open("resources/imgs/Mpho.jpg")
				st.image(Mpho)
			with text_column:
				st.subheader("Team Lead: Mpho Tjale")
				st.write("Professional in charge of guiding, monitoring and leading the entire team")

		with st.container():
			image_column, text_column = st.columns((1, 2))
			with image_column:
				Tshepo = Image.open("resources/imgs/Tshepo.JPG")
				st.image(Tshepo)
			with text_column:
				st.subheader("Creative Technologist: Tshepo")
				st.write("Primarily technology-focused individual who develop information technology solutions for digital innovation projects. Collaborate with production and marketing departments, design software prototypes, and enhance digital user-experiences")
		with st.container():
			image_column, text_column = st.columns((1, 2))
			with image_column:
				Nyiko = Image.open("resources/imgs/Nyiko.jpg")
				st.image(Nyiko)
			with text_column:
				st.subheader("Deputy Technical Lead: Nyiko")
				st.write("Help Oversee the company's technical team and all projects they undertake, analyze briefs, write progress reports and identify risks")
      	
		with st.container():
			image_column, text_column = st.columns((1, 2))
			with image_column:
				Ayanda = Image.open("resources/imgs/Ayanda.png")
				st.image(Ayanda)
			with text_column:
				st.subheader("Communications Lead: Ayanda")
				st.write("Ensure effective communication, promotion, marketing, engagement, and involvement both within and out of the organisation to support our goals. ")

		with st.container():
			image_column, text_column = st.columns((1, 2))
			with image_column:
				Abidence = Image.open("resources/imgs/Abidance.jpg")
				st.image(Abidence)
			with text_column:
				st.subheader("Technical Lead: Abidence")
				st.write("Oversee the company's technical team and all projects they undertake, analyze briefs, write progress reports and identify risks")
    
		with st.container():
			image_column, text_column = st.columns((1, 2))
			with image_column:
				Mathapelo = Image.open("resources/imgs/Mathapelo.jpg")
				st.image(Mathapelo)
			with text_column:
				st.subheader("Design Director: Mathapelo")
				st.write("Ensure that all products and experiences are delivered on time, on budget, and to the highest standards of quality. In addition to communicating a company's creative vision to design teams and stakeholders, supervises the entire design process and all-important technical decisions")

			image_column, text_column = st.columns((1, 2))
			with image_column:
				Emmanual = Image.open("resources/imgs/West.jpg")
				st.image(Emmanual)
			with text_column:
				st.subheader("Operations & Projects: Emmanuel")
				st.write("Establish project plans, tasks, resources, and ownership to ensure timelines are met")

#---------------------------------------------------------------------------------
# -----CONTACT--------
	if selection == "Contact Us":
		st.header("Get In Touch With Us!")
		collage = Image.open("resources/imgs/Collage2.png")
		st.image(collage)
		st.write("---")
		st.write("##")

		contact_form = """
  		<form action="https://formsubmit.co/tshepoelifa238@gmail.com" method="POST">
			<input type="hidden" name="captcha" value="false">
     		<input type="text" name="name" placeholder="Your name" required>
     		<input type="email" name="email" placeholder="Your email" required>
			<textarea name="message" placeholder="Your message here" required></textarea>
     		<button type="submit">Send</button>
		</form>
     	"""
		left_column, right_column = st.columns(2)
		with left_column:
			st.markdown(contact_form, unsafe_allow_html=True)
		with right_column:
			st.empty()





# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
