

import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import plotly.express as px

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url('https://anhdepfree.com/wp-content/uploads/2019/05/50-anh-background-dep-nhat-4.jpg');
background-size: cover;
}
[data-testid="stHeader"] {
background-color: rgba(0,0,0,0);

}

[data-testid="stToolbar"] {
right: 2rem;
}

[data-testid="stSidebar"] {
background-image: url('https://images.pexels.com/photos/2088203/pexels-photo-2088203.jpeg?auto=compress&cs=tinysrgb&w=600');
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


	st.info("This application is all about tweet sentiment analysis of climate change. It is able to classify whether" 
			 "or not a person believes in climate change, based on their novel tweet data.")


	# Building out the "Tweet Sentitment analysis " page		
	if selection == "Tweet analysis":
		st.info("This app analyses sentiments on climate change based on tweet data")
		#top level filters
		#message_filter = st.selectbox("Select the message", pd.unique(raw['sentiment']))
		# dataframe filter
		#df = raw[raw['sentiment']== message_filter] 
		st.markdown("### Tweet distribution")
		sentiment = raw['sentiment'].value_counts()
		sentiment = pd.DataFrame({'Sentiment':sentiment.index, 'Tweets':sentiment.values})
	
		# create two columns for charts
		fig_col1, fig_col2 = st.columns(2)
		
		with fig_col1:
			fig = fig = px.bar(sentiment, x='Sentiment', y='Tweets', color = 'Tweets', height= 500)
			st.plotly_chart(fig)
	
       	#
		with fig_col2:
			fig = px.pie(sentiment, values= 'Tweets', names= 'Sentiment')
			st.plotly_chart(fig)
			
	

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
from streamlit_option_menu import option_menu
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.metrics import f1_score
import model_app
import base64
import numpy as np





# Data dependencies
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import plotly.express as px

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url('https://anhdepfree.com/wp-content/uploads/2019/05/50-anh-background-dep-nhat-4.jpg');
background-size: cover;
}
[data-testid="stHeader"] {
background-color: rgba(0,0,0,0);

}

[data-testid="stToolbar"] {
right: 2rem;
}

[data-testid="stSidebar"] {
background-image: url('https://images.pexels.com/photos/2088203/pexels-photo-2088203.jpeg?auto=compress&cs=tinysrgb&w=600');
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Vectorizer

#news_vectorizer = open("resources/tfidfvect.pkl","rb")
#tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
#rfc_vectorizer = open("resources/rfc_TfidfVectorizer.pkl","rb")
#tweet_rfc = joblib.load(rfc_vectorizer)

news_vectorizer = open("resources/Count_Vectorizer.pkl","rb")
tweet_vect = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
#upload_file 


# Load your raw data
#raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual appdef random_char(y):

news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
rfc_vectorizer = open("resources/rfc_TfidfVectorizer.pkl","rb")
tweet_rfc = joblib.load(rfc_vectorizer)




# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app

def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages

	#st.subheader("Climate change tweet classification")
	
	#st.info("This application is all about tweet sentiment analysis of climate change. It is able to classify whether" 
			# "or not a person believes in climate change, based on their novel tweet data.")

	#st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	#with open("resources/imgs/testing_bck.jpg","rb") as background_img:
	#	encoded_string = base64.b64encode(background_img.read())

	# st.markdown(
    #f"""
    #<style>
    #.stApp {{
     #  background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
	#	background-attachment: fixed;
     #  background-size: cover
    #}}
   	#</style>
    #""",
    # unsafe_allow_html=True
    #)
	
	options = ["Background","About us","Know your file","Text tweet prediction","File tweet classification","Conclusion"]
	selection = st.sidebar.selectbox("Lets interact", options)

	if selection == "Conclusion":
		st.success("Some very good news, we successfully deployed the AI-Platform Twitter classification to production today! Big thanks to the team for helping us get this over the line. It was a pleasure to work off of a well written code base, and the requested changes were delivered on time and to spec.")
		st.info("Thynk Data allows us to improve on all aspects of risk management by providing a single toolkit for data analysis and preparation, modelling, deployment and monitoring. It allows us to use the latest tools and techniques, without sacrificing the transparency, robustness, customisation and efficiency we expect.")

	if selection == "About us":
		st.subheader("Our Story")
		st.info("Our CEO, Dr Craig Nyatondo, an internationally published data scientist and at the time computer science lecturer at the University of Stellenbosch, founded ml4africa.com (Machine Learning for Africa) in 2013 after he saw the potential of his field of study to impact communities in South Africa and beyond. One year later Thynk Data with its agile and innovative business model was born as the result of a keen understanding of the predictive analysis needs of governmental stakeholders as well as corporate clients. The integration of theory and praxis lies at the heart of who Thynk Data is. Our data crafters are accomplished engineers, mathematicians and scientists, who are much respected in their respective fields of specialisation.")

		st.subheader("We belive in Purpose before profit")
		st.write("•		Our purpose is to create value by collaboratively crafting elegant, data-driven solutions for significant problems. ")
		st.write("• 	To achieve this, we subscribe to the values of Trustworthy Leadership, Collaborative Learning and Creative Craftsmanship..")
		st.write("•		To our clients, Thynk Data promises to be Trendsetters, Academically Excellent, Agile and Adaptable and Deeply Immersed")


		st.subheader("Meet the Team")
		fig_col1, fig_col2,fig_col3,fig_col4,fig_col5 = st.columns(5)
		
		with fig_col1:
			image_climate = Image.open(os.path.join("resources/imgs/Craig.jpg"))
			image_climate = image_climate.resize((300,300))
			st.image(image_climate, caption='Chief Execitive Officer: Dr Craig Nyatondo ')

		with fig_col2:
			image_climate = Image.open(os.path.join("resources/imgs/Caitlin.jpg"))
			image_climate = image_climate.resize((300,300))
			st.image(image_climate, caption='Chief Infrmation Officer: caitlin Mclaren')

		with fig_col3:
			image_climate = Image.open(os.path.join("resources/imgs/Karabo.jpg"))
			image_climate = image_climate.resize((300,300))
			st.image(image_climate, caption= 'Senior Data Engineer: Karabo Ratona')

		with fig_col4:
			image_climate = Image.open(os.path.join("resources/imgs/Nomonde.jpg"))
			image_climate = image_climate.resize((300,300))
			st.image(image_climate, caption='Senior Interface Developer: Nomonde Mraqisa')

		with fig_col5:
			image_climate = Image.open(os.path.join("resources/imgs/Mamtie.jpg"))
			image_climate = image_climate.resize((300,300))
			st.image(image_climate, caption='Lead full stack Engineer: Mamutele Phosa')

		st.subheader("Recognition for Thynk Data")
		st.info("Thynk Data is a Amazon Gold Partner and Amazon Independent Software Vendor. In 2019 Thynk Data was named a finalist in the Amazon AI Partner of the Year Awards. In 2016 Thynk Data was named the Most Innovative Business in Afica.")

	
		
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
		st.subheader("Classifying tweets using models")
		st.info("In this section we will be classifying tweets using the listed models. The models are listed from best performing to least performing. Keeping in mind that you can select one model at a time")
		model = st.radio(
    	"Select a model to classifiy your tweet",
    		('Logistic_regression','Naive_Bayes','Linear_Support_Vector','Random Forest','K Neighbors' ))
		# Creating a text box for user input
		

		if model == 'Random Forest' :
			st.success("The random forest is a classification algorithm consisting of many decisions trees. It uses bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree")
			tweet_text = st.text_area("Type a tweet")
			tweet_text = model_app.cleaning_text(tweet_text)
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
			st.success("The logisic regression model estimates the probability of an event occurring, such as voted or didn’t vote, based on a given dataset of independent variables. ne of the main advantages of logistic regre is that it is one of the most efficient algorithms ")
			tweet_text = st.text_area("Type a tweet")
			tweet_text = model_app.cleaning_text(tweet_text)
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
			st.success("The random forest is a classification algorithm consisting of many decisions trees. It uses bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree")
			tweet_text = st.text_area("Type a tweet")
			tweet_text = model_app.cleaning_text(tweet_text)
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
			st.success("Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. The Bayes’ Theorem finds the probability of an event occurring given the probability of another event that has already occurred.")
			tweet_text = st.text_area("Type a tweet")
			tweet_text = model_app.cleaning_text(tweet_text)
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
			st.success("The objective of Support Vector Machine algorithm is to find a hyperplane in an N-dimensional space that distinctly classifies the data points. The dimension of the hyperplane depends upon the number of features. The supervised machine learning algorithm can be used for both classification and regression.")
			tweet_text = st.text_area("Type a tweet")
			tweet_text = model_app.cleaning_text(tweet_text)
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



	st.subheader("Climate change tweet classification")


	# Creating sidebar with selection box -
	# you can create multiple pages this way

	options = ["About us","Background", "App tour", "Tweet analysis", "Prediction", "Clonclusion"]
	selection = st.sidebar.selectbox("Page Menu", options)

	# Building out the About us page
	if selection == "About us":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")
		image = Image.open(os.path.join("resources/imgs/twitter_logo.jpg"))
		st.image(image, caption='Sunrise by the mountains')

	# Building out the Background page
	if selection == "Background":
		st.title("ThynkData partners with you to build an industry-leading roadmap for change and innovation.")
		# You can read a markdown file from supporting resources folder
		st.info("With our proven process we identify business cases in an engaging and collaborative way." " "
		 "We also assist in quantifying the business value of such potential projects, preventing wasted expenditure.")
		st.write("This is how we build with you a Data-Driven Enterprise:")
		image = Image.open(os.path.join("resources/imgs/strategic_planning.jpg"))
		st.image(image, caption='')
		st.title("Machine Learning is our profession")
		st.info("We are experts in the latest Machine Learning and modelling techniques" " " 
		"and are able to apply the best fit to your business problem.")
		st.write("•	ThynkData designs, implements and maintains infrastructure " " "
		"to run Machine Learning models on enterprise scale.") 
		st.write("•	Our production environment ensures auditable data governance, " " "
		"robust quality testing and the monitoring of model performance.")
		st.write("•	ThynkData assists in the end-to-end implementation " " "
		"(in the cloud, on premise or in hybrid configurations) of the most " " "
		"optimal infrastructure for specific industry and business needs.")

		st.info("This is how we develop solutions to your challenges:")
		st.write("ThynkData Development Process")
		image = Image.open(os.path.join("resources/imgs/ThynkData_Dev_Process.jpg"))
		st.image(image, caption='')
	
	

	# Building out the App tour page
	if selection == "App tour":
		selected = option_menu(
			menu_title="Main Menu",
			options=["About us", "Background", "Twitter analysis", "Prediction", "Conclusion/ Credit"],
			icons=["people-fill", "book-half", "bar-chart-line-fill", "graph-up"],
			menu_icon="cast",
			default_index=0,
			orientation="horizontal",
		)
		
		st.title("Hi There :wave:")
		st.title("Welcome to our App :smile:")
		st.write("" * 34)

		
			
		


		

		



###########################################################################################################
###########################################################################################################

	# Building out the Tweet Sentitment analysis page		
	if selection == "Tweet analysis":
		st.info("This app analyses sentiments on climate change based on tweet data")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
		#top level filters
		#message_filter = st.selectbox("Select the message", pd.unique(raw['sentiment']))
		# dataframe filter
		#df = raw[raw['sentiment']== message_filter] 
		st.markdown("### Tweet distribution")
		sentiment = raw['sentiment'].value_counts()
		sentiment = pd.DataFrame({'Sentiment':sentiment.index, 'Tweets':sentiment.values})
	
		# create two columns for charts
		fig_col1, fig_col2 = st.columns(2)
		
		with fig_col1:
			fig = fig = px.bar(sentiment, x='Sentiment', y='Tweets', color = 'Tweets',  height= 500)
			#plt.bar(x_pos, height, color=['black', 'red', 'green', 'blue', 'cyan'])
			#x_pos = np.arange(len(bars))
			st.plotly_chart(fig)
	
       	#
		with fig_col2:
			fig = px.pie(sentiment, values= 'Tweets', names= 'Sentiment')
			#fig.plt(np.arrange(0,11), color = 'yellow')
			st.plotly_chart(fig)
			
		

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		model = st.radio(
    	"Select a model to classifiy your tweet",
    		('Random Forest Classifier', 'Logistic_regression'))
		# Creating a text box for user input
		# upload a file
		data = st.radio(
    	"How do you want to load data",
    		('Upload tweets samples', 'Type your tweet'))

		if data == 'Upload tweets samples' :
			upload_file = st.file_uploader("Upload file")
		else:
			tweet_text = st.text_area("Type a tweet")

		if model == 'Random Forest Classifier':
			if st.button("Classify"):
				# Transforming user input with vectorizer
				if data == 'Upload tweets samples' :
					rfc_file = tweet_rfc.transform([upload_file]).toarray()
				else:
					rfc_text = tweet_rfc.transform([tweet_text]).toarray()
				
				# Load your randomfc_model.pkl file 
				predictor = joblib.load(open(os.path.join("resources/randomfc_model.pkl"),"rb"))
				if data == 'Upload tweets samples' :
					prediction_file  = predictor.predict(rfc_file)
				else:
					prediction = predictor.predict(rfc_text)
				# When model has successfully run, will print prediction
				st.success("Text Categorized as: {}".format(prediction))
	
		if model == 'Logistic_regression' :
			if st.button("Classify"):
				# Transforming user input with vectorizer
				if data == 'Upload tweets samples' :
					vect_file = tweet_cv.transform([upload_file]).toarray()
				else:
					vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your Logistic_regression.pkl file 
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# When model has successfully run, will print prediction
				st.success("Text Categorized as: {}".format(prediction))

	#Building out the predication page
	if selection == "Random Forest Classifier":
		st.info("Just a little bit about the random classifyer model")
		
		image = Image.open(os.path.join("resources/imgs/twitter_logo.jpg"))
		st.image(image, caption='Sunrise by the mountains')
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			predictor = joblib.load(open(os.path.join("resources/randomfc_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

	


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()


