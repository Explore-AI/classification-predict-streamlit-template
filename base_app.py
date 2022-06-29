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
import io
from streamlit_option_menu import option_menu


# Data dependencies
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import contractions
import matplotlib.pyplot as plt
#import seaborn as sns

# Vectorizer
news_vectorizer = open("resources/vectoriser-ngram-(1,2).pickle","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
##define function to preprocess data
def preprocess(textdata):
    processedTweet = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"http\S+"
    punctuations      = "[^a-zA-Z#@_]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern,'newsurl',tweet)
        #expand contractions
        tweet= contractions.fix(tweet)              
        # Replace all punctuation.
        tweet = re.sub(punctuations, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            #if word not in stopwordlist:
            if word not in stopwords.words('english'):
                if len(word)>3:
                # Lemmatizing the word.
                    word = wordLemm.lemmatize(word)
                    tweetwords += (word+' ')
            
        processedTweet.append(tweetwords)

		
        return   processedTweet
#define function to export csv file
def convert_df(df):
   	return df.to_csv().encode('utf-8')
def predict(path, model, feature):
	predictor = joblib.load(open(os.path.join(path,model),"rb"))
	prediction = predictor.predict(feature)
	return prediction
#define function to format graph
def my_fmt(x):
    return '{:.4f}%\n({:.0f})'.format(x, total*x/100)

logo = Image.open("resources/imgs/default_edited.png")
oluyemi_new= Image.open('resources/imgs/yemi edsa picture.JPG')
oluyemi=oluyemi_new.resize((180,250))
joshua_new= Image.open('resources/imgs/Eujosh_pic_new.JPEG')
joshua=joshua_new.resize((180,250))
abiola_new= Image.open('resources/imgs/Abiola_pic.JPEG')
abiola=abiola_new.resize((180,250))
ifeoluwa_new= Image.open('resources/imgs/ifeoluwa_pic_new.JPEG')
ifeoluwa=ifeoluwa_new.resize((180,250))
lawson_new=Image.open('resources/imgs/Lawson_pic.JPEG')
lawson=lawson_new.resize((180,250))
stephen_new=Image.open('resources/imgs/Stephen_pic_new.JPEG')
stephen=stephen_new.resize((180,250))
welcome_message='<p style="font-family:sans-serif; color:Black; font-size: 20px;">Welcome, we are glad to have you here.\
			Kindly use Navigation on the side to find your way around. Enjoy your stay</p>'

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title ("AIORIGIN")
	
	

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	Navigation = ["About Us", "Classifier", "Information"]
	selection = st.sidebar.selectbox("Navigation path", Navigation)

	#Building the "About us" page
	if selection== "About Us":
		st.image(logo)
		st.markdown(welcome_message,unsafe_allow_html=True)
		st.info("Meet the Team")
		st.image(image=[oluyemi, joshua,abiola,ifeoluwa,lawson,stephen,],
		caption=['Oluyemi Alabi','Joshua Umukoro', 'Abiola Akinwale',
		'Ifeoluwa Adeoti','Lawson Umukoro','Stephen Tshiani'])
		#if st.button("Go to classifier"):
			#open sidebar.selectbox("Make Prediction")

	# Building out the "Information" page
	if selection == "Information":
		st.info("Classifier Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("The Multinomial Naive-Bayes model was used in the development of the application\
			it uses the bayes principle. You can have a look at the raw data used and also the model\
				 performace as compared to other models.")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show sample of raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
		st.checkbox('View model Perfomance graph')


	# Building out the predication page
	if selection == "Classifier":
		st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification")
		model_selection = st.radio("Please choose a model", ["Logistic Regression" ,"Multinomial Naive Bayes (Recommended)",\
			"Linear Support Vector Classifier"], help= 'Select a model that will be used for prediction')

		st.info("NOTE: If you would like to analyse large amount of tweets at a go, kindly use the upload button on the side.  \
			File to be uploaded must be '.csv' and have two columns. First column \
				should be named 'index' while second column named 'tweets'. Maximum number of rows is '15,000' ")
		st.info("For single tweets, enter the tweet to be analysed in the text area below")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Tweet","Type Here")


		if st.button("Classify"):
			#tweet_text= preprocess(tweet_text)
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			#predictor = joblib.load(open(os.path.join("resources/model_5.pkl"),"rb"))
			#prediction = predictor.predict(vect_text)

			if model_selection == "Logistic Regression":				
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				prediction= predict('resources','Sentiment-LR.pickle', vect_text)
				output = {-1: "an Anti-climate tweet", 0: "a Neutral tweet", 1: "a Pro-climate tweet", 2: "Climate change News"}
				f = output.get(prediction[0])	
			# When model has successfully run, will print prediction
				st.success("Tweet is Categorized as: {}".format(f))

			elif model_selection == "Multinomial Naive Bayes (Recommended)":
				prediction=predict('resources','mnb.pickle', vect_text)
				output = {-1: "an Anti-climate tweet", 0: "a Neutral tweet", 1: "a Pro-climate tweet", 2: "Climate change News"}
				f = output.get(prediction[0])	
			# When model has successfully run, will print prediction
				st.success("Tweet is Categorized as: {}".format(f))

			elif model_selection == "Linear Support Vector Classifier":
				prediction=predict('resources','svc_model.pickle', vect_text)
				output = {-1: "an Anti-climate tweet", 0: "a Neutral tweet", 1: "a Pro-climate tweet", 2: "Climate change News"}
				f = output.get(prediction[0])	
			# When model has successfully run, will print prediction
				st.success("Tweet is Categorized as: {}".format(f))
			
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			#output = {-1: "an Anti-climate tweet", 0: "a Neutral tweet", 1: "a Pro-climate tweet", 2: "Climate change News"}
			#f = output.get(prediction[0])	
			# When model has successfully run, will print prediction
			#st.success("Tweet is Categorized as: {}".format(f))
		
			## for the large test file, option to upload the file
		upload_file= st.sidebar.file_uploader(
			label="Upload the csv file containing tweets here",
    		type="csv",
   			accept_multiple_files=False,
    		help='''Upload a csv file that contains tweet.     
        	required structure:     
        	first column = index;        
        	second column = tweets;      
        	first row = column headers;     
        	length = max. 15,000 rows
        	''')   
		if upload_file is not None:
			tweet_df = pd.read_csv(upload_file)
			preprocess(tweet_df['tweets'])
			vect_df = tweet_cv.transform(tweet_df['tweets']).toarray()

			if model_selection =="Multinomial Naive Bayes (Recommended)":
				#tweet_df = pd.read_csv(upload_file)
				#vect_df = tweet_cv.transform(tweet_df['tweets']).toarray()
				classification=predict('resources','mnb.pickle', vect_df)
				#predicted = joblib.load(open(os.path.join("resources/mnb.pickle"),"rb"))
				#classification = predicted.predict(vect_df)

			elif model_selection == "Logistic Regression":
				#vect_df = tweet_cv.transform(tweet_df['tweets']).toarray()
				classification=predict('resources','Sentiment-LR.pickle', vect_df)
				#predicted = joblib.load(open(os.path.join("resources/Sentiment-LR.pickle"),"rb"))
				#classification = predicted.predict(vect_df)

			elif model_selection == "Linear Support Vector Classifier":
				#tweet_df = pd.read_csv(upload_file)
				#vect_df = tweet_cv.transform(tweet_df['tweets']).toarray()
				classification=predict('resources','svc_model.pickle', vect_df)
				#predicted = joblib.load(open(os.path.join("resources/svc_model.pickle"),"rb"))
				#classification = predicted.predict(vect_df)


			result = pd.DataFrame(classification, columns = ['sentiment'])
			result['tweets'] = tweet_df['tweets']
			result = result[['tweets', 'sentiment']]
			m={'sentiment':{1:'Pro-climate change', 2: 'Climate change news', 
			0:'Neutral', -1: 'Anti-climate change'}}
			result.replace(m, inplace=True)
			classified=convert_df(result)
			st.download_button("Press to Download result file",
			classified,"file.csv","text/csv",
			key='download-csv')
			graph=result['sentiment'].value_counts().plot(kind='pie')
			graph= 'pie.png'
			img = io.BytesIO()
			plt.savefig(img, format='png')
			st.download_button(
				label="Download chart",
        		data=img,
        		file_name=graph,
        		mime="image/png"
    )

			


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
