"""
	This streamlit web app utilises various models to analyse 
	the sentiment of tweets regarding climate change and 
	classify them into one of the following classes: 
	2: The tweet is a message of factual news
	1: The author believes in climate change
	0: The message has a neutral sentiment
	-1: The author does not believe in climate change
"""
#---------------------------------------------------------------
# Streamlit dependencies
import streamlit as st

# Data dependencies
import spacy
nlp = spacy.load('en_core_web_sm')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Enter your code here:
dftrain = pd.read_csv('train.csv')
dftest = pd.read_csv('test.csv')

# splitting the model
from sklearn.model_selection import train_test_split

X = dftrain['message']  # this time we want to look at the text
y = dftrain['sentiment']

# word count analysis
word_count = dftrain['message'].apply(lambda x: len(x.split()))
dftrain['word_count'] = word_count

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# since i am going to use the tfidtransformer and linearsvc, i should thus create their objects first

# create tfidf object
from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf_transformer = TfidfTransformer()

# create linearsvc object
from sklearn.svm import LinearSVC
# clf = LinearSVC()

from sklearn.pipeline import Pipeline

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)  

# changing background colour
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css('style.css')

#@st.cache(suppress_st_warning=True)

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# dictionary of predictions and definitions
	pred_values = {2:'This is news',
	1:'The author believes in climate change',
	0:'The tweet is neutral in regards climate change',
	-1:'The author does not believe in climate change'
	}
	
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "Exploratory Data Analysis", "Prediction", 'Test Shop', 'Our People' ]
	selection = st.sidebar.radio("Choose Option", tuple(options))


	# Building out the "Information" page
	if selection == "Information":
		image = Image.open('images\markus-spiske-rxo6PaEhyqQ-unsplash.jpg')
		st.info("General Information")
		st.image(image, caption='Climate change, photo by Markus Spiske' )#, use_column_width=True)
	
		# You can read a markdown file from supporting resources folder
		st.markdown("The purpose of this web app is to demonstrate the functionality and performance \n of various models on tweet analysis and classification specifically for climate change.")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(dftrain[['sentiment','message','tweetid']].head(10)) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			# vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			# predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			# prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			prediction =text_clf.predict([tweet_text])
			st.success("Text Category: {}".format(pred_values[prediction[0]]))
	if selection == "Exploratory Data Analysis":
		# boxplots for word count analysis
		# create subplots
		plt.figure(figsize=(1,1))

		fig, axs = plt.subplots(1, 4, sharey = True)
		fig.suptitle('Boxplots for word count of each class')

		# class 2 plot
		y2 = dftrain[dftrain['sentiment'] == 2]['word_count']
		axs[0].boxplot(y2)
		axs[0].set_xlabel('class 2')

		# class 1 plot
		y1 = dftrain[dftrain['sentiment'] == 1]['word_count']
		axs[1].boxplot(y1)
		axs[1].set_xlabel('class 1')

		# class 0 plot
		y0 = dftrain[dftrain['sentiment'] == 0]['word_count']
		axs[2].boxplot(y0)
		axs[2].set_xlabel('class 0')

		# class -1 plot
		y_1 = dftrain[dftrain['sentiment'] == -1]['word_count']
		axs[3].boxplot(y_1)
		axs[3].set_xlabel('class -1')

		axs[0].set_ylabel('Word Count')

		st.pyplot()

		st.markdown('The boxplots of word count show distinct properties for each class.\n The presence of outliers, varying medians and range sizes imply \n that the word count property will add substantial value to model training.')
		#A bar graph comparing the frequency of each sentiment
		dftrain['sentiment'].value_counts().plot(kind = 'bar')
		plt.xticks(rotation='horizontal')
		plt.xlabel('Sentiments')
		plt.ylabel('Sentiment counts')
		plt.title('Sentiment Value Counts')
		st.pyplot()

		st.markdown('This graph shows that these four classes are imbalanced, which affects the accuracy of the model negatively. This shows that resambling is necessary before training a model with this data.')
	if selection == 'Test Shop':
		st.markdown('In this section im just testing things out,\n dont know what i should put in and how i should do it but it will be here for now')

		# testing selectbox
		st.markdown('Testing selectbox')
		selbox = ['a','b','x']
		sb = st.selectbox('choose', selbox)
		st.write(sb)

		# testing the csv uploader
		uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
		if uploaded_file is not None:
			dat1 = pd.read_csv(uploaded_file)
			st.write(dat1.head())

		# testing images
		image = Image.open('images\markus-spiske-rxo6PaEhyqQ-unsplash.jpg')

		st.image(image, caption='Climate change, photo by Markus Spiske' , use_column_width=True)
	if selection == 'Our People':
		st.markdown('Project Owner: EDSA')
		st.markdown('Scrum Master : Noluthando Khumalo')
		st.markdown('Developer: Itumeleng Ngoetjana')
		st.markdown('Designer : Thavha Tsiwana')
		st.markdown('Designer : Pontsho Mokone')
		st.markdown('Tester : Tumelo Mokubi')


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
