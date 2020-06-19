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
#---------------------------------------------------------------
# Streamlit dependencies
import streamlit as st

# Data dependencies
import spacy
nlp = spacy.load('en_core_web_sm')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")


	
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "Exploratory Data Analysis", "Prediction" ]
	selection = st.sidebar.selectbox("Choose Option", options)


	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("The purpose of this web app is to demonstrate the functionality and performance \n of various models on tweet analysis and classification specifically for climate change.")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			#prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			prediction =text_clf.predict([tweet_text])
			st.success("Text Categorized as: {}".format(prediction))
	if selection == "Exploratory Data Analysis":
		# boxplots for word count analysis
		# create subplots
		fig, axs = plt.subplots(1, 4, sharey = True)

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
		st.pyplot()

		# the histogram plots for word counts
		fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = 'col', sharey = 'row')
		ax1.hist(y2)
		ax1.set_title('class 2')
		ax2.hist(y1, color = 'red')
		ax2.set_title('class 1')
		ax3.hist(y0, color = 'green')
		ax3.set_title('class 0')
		ax4.hist(y_1, color = 'purple')
		ax4.set_title('class -1')

		st.pyplot()

		#A bar graph comparing the frequency of each sentiment
		dftrain['sentiment'].value_counts().plot(kind = 'bar')
		plt.xticks(rotation='horizontal')
		plt.xlabel('Sentiments')
		plt.ylabel('Sentiment counts')
		plt.title('Sentiment Value Counts')
		st.pyplot()

		st.markdown('This graph shows that these four classes are imbalanced, which affects the accuracy of the model negatively. This shows that resambling is necessary before training a model with this data.')


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
