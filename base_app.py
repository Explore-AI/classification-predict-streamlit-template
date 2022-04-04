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
from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator
import matplotlib.pyplot as plt
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
df_train = pd.read_csv("data/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "Data Insights"]
	selection = st.sidebar.selectbox("Choose Option", options)

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
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

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
			st.success("Text Categorized as: {}".format(prediction))

	# Building the EDA Page
	if selection == "Data Insights":
		st.title("Data Insights")
		# Clean Data
		wn = WordNetLemmatizer()

		def text_preprocessing(review):
			""" Takes into text data and preprocesses before returning text data."""
			review = re.sub('[^a-zA-Z]', ' ', review)
			review = review.lower()
			review = review.split()
			review = [wn.lemmatize(word) for word in review if not word in stopwords.words('english')]
			review = ' '.join(review)
			return review

		# Clean data
		df_train['message']= df_train['message'].apply(text_preprocessing)
		df = df_train.copy()

		# Generate Wordcloud imaages
		# Create DataFrame for Each Sentiment
		df_sent1 = df[df['sentiment']==1]
		df_sent0 = df[df['sentiment']==0]
		df_sentneg = df[df['sentiment']==-1]
		df_sent2 = df[df['sentiment']==2]

		tweet_All = " ".join(review for review in df_train.message)
		tweet_sent0 = " ".join(review for review in df_sent0.message)
		tweet_sent1 = " ".join(review for review in df_sent1.message)
		tweet_sentneg = " ".join(review for review in df_sentneg.message)
		tweet_sent2 = " ".join(review for review in df_sent2.message)

		fig, ax = plt.subplots(5, 1, figsize  = (30,30))
		# Create and generate a word cloud image:
		wordcloud_ALL = WordCloud(max_font_size=50, max_words=100, background_color="black").generate(tweet_All)
		wordcloud_sent0 = WordCloud(max_font_size=50, max_words=100, background_color="black").generate(tweet_sent0)
		wordcloud_sent1 = WordCloud(max_font_size=50, max_words=100, background_color="black").generate(tweet_sent1)
		wordcloud_sentneg = WordCloud(max_font_size=50, max_words=100, background_color="black").generate(tweet_sentneg)
		wordcloud_sent2 = WordCloud(max_font_size=50, max_words=100, background_color="black").generate(tweet_sent2)

		# Display the generated image:
		ax[0].imshow(wordcloud_ALL, interpolation='bilinear')
		ax[0].set_title('All Tweets', fontsize=30)
		ax[0].axis('off')
		ax[1].imshow(wordcloud_sent0, interpolation='bilinear')
		ax[1].set_title('Neutral Tweets',fontsize=30)
		ax[1].axis('off')
		ax[2].imshow(wordcloud_sent1, interpolation='bilinear')
		ax[2].set_title('Pro Climate Change',fontsize=30)
		ax[2].axis('off')
		ax[3].imshow(wordcloud_sentneg, interpolation='bilinear')
		ax[3].set_title('Anti Climate Change',fontsize=30)
		ax[3].axis('off')
		ax[4].imshow(wordcloud_sent2, interpolation='bilinear')
		ax[4].set_title('News Tweets',fontsize=30)
		ax[4].axis('off')
		st.pyplot(fig)


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
