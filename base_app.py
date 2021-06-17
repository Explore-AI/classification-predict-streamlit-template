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
import re
import string
import inflect
import unicodedata
from nltk.corpus import stopwords

# Vectorizer
news_vectorizer = open("resources/vectoriser.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# preprocessing the raw data

         # first function
def remove_URL(words):
    """This function take a string as an input and removes any url that are present in that string"""
    
    return re.sub(r"http\S+", "", words)

raw['message'] = raw['message'].apply(remove_URL)

       # third function
def remove_punctuations(tokenized_words):
    """This function take a string/list as an input and removes all the punctuations"""
    words = str.maketrans('', '', string.punctuation)
    return tokenized_words.translate(words)

def replace_slang(text):       
    return ' '.join([word.replace('rt', '') for word in text.split()])

def remove_non_ascii(tokenized_words):
    """Remove non-ASCII characters from list of tokenized words"""

    return ''.join([unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in tokenized_words])

def to_lowercase(tokenized_words):
    """Convert all characters to lowercase from list of tokenized words"""
    
    return ''.join([word.lower() for word in tokenized_words])

def replace_numbers(tokenized_words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    inflector = inflect.engine()

    return ''.join([inflector.number_to_words(word) if word.isdigit() else word for word in tokenized_words])

def remove_other(text):
    
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    return text

def remove_emoji(words):
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"
                u"\U0001F300-\U0001F5FF"
                u"\U0001F680-\U0001F6FF"
                u"\U0001F1E0-\U0001F1FF"
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', words)

def clean_text_data(tokenized_words):
    """Clean each word from list of tokenized words"""
    
    tokenized_words = remove_non_ascii(tokenized_words)
    tokenized_words = remove_emoji(tokenized_words)
    tokenized_words = replace_numbers(tokenized_words)
    tokenized_words = to_lowercase(tokenized_words)
    tokenized_words = remove_punctuations(tokenized_words)
    tokenized_words = remove_other(tokenized_words)
    tokenized_words = replace_slang(tokenized_words)
    
    return tokenized_words

raw['message'] = raw['message'].apply(clean_text_data)

    #fourth function
def remove_stopwords(text):
    """Remove stop words from list of tokenized words"""
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

raw['message'] = raw['message'].apply(remove_stopwords)



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification by AM1")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
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
			predictor = joblib.load(open(os.path.join("resources/trained.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
