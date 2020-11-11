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
# Streamlit dependencies
import streamlit as st
import joblib,os
import spacy_streamlit
import spacy
import nltk
import string
import re
import contractions
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from wordcloud import WordCloud, ImageColorGenerator
import warnings
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

warnings.filterwarnings(action = 'ignore') 

nlp = spacy.load('en')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file


# Load your raw data
raw = pd.read_csv("resources/train.csv")
retweet = 'RT'
import streamlit.components.v1 as components

def mentions(x):
    x = re.sub(r"(?:\@|https?\://)\S+", "", x)
    return x

def remove_punc(x):
    x = re.sub(r"([^A-Za-z0-9]+)"," ",x)
    return x

def StopWords():
    stop_words = set(stopwords.words('english'))
    return stop_words

st.cache(suppress_st_warning=True)
def word_count(train):
    cnt = Counter()
    for message in train['message'].values:
        for word in message:
            cnt[word] +=1
    return cnt.most_common(20)

st.cache(suppress_st_warning=True)
def data_cleaning(df):
    wnl = WordNetLemmatizer()
    df['message'] = df['message'].apply(mentions)
    df['message'] = df['message'].apply(lambda x: contractions.fix(x))
    df['message'] = df['message'].str.replace(r"http\S+|www.\S+", "", case=False)
    df['message'] = df['message'].map(lambda x: remove_punc(str(x)))
    df['message'] = df['message'].apply(word_tokenize)
    df['message'] = df['message'].apply(lambda x: [word for word in x if word not in retweet])
    df['message'] = df['message'].apply(lambda x : [word.lower() for word in x])
    df['message'] = df['message'].apply(lambda x: [word for word in x if word not in StopWords()])
    df['pos_tags'] = df['message'].apply(nltk.tag.pos_tag)
    df['wordnet_pos'] = df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    df['message'] = df['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
    return df

def pro_mostpopular(df):
    pro_popular = df[df['sentiment'] == 1]
    pro_pop = word_count(pro_popular)
    return pro_pop

def anti_mostpopular(df):
    anti_popular = df[df['sentiment']== -1]
    anti_pop = word_count(anti_popular)
    return anti_pop

def neutral_mostpopular(df):
    neutral = df[df['sentiment']==0]
    neutral_pop = word_count(neutral)
    return neutral_pop
def news_mostpopular(df):
    news = df[df['sentiment']==2]
    news_pop = word_count(news)
    return news_pop

st.cache(suppress_st_warning=True)
def popularwords_visualizer(popular, title):
    plt.bar(range(len(popular)), [val[1] for val in popular], align='center')
    plt.xticks(range(len(popular)), [val[0] for val in popular])
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylabel("# of times the word appeard")
    plt.xlabel("Most popular word (Descending)")
    st.pyplot()


def wordcloud_visualizer(df,title):
    words = df['message']
    allwords = []
    for wordlist in words:
        allwords += wordlist
        
    mostcommon = FreqDist(allwords).most_common(1000)
    wordcloud = WordCloud(width=1000, height=800, background_color='white').generate(str(mostcommon))
    fig = plt.figure(figsize=(30,10), facecolor='white')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(title)
    plt.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig)
# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Text Classification", "Information","About Predict","Exploratory Data Analysis","Word-clouds"]
	selection = st.sidebar.selectbox("Choose Option", options)
	
	if selection == "About Predict":
		markup(selection)
		components.html(
			"""
			<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">
			<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
			<script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ho+j7jyWK8fNQe+A12Hb8AhRq26LrZ/JpcUGGOn+Y7RsweNrtN/tE3MoK7ZeZDyx" crossorigin="anonymous"></script>
			<div id="accordion">
			<div class="card">
				<div class="card-header" id="headingOne">
				<h5 class="mb-0">
					<button class="btn btn-link" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
					Collapsible Group Item #1
					</button>
				</h5>
				</div>
				<div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
				<div class="card-body">
					Collapsible Group Item #1 content
				</div>
				</div>
			</div>
			<div class="card">
				<div class="card-header" id="headingTwo">
				<h5 class="mb-0">
					<button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
					Collapsible Group Item #2
					</button>
				</h5>
				</div>
				<div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
				<div class="card-body">
					Collapsible Group Item #2 content
				</div>
				</div>
			</div>
			</div>
			""",
			height=600,
		)
  
	
	# Building out the "Information" page
	if selection == "Information":
		markup(selection)
		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
       
	# Building out the predication page
	if selection == "Text Classification":
		markup(selection)
		# Creating a text box for user input
		
		tweet_text = st.text_area("Enter Text","Type Here")
		if st.button("Classify"):
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/SVCpipeline.pkl"),"rb"))
			prediction = predictor.predict([tweet_text])

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))
			models = ["en_core_web_sm",'en_core_web_md']
			#docx = nlp(text_raw)
			#spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)
			spacy_streamlit.visualize(models, tweet_text)
   

	if selection == "Exploratory Data Analysis":
		markup(selection)
		train = data_cleaning(raw)
		train_pop = word_count(train)
		popularwords_visualizer(train_pop," 20 Most Popular words in the training data set")
		#Gettting popular words for pro
		pro_popular = pro_mostpopular(train)
		#Visualizing the 20 most popular pro tweets
		popularwords_visualizer(pro_popular,"20 Most popular word in the pro class")
		#Getting the Anti most popular words
		anti_pop = anti_mostpopular(train)
		popularwords_visualizer(anti_pop,"20 Most popular word in the anti class")
		neutral_pop = neutral_mostpopular(train)
		popularwords_visualizer(neutral_pop,"20 Most popular word in the neutral class")
		news_pop = news_mostpopular(train)
		popularwords_visualizer(news_pop,"20 Most popular word in the news class")
	elif selection == "Word-clouds":
		markup("Word Clouds")
		clean_data = data_cleaning(raw)
		wordcloud_visualizer(clean_data,"Word Cloud For the training set")

		pro_data = clean_data[clean_data['sentiment']==1]
		wordcloud_visualizer(pro_data,"Word Cloud for pro sentiment class")
  
		anit_data = clean_data[clean_data['sentiment']==-1]
		wordcloud_visualizer(anit_data,"Word Cloud for anti sentiment class")

		neut_data = clean_data[clean_data['sentiment']==0]
		wordcloud_visualizer(neut_data,"Word Cloud for neutral sentiment class")
  
		news_data = clean_data[clean_data['sentiment']==-1]
		wordcloud_visualizer(news_data,"Word Cloud for news sentiment class")
  
  		
def markup(selection):
    html_temp = """<div style="background-color:{};padding:10px;border-radius:10px; margin-bottom:15px;"><h1 style="color:{};text-align:center;">"""+selection+"""</h1></div>"""
    st.markdown(html_temp, unsafe_allow_html=True)

#Getting the WordNet Parts of Speech
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN    

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
