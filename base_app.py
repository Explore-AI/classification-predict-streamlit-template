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

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

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

st.cache(suppress_st_warning=True)
def mentions(x):
    x = re.sub(r"(?:\@|https?\://)\S+", "", x)
    return x

st.cache(suppress_st_warning=True)
def remove_punc(x):
    x = re.sub(r"([^A-Za-z0-9]+)"," ",x)
    return x

st.cache(suppress_st_warning=True)
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

st.cache(suppress_st_warning=True)
def pro_mostpopular(df):
    pro_popular = df[df['sentiment'] == 1]
    pro_pop = word_count(pro_popular)
    return pro_pop

st.cache(suppress_st_warning=True)
def anti_mostpopular(df):
    anti_popular = df[df['sentiment']== -1]
    anti_pop = word_count(anti_popular)
    return anti_pop

st.cache(suppress_st_warning=True)
def neutral_mostpopular(df):
    neutral = df[df['sentiment']==0]
    neutral_pop = word_count(neutral)
    return neutral_pop

st.cache(suppress_st_warning=True)
def news_mostpopular(df):
    news = df[df['sentiment']==2]
    news_pop = word_count(news)
    return news_pop

st.cache(suppress_st_warning=True)
def popularwords_visualizer(data):
    news = news_mostpopular(data)
    pro = pro_mostpopular(data)
    anti=anti_mostpopular(data)
    neutral = neutral_mostpopular(data)
    
    #Creating the Subplots for Most Popular words
    fig, axs = plt.subplots(2, 2)
    
    plt.setp(axs[-1, :], xlabel='Most popular word (Descending)')
    plt.setp(axs[:, 0], ylabel='# of times the word appeard')
    axs[0,0].bar(range(len(news)),[val[1] for val in news],align='center')
    axs[0,0].set_xticks(range(len(news)), [val[0] for val in news])
    #axs[0.0].set_xticks(rotation=90)
    #axs[0.0].set_title("20 Most Popular words in the training data set")
    #axs[0.0].set_ylabel("# of times the word appeard")
    #axs[0.0].xlabel("Most popular word (Descending)")
    
    axs[0,1].bar(range(len(neutral)),[val[1] for val in neutral],align='center')
    axs[0,1].set_xticks(range(len(neutral)), [val[0] for val in neutral])
    #axs[0,1].set_xticks(rotation=90)
    #axs[0,1].set_title("20 Most Popular words in the training data set")
    #axs[0,1].set_ylabel("# of times the word appeard")
    #axs[0,1].xlabel("Most popular word (Descending)")
    
    axs[1,0].bar(range(len(pro)),[val[1] for val in pro],align='center')
    axs[1,0].set_xticks(range(len(pro)), [val[0] for val in pro])
    #axs[1,0].set_xticks(rotation=90)
    #axs[1,0].set_title("20 Most Popular words in the pro data set")
    #axs[1,0].set_ylabel("# of times the word appeard")
    #axs[1,0].xlabel("Most popular word (Descending)")
    
    axs[1,1].bar(range(len(anti)),[val[1] for val in anti],align='center')
    axs[1,1].set_xticks(range(len(anti)), [val[0] for val in anti])
    fig.tight_layout(pad=3)
    st.pyplot(fig)

st.cache(suppress_st_warning=True)
def wordcloud_visualizer(df,title):
    words = df['message']
    allwords = []
    for wordlist in words:
        allwords += wordlist
        
    mostcommon = FreqDist(allwords).most_common(1000)
    wordcloud = WordCloud(width=1000, height=800, background_color='white').generate(str(mostcommon))
    fig = plt.figure(figsize=(30,10), facecolor='white')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(title,fontsize='xx-large')
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
		models =["Support Vector Classifier","Logisitic Regression Classifier","Optimized Support Vector Classifier"]
		modelselection = st.selectbox("Choose Predictive Model",models)
		if modelselection =="Support Vector Classifier":
			tweet_text = st.text_area("Enter Text","Type Here")
			if st.button("Classify with SVC model"):
				predictor = joblib.load(open(os.path.join("resources/SVCpipeline.pkl"),"rb"))
				prediction = predictor.predict([tweet_text])
				st.success("Text Categorized as: {}".format(prediction))
		elif modelselection == "Logisitic Regression Classifier":
			prediction_text = st.text_area("Classify text with Logistic regression Classifer","Enter Text here")
			if st.button("Classify with Logistic regressor"):
				pred = joblib.load(open(os.path.join("resources/LogisticReg (1).pkl"),"rb"))
				predict = pred.predict([prediction_text])
				st.success("Text Categorized as: {}".format(predict))
		elif modelselection =="Optimized Support Vector Classifier":
			prediction_text = st.text_area("Classify text with Optimized Support Vector Classifier","Enter Text here")
			if st.button("Classify with Logistic regressor"):
				pred = joblib.load(open(os.path.join("resources/SVCGrid (1).pkl"),"rb"))
				predict = pred.predict([prediction_text])
				st.success("Text Categorized as: {}".format(predict))
			#models = ["en_core_web_sm",'en_core_web_md']
			#docx = nlp(tweet_text)
			#spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)
			#spacy_streamlit.visualize(models, tweet_text)
   

	if selection == "Exploratory Data Analysis":
		markup(selection)
		train = data_cleaning(raw)
		popularwords_visualizer(train)
  
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
  
st.cache(suppress_st_warning=True) 		
def markup(selection):
    html_temp = """<div style="background-color:{};padding:10px;border-radius:10px; margin-bottom:15px;"><h1 style="color:{};text-align:center;">"""+selection+"""</h1></div>"""
    st.markdown(html_temp, unsafe_allow_html=True)

#Getting the WordNet Parts of Speech
st.cache(suppress_st_warning=True)
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
