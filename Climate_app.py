import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import string
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
#emoji cloud
# from deepmoji import DeepMoji
import string
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
from sklearn.metrics import f1_score
from wordcloud import WordCloud
from collections import  Counter
from sklearn.feature_extraction.text import CountVectorizer
import emoji


warnings.filterwarnings("ignore", category=DeprecationWarning)
import nltk
#nltk.download('all')

nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))



 # Add a title
st.title('Climate Changes Twitter Sentiment')
# Add some text
st.markdown('Climate changes are crucial for us to observe on what is happening in the world.')
st.markdown("This work is divided into crucial sections:")
st.markdown("# 1. Preproccessing Data")
st.markdown('### Upload and view the data for this Project')
train =  pd.read_csv('train.csv')
test =  pd.read_csv('test.csv')

df = train.copy()
#Create a checkbox to provide a solution to show the dataframe
if st.checkbox('Show both the Train and Test dataframe'):
	st.write(train)
	st.write(test)

dataset_name = st.sidebar.selectbox('Select Dataset', ("Train", "Test"))
Classifier_name = st.sidebar.selectbox('Select Classifier', ("KNN", "SVM", "XGB", "Random Forest"))

# Create class distribution dataframe
sentiment_counts = df.groupby('sentiment').size().reset_index(name='counts')


if st.checkbox('Show sentiment counts'):
	st.write(sentiment_counts)

st.markdown(" ### Defining class Labels for each sentiment class")


def sentiment(df):
  
  sentiment = df['sentiment']
  sentiment_class = []

  for i in sentiment :
      if i == 1 :
          sentiment_class.append('Pro')
      elif i == 0 :
          sentiment_class.append('Neutral')
      elif i == -1 :
          sentiment_class.append('Anti')
      else :
          sentiment_class.append('News')

  df['sentiment'] = sentiment_class
    
  return df
df = sentiment(df)

if st.checkbox('Show sentiment as Pro, Anti, Neutral and News'):
	st.write(df)

# Create class distribution dataframe
sent_count = df.groupby('sentiment').size().reset_index(name='counts')

if st.checkbox('Group the sentiments into a new dataframe'):
	st.write(sent_count)

# Extract the hashtags from the tweets

def extract_hashtags(df):
  df['hashtags'] = df['message'].str.findall(r'#.*?(?=\s|$)')
  df['hashtags'] = df['hashtags'].apply(lambda x: np.nan if len(x)==0 else [x.lower() for x in x])
    
  return df

df = extract_hashtags(df)

if st.checkbox('Extract the hashtags from the tweets'):
	st.write(df)

st.markdown('### Emoji extraction of the train data for each message')

# Create a function for emoji extraction
def extract_emojis(s):
  return ''.join(c for c in s if c in emoji.UNICODE_EMOJI)

#extracting emojis on train data
df['emoji'] = df['message'].apply(extract_emojis)

if st.checkbox('Extract the emojis from the tweets'):
	st.write(df[df['emoji']!='']['emoji'])

# Create the function to extract the emojis from data
def extract_emojis(df):
  for char in df:
    if char in emoji.UNICODE_EMOJI:
      return True
    else:
      return False

df['emoji'] = df['message'].apply(extract_emojis)

if st.checkbox('Convert emojies on the dataframe to text '):
	st.write(df[df['emoji']==True])


#convert emojies on the dataframe to text 
def text_emoji(txt):
  emoji_converter = emoji.demojize(txt, delimiters=("", ""))
  return emoji_converter

st.markdown("### Converting emojies into Text")

#convert emojies on the dataframe to text from the train data
df['message'] = df['message'].apply(text_emoji)

if st.checkbox('Emojies on the dataframe to text from the train data'):
	st.write(df[df['emoji']==True])


# remove special characters, numbers, punctuations from train data
df['message'] = df['message'].str.replace("[^a-zA-Z#]", " ")

#removing short words from train data
df['message'] = df['message'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

#removing short words from train data
df['message'] = df['message'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

st.markdown("### Tweet Cleaning")


if st.checkbox('Cleaning the text of train data from the train data'):
	st.write(df['message'])

#Replace the word https with nothing: train
df['message'] = df['message'].str.replace('https', '')


# Remove Line breaks: train
df['message']=df['message'].replace('\n', ' ')
# st.write(df)
# Remove Line breaks: test
test['message']=test['message'].replace('\n', ' ')

if st.checkbox('After more cleaning of the text of train data from the train data'):
	st.write(df)

st.markdown('### Parts of Speach tagging')

#tokenizinging tweet train data
df['Tokenized_tweet'] = df['message'].apply(lambda x: x.split())

if st.checkbox('Tokenization'):
	st.write(df['Tokenized_tweet'])

#stop words Removal from train data
df['stopwords_removed'] = df['Tokenized_tweet'].apply(lambda x: [word for word in x if word not in STOPWORDS])

if st.checkbox('STOPWORDS'):
	st.write(df['stopwords_removed'])

#Speech Tagging with the train dataAdaBoostClassifier
df['pos_tags'] = df['stopwords_removed'].apply(nltk.tag.pos_tag)

if st.checkbox('Speech Tagging'):
	st.write(df['pos_tags'])

st.markdown("Exploratory Data Analysis EDA")

plt.figure(figsize=(12,6))
sns.countplot(x='sentiment',data=df, palette="Blues_d")
st.pyplot()
# # # chart_data = pd.DataFrame(
# # #     np.random.randn(20, 3),
# # #     columns=['a', 'b', 'c'])

# # # st.line_chart(chart_data)

# # # map_data = pd.DataFrame(
# # #     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
# # #     columns=['lat', 'lon'])

# # # st.map(map_data)

# # # Create a title
# # # st.title("First Streamlit Web Application")
# # # # adding markdown
# # # st.markdown("Markdown")
# # # # we can use # mark like we are using in jupyter notebook
# # # st.markdown("### Markdown line.")
# # # # Create a side bar
# # # dataset_name = st.sidebar.selectbox('Select DAtaset', ("Train", "Test"))
# # # Classifier_name = st.sidebar.selectbox('Select Classifier', ("KNN", "SVM", "XGB", "Random Forest"))

# # # def get_dataset('dataset_name'):
# # # st.text('To see the the Twitter dataframe you have to click below to checkbox:')
# # # import numpy as np
# # # df = pd.read_csv("train.csv")
# # # #Create a checkbox to provide a solution to show the dataframe
# # # if st.checkbox('Show dataframe'):
# # #     st.write(df)
# # #  use st.selectbox to choose from a series or a list.
# # # import numpy as np
# # # df = pd.read_csv("train.csv")
# # # option = st.selectbox('Which Club do you like best?',
# # #      df['sentiment'].unique())  
# # # 'You selected: ', option

# # ##############################################################################################################
# # # import streamlit as st
# # # # NLP Pkgs
# # # from textblob import TextBlob
# # # import pandas as pd 
# # # # Emoji
# # # import emoji

# # # # Web Scraping Pkg
# # # from bs4 import BeautifulSoup
# # # from urllib.request import urlopen

# # # # Fetch Text From Url
# # # @st.cache
# # # def get_text(raw_url):
# # # 	page = urlopen(raw_url)
# # # 	soup = BeautifulSoup(page)
# # # 	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
# # # 	return fetched_text




# # # def main():
# # # 	"""Sentiment Analysis Emoji App """

# # # 	st.title("Sentiment Analysis Emoji App")

# # # 	activities = ["Sentiment","Text Analysis on URL","About"]
# # # 	choice = st.sidebar.selectbox("Choice",activities)

# # # 	if choice == 'Sentiment':
# # # 		st.subheader("Sentiment Analysis")
# # # 		st.write(emoji.emojize('Everyone :red_heart: Streamlit ',use_aliases=True))
# # # 		raw_text = st.text_area("Enter Your Text","Type Here")
# # # 		if st.button("Analyze"):
# # # 			blob = TextBlob(raw_text)
# # # 			result = blob.sentiment.polarity
# # # 			if result > 0.0:
# # # 				custom_emoji = ':smile:'
# # # 				st.write(emoji.emojize(custom_emoji,use_aliases=True))
# # # 			elif result < 0.0:
# # # 				custom_emoji = ':disappointed:'
# # # 				st.write(emoji.emojize(custom_emoji,use_aliases=True))
# # # 			else:
# # # 				st.write(emoji.emojize(':expressionless:',use_aliases=True))
# # # 			st.info("Polarity Score is:: {}".format(result))
			
# # # 	if choice == 'Text Analysis on URL':
# # # 		st.subheader("Analysis on Text From URL")
# # # 		raw_url = st.text_input("Enter URL Here","Type here")
# # # 		text_preview_length = st.slider("Length to Preview",50,100)
# # # 		if st.button("Analyze"):
# # # 			if raw_url != "Type here":
# # # 				result = get_text(raw_url)
# # # 				blob = TextBlob(result)
# # # 				len_of_full_text = len(result)
# # # 				len_of_short_text = round(len(result)/text_preview_length)
# # # 				st.success("Length of Full Text::{}".format(len_of_full_text))
# # # 				st.success("Length of Short Text::{}".format(len_of_short_text))
# # # 				st.info(result[:len_of_short_text])
# # # 				c_sentences = [ sent for sent in blob.sentences ]
# # # 				c_sentiment = [sent.sentiment.polarity for sent in blob.sentences]
				
# # # 				new_df = pd.DataFrame(zip(c_sentences,c_sentiment),columns=['Sentence','Sentiment'])
# # # 				st.dataframe(new_df)

# # # 	if choice == 'About':
# # # 		st.subheader("About:Sentiment Analysis Emoji App")
# # # 		st.info("Built with Streamlit,Textblob and Emoji")
# # # 		st.text("Jesse E.Agbe(JCharis")
# # # 		st.text("Jesus Saves@JCharisTech")







# # # if __name__ == '__main__':
# # # 	main()

# # #########################################################################################################

# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import pydeck as pdk
# # import plotly.express as px

# # DATE_TIME = "date/time"
# # DATA_URL = (
# #     "/path/to/Motor_Vehicle_Collisions_-_Crashes.csv"
# # )

# # st.title("Motor Vehicle Collisions in New York City")
# # st.markdown("This application is a Streamlit dashboard that can be used "
# #             "to analyze motor vehicle collisions in NYC 🗽💥🚗")


# # @st.cache(persist=True)
# # def load_data(nrows):
# #     data = pd.read_csv(DATA_URL, nrows=nrows, parse_dates=[['CRASH_DATE', 'CRASH_TIME']])
# #     data.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
# #     lowercase = lambda x: str(x).lower()
# #     data.rename(lowercase, axis="columns", inplace=True)
# #     data.rename(columns={"crash_date_crash_time": "date/time"}, inplace=True)
# #     #data = data[['date/time', 'latitude', 'longitude']]
# #     return data

# # data = load_data(15000)
# # data[['latitude','longitude']].to_csv('lat_long.csv', index=False)


# # st.header("Where are the most people injured in NYC?")
# # injured_people = st.slider("Number of persons injured in vehicle collisions", 0, 19)
# # st.map(data.query("injured_persons >= @injured_people")[["latitude", "longitude"]].dropna(how="any"))

# # st.header("How many collisions occur during a given time of day?")
# # hour = st.slider("Hour to look at", 0, 23)
# # original_data = data
# # data = data[data[DATE_TIME].dt.hour == hour]
# # st.markdown("Vehicle collisions between %i:00 and %i:00" % (hour, (hour + 1) % 24))

# # midpoint = (np.average(data["latitude"]), np.average(data["longitude"]))
# # st.write(pdk.Deck(
# #     map_style="mapbox://styles/mapbox/light-v9",
# #     initial_view_state={
# #         "latitude": midpoint[0],
# #         "longitude": midpoint[1],
# #         "zoom": 11,
# #         "pitch": 50,
# #     },
# #     layers=[
# #         pdk.Layer(
# #         "HexagonLayer",
# #         data=data[['date/time', 'latitude', 'longitude']],
# #         get_position=["longitude", "latitude"],
# #         auto_highlight=True,
# #         radius=100,
# #         extruded=True,
# #         pickable=True,
# #         elevation_scale=4,
# #         elevation_range=[0, 1000],
# #         ),
# #     ],
# # ))
# # if st.checkbox("Show raw data", False):
# #     st.subheader("Raw data by minute between %i:00 and %i:00" % (hour, (hour + 1) % 24))
# #     st.write(data)

# # st.subheader("Breakdown by minute between %i:00 and %i:00" % (hour, (hour + 1) % 24))
# # filtered = data[
# #     (data[DATE_TIME].dt.hour >= hour) & (data[DATE_TIME].dt.hour < (hour + 1))
# # ]
# # hist = np.histogram(filtered[DATE_TIME].dt.minute, bins=60, range=(0, 60))[0]
# # chart_data = pd.DataFrame({"minute": range(60), "crashes": hist})

# # fig = px.bar(chart_data, x='minute', y='crashes', hover_data=['minute', 'crashes'], height=400)
# # st.write(fig)

# # st.header("Top 5 dangerous streets by affected class")
# # select = st.selectbox('Affected class', ['Pedestrians', 'Cyclists', 'Motorists'])

# # if select == 'Pedestrians':
# #     st.write(original_data.query("injured_pedestrians >= 1")[["on_street_name", "injured_pedestrians"]].sort_values(by=['injured_pedestrians'], ascending=False).dropna(how="any")[:5])

# # elif select == 'Cyclists':
# #     st.write(original_data.query("injured_cyclists >= 1")[["on_street_name", "injured_cyclists"]].sort_values(by=['injured_cyclists'], ascending=False).dropna(how="any")[:5])

# # else:
# #     st.write(original_data.query("injured_motorists >= 1")[["on_street_name", "injured_motorists"]].sort_values(by=['injured_motorists'], ascending=False).dropna(how="any")[:5])

# ########################################################################################
import streamlit as st
import pandas as pd

# data = pd.read_csv("train.csv")
# # Create a list of possible values and multiselect menu with them in it.
# sentiment = data['sentiment'].unique()
# selected_sentiment = st.multiselect('Select sentiment', sentiment)

# # Mask to filter dataframe
# filter_sentiments = data['sentiment'].isin(selected_sentiment)

# data = data[filter_sentiments]
# ##############################################################################################

# st.title(project_title)

# # Graphing Function #####
# z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
# z = z_data.values
# sh_0, sh_1 = z.shape
# x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
# fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
# fig.update_layout(title='IRR', autosize=False,
#                   width=800, height=800,
#                   margin=dict(l=40, r=40, b=40, t=40))
# st.plotly_chart(fig)