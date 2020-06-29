# Native dependencies
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from model import sentiment_desc

stop = stopwords.words('english')

def clean_text(df):
    """
    This function cleans tweets on the 'messages' column.

    Parameters: 
    df (obj): Data frame.

    Returns:
    Dataframe with cleaned tweets.

    """
    # Lowering all the text
    df.message = df.message.apply(lambda x: " ".join(x.lower() for x in x.split()))
    # Removing mentions
    df.message = df.message.apply(lambda x: re.sub("(@[A-Za-z0-9]+)","",x))
    # Removing short words
    df.message = df.message.apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    # Removing https/http links
    df.message = df.message.apply(lambda x: re.sub('http[s]?://\S+', '', x))
    # Removing punctuation, with the exception of hashtags
    df.message = df.message.str.replace("[^a-zA-Z#]", " ")
    # Removing numbers
    df.message = df.message.apply(lambda x: re.sub('\d+','',x.lower()))
    # Replace sentiment with words
    df.sentiment = df.sentiment.apply(lambda x: sentiment_desc[str(x)])
    df['message'] = df['message'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    return df

"""
def clean_data(df):
	df = df.copy()
	links = 'http[s]?://\S+'
	hashtags = '#\w+\S'
	numbers = '\d+'
	mentions = '@\w+\S'
	symbols = '\W'
	retweets = '^rt\s'
	patterns = [links,hashtags,numbers,mentions,symbols, retweets]
	for pattern in patterns:
		df.message = df.message.apply(lambda x: re.sub(pattern,' ', x.lower()))


	df.message = df.message.apply(lambda x: " ".join(x for x in x.split() if x not in stopwords.words('english')))

	df.sentiment = df.sentiment.apply(lambda x: sentiment_desc[str(x)])

		
	return df
"""
def remove_stop_words(df):
	df = df.copy()
	stop = stopwords.words('english')
	df.message = df.message.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
	return df

def find_roots(df, st):
	df = df.copy()
	st = st
	df.message.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
	return df

def vectorize_features(df,v):
    df = df.copy()
    messages = list(df.message)
    vector_x = v.fit_transform(messages)
    return vector_x