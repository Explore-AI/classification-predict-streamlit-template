#streamlit dependencies
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nlppreprocess import NLP as nlp

def cleaner(line):

    '''This function takes raw text as input, cleans it by removing any noise added to it.
       This includes: punctuation, "#", "@" , digits, ...
       The function will also find the root form of each word, and return the clean tweet'''

    line = re.sub(r'^RT ','', re.sub(r'https://t.co/\w+', '', line).strip()) 
    punctuation = re.compile("[.;:!\'’‘“”?,\"()\[\]]")
    tweet = punctuation.sub("", line.lower()) 
    nlp_for_stopwords = nlp(replace_words=True, remove_stopwords=True, remove_numbers=True, remove_punctuations=False) 
    tweet = nlp_for_stopwords.process(tweet)
    tweet = tweet.split()  
    pos = pos_tag(tweet)
    lemmatizer = WordNetLemmatizer()
    tweet = ' '.join([lemmatizer.lemmatize(word, po[0].lower()) if po[0].lower() in ['n', 'r', 'v', 'a'] else word for word, po in pos])

    return tweet