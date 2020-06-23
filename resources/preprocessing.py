"""

    Helper Functions.

    Author: JHB-EN2.

    Description: These helper functions are to be used to clean data for predictions
    and EDA 

"""
import spacy
nlp = spacy.load('en_core_web_sm')

class TweetPreprocessing:
    
    # def __init__(self):
    #     self.tweet = tweet
    
    def lemmatizeTweet(self, tweet):
        lemmatized_tweet = list()
        doc = nlp(tweet)
        for token in doc:
            lemmatized_tweet.append(token.lemma_)
        lemmatized_tweet = ' '.join(lemmatized_tweet)
        return lemmatized_tweet
    
    def tweetPreprocessor(self, tweet):
        clean_tweat = tweet.replace('\n', ' ')
        clean_tweat = re.sub('\w*\d\w*', ' ', clean_tweat)
        clean_tweat = re.sub(r'[::.,_()/\{}"?!¬¦ãÃâÂ¢\d]', '', clean_tweat)
        return clean_tweat
    
    def tweetTokenizer(self, tweet):
        tokenizer = TweetTokenizer(preserve_case = False, strip_handles = True)
        tokenized_tweet = tokenizer.tokenize(tweet)
        return tokenized_tweet
    
    def removeStopWords(self, tweet):
        clean_tweet = list()
        for token in tweet:
            if token not in stopwords.words('english'):
                if token.startswith('#') == False:
                    if len(token) > 1:
                        clean_tweet.append(token)
        return clean_tweet
    
# tweet = train_data.loc[0, 'message']
# tweet = tweet_preproc.lemmatizeTweet(tweet)
# tweet = tweet_preproc.tweetPreprocessor(tweet)
# tweet = tweet_preproc.tweetTokenizer(tweet)
# tweet = tweet_preproc.removeStopWords(tweet)
