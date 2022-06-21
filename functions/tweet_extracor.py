# importing neccessary modules and packages
import pandas as pd
import csv
import re 
import string
import preprocessor as p
 
consumer_key = 'Rs5p7znGjoEWNC5PmkxwPENC1'
consumer_secret = 'e8tAkkfGaMppRnaUJnCaWmfr9x7zXOmKrG2NkGv3nUYrShUqDB'
# access_key= <enter key>
# access_secret = <enter key>
 
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_key, access_secret)
 
# api = tweepy.API(auth,wait_on_rate_limit=True)
 
csvFile = open('messages', 'a')
csvWriter = csv.writer(csvFile)
 
search_words = "#"      # enter your words
new_search = search_words + " -filter:retweets"
 
# for tweet in tweepy.Cursor(api.search,q=new_search,count=100,
                        #    lang="en",
                        #    since_id=0).items():
    # csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8'),tweet.user.screen_name.encode('utf-8'), tweet.user.location.encode('utf-8')])