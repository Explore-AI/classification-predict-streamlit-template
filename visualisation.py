# Streamlit dependencies
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud

import pandas as pd

#import for imbalance
from sklearn.utils import resample

def word_cloud(df,class_no,class_name):
  """
  This function generates word cloud visualizations across different classes.

  Parameters:
    df (obj): Data frame.
    class_no (int): Class number
    class_name (obj): Class name

   Returns:
    word cloud visual
  """

  sentiment_class = ' '.join([text for text in df['message'][df['sentiment'] == class_no]])
  from wordcloud import WordCloud
  wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110,
                        background_color="white").generate(sentiment_class)

  plt.figure(figsize=(10, 7))
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.title('WordCloud for' + " " + class_name)
  plt.axis('off')
  st.pyplot()

def visualize_data(df):
    df = df.copy()

    st.write("### **You can view WordClouds of commonly used words per category using the filter below.**")
    options = st.multiselect('Select tweet category to visualize with Wordcloud:', ['Pro', 'Anti', 'Neutral', 'News'], ['Pro'])
    for choice in options:
        st.subheader(f'{choice} Tweets')
        word_cloud(df,choice,f'{choice} Tweets')


    st.subheader("Observations")

    st.write("""
        * Climate change seems to be the most frequently used word in all the tweet classes.
        * There is an overlap in frequest words among the classes, however, not much distinction can be drawn from the wordcloud.
        * Since WordCloud does not show how frequently a word appears, we will create a frequent words dictionary with top 20 counts for much better information extrapolation.
    """)


    df_majority = df[(df.sentiment=="Pro") |
                          (df.sentiment=="Neutral") |
                          (df.sentiment =="News")]
    df_minority = df[df.sentiment == "Anti"]

    #Upsample minority class
    df_minority_upsampled= resample(df_minority,replace= True,
                                n_samples= 4000, random_state =42) #sample with replacement

    #Combine majority class with upsampled minority class
    df_upsampled = pd.concat ([df_majority,
                              df_minority_upsampled])

    st.write("### **The barplot below shows the message distribution over the sentiments.**")

    dist_class = df['sentiment'].value_counts()
    fig, (ax1 )= plt.subplots(1, figsize=(8,4))
    sns.barplot(x=dist_class.index, y=dist_class, ax=ax1).set_title("Tweet message distribution over the sentiments")
    st.pyplot()

    #Display new class counts
    st.table(df_upsampled.sentiment.value_counts())
    st.subheader("Observations")
    st.write("* From the above diagram we can see that there are move people who are Pro Climate Change, and the least are those who are Anti Climate Change.")





"""
def plot_common_words(df, sentiment):
    df = df.copy()
    # Extracting tweets for each tweet class
    anti_tweets = [text for text in df['message']
                   [df['sentiment'] == sentiment]]

    # Value counts: top 20 most appearing words for each tweet class
    anti_series = pd.Series(' '.join(anti_tweets).split()).value_counts()[:20]

    plt.figure(figsize=(10, 7))
    anti_series.plot.bar()
    plt.xlabel('common words')
    plt.ylabel('value counts')
    plt.title(f'Top 20 Common words for {sentiment} Climate Change Tweets')
    st.pyplot()
"""

def common_words(df, class_no, class_name):
  """
  This is a function to extract top 20 comon words per class.

    Parameters:
    df (obj): Data frame.
    class_no (int): Class number
    class_name (obj): Class name

    Returns:
    Bar plot for the 20 most used words in the tweets.
    """
  name =[text for text in df['message'][df['sentiment'] == class_no]]
  series=pd.Series(' '.join(name).split()).value_counts()[:20]
  new_df=pd.DataFrame(data=series, columns=['count']).reset_index()

  plt.figure(figsize=(10, 7))
  ax=sns.barplot(x=new_df['count'],y=new_df['index'],data=new_df)
  plt.xlabel('value counts')
  plt.ylabel('common words')
  plt.title('Top 20 Common words for'+ ' '+class_name)
  st.pyplot()
