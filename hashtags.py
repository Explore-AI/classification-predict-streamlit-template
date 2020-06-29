import streamlit as st
import nltk
# Visualisations
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import re

# function to collect hashtags


def hashtag_extract(data):
    """
    Function to extact hashtags.

    Parameter(s):
      data (obj): a dataframe object

    Returns:
    List of hashtags
    """
    hashtags = []
    # Loop over the words in the tweet
    for i in data:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags


def common_tags(class_list, name):
    """
    Function to plot top 10 common hashtags.

    Returns:
    Bar plot of common hashtags.
    """
    a = nltk.FreqDist(class_list)
    d = pd.DataFrame({'Hashtag': list(a.keys()),
                      'Count': list(a.values())})
    # selecting top 10 most frequent hashtags
    d = d.nlargest(columns="Count", n=10)
    plt.figure(figsize=(10, 12))
    ax = sns.barplot(data=d, x='Count', y='Hashtag')
    plt.xlabel('counts')
    plt.ylabel('Hashtags')
    plt.title('Top 10 Common Hashtags for' + ' ' + name)
    st.pyplot()


def extract_hash(df):

    HT_pro = hashtag_extract(df['message'][df['sentiment'] == 'Pro'])

    # extracting hashtags from anti climate change tweets
    HT_anti = hashtag_extract(df['message'][df['sentiment'] == 'Anti'])

    # extracting hashtags from neutral tweets
    HT_neutral = hashtag_extract(df['message'][df['sentiment'] == 'Neutral'])
    # unnesting list
    HT_pro = sum(HT_pro, [])
    HT_anti = sum(HT_anti, [])
    HT_neutral = sum(HT_neutral, [])
    dict = {'Pro': HT_pro,'Anti':HT_anti,'Neutral':HT_neutral}
    st.write("### **You can view the most popular Hashtags on Tweets about Climate Change using the filter below.**")
    options = st.multiselect('Select tweet category to visualize hashtags:', ['Pro', 'Anti', 'Neutral'], ['Pro'])
    for choice in options:
        st.subheader(f'{choice} Tweets Hashtags on climate change')
        common_tags(dict[choice], f'{choice} Climate Change Tweets')
