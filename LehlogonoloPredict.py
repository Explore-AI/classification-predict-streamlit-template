#!/usr/bin/env python
# coding: utf-8

# **Imports**

# In[63]:


import spacy
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
nlp = spacy.load('en_core_web_sm')


# **Load Data**

# In[64]:


data = pd.read_csv('train1.csv')


# **Data Cleaning**

# **1.1 Uppercase to lowercase**

# In[65]:


data['message_2'] = data['message'].str.lower() 


# **Unwanted Characters**

# In[66]:


data['message_3'] = data['message_2'].apply(lambda char: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", char)) 


# In[67]:


data['message_4'] = data['message_3'].apply(lambda char: re.sub(r"\d+", "", char))


# **Possessive Pronoun not sure of neccesity**

# In[68]:


data['message_5'] = data['message_4'].str.replace("'s", "")


# **Lemmetazation**

# In[69]:


nltk.download('punkt')
nltk.download('wordnet')


# In[70]:


wordnet_lemmatizer = WordNetLemmatizer()


# In[71]:


nrows = len(data)
lemmatized_text_list = []

for row in range(0, nrows):
    lemmatized_list = []
    text = data.loc[row]['message_4']
    text_words = text.split(" ")
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
    lemmatized_text = " ".join(lemmatized_list)
    lemmatized_text_list.append(lemmatized_text)


# In[72]:


data['message_6'] = lemmatized_text_list


# **Stop Words**

# In[73]:


nltk.download('stopwords')


# In[74]:


stop_words = list(stopwords.words('english'))


# In[75]:


stops = r"\b" + word + r"\b"


# In[76]:


data['message_7'] = data['message_6']
for stop_word in stop_words:
    stops_stopword = r"\b" + stop_word + r"\b"
    data['message_7'] = data['message_7'].str.replace(stops_stopword, '')


# In[77]:


data


# **Clean Data**

# In[ ]:


data = data.drop(['message','message_2','message_3','message_4','message_5','message_6'] , axis = 0)
data = data.rename(columns={'message_7': 'messages'})


# In[ ]:




