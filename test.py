# Streamlit dependencies
import streamlit as st
import joblib,os 


# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer


#import classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud

st.title('Classifier example')

st.write('''
# Explore different classifiers
which one is the best?
''')

#different wigets and assign a variabble with different datasets
dataset_name = st.sidebar.selectbox("Select Dataset",("train","test")) 
#st.write(dataset_name)

classifier_name=st.sidebar.selectbox("select classifier", ("KNN","SVN", "Random Forest"))

#define a function

if dataset_name=="train":
    data =  pd.read_csv('data/train.csv')

    # Create a variable to store the text
    data = data[data['sentiment'] == data['sentiment'].unique]
    cv = CountVectorizer()
    cv_data = cv.fit_transform(data.message)
    dict_ = {k:v for k,v in sorted(cv.vocabulary_.items(),key = lambda item: item[1],reverse =True)}
    word_list = [word for word in dict_.keys()][:amount]
    message = " ".join(word for word in word_list)
    # Instantiate wordcloud object
    word_cloud = WordCloud(collocations =False,
                          background_color = 'black',
                          width=400, 
                          height=300, 
                          contour_width=2, 
                          contour_color='steelblue')
    # Create Plot
    size = data.sentiment.nunique()
    amount = 500  #Change this number to reduce or increase the amount of words plotted 
    plot = list(data.sentiment.unique())
    fig = plt.figure(figsize=(50, 50 * size // 2))
    for index, column in enumerate(plot):
        ax = fig.add_subplot(size, 2, index + 1)
        wordcloud = word_cloud(data, column, amount)
        ax.imshow(wordcloud)
        plt.title('{} Most Frequent Words {} sentiment'.format(amount,column), size = 40, pad =15)
        ax.axis('off')                      

    X = data.message  
    y = data.sentiment

else: 

    data = pd.read_csv('data/test_with_no_labels.csv')
    #X=data.message
        




'''def add_param(clf_name):
    params = dict()
    if clf_name == 'KNN':
        K =  st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clf_name == 'SVN': 
        C =  st.sidebar.slider('c', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params
params = add_param(classifier_name)


#create the actual classifier
def get_classifier(clf_name, params):
    if clf_name == 'KNN':
        clf= KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'SVN': 
        clf = SVC(C=params['C'])
    else:
        
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                    max_depth=params['max_depth'], random_state=123)

    return clf

clf = get_classifier(classifier_name,params)

#classifications

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=123)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write("clasifier = {classifier_name}")
st.write("accuracy={acc}")



st.write("shape of dataset", X.shape)
st.write("number of classes", len(np.unique(y))) 

'''
