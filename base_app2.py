# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image

# Data dependencies
# Packages for data analysis
import numpy as np
import pandas as pd

# Packages for visualisations
import matplotlib.pyplot as plt
import seaborn as sns

# Packages for preprocessing
import re
from nltk import word_tokenize
import emoji
from ftfy import fix_text
from nltk.stem.snowball import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')

# Packages for saving models
import pickle

###########################################################################
# Functions, dictionaries and lists used to prepare the input for modeling
###########################################################################

# Function to extract sentiment
def sentiment_score(text):
    """ A function that determines the sentiment of a text string.

        Parameters
        ----------
        text: Text string.

        Returns
        -------
        sentiment:  String indicating the sentiment of the input string.
    """

    sid = SentimentIntensityAnalyzer()
    s = sid.polarity_scores(text)['compound']
    if s<-0.05:
        sentiment='negative'
    elif s>0.05:
        sentiment='positive'
    else:
        sentiment='neutral'

    return sentiment

# Load list of unique news related handles
with open('Lists_and_dictionaries/news_file.pkl', 'rb') as file:
    news = pickle.load(file)

# Read in created hashtag text file and create a hashtags dictionary
# Keys
hash_file = [line.rstrip('\n') for line in open('Lists_and_dictionaries/hash_file.txt')]
hash_file = [i.center(len(i)+2) for i in hash_file]
# Values
hash_file_clean = [line.rstrip('\n') for line in open('Lists_and_dictionaries/hash_file_clean.txt')]
hash_file_clean = [i.center(len(i)+2) for i in hash_file_clean]

hashtags = {hash_file[i]: hash_file_clean[i] for i in range(len(hash_file))}
hashtags.update({'todayinmaker ':'today in maker'})#this is added to differentiate it from ' todayinmaker ' because this 1 occurs at start of tweet

# Function to substitute hastags with separated words
def expand_hashtags(df,column_name):
    """ A funtion that expands the hashtag words into separate words.

        Parameters
        ----------
        df:          Dataframe containing the text column to be transformed.
        column_name: Name of the column containing the text data.

        Returns
        -------
        df:  Dataframe containg the updated text column

        Example
        -------
        #iamgreat returns 'i am great'
    """

    df[column_name] = df[column_name].str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r"[#]",'',x))
    for word in hashtags.keys():
            df[column_name] = df[column_name].apply(lambda x: re.sub(word,hashtags[word],x))
    return df

# Import dictionary of contractions
with open('Lists_and_dictionaries/contractions_dict.pkl', 'rb') as file:
    contractions = pickle.load(file)

# Import dictionary of contractions
with open('Lists_and_dictionaries/short_dict.pkl', 'rb') as file:
    short = pickle.load(file)

# Cleaning Function
def cleanup(raw):
    """ A function that 'cleans' tweet data. The text gets modified by:
        - being lower cased,
        - removing urls,
        - removing bad unicode,
        - replacing emojis with words,
        - removing twitter non news related handles,
        - removing punctuation,
        - removing vowels repeated at least 3 times,
        - replacing sequences of 'h' and 'a', as well as 'lol' with 'laugh',
        - adding sentiment

        Parameters
        ----------
        raw: Text string.

        Returns
        -------
        raw:  Modified clean string
    """

    # Convert to lowercase
    raw = raw.lower()

    # Fix strange characters
    raw = fix_text(raw)

    # Substitute hastags with separated words
    for w in hashtags.keys():
        raw = re.sub(w,hashtags[w]+' ',re.sub(r"#",'',raw))

    # Replace contracted words with full word
    raw = ' '.join([contractions[w.lower()] if w.lower() in contractions.keys() else w for w in raw.split()])

    # Replacing shortened words with full words
    for w in short.keys():
        raw = re.sub(w,short[w],raw)

    # Removing urls
    raw = re.sub(r'https\S+','url',raw)
    raw = re.sub(r'www\S+', 'url',raw)

    # Replace emojis with their word meaning
    raw = emoji.demojize(raw)

    # Remove twitter non news related handles
    raw = ' '.join([y for y in raw.split() if y not in [x for x in re.findall(r'@[\w]*',raw) if x not in news]])

    # Add sentiment
    raw = raw + ' ' + sentiment_score(raw)

    # Remove punctuation
    raw = re.sub(r"[^A-Za-z ]*",'',raw)

    # Remove vowels repeated at least 3 times ex. Coooool > Cool
    raw = re.sub(r'([aeiou])\1+', r'\1\1', raw)

    # Replace sequence of 'h' and 'a', as well as 'lol' with 'laugh'
    raw = re.sub(r'ha([ha])*', r'laugh', raw)
    raw = re.sub(r'he([he])*', r'laugh', raw)
    raw = re.sub(r"lol([ol])*", r'laugh', raw)
    raw = re.sub(r"lo([o])*l", r'laugh', raw)

    return raw

###########################################################################
###########################################################################

# Stemming and tokenization class
class StemAndTokenize:
    def __init__(self):
        self.ss = SnowballStemmer('english')
    def __call__(self, doc):
        return [self.ss.stem(t) for t in word_tokenize(doc)]

# Dictionary of classes and their hastags
class_dict = {-1:'Anti',0:'Neutral',1:'Pro',2:'News'}


# Load your raw data
raw = pd.read_csv("resources/train.csv")
def main():

    #title and subheader
    st.markdown("![Image of Yaktocat](https://github.com/Xenaschke/classification-predict-streamlit-template/blob/master/images/logos.PNG?raw=true.PNG)")
    #creating side menu
    options = ["About the app","Data insights","Data Visualisation","Classify tweets","Model Perfomance"]
    selection = st.sidebar.selectbox("Menu Options", options)
    #model Perfomance page
    if selection == "Model Perfomance":
        st.title("Classification report")
        st.markdown("A classification report measure the quality of the predictions made by a classification algorithm.it indicates how many predictions are True and how many are False. The report also uses the True Positives(TP), False Positives(FP), True Negatives(TN) and False Negatives(FN) to show the main classification metrics,i.e precision, recall and f1-score on a per-class basis. These are the same concepts used in the confusion matrix above.")
        st.markdown("**Precision** : The ability of a classifier to not label an instance positive when it is actually negative. So it considers how                  accurate a classifier is in predicting positive cases.For each class it is defined as the ratio of true positives to the sum of true and false positives:")
        st.markdown("precision = TP/(TP + FP)")
        st.markdown("**Recall** : The ability of a classifier to find all positive instances. It considers the fraction of positives that were                     correctly identified. For each class it is defined as the ratio of true positives to the sum of true positives and false negatives:")
        st.markdown("recall = TP/(TP + FN)")
        st.markdown("**F1 Score** : A weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. As a rule                 of thumb, the weighted average of F1 should be used to compare classifier models")
        st.markdown("F1 Score = 2 x (Recall x Precision) / (Recall + Precision)")
        st.markdown(" ")
        st.markdown("**Classification Report from Logistic Regression Model**")
        st.image(Image.open("images/lr.PNG"))
        st.image(Image.open("images/na.PNG"))
        st.image(Image.open("images/svm.PNG"))
        st.image(Image.open("images/rf.PNG"))
        st.image(Image.open("images/knn.PNG"))
        st.markdown("The `F1 score` is our main metric that we use to decide on the best model to use.")

  #building the Information page
    if selection == "About the app":
        st.title("About the app")
        st.markdown("![Image of Yaktocat](https://abcsplash-bc-a.akamaized.net/4477599164001/201604/4477599164001_4864948520001_4863149671001-vs.jpg?pubId=4477599164001.jpg)")
        st.markdown("While climate is a measure of the average weather over a period of time, climate change means a change in the measures of climate, such as temperature, rainfall, or wind, lasting for an extended period – decades or longer. Man made climate change is the phenomena that humans cause or contribute towards the change in climate.")
        st.markdown("This app is useful for classifying whether or not a person believes in climate change, based on their tweet(s). The app is created to help companies determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received. To determine how tweets percieve climate change, the app gives users a choice to use a model of their choice.")

        # You can read a markdown file from supporting resources folder



        st.video("images/global.mp4")
       

    if selection == "Data insights":
        st.title("Data insights")
        st.markdown("Table of variable description")
        st.markdown("![Image](https://github.com/Xenaschke/classification-predict-streamlit-template/blob/master/images/image1.PNG?raw=true.PNG)")
        st.markdown("Table of class description")
        st.markdown("![Image](https://github.com/Xenaschke/classification-predict-streamlit-template/blob/master/images/image2.PNG?raw=true.PNG)")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page
        x = st.slider('number of tweets')
        if st.checkbox('show Pro tweets'):
            st.write(raw[['sentiment','message']][raw['sentiment']==1].head(x))
        if st.checkbox('show Anti tweets'):
            st.write(raw[['sentiment','message']][raw['sentiment']==-1].head(x))
        if st.checkbox('show Neutral tweets'):
            st.write(raw[['sentiment','message']][raw['sentiment']==0].head(x))
        if st.checkbox('show News tweets'):
            st.write(raw[['sentiment','message']][raw['sentiment']==2].head(x))
        


    if selection== "Classify tweets":
        st.title("Classify tweets")
        st.markdown("![Image of Yaktocat](https://github.com/Xenaschke/classification-predict-streamlit-template/blob/master/images/tweets.PNG?raw=true.PNG)")
        models = pd.DataFrame({'model name': ['Logistic Regression', 'Naive Bayes','Linear SVM','Random Forest', 'K Nearest Neighbors']})
        model_sel=st.selectbox('Select a model', models['model name'])

        #building the Logistic Regression
        if model_sel == "Logistic Regression":
            st.info("Prediction with Logistic Regression Model")
            #st.markdown("
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Preparing text for the model
                vect_text = [cleanup(tweet_text)]
                # Load .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/Logistic_regression.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                st.success("Text Categorized as: {}".format(class_dict[prediction[0]]))
                # if prediction==-1:
                #     st.success("Anti")
                # elif prediction==0:
                #     st.success("Neutral")
                # elif prediction == 1:
                #     st.success("Pro")
                # else:
                #     st.success("News")

        #building the Naive Bayes
        if model_sel == "Naive Bayes":
            st.info("Prediction with Naive Bayes Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Preparing text for the model
                vect_text = [cleanup(tweet_text)]
                # Load .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/Naive_bayes.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                st.success("Text Categorized as: {}".format(class_dict[prediction[0]]))
                # if prediction==-1:
                #     st.success("Anti")
                # elif prediction==0:
                #     st.success("Neutral")
                # elif prediction == 1:
                #     st.success("Pro")
                # else:
                #     st.success("News")

        #building the Linear SVM
        if model_sel == "Linear SVM":
            st.info("Prediction with Linear SVM Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Preparing text for the model
                vect_text = [cleanup(tweet_text)]
                # Load .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/SVM.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                st.success("Text Categorized as: {}".format(class_dict[prediction[0]]))
                # if prediction==-1:
                #     st.success("Anti")
                # elif prediction==0:
                #     st.success("Neutral")
                # elif prediction == 1:
                #     st.success("Pro")
                # else:
                #     st.success("News")

        #building the Random Forest
        if model_sel == "Random Forest":
            st.info("Prediction with Random Forest Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Preparing text for the model
                vect_text = [cleanup(tweet_text)]
                # Load .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/Random_forest.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                st.success("Text Categorized as: {}".format(class_dict[prediction[0]]))
                # if prediction==-1:
                #     st.success("Anti")
                # elif prediction==0:
                #     st.success("Neutral")
                # elif prediction == 1:
                #     st.success("Pro")
                # else:
                #     st.success("News")

        #building the KNN
        if model_sel == "K Nearest Neighbors":
            st.info("Prediction with K Nearest Neighbors Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Preparing text for the model
                vect_text = [cleanup(tweet_text)]
                # Loading .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/lr.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                st.success("Text Categorized as: {}".format(class_dict[prediction[0]]))
                # if prediction==-1:
                #     st.success("Anti")
                # elif prediction==0:
                #     st.success("Neutral")
                # elif prediction == 1:
                #     st.success("Pro")
                # else:
                #     st.success("News")

    #building the Draw
    if selection == "Data Visualisation":
        st.title("Data Visualisation")
        if st.checkbox("show/hide"):
            plt.figure(figsize=(8.5,5))
            raw['sentiment'].replace({-1: 'Anti',0:'Neutral',1:'Pro',2:'News'}).value_counts().plot(kind='bar',figsize=(8.5,5), color='tan')
            plt.title('Number of types of comments')
            plt.xlabel('Comment type')
            plt.ylabel('Number of comments')
            st.pyplot()
            plt.figure(figsize=(8.5,5))
            sns.distplot(raw['sentiment'],color='g',kde_kws={'bw':0.1}, bins=100, hist_kws={'alpha': 0.4})
            plt.title('Distribution graph for different classes')
            st.pyplot()

            plt.figure(figsize=(10,10))
            names = ['Pro','News','Neutral','Anti']
            raw['sentiment'].replace({-1: 'Anti',0:'Neutral',1:'Pro',2:'News'}).value_counts().plot(kind='pie', labels=names, autopct='%1.1f%%')
            plt.title('Number of types of comments')
            st.pyplot()

        df_analyse = raw.copy()
        sid = SentimentIntensityAnalyzer()
        df_analyse['compound']  =  df_analyse['message'].apply(lambda x: sid.polarity_scores(x)['compound'])
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 4))
        plt.figtext(.51,.95, 'Distribution of the tweets sentiment\n', fontsize=20, ha='center',fontweight='bold')
        ax1.hist(df_analyse['compound'], bins=15, edgecolor='k',color='lightblue')
        plt.figtext(0.23, 0.06, 'sentiment score', horizontalalignment='left',fontsize = 12)
        fig.text(0.00001, 0.5, 'number of tweets in sentiment', va='center', rotation='vertical',fontsize=12)
        plt.figtext(0.02, 0.0001, 'figure 1: positive, negative and neutral sentiment', horizontalalignment='left',fontsize = 14,style='italic')

        bins = np.linspace(-1, 1, 30)
        ax2.hist([df_analyse['compound'][df_analyse['compound'] > 0], df_analyse['compound'][df_analyse['compound'] < 0]], bins, label=['Positive sentiment', 'Negative sentiment'])
        plt.xlabel('sentiment score',fontsize=12)
        ax2.legend(loc='upper right')
        plt.figtext(0.75, 0.0001, 'figure 2: positive and negative sentiment', horizontalalignment='right',fontsize = 14,style='italic')
        plt.tight_layout()
        st.pyplot()


# Required to let Streamlit instantiate our web app.

        #file = joblib.load(open(os.path.join("Common_words_pro"),"rb"))
        
        
# Required to let Streamlit instantiate our web app.  

if __name__ == '__main__':
	main()
