# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image

import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')

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
    # Dictionary of contractions
    contractions = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"wasn't": "was not",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we'll":"we will",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
}
    short = {' BD ': ' Big Deal ',
 ' abt ':' about ',
 ' ab ': ' about ',
 ' fav ': ' favourite ',
 ' fab ': ' fabulous ',
 ' smh ': ' shaking my head ',
 ' u ': ' you ',
 ' c ': ' see ',
 ' anon ': ' anonymous ',
 ' ac ': ' aircon ',
 ' a/c ': ' aircon ',
 ' yo ':' year old ',
 ' n ':' and ',
 ' nd ':' and ',
 ' 2 ': ' to ',
 ' w ': ' with ',
 ' w/o ': ' without ',
 ' r ': ' are ',
 ' rip ':' rest in peace ',
 ' 4 ' : ' for ',
' BF ': ' Boyfriend ',
' BRB ': ' Be Right Back ',
' BTW ': ' By The Way ',
' GF ': ' Girlfriend ',
' HBD ': ' Happy Birthday ',
' JK ': ' Just Kidding ',
' K ':' Okay ',
' LMK ': ' Let Me Know ',
' LOL ': ' Laugh Out Loud ',
' HA ':' laugh ',
' MYOB ': ' Mind Your Own Business ',
' NBD ': ' No Big Deal ',
' NVM ': ' Nevermind ',
' Obv ':' Obviously ',
' Obvi ':' Obviously ',
' OMG ': ' Oh My God ',
' Pls ': ' Please ',
' Plz ': ' Please ',
' Q ': ' Question ', 
' QQ ': ' Quick Question ',
' RLY ': ' Really ',
' SRLSY ': ' Seriously ',
' TMI ': ' Too Much Information ',
' TY ': ' Thank You, ',
' TYVM ': ' Thank You Very Much ',
' YW ': ' You are Welcome ',
' FOMO ': ' Fear Of Missing Out ',
' FTFY ': ' Fixed This For You ',
' FTW ': ' For The Win ',
' FYA ': ' For Your Amusement ',
' FYE ': ' For Your Entertainment ',
' GTI ': ' Going Through It ',
' HTH ': ' Here to Help ',
' IRL ': ' In Real Life ',
' ICYMI ': ' In Case You Missed It ',
' ICYWW ': ' In Case You Were Wondering ',
' NBC ': ' Nobody Cares Though ',
' NTW ': ' Not To Worry ',
' OTD ': ' Of The Day ',
' OOTD ': ' Outfit Of The Day ',
' QOTD ': ' Quote of the Day ',
' FOTD ': ' Find Of the Day ',
' POIDH ': ' Pictures Or It Did ntt Happen ',
' YOLO ': ' You Only Live Once ',
' AFAIK ': ' As Far As I Know ',
' DGYF ': ' Dang Girl You Fine ',
' FWIW ': ' For What It is Worth ',
' IDC ': ' I Do not Care ',
' IDK ': ' I Do not Know ',
' IIRC ': ' If I Remember Correctly ',
' IMHO ': ' In My Honest Opinion ',
' IMO ': ' In My Opinion ',
' Jelly ': ' Jealous ',
' Jellz ': ' Jealous ',
' JSYK ': ' Just So You Know ',
' LMAO ': ' Laughing My Ass Off ',
' LMFAO ': ' Laughing My Fucking Ass Off ',
' NTS ': ' Note to Self ',
' ROFL ': ' Rolling On the Floor Laughing ',
' ROFLMAO ': ' Rolling On the Floor Laughing My Ass Off ',
' SMH ': ' Shaking My Head ',
' TBH ': ' To Be Honest ',
' TL;DR ':  ' Too Long; Did not Read ',
' TLDR ':  ' Too Long; Did not Read ',
' YGTR ': ' You Got That Right ',
' AYKMWTS ': ' Are You Kidding Me With This Shit ',
' BAMF ': ' Bad Ass Mother Fucker ',
' FFS ': ' For Fuck Sake ',
' FML ': ' Fuck My Life ',
' HYFR ': ' Hell Yeah Fucking Right ',
' IDGAF ': ' I Do not Give A Fuck ',
' NFW ': ' No Fucking Way ',
' PITA ': ' Pain In The Ass ',
' POS ': ' Piece of Shit ',
' SOL ': ' Shit Outta Luck ',
' STFU ': ' Shut the Fuck Up ',
' TF ': ' The Fuck ',
' WTF ': ' What The Fuck ',
' BFN ': ' Bye For Now ',
' CU ': ' See You ',
' IC ': ' I see ',
' CYL ': ' See You Later ',
' GTG ': ' Got to Go ',
' OMW ': ' On My Way ',
' RN ': ' Right Now ',
' TTYL ': ' Talk To You Later ',
' TYT ': ' Take Your time ',
' CC ': ' Carbon Copy ',
' CX ': ' Correction ',
' DM ': ' Direct Message ',
' FB ': ' Facebook ',
' FBF ': ' Flash-Back Friday ',
' FF ': ' Follow Friday ',
' HT ': ' Tipping my hat ',
' H/T ': ' Tipping my hat ',
' IG ': ' Instagram ',
' Insta ': ' Instagram ',
' MT ':' Modified Tweet ',
' OH ': ' Overheard ',
' PRT ': ' Partial Retweet ',
' RT ': ' Retweet ',
'rt ' : ' retweet ',
' SO ':' Shout Out ',
' S/O ': ' Shout Out ',
' TBT ': ' Throw-Back Thursday ',
' AWOL ': ' Away While Online ',
' BFF ': ' Best Friend Forever ',
' NSFW ': ' Not Safe For Work ',
' OG ': ' Original Gangster ',
' PSA ': ' Public Service Announcement ',
' PDA ': ' Public Display of Affection '}

    short = dict((key.lower(), value.lower()) for key,value in short.items())
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
    st.image(Image.open("images/logos.PNG"))
    #creating side menu
    options = ["About the app","Data insights","Data Visualisation","Model Perfomance","Classify tweets"]
    selection = st.sidebar.selectbox("Menu Options", options)
    #model Perfomance page
    if selection == "Model Perfomance":
        st.title("Classification report")
        st.markdown("A classification report measure the quality of the predictions made by a classification algorithm.it indicates how many predictions are True and how many are False. The report also uses the True Positives(TP), False Positives(FP), True Negatives(TN) and False Negatives(FN) to show the main classification metrics, i.e precision, recall and f1-score on a per-class basis. These are the same concepts used in the confusion matrix above.")
        st.markdown("**Precision** : The ability of a classifier to not label an instance positive when it is actually negative. So it considers how                  accurate a classifier is in predicting positive cases.For each class it is defined as the ratio of true positives to the sum of true and false positives:")
        st.markdown("precision = TP/(TP + FP)")
        st.markdown("**Recall** : The ability of a classifier to find all positive instances. It considers the fraction of positives that were                     correctly identified. For each class it is defined as the ratio of true positives to the sum of true positives and false negatives:")
        st.markdown("recall = TP/(TP + FN)")
        st.markdown("**F1 Score** : A weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. As a rule                 of thumb, the weighted average of F1 should be used to compare classifier models")
        st.markdown("F1 Score = 2 x (Recall x Precision) / (Recall + Precision)")
        st.markdown(" ")
        st.markdown("**Classification Report from Logistic Regression Model**")
        st.image(Image.open("images/lr.png"))
        #st.markdown("**Classification Report from Logistic Regression Model**")
        st.image(Image.open("images/na.png"))
        #st.markdown("**Classification Report from Logistic Regression Model**")
        st.image(Image.open("images/svm.png"))
        st.markdown("**Classification Report from Random Forest Model**")
        st.image(Image.open("images/rf.png"))
        st.markdown("**Classification Report from Neural Networks Model**")
        st.image(Image.open("images/nn.png"))
        st.markdown("The `F1 score` is our main metric that we use to decide on the best model to use.")

  #building the Information page
    if selection == "About the app":
        st.title("About the app")
        st.image(Image.open("images/earth.jpg"))
        st.markdown("While climate is a measure of the average weather over a period of time, climate change means a change in the measures of climate, such as temperature, rainfall, or wind, lasting for an extended period – decades or longer. Man made climate change is the phenomena that humans cause or contribute towards the change in climate.")
        st.video("images/global.mp4")
        st.markdown("This app is useful for classifying whether or not a person believes in climate change, based on their tweet(s). The app is created to help companies determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received. To determine how tweets percieve climate change, the app gives users a choice to use a model of their choice.")

        # You can read a markdown file from supporting resources folder



        
       

    if selection == "Data insights":
        st.title("Data insights")
        st.subheader("Descriptions")
        st.markdown("Table of variable description")
        st.markdown("![Image](https://github.com/Xenaschke/classification-predict-streamlit-template/blob/master/images/image1.PNG?raw=true.PNG)")
        st.markdown("Table of class description")
        st.markdown("![Image](https://github.com/Xenaschke/classification-predict-streamlit-template/blob/master/images/image2.PNG?raw=true.PNG)")
        st.subheader("Raw data")
        if st.checkbox('Show'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page
        st.subheader("Raw data for each class")
        x = st.slider('Choose the number of tweets to show')
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
        models = pd.DataFrame({'model name': ['Logistic Regression', 'Naive Bayes','Linear SVM','Random Forest', 'K Nearest Neighbors','Neural_network']})
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
<<<<<<< HEAD

=======
                st.success("Accuracy of this model is: 76%")
>>>>>>> a3a1e9e9df53ec7567fe06618476c0e57ca6fc7a

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
<<<<<<< HEAD

=======
                st.success("Accuracy of this model is: 73%")
                
>>>>>>> a3a1e9e9df53ec7567fe06618476c0e57ca6fc7a

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
<<<<<<< HEAD

=======
                st.success("Accuracy of this model is: 78%")
>>>>>>> a3a1e9e9df53ec7567fe06618476c0e57ca6fc7a

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
<<<<<<< HEAD

=======
                st.success("Accuracy of this model is: 69%")
>>>>>>> a3a1e9e9df53ec7567fe06618476c0e57ca6fc7a

        #building the KNN
        if model_sel == "K Nearest Neighbors":
            st.info("Prediction with K Nearest Neighbors Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Preparing text for the model
                vect_text = [cleanup(tweet_text)]
                # Loading .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/KNN.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                st.success("Text Categorized as: {}".format(class_dict[prediction[0]]))
<<<<<<< HEAD

=======
                st.success("Accuracy of this model is: 73%")
<<<<<<< HEAD
>>>>>>> a3a1e9e9df53ec7567fe06618476c0e57ca6fc7a
=======
        #building the KNN
        if model_sel == "Neural_network":
            st.info("Prediction with Neural_network Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Preparing text for the model
                vect_text = [cleanup(tweet_text)]
                # Loading .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/Neural_network.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                st.success("Text Categorized as: {}".format(class_dict[prediction[0]]))
                st.success("Accuracy of this model is: 73%")                
>>>>>>> dev

    #building the Draw
    if selection == "Data Visualisation":
        st.title("Data Visualisation")
        visualss= st.radio("Select a visual you would like to see",("A graph of number of tweets per class","A pie chart of proportion of tweets per class","Graphs of distribution of tweets sentiment scores"))
        if visualss=="A graph of number of tweets per class":
            plt.figure(figsize=(8.5,5))
            raw['sentiment'].replace({-1: 'Anti',0:'Neutral',1:'Pro',2:'News'}).value_counts().plot(kind='bar',figsize=(8.5,5), color="ForestGreen")
            plt.xlabel('Sentiment class', fontsize = 10)
            plt.xticks(rotation='horizontal')
            plt.ylabel('Number of tweets', fontsize = 10)
            plt.figtext(0.12, 0.00000000001, '', horizontalalignment='left', fontsize = 14,style='italic')
            st.pyplot()
        elif visualss == "A pie chart of proportion of tweets per class":
            plt.figure(figsize=(11,11))
            names = ['Pro','News','Neutral','Anti']
            perc = raw['sentiment'].replace({-1: 'Anti',0:'Neutral',1:'Pro',2:'News'}).value_counts()
            perc.name = ''
            perc.plot(kind='pie', labels=names, autopct='%1.1f%%')
            plt.figtext(0.12, 0.1, '', horizontalalignment='left',fontsize = 14,style='italic')
            plt.legend(raw['sentiment'].replace({-1: 'Anti: Does not believe in manmade climate change',
                                                      0:'Neutral: Neither believes nor refutes manmade climate change',
                                                      1:'Pro:Believe in manmade climate change',2:'News: Factual News about climate change'}), bbox_to_anchor=(2,0.7), loc="right")
            st.pyplot()
        elif visualss== "Graphs of distribution of tweets sentiment scores":
            sid = SentimentIntensityAnalyzer()
            df_analyse = raw.copy()
            df_news = df_analyse[df_analyse['sentiment']==2] #extract all news and separate them from positive,neut and neg
            df_analyse = df_analyse[df_analyse['sentiment'] != 2]
            df_analyse['compound']  =  df_analyse['message'].apply(lambda x: sid.polarity_scores(x)['compound'])
            df_analyse['comp_score'] = df_analyse['compound'].apply(lambda c: 'pos' if c >0 else 'neg' if c<0 else 'neu')
            
            fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 4))
            plt.figtext(.51,.95, 'Distribution of the tweets sentiment scores\n', fontsize=20, ha='center',fontweight='bold')

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

<<<<<<< HEAD
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

=======
# Required to let Streamlit instantiate our web app.

        #file = joblib.load(open(os.path.join("Common_words_pro"),"rb"))
        
        
>>>>>>> a3a1e9e9df53ec7567fe06618476c0e57ca6fc7a
# Required to let Streamlit instantiate our web app.  

if __name__ == '__main__':
	main()
