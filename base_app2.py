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
#loading clean df
clean_df=pd.read_csv("resources/clean_df.csv")

def main():

    #title and subheader
    st.image(Image.open("images/logos.PNG"))
    #creating side menu
    options = ["About the app","Data insights","Data visualisation","Model perfomance","Classify tweets"]
    selection = st.sidebar.selectbox("Menu Options", options)
    #model Perfomance page
    if selection == "Model perfomance":
        st.title("Classification report")
        st.markdown("A classification report measure the quality of the predictions made by a classification algorithm.it indicates how many predictions are True and how many are False. The report also uses the True Positives(TP), False Positives(FP), True Negatives(TN) and False Negatives(FN) to show the main classification metrics, i.e precision, recall and f1-score on a per-class basis. These are the same concepts used in the confusion matrix above.")
        st.markdown("**Precision** : The ability of a classifier to not label an instance positive when it is actually negative. So it considers how                  accurate a classifier is in predicting positive cases.For each class it is defined as the ratio of true positives to the sum of true and false positives.")
        st.markdown("Precision = TP/(TP + FP)")
        st.markdown("**Recall** : The ability of a classifier to find all positive instances. It considers the fraction of positives that were                     correctly identified. For each class it is defined as the ratio of true positives to the sum of true positives and false negatives.")
        st.markdown("Recall = TP/(TP + FN)")
        st.markdown("**F1 Score** : A weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. As a rule                 of thumb, the weighted average of F1 should be used to compare classifier models")
        st.markdown("F1 Score = 2 x (Recall x Precision) / (Recall + Precision)")
        st.markdown(" To get a classification report of your model of interest, select the model:")
        report = pd.DataFrame({'report name': ['Logistic Regression', 'Naive Bayes','Support Vector Machine','Random Forest', 'K Nearest Neighbors','Neural network']})
        model_sel=st.selectbox('Select a model', report['report name'])
        if model_sel == 'Logistic Regression':
            st.markdown("**Classification Report from Logistic Regression Model**")
            st.image(Image.open("images/lr.png"))
        if model_sel =='Naive Bayes':
            st.markdown("**Classification Report from Naive Bayes Model**")
            st.image(Image.open("images/na.png"))
        if model_sel =='Support Vector Machine':
            st.markdown("**Classification Report from Support Vector Machine Model**")
            st.image(Image.open("images/svm.png"))
        if model_sel =='Random Forest':
            st.markdown("**Classification Report from Random Forest Model**")
            st.image(Image.open("images/rf.png"))
        if model_sel =='Neural network':
            st.markdown("**Classification Report from Neural Networks Model**")
            st.image(Image.open("images/nn.png"))
        if model_sel =='K Nearest Neighbors':
            st.markdown("**Classification Report from K Nearest Neighbors Model**")
            st.image(Image.open("images/knn.png"))
        st.markdown("The `F1 score` is recommended metric to use to decide on the best model to use.")

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
        st.subheader("Raw data and clean data for each class")
        x = st.slider('Choose the number of tweets to show')
        if st.checkbox('show Pro tweets'):
            st.markdown("raw data")
            st.write(raw[['sentiment','message']][raw['sentiment']==1].head(x))
            st.markdown("clean data")
            st.write(clean_df[['sentiment','message']][clean_df['sentiment']==1].head(x))
        if st.checkbox('show Anti tweets'):
            st.markdown("raw data")
            st.write(raw[['sentiment','message']][raw['sentiment']==-1].head(x))
            st.markdown("clean data")
            st.write(clean_df[['sentiment','message']][clean_df['sentiment']==-1].head(x))
        if st.checkbox('show Neutral tweets'):
            st.markdown("raw data")
            st.write(raw[['sentiment','message']][raw['sentiment']==0].head(x))
            st.markdown("clean data")
            st.write(clean_df[['sentiment','message']][clean_df['sentiment']==0].head(x))
        if st.checkbox('show News tweets'):
            st.markdown("raw data")
            st.write(raw[['sentiment','message']][raw['sentiment']==2].head(x))
            st.markdown("clean data")
            st.write(clean_df[['sentiment','message']][clean_df['sentiment']==2].head(x))
            



    if selection== "Classify tweets":
        st.title("Classify tweets")
        st.image(Image.open("images/tweets.PNG"))
        models = pd.DataFrame({'model name': ['Logistic Regression', 'Naive Bayes','Support Vector Machine','Random Forest', 'K Nearest Neighbors','Neural network']})
        model_sel=st.selectbox('Select a model', models['model name'])

        #building the Logistic Regression
        if model_sel == "Logistic Regression":
            st.markdown(open('resources/lr.md').read())
            st.info("Prediction with Logistic Regression Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Preparing text for the model
                vect_text = [cleanup(tweet_text)]
                # Load .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/Logistic_regression.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                st.success("Tweet is categorized as: {}".format(class_dict[prediction[0]]))
                st.success("Accuracy of this model is: 76%")
                st.success("F1 score of this model is: 0.80")           

        #building the Naive Bayes
        if model_sel == "Naive Bayes":
            st.markdown(open('resources/nb.md').read())
            st.info("Prediction with Naive Bayes Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Preparing text for the model
                vect_text = [cleanup(tweet_text)]
                # Load .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/Naive_bayes.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                st.success("Tweet is categorized as: {}".format(class_dict[prediction[0]]))
                st.success("Accuracy of this model is: 73%")
                st.success("F1 score of this model is: 0.75")

        #building the Linear SVM
        if model_sel == "Support Vector Machine":
            st.markdown(open('resources/svm.md').read())
            st.info("Prediction with Support Vector Machine (SVM) Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Preparing text for the model
                vect_text = [cleanup(tweet_text)]
                # Load .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/SVM.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                st.success("Tweet is categorized as: {}".format(class_dict[prediction[0]]))
                st.success("Accuracy of this model is: 78%")
                st.success("F1 score of this model is: 0.82")

        #building the Random Forest
        if model_sel == "Random Forest":
            st.markdown("## Random Forest")
            st.markdown("### The basic concept")
            st.markdown("This model fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.")
            st.markdown("- **Pro:** The predictive performance can compete with the best supervised learning algorithms.")
            st.markdown("- **Con:**  This model requires more computational resources and is less intuitive.")
            st.info("Prediction with Random Forest Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Preparing text for the model
                vect_text = [cleanup(tweet_text)]
                # Load .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/Random_forest.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                st.success("Tweet is categorized as: {}".format(class_dict[prediction[0]]))
                st.success("Text Categorized as: {}".format(class_dict[prediction[0]]))
                st.success("Accuracy of this model is: 69%")
                st.success("F1 score of this model is: 0.79")

        #building the KNN
        if model_sel == "K Nearest Neighbors":
            st.markdown("## Nearest Neighbors")
            st.markdown("### The basic concept")
            st.markdown(" The K Nearest Neighbors model is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions).")
            st.markdown("- **Pro:** The model is robust with regard to the search space; for instance, classes don't have to be linearly separable.")
            st.markdown("- **Con:**  This model does not learn anything from the training of data and simply uses the training data itself for classification.")
            st.info("Prediction with K Nearest Neighbors Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Preparing text for the model
                vect_text = [cleanup(tweet_text)]
                # Loading .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/KNN.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                st.success("Tweet is categorized as: {}".format(class_dict[prediction[0]]))
                st.success("Accuracy of this model is: 73%")
                st.success("F1 score of this model is: 0.75")

        #building the Neural network
        if model_sel == "Neural network":
            st.markdown("## Neural network")
            st.markdown("### The basic concept")
            st.markdown(" A neural network consists of units (neurons), arranged in layers, which convert an input vector into some output. Each unit takes an input, applies a (often nonlinear) function to it and then passes the output on to the next layer.")
            st.markdown("- **Pro:** The model requires less formal statistical training")
            st.markdown("- **Con:**  Neural networks usually require much more data than traditional machine learning algorithms.")
            st.info("Prediction with Neural_network Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Preparing text for the model
                vect_text = [cleanup(tweet_text)]
                # Loading .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/Neural_network.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                st.success("Tweet is categorized as: {}".format(class_dict[prediction[0]]))
                st.success("Accuracy of this model is: 73%")
                st.success("F1 score of this model is: 0.79")
    #building the Draw
    if selection == "Data visualisation":
        st.title("Data Visualisation")
        st.markdown("The graphs shown below illustrate how the data that was used to train the models looks like in terms of the distribution of the predictor variables and the predicted variable. To view a particular graph you can use the buttons below.")
        visualss= st.radio("Select a visual you would like to see",("Proportion of tweets in each class",'Number of urls per class','Kernel distribution of number of words per class','Overall distribution of sentiment scores','Distribution of the sentiment scores per class',"Most common words in Pro class","Most common words in Neutral class","Most common words in Anti class","Most common words in News class"))
        if visualss=="Proportion of tweets in each class":
            st.image(Image.open("images/pie.PNG"))
        elif visualss =='Kernel distribution of number of words per class':
            st.image(Image.open("images/kernel.PNG"))
        elif visualss== 'Number of urls per class':
            st.image(Image.open("images/urls.PNG"))
        elif visualss== 'Overall distribution of sentiment scores':   
            st.image(Image.open("images/distribution.PNG"))
        elif visualss== 'Distribution of the sentiment scores per class':
            st.image(Image.open("images/compound.PNG"))  
        elif visualss=="Most common words in Pro class":
            st.image(Image.open("images/pro.PNG"))
        elif visualss=="Most common words in Neutral class":
            st.image(Image.open("images/neutral.PNG"))
        elif visualss=="Most common words in Anti class":
            st.image(Image.open("images/anti.PNG"))
        elif visualss=="Most common words in News class":
            st.image(Image.open("images/news.PNG"))
       

# Required to let Streamlit instantiate our web app.

        #file = joblib.load(open(os.path.join("Common_words_pro"),"rb"))

# Required to let Streamlit instantiate our web app.

if __name__ == '__main__':
	main()
