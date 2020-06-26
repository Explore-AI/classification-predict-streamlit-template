import streamlit as st 
import pandas as pd
import numpy as np
import joblib,os
from nltk import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
#nltk.download('vader_lexicon')

#stemming class 
class StemAndTokenize:
    def __init__(self):
        self.ss = SnowballStemmer('english')
    def __call__(self, doc):
        return [self.ss.stem(t) for t in word_tokenize(doc)]

# Load your raw data
raw = pd.read_csv("resources/train.csv")
def main():
    
    #title and subheader 
    st.markdown("![Image of Yaktocat](https://github.com/Xenaschke/classification-predict-streamlit-template/blob/master/images/logos.PNG?raw=true.PNG)")
    #creating side menu
    options = ["About the app","Data insights","Data Visualisation","Classify tweets"]
    selection = st.sidebar.selectbox("Menu Options", options)

    #building the Information page
    if selection == "About the app":
        st.title("About the app")
        st.markdown("![Image of Yaktocat](https://abcsplash-bc-a.akamaized.net/4477599164001/201604/4477599164001_4864948520001_4863149671001-vs.jpg?pubId=4477599164001.jpg)")
        st.markdown("While climate is a measure of the average weather over a period of time, climate change means a change in the measures of climate, such as temperature, rainfall, or wind, lasting for an extended period – decades or longer.")
        st.markdown("This app is useful for classifying whether or not a person believes in climate change, based on their tweet(s). The app is created to help companies determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received. To determine how tweets percieve climate change, the app gives users a choice to use a model of their choice.")
        # You can read a markdown file from supporting resources folder
        
            
    if selection == "Data insights":
        st.title("Data insights")
        st.markdown("Table of variable description")
        st.markdown("![Image](https://github.com/Xenaschke/classification-predict-streamlit-template/blob/master/images/image1.PNG?raw=true.PNG)")
        st.markdown("Table of class description")
        st.markdown("![Image](https://github.com/Xenaschke/classification-predict-streamlit-template/blob/master/images/image2.PNG?raw=true.PNG)")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page
    
    if selection== "Classify tweets":
        st.title("Classify tweets")
        st.markdown("![Image of Yaktocat](https://github.com/Xenaschke/classification-predict-streamlit-template/blob/master/images/tweets.PNG?raw=true.PNG)")
        models = pd.DataFrame({'model name': ['Logistic Regression', 'Naive Bayes','Linear SVM','Random Forest', 'K Nearest Neighbors']})
        model_sel=st.selectbox('Select a model', models['model name'])
    
        #building the Logistic Regression
        if model_sel == "Logistic Regression":
            st.info("Prediction with Logistic Regression Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Transforming user input into a list
                vect_text = [tweet_text]
                # Load .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/Logistic_regression.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                #st.success("Text Categorized as: {}".format(prediction))
                if prediction==-1:
                    st.success("Anti")
                elif prediction==0:
                    st.success("Neutral")
                elif prediction == 1:
                    st.success("Pro")
                else:
                    st.success("News")

        #building the Naive Bayes
        if model_sel == "Naive Bayes":
            st.info("Prediction with Naive Bayes Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Transforming user input into a list
                vect_text = [tweet_text]
                # Load .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/Naive_bayes.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                #st.success("Text Categorized as: {}".format(prediction))
                if prediction==-1:
                    st.success("Anti")
                elif prediction==0:
                    st.success("Neutral")
                elif prediction == 1:
                    st.success("Pro")
                else:
                    st.success("News")                
        
        #building the Linear SVM
        if model_sel == "Linear SVM":
            st.info("Prediction with Linear SVM Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Transforming user input into a list
                vect_text = [tweet_text]
                # Load .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/SVM.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                #st.success("Text Categorized as: {}".format(prediction))
                if prediction==-1:
                    st.success("Anti")
                elif prediction==0:
                    st.success("Neutral")
                elif prediction == 1:
                    st.success("Pro")
                else:
                    st.success("News")                
        #building the Random Forest
        if model_sel == "Random Forest":
            st.info("Prediction with Random Forest Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Transforming user input into a list
                vect_text = [tweet_text]
                # Load .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/Random_forest.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                #st.success("Text Categorized as: {}".format(prediction))
                if prediction==-1:
                    st.success("Anti")
                elif prediction==0:
                    st.success("Neutral")
                elif prediction == 1:
                    st.success("Pro")
                else:
                    st.success("News")                
        #building the KNN
        if model_sel == "K Nearest Neighbors":
            st.info("Prediction with K Nearest Neighbors Model")
            tweet_text = st.text_area("Enter your tweet ","Type Here 🖊")
            if st.button("Classify"):
                # Transforming user input into a list
                vect_text = [tweet_text]
                # Loading .pkl file with the model of your choice + make predictions
                predictor = joblib.load(open(os.path.join("models/lr.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
                #st.success("Text Categorized as: {}".format(prediction))
                if prediction==-1:
                    st.success("Anti")
                elif prediction==0:
                    st.success("Neutral")
                elif prediction == 1:
                    st.success("Pro")
                else:
                    st.success("News")                
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
if __name__ == '__main__':
	main()



