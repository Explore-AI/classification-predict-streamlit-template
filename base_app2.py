import streamlit as st 
import pandas as pd
import numpy as np
import joblib,os
from nltk import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns


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
    st.title("Tweet Classifer App")
    st.subheader("Climate change tweet classification")
    #creating side menu
    options = ["Information","Lr","Lsvm","Rf","Knn","Nn","Visuals"]
    selection = st.sidebar.selectbox("Menu Options", options)

    #building the Information page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("This climate change tweet classifier app is useful for classifying whether or not a person believes in climate change, based on their tweet. \n The App is created to help companies determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received. The App gives users a choice to use a model of their choice to determine how the tweet(s) they have percieve(s) climate change')
        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page

    #building the Logistic Regression
    if selection == "Lr":
        st.info("Prediction with Logistic Regression Model")
        tweet_text = st.text_area("enter text ","Type Here")
        if st.button("Classify"):
            # Transforming user input into a list
            vect_text = [tweet_text]
            # Load .pkl file with the model of your choice + make predictions
            predictor = joblib.load(open(os.path.join("lr.pkl"),"rb"))
            prediction = predictor.predict(vect_text)
            st.success("Text Categorized as: {}".format(prediction))

    #building the Naive Bayes
    if selection == "Nn":
        st.info("Prediction with Naive Bayes Model")
        tweet_text = st.text_area("enter text ","Type Here")
        if st.button("Classify"):
            # Transforming user input into a list
            vect_text = [tweet_text]
            # Load .pkl file with the model of your choice + make predictions
            predictor = joblib.load(open(os.path.join("lr.pkl"),"rb"))
            prediction = predictor.predict(vect_text)
            st.success("Text Categorized as: {}".format(prediction))
        
    #building the Linear SVM
    if selection == "Lsvm":
        st.info("Prediction with Linear SVM Model")
        tweet_text = st.text_area("enter text ","Type Here")
        if st.button("Classify"):
            # Transforming user input into a list
            vect_text = [tweet_text]
            # Load .pkl file with the model of your choice + make predictions
            predictor = joblib.load(open(os.path.join("Linear_SVM_base_model.pkl"),"rb"))
            prediction = predictor.predict(vect_text)
            st.success("Text Categorized as: {}".format(prediction))
            
        #building the Draw
    if selection == "Visuals":
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
    #building the Random Forest
    if selection == "Rf":
        st.info("Prediction with Random Forest Model")
        tweet_text = st.text_area("enter text ","Type Here")
        if st.button("Classify"):
            # Transforming user input into a list
            vect_text = [tweet_text]
            # Load .pkl file with the model of your choice + make predictions
            predictor = joblib.load(open(os.path.join("lr.pkl"),"rb"))
            prediction = predictor.predict(vect_text)
            st.success("Text Categorized as: {}".format(prediction))        

    #building the KNN
    if selection == "Knn":
        st.info("Prediction with K Nearest Neighbors")
        tweet_text = st.text_area("enter text ","Type Here")
        if st.button("Classify"):
            # Transforming user input into a list
            vect_text = [tweet_text]
            # Loading .pkl file with the model of your choice + make predictions
            predictor = joblib.load(open(os.path.join("lr.pkl"),"rb"))
            prediction = predictor.predict(vect_text)
            st.success("Text Categorized as: {}".format(prediction))


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()



