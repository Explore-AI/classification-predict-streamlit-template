import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import string
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
#emoji cloud
# from deepmoji import DeepMoji
import string
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
from sklearn.metrics import f1_score
from wordcloud import WordCloud
from collections import  Counter
from sklearn.feature_extraction.text import CountVectorizer
import emoji
from PIL import Image
from wordcloud import WordCloud
import plotly.express as px
from sklearn.svm import LinearSVC
from sklearn.metrics import plot_confusion_matrix
import joblib,os

warnings.filterwarnings("ignore", category=DeprecationWarning)
import nltk
#nltk.download('all')

nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# Vectorizer
news_vectorizer = open("resources/vect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

 # Add a title
# st.title('Climate Changes Twitter Sentiment')

# Add an Image to the web app
image =  Image.open("img.png")
st.image(image, use_column_width = True)

# Read the train data
train =  pd.read_csv('train1.csv')
# Make a copy of the data
df = train.copy()

def main():
  activities = ['Prediction', 'EDA', 'model', 'About us']
  option=st.sidebar.selectbox("Selection option:", activities)

  if option=='Prediction':

    # st.header("Tweet prediction")
    st.info("Prediction with Classification ML Models")

    option1 = st.selectbox('Choose the model to make the Prediction with:',("Logistic Regression", "KNN", "SVM", "Naive Bayes", "Random Forest", "LinearSVC", "GBC", "SGDC"))
    st.write('You selected:', option1)
    if option1=="Logistic Regression":
      # Creating a text box for user input
      tweet_text = st.text_area("Enter Text","Type Here or Copy and Paste Tweet Here")

      if st.button("Classify"):
        # Transforming user input with vectorizer
        vect_text = tweet_cv.transform([tweet_text]).toarray()
        # Load your .pkl file with the model of your choice + make predictions
        # Try loading in multiple models to give the user a choice
        predictor = joblib.load(open(os.path.join("resources/lr_model_.pkl"),"rb"))
        prediction = predictor.predict(vect_text)

        if prediction == 1 :

          st.success("Text Categorized as: Pro ")
        elif prediction == 0 :

          st.success("Text Categorized as: Neutral ")
        elif prediction == -1 :

          st.success("Text Categorized as: Anti ")
        else :
          st.success("Text Categorized as: News ")
        

        # When model has successfully run, will print prediction
        # You can use a dictionary or similar structure to make this output
        # more human interpretable.
        # st.success("Text Categorized as: {}".format(prediction))


    if option1=="Naive Bayes":
      # Creating a text box for user input
      tweet_text = st.text_area("Enter Text","Type Here or Copy and Paste Tweet Here")

      if st.button("Classify"):
        # Transforming user input with vectorizer
        vect_text = tweet_cv.transform([tweet_text]).toarray()
        # Load your .pkl file with the model of your choice + make predictions
        # Try loading in multiple models to give the user a choice
        predictor = joblib.load(open(os.path.join("resources/nb_model_.pkl"),"rb"))
        prediction = predictor.predict(vect_text)

        # When model has successfully run, will print prediction
        # You can use a dictionary or similar structure to make this output
        # more human interpretable.
        if prediction == 1 :
          st.success("Text Categorized as: Pro ")
        elif prediction == 0 :
          st.success("Text Categorized as: Neutral ")
        elif prediction == -1 :
          st.success("Text Categorized as: Anti ")
        else :
          st.success("Text Categorized as: News ")
        
        #st.success("Text Categorized as: {}".format(prediction))

    if option1=="KNN":
      # Creating a text box for user input
      tweet_text = st.text_area("Enter Text","Type Here or Copy and Paste Tweet Here")

      if st.button("Classify"):
        # Transforming user input with vectorizer
        vect_text = tweet_cv.transform([tweet_text]).toarray()
        # Load your .pkl file with the model of your choice + make predictions
        # Try loading in multiple models to give the user a choice
        predictor = joblib.load(open(os.path.join("resources/knc_model_.pkl"),"rb"))
        prediction = predictor.predict(vect_text)

        # When model has successfully run, will print prediction
        # You can use a dictionary or similar structure to make this output
        # more human interpretable.
        if prediction == 1 :
          st.success("Text Categorized as: Pro ")
        elif prediction == 0 :
          st.success("Text Categorized as: Neutral ")
        elif prediction == -1 :
          st.success("Text Categorized as: Anti ")
        else :
          st.success("Text Categorized as: News ")
        #st.success("Text Categorized as: {}".format(prediction))

    if option1=="Random Forest":
      # Creating a text box for user input
      tweet_text = st.text_area("Enter Text","Type Here or Copy and Paste Tweet Here")

      if st.button("Classify"):
        # Transforming user input with vectorizer
        vect_text = tweet_cv.transform([tweet_text]).toarray()
        # Load your .pkl file with the model of your choice + make predictions
        # Try loading in multiple models to give the user a choice
        predictor = joblib.load(open(os.path.join("resources/rfc_model.pkl"),"rb"))
        prediction = predictor.predict(vect_text)

        # When model has successfully run, will print prediction
        # You can use a dictionary or similar structure to make this output
        # more human interpretable.
        if prediction == 1 :
          st.success("Text Categorized as: Pro ")
        elif prediction == 0 :
          st.success("Text Categorized as: Neutral ")
        elif prediction == -1 :
          st.success("Text Categorized as: Anti ")
        else :
          st.success("Text Categorized as: News ")
        #st.success("Text Categorized as: {}".format(prediction))

    if option1=="LinearSVC":
      # Creating a text box for user input
      tweet_text = st.text_area("Enter Text","Type Here or Copy and Paste Tweet Here")

      if st.button("Classify"):
        # Transforming user input with vectorizer
        vect_text = tweet_cv.transform([tweet_text]).toarray()
        # Load your .pkl file with the model of your choice + make predictions
        # Try loading in multiple models to give the user a choice
        predictor = joblib.load(open(os.path.join("resources/lsvc_model.pkl"),"rb"))
        prediction = predictor.predict(vect_text)

        # When model has successfully run, will print prediction
        # You can use a dictionary or similar structure to make this output
        # more human interpretable.

        if prediction == 1 :
          st.success("Text Categorized as: Pro ")
        elif prediction == 0 :
          st.success("Text Categorized as: Neutral ")
        elif prediction == -1 :
          st.success("Text Categorized as: Anti ")
        else :
          st.success("Text Categorized as: News ")
        
        #st.success("Text Categorized as: {}".format(prediction))

    if option1=="SVM":
      # Creating a text box for user input
      tweet_text = st.text_area("Enter Text","Type Here or Copy and Paste Tweet Here")

      if st.button("Classify"):
        # Transforming user input with vectorizer
        vect_text = tweet_cv.transform([tweet_text]).toarray()
        # Load your .pkl file with the model of your choice + make predictions
        # Try loading in multiple models to give the user a choice
        predictor = joblib.load(open(os.path.join("resources/svm_model.pkl"),"rb"))
        prediction = predictor.predict(vect_text)

        # When model has successfully run, will print prediction
        # You can use a dictionary or similar structure to make this output
        # more human interpretable.

        if prediction == 1 :
          st.success("Text Categorized as: Pro ")
        elif prediction == 0 :
          st.success("Text Categorized as: Neutral ")
        elif prediction == -1 :
          st.success("Text Categorized as: Anti ")
        else :
          st.success("Text Categorized as: News ")
        

        #st.success("Text Categorized as: {}".format(prediction))

    if option1=="GBC":
      # Creating a text box for user input
      tweet_text = st.text_area("Enter Text","Type Here or Copy and Paste Tweet Here")

      if st.button("Classify"):
        # Transforming user input with vectorizer
        vect_text = tweet_cv.transform([tweet_text]).toarray()
        # Load your .pkl file with the model of your choice + make predictions
        # Try loading in multiple models to give the user a choice
        predictor = joblib.load(open(os.path.join("resources/gbc_model.pkl"),"rb"))
        prediction = predictor.predict(vect_text)

        # When model has successfully run, will print prediction
        # You can use a dictionary or similar structure to make this output
        # more human interpretable.

        if prediction == 1 :
          st.success("Text Categorized as: Pro ")
        elif prediction == 0 :
          st.success("Text Categorized as: Neutral ")
        elif prediction == -1 :
          st.success("Text Categorized as: Anti ")
        else :
          st.success("Text Categorized as: News ")
        
        # st.success("Text Categorized as: {}".format(prediction))
    if option1=="SGDC":
      # Creating a text box for user input
      tweet_text = st.text_area("Enter Text","Type Here or Copy and Paste Tweet Here")

      if st.button("Classify"):
        # Transforming user input with vectorizer
        vect_text = tweet_cv.transform([tweet_text]).toarray()
        # Load your .pkl file with the model of your choice + make predictions
        # Try loading in multiple models to give the user a choice
        predictor = joblib.load(open(os.path.join("resources/sgdc_model.pkl"),"rb"))
        prediction = predictor.predict(vect_text)

        # When model has successfully run, will print prediction
        # You can use a dictionary or similar structure to make this output
        # more human interpretable.

        if prediction == 1 :
          st.success("Text Categorized as: Pro ")
        elif prediction == 0 :
          st.success("Text Categorized as: Neutral ")
        elif prediction == -1 :
          st.success("Text Categorized as: Anti ")
        else :
          st.success("Text Categorized as: News ")
       
  if option=="EDA":
  	 data=st.file_uploader("Upload Dataset and View:", type=['csv', 'xlsx', 'text','json'])
  	 st.success('Data uploaded successfully')
  	 if data is not None:

  	 	df1=pd.read_csv(data)
  	 	st.dataframe(df1.head(10))

  	 df = train.copy()
  	 sentiment_counts = df.groupby('sentiment').size().reset_index(name='counts')
  	 def sentiment(df):
  	 	sentiment = df['sentiment']
  	 	sentiment_class = []
  	 	for i in sentiment :
  	 		if i == 1 :
  	 			sentiment_class.append('Pro')
  	 		elif i == 0 :
  	 			sentiment_class.append('Neutral')
  	 		elif i == -1 :
  	 			sentiment_class.append('Anti')
  	 		else :
  	 			sentiment_class.append('News')
  	 	df['sentiment'] = sentiment_class
  	 	return df
  	 df = sentiment(df)


  	 sent_count = df.groupby('sentiment').size().reset_index(name='counts')
  	 st.sidebar.subheader('Exploratory Data Analysis')
  	 if st.sidebar.checkbox('View sentiment count'):
  	 	st.write(sent_count)

  	 # Create a function for emoji extraction
  	 def extract_emojis(s):
  	 	return ''.join(c for c in s if c in emoji.UNICODE_EMOJI)
  	 #extracting emojis on train data
  	 df['emoji'] = df['message'].apply(extract_emojis)
  	 # Create the function to extract the emojis from data
  	 def extract_emojis(df):
  	 	for char in df:
  	 		if char in emoji.UNICODE_EMOJI:
  	 			return True
  	 		else:
  	 			return False
  	 df['emoji'] = df['message'].apply(extract_emojis)

  	 #convert emojies on the dataframe to text
  	 def text_emoji(txt):
  	 	emoji_converter = emoji.demojize(txt, delimiters=("", ""))
  	 	return emoji_converter
  	 # remove special characters, numbers, punctuations from train data
  	 df['message'] = df['message'].str.replace("[^a-zA-Z#]", " ")
  	 #removing short words from train data
  	 df['message'] = df['message'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
  	 #removing short words from train data
  	 df['message'] = df['message'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
  	 #Replace the word https with nothing: train
  	 df['message'] = df['message'].str.replace('https', '')
  	 # Remove Line breaks: train
  	 df['message']=df['message'].replace('\n', ' ')

  	 #st.write(df)
  	 st.sidebar.subheader("Visualising the Dataset")
  	 if st.sidebar.checkbox('Sentiments'):
  	 	fig, ax = plt.subplots()
  	 	ax  =sns.countplot(x='sentiment',data=df, palette="Blues_d")
  	 	st.pyplot(fig)

  	 if st.sidebar.checkbox("Tweet Length Distribution"):
  	 	df['tweet length'] = df['message'].apply(len)
  	 	fig, ax = plt.subplots()
  	 	ax = sns.FacetGrid(df,col='sentiment')
  	 	ax.map(plt.hist,'tweet length')
  	 	st.pyplot(ax)

  	 # #Top 10 Hashtags from the tweets
  	 # hashtags_pro = []
  	 # for message in Pro:
  	 # 	hashtag = re.findall(r"#(\w+)", message)
  	 # 	hashtags_pro.append(hashtag)

  	 # hashtags_pro = sum(hashtags_pro,[])
  	 # a = nltk.FreqDist(hashtags_pro)
  	 # d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
  	 # d = d.nlargest(columns="Count", n = 10)
  	 # if checkbox("Top 10 Hashtags in Pro Tweets"):
  	 # 	fig, ax =  plt.figure(figsize=(10,5))
  	 # 	ax = sns.barplot(data=d, x= "Hashtag", y = "Count",palette=("Blues_d"))
  	 # 	plt.setp(ax.get_xticklabels(),rotation='vertical', fontsize=10)
  	 # 	plt.title('Top 10 Hashtags in Pro Tweets', fontsize=14) 
  	 # 	st.pyplot(ax)
  	 

  	 import collections
  	 df['temp_list'] = df['message'].apply(lambda x:str(x).split())
  	 top = collections.Counter([item for sublist in df['temp_list'] for item in sublist])
  	 temp = pd.DataFrame(top.most_common(20))
  	 temp.columns = ['Common_words','count']
  	 temp_list = temp.style.background_gradient(cmap='Blues')

  	 if st.sidebar.checkbox("Frequent Words"):
  	 	fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Tweets', orientation='h', width=700, height=700)
  	 	st.write(fig)
  	 	st.pyplot()

  	 #most common words Sentiment wise
  	 Pro = df[df['sentiment']=='Pro']
  	 News = df[df['sentiment']=='News']
  	 Neutral = df[df['sentiment']=='Neutral']
  	 Anti =df[df['sentiment']=='Anti']

  	 #MosT common positive words based on sentiment
  	 top = collections.Counter([item for sublist in Pro['temp_list'] for item in sublist])
  	 pro_tweet = pd.DataFrame(top.most_common(20))
  	 pro_tweet.columns = ['Common_words','count']

  	 if st.sidebar.checkbox("Figure representing common words in Pro tweets"):
  	 	fig = px.bar(pro_tweet, x="count", y="Common_words", title='Commmon used words in Pro tweet', orientation='h', 
             width=700, height=700)
  	 	st.write(fig)
  	 	st.pyplot()
	
  elif option=="model":
  	st.header("Models Building")
  	df = train.copy()
  	if st.sidebar.checkbox("Data we will be working with"):
  		st.write(df)
  		# Diving the data into X and Y variables.
  	X=df["message"]
  	y=df["sentiment"]

  	Classifier_name = st.sidebar.selectbox("Preferred Classifier:", ("LR", "KNN", "SVM", "Naive Bayes", "Random Forest", "LinearSVC", "GBC", "SGDC"))
  	def add_parameter(name_of_cls):
  		param=dict()
  		if name_of_cls=="LR":
  			C=st.sidebar.slider("C",0.01, 1.0)
  			max_iter=st.sidebar.slider("max_iter",1, 1000)
  			penalty=st.sidebar.radio("penalty", ("l2", "l1", "elasticnet", "None"), key = 'l2')
  			random_state = st.sidebar.number_input("random_state", 1, 1000, step = 1, key = "random_state")
  			solver=st.sidebar.radio("solver", ("newton-cg", "lbfgs", "liblinear", "sag", "saga"), key = 'lbfgs')
  			param["C"]=C
  			param["solver"]=solver
  			param["max_iter"]=max_iter
  			param["penalty"]=penalty
  			param["random_state"]=random_state
  		if name_of_cls=="Random Forest":
  			n_estimators=st.sidebar.slider("n_estimators", 1, 1000)
  			random_state=st.sidebar.slider("random_state", 0, 100)
  			param["n_estimators"]=n_estimators
  			param["random_state"]=random_state
  		if name_of_cls=="SVM":
  			C=st.sidebar.slider("C", 0.01,15)
  		if name_of_cls=="KNN":
  			n_neighbors=st.sidebar.slider("n_neighbors", 1, 100)
  			algorithm=st.sidebar.radio("algorithm", ("auto", "ball_tree", "kd_tree", "brute"), key = 'auto')
  			param["n_neighbors"]=n_neighbors
  		if name_of_cls=="LinearSVC":
  			max_iter=st.sidebar.slider("max_iter",1, 1000)
  			random_state=st.sidebar.number_input("random_state", 1, 1000, step = 1, key = "random_state")
  			C=st.sidebar.slider("C", 0.01,1.0)
  			param["random_state"]=random_state
  			param["max_iter"]=max_iter
  			param['C']=C
  		if name_of_cls=="GBC":
  			n_estimators=st.sidebar.slider("n_estimators", 1, 100)
  			param["n_estimators"]=n_estimators
  		if name_of_cls=="SGDC":
  			max_iter=st.sidebar.slider("max_iter",1, 5000)
  			param["max_iter"]=max_iter
  		return param
  	param=add_parameter(Classifier_name)


  	def get_classifier(name_of_cls, param):

  		clf=None
  		if name_of_cls=="KNN":
  			clf=KNeighborsClassifier(n_neighbors=param["n_neighbors"], algorithm=param["algorithm"], leaf_size=param["leaf_size"])
  		elif name_of_cls=="LR":
  			clf=LogisticRegression(C=param["C"], max_iter=param['max_iter'], penalty=param["penalty"], solver=param["solver"])
  		elif name_of_cls=="Naive Bayes":
  			clf = MultinomialNB()
  		elif name_of_cls=="Random Forest":
  			clf=RandomForestClassifier(n_estimators=param["n_estimators"], random_state=param["random_state"])
  		elif name_of_cls=="LinearSVC":
  			clf=LinearSVC(C=param["C"], random_state=param["random_state"], max_iter=param["max_iter"])
  		elif name_of_cls=="GBC":
  			clf=GradientBoostingClassifier(n_estimators=param["n_estimators"], learning_rate=1.0, max_depth=1)
  		elif name_of_cls=="SGDC":
  			clf=SGDClassifier(tol=0.01)
  		else:
  			st.warnings("Select your choice of algorithm.")
  		return clf
  	clf=get_classifier(Classifier_name,param)

  	vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, stop_words="english")
  	X_vectorized = vectorizer.fit_transform(X)
  	X_train,X_val,y_train,y_val = train_test_split(X_vectorized,y,test_size=.3,shuffle=True, stratify=y, random_state=11)
  	
  	clf.fit(X_train, y_train)
  	y_pred=clf.predict(X_val)

  	st.write("### Name of classifier:",Classifier_name)
  	if st.checkbox("View the Prediction sentiments"):
  		st.write("Prediction of the sentiments:",y_pred)
  		accuracy=metrics.classification_report(y_val,y_pred)
  		confusion_matrix=metrics.confusion_matrix(y_val,y_pred)

  	if st.checkbox("View Classification Report"):
  		st.write("Accuracy:",accuracy)

  	if st.checkbox("View the Confusion Matrix"):
  		st.write("Confusion Matrix:",confusion_matrix)

  	
  	if st.checkbox("Display heatmap for Confusion Matrix"):
  		fig, ax = plt.subplots()
  		ax = sns.heatmap(confusion_matrix, annot=True)
  		st.pyplot(fig)
 
  	if Classifier_name=="LR":
  		if st.sidebar.checkbox("Description of the model parameters"):
  			st.markdown("""**penalty: {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’** <br>
Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties. ‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’ (not supported by the liblinear solver), no regularization is applied.<br>

**C : float, default=1.0** <br>
Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.<br>

**random_state: int, RandomState instance, default=None**<br>
Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data. See Glossary for details.<br>

**solver: {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’**<br>
Algorithm to use in the optimization problem.<br>

- For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.<br>

- For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.<br>

- ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty<br>

- ‘liblinear’ and ‘saga’ also handle L1 penalty<br>

- ‘saga’ also supports ‘elasticnet’ penalty<br>

- ‘liblinear’ does not support setting penalty='none'<br>

**max_iter : int, default=100**<br>
Maximum number of iterations taken for the solvers to converge.<br>
    """, unsafe_allow_html=True)
  	if Classifier_name=="KNN":
  		if st.sidebar.checkbox("Description of the model parameters"):
  			st.markdown("""**n_neighbors**: int, default=5<br>
Number of neighbors to use by default for kneighbors queries.<br>

**algorithm** : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’<br>
- Algorithm used to compute the nearest neighbors:<br>

- `ball_tree` will use BallTree<br>

- `kd_tree` will use KDTree<br>

- `brute` will use a brute-force search.<br>

- `auto` will attempt to decide the most appropriate algorithm based on the values passed to fit method.<br>

**leaf_size**: int, default=30
Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.<br>


    """, unsafe_allow_html=True)
  	if Classifier_name=="Random Forest":
  		if st.sidebar.checkbox("Description of the model parameters"):
  			st.markdown("""**n_estimators : int, default=100**<br>
The number of trees in the forest.<br>

**random_state : int or RandomState, default=None**<br>
Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features).<br>
    """, unsafe_allow_html=True)
  	if Classifier_name=="LinearSVC":
  		if st.sidebar.checkbox("Description of the model parameters"):
  			st.markdown("""**C : float, default=1.0**<br>
Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.<br>

**random_state : int or RandomState instance, default=None**
Controls the pseudo random number generation for shuffling the data for the dual coordinate descent (if dual=True). When dual=False the underlying implementation of LinearSVC is not random and random_state has no effect on the results. Pass an int for reproducible output across multiple function calls.<br>

**max_iter : int, default=1000**<br>
The maximum number of iterations to be run.<br>
    """, unsafe_allow_html=True)
  	if Classifier_name=="GBC":
  		if st.sidebar.checkbox("Description of the model parameters"):
  			st.markdown("""**n_estimators : int, default=100**<br>
The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.<br>
    """, unsafe_allow_html=True)
  	if Classifier_name=="SGDC":
  		if st.sidebar.checkbox("Description of the model parameters"):
  			st.markdown("""**max_iter : int, default=1000**<br>
The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit method.<br>    
	""", unsafe_allow_html=True)



  elif option=="About us":
  	st.markdown("<h1 style='text-align: center; color: blue;'>About Us</h1>", unsafe_allow_html=True)
  	image =  Image.open("About us page.png")
  	st.image(image, use_column_width=True)

  	



    
if __name__ == '__main__':
	main()