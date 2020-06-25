"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
#import markdown

# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/datasets/train.csv")
eda = pd.read_csv("resources/datasets/eda.csv", sep ='\t')


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change belief classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "EDA", "Insights", "Prediction"]
	selection = st.sidebar.selectbox("Choose Option", options)



	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown(open("resources/info.md").read())
		
		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the "Prediction" page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a selection box to choose different models
		models = ['Support Vector Classifier','Logistic Regression']
		classifiers = st.selectbox("Choose a classifier", models)
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):

			if classifiers == 'Support Vector Classifier':
				# Transforming user input with vectorizer
				vect_text = [tweet_text]#.toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/linear_svc.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				
			elif classifiers == 'Logistic Regression':
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if prediction == -1:
				result = 'Anti'
			elif prediction == 0:
				result = 'Neutral'
			elif prediction == 1:
				result = 'Pro'
			else:
				result = 'News'
			
			st.success("Text Categorized as: {}".format(result))
			
	# Building out the "EDA" page
	if selection == "EDA":
		target_map = {-1:'Anti', 0:'Neutral', 1:'Pro', 2:'News'}
		eda['target'] = eda['sentiment'].map(target_map)

		st.info("Exploratory Data Analysis")
		
		st.write('Place markdown here')
		
		# Plot count of sentiments
		fig, ax = plt.subplots()
		fig = sns.countplot(data = eda, x = 'target', palette = {'Pro':'#CCCC00', 'News':'teal', 'Neutral':'teal', 'Anti':'teal'})
		plt.xlabel('Sentiment')
		plt.ylabel('Number of Observations')
		plt.title('Count of Observations by Class\n')
		ax.set_xticklabels(['Anti', 'Neutral', 'Pro', 'News'])
		plt.xticks(rotation=45)
		st.pyplot()
		st.markdown(open("resources/eda.md").read(),unsafe_allow_html=False)

		
		# Sentiment Scores
		# fig, axes = plt.subplots(1, 4, figsize = (18, 6), sharey = True)
		# palette = {'Pro':'#CCCC00', 'News':'teal', 'Neutral':'teal', 'Anti':'teal'}
		# for i, column in enumerate(nltk_scores.keys()):
		# 	g = sns.barplot(data=eda, x='target', y=column, ax=axes[i], palette=palette)
		# 	g.set_title(column)
		# 	g.set_ylabel(' ')
		# 	g.set_xlabel(' ')
		# st.pyplot()

		# Subjectivity and Polarity
		# fig, axes = plt.subplots(1, 2, figsize = (12, 5), sharey = True)
		# for i, column in enumerate(['subjectivity', 'polarity']):
		# 	g = sns.barplot(data = EDA, x = 'target', y = column, ax = axes[i], palette = {'Pro':'#CCCC00', 'News':'teal', 'Neutral':'teal', 'Anti':'teal'});
		# 	g.set_title(column)
		# 	g.set_ylabel(' ')
		# 	g.set_xlabel(' ')

		# Scatter Plots
		data = eda.groupby('target')[['negative', 'positive', 'neutral', 'compound', 'polarity', 'subjectivity']].mean().reset_index()
		def plotScatter(x, y, df):
			fig = plt.figure(figsize = (8, 5))
			g = sns.scatterplot(data = df, x = x, y = y, hue = 'target', legend = False, palette = {'Pro':'#CCCC00', 'News':'teal', 'Neutral':'teal', 'Anti':'teal'})

			# add annotations one by one with a loop
			for line in range(0,data.shape[0]):
				g.text(data[x][line], data[y][line], data['target'][line], 
						horizontalalignment='left', size='large', color='black')
			return g

		fig = plotScatter(x = 'compound', y = 'polarity', df = data)
		st.pyplot()

		fig = plt.figure(figsize = (8, 5))
		g = sns.scatterplot(data = eda, x = 'subjectivity', y = 'polarity', color = 'teal', hue = 'target')
		g.arrow(0, 0.0, 0.99, 1, fc = 'black', ec = '#CCCC00')
		g.arrow(0, 0.0, 0.99, -1, fc = 'black', ec = '#CCCC00')
		st.pyplot()

		fig = plotScatter(x = 'subjectivity', y = 'polarity', df = data)
		st.pyplot()

		# Sentiment Distributions
		columns = ['polarity', 'compound']
		fig, axes = plt.subplots(1, len(columns), figsize = (18, 5), sharey = True)
		for i, column in enumerate(columns):
			sns.distplot(eda[column], ax = axes[i])
		st.pyplot()

		sum_of_pol_and_comp = eda['polarity'].add(eda['compound'])
		fig = sns.distplot(sum_of_pol_and_comp)
		st.pyplot()

		eda['pol_plus_comp'] = sum_of_pol_and_comp
		data = eda.groupby('target')[['pol_plus_comp', 'subjectivity']].mean().reset_index()
		plotScatter(x = 'subjectivity', y = 'pol_plus_comp', df = data)
		st.pyplot()

	# Building out the "Insights" page
	if selection == "Insights":
		insight_df = pd.read_csv("resources/datasets/interactive.csv", sep ='\t')
		st.info("Insights from Word Clouds")
		
		# Building vocabulary
		vocab = list()
		ignore = ['rt', 'urlweb', 'htt', 'ht','pron','amp','https','http','mr']
		for tweet in insight_df['tweets']:
			for token in tweet:
				if token not in ignore:
					vocab.append(token)
		
		
		
		# Generate a list of the most common words
		most_common_words = Counter(vocab).most_common()
		n = st.sidebar.slider('Top n words to include in Wordcloud', min_value=100, max_value=13000, value=12000, step=None, format=None, key=None)
		most_common = list()
		for word in most_common_words[:n]:
			most_common.append(word[0])
		st.write(vocab)
		def removeInfrequentWords(tweet):
			pre_proc_tweet = list()
			for token in tweet:
				if token in most_common:
					pre_proc_tweet.append(token)
			return pre_proc_tweet

		insight_df['tweets'] = insight_df['tweets'].map(removeInfrequentWords)

		target_map = {-1:'Anti', 0:'Neutral', 1:'Pro', 2:'News'}
		insight_df['target'] = insight_df['sentiment'].map(target_map)

		def getPolarityScores(tweet):
			tweet = ' '.join(tweet)
			sid = SentimentIntensityAnalyzer()
			scores = sid.polarity_scores(tweet)
			return scores

		nltk_scores = dict(compound = list(), negative = list(), neutral = list(), positive = list())
		for tweet in insight_df['tweets']:
			output = getPolarityScores(tweet)
			nltk_scores['compound'].append(output['compound'])
			nltk_scores['negative'].append(output['neg'])
			nltk_scores['neutral'].append(output['neu'])
			nltk_scores['positive'].append(output['pos'])

		if 'compound' in insight_df.columns:
			insight_df.drop(['compound', 'negative', 'neutral', 'positive'], axis = 1, inplace = True)
			insight_dfe = pd.concat([insight_df, pd.DataFrame(nltk_scores)], axis = 1)
		else:
			insight_df = pd.concat([insight_df, pd.DataFrame(nltk_scores)], axis = 1)

		sentiment_scores = [TextBlob(tweet).sentiment for tweet in insight_df['message']]

		pol = list()
		subj = list()
		for scores in sentiment_scores:
			pol.append(scores.polarity)
			subj.append(scores.subjectivity)

		insight_df['polarity'] = pol
		insight_df['subjectivity'] = subj


		
		# Create selections for each class
		insight_options = ["Overview", "News", "Pro", "Neutral", "Anti"]
		insights = st.sidebar.selectbox("Choose Sentiment", insight_options)

		# Most Common
		hundred_most_common_words_count = Counter(vocab).most_common(50)
		hundred_most_common_words = list()
		for word in hundred_most_common_words_count:
			token = word[0]
			hundred_most_common_words.append(token)
		st.write(hundred_most_common_words)
		# Creating sliders for sentiment tuning
		neg_upper = st.sidebar.slider('Negative Sentiment Upper Bound', min_value=-1.0, max_value=0.0, value=-0.2, step=0.01)
		pos_lower = st.sidebar.slider('Positive Sentiment Lower Bound', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
		
		
		if insights == "Overview":			
			# Plotting Compound = Neutral
			data = insight_df[(insight_df['compound'] > -neg_upper) & (insight_df['compound'] < pos_lower)]
			
			# Building word list
			words = list()
			for tweet in data['tweets']:
				for token in tweet:
					if token not in hundred_most_common_words:
						words.append(token)
			words = ' '.join(words)

			wordcloud = WordCloud(contour_width=3, contour_color='steelblue').generate(words)

			# Display the generated image:
			plt.imshow(wordcloud, interpolation='bilinear')
			plt.axis("off")
			plt.margins(x=0, y=0)
			st.pyplot()

			# st.write(data)
			cloud = plotWordCloud(data)
			st.pyplot()
			# print('Sentiment = Negative')
			# data = eda[eda['compound'] < neg_upper]
			# plotWordCloud(data)

			# print('Sentiment = Positive')
			# data = eda[eda['compound'] > pos_lower]
			# plotWordCloud(data)

		elif insights == "News":
			st.write('graph here')
		elif insights == "Pro":
			st.write('graph here')
		elif insights == "Neutral":
			st.write('graph here')
		else:
			st.write('graph here')



# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
