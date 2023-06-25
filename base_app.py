"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
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

# Data dependencies
import pandas as pd
import string

# Pretty graphs
import matplotlib.pyplot as plt



# For background image
import base64


# Vectorizer
@st.cache_resource
def load_model(url, name):
	vectorizer = open(url, name) 
	model = joblib.load(vectorizer) # load vectorizer from the pkl file
	return model

tweet_cv = load_model("resources/vectorizer.pkl","rb")

# Load your raw data


@st.cache_data  # Add the caching decorator
def load_data(url):
    df = pd.read_csv(url)
    return df

df = load_data("resources/train.csv")



#Separating positive and negative tweets for pie chart 
data_disbelief = df[df['sentiment'] == -1]
data_no_belief = df[df['sentiment'] == 0]
data_belief = df[df['sentiment'] == 1]
data_high_belief = df[df['sentiment'] == 2]


# Function to add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )


# The main function where we will build the actual app
def main():
	"""TechIntel Tweet Classifier App with Streamlitt """
	# Add background image
	add_bg_from_local('resources/TechIntelCrop_o30.png')
	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("TechIntel Tweet Classifier")

	st.subheader("Climate Change Tweet Sentiment Classifier")

	# Adds logo to sidebar
	st.sidebar.image('resources/TechIntelCrop.png')
	# Creating sidebar with selection box -
	# you can create multiple pages this way

	options = ["Classifier", "How does it work?", "Statistics", "About TechIntel", "FAQs", "Feedback"]

	selection = st.sidebar.selectbox("Choose Page", options)

	# Building out the predication page
	if selection == "Classifier":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter tweet here")

		if st.button("Click here for result"):
			# Preprocess the data
			# Change to lowercase
			tweet_text = tweet_text.lower()
			# Remove urls
			pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
			subs_url = r'url-web'
			tweet_text = tweet_text.replace(pattern_url, subs_url)
			# Remove punctuation
			def remove_punctuation(post):
				return ''.join([l for l in post if l not in string.punctuation])

			tweet_text = remove_punctuation(tweet_text)
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text])
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/SVCmodel.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if prediction == -1 or prediction == 3:
				st.success("This tweet suggests that this person believes in conspiracy theories about climate change. :question: :question: :question:")	
			elif prediction == 0:
				st.success("This tweet suggests that this person is neutral about climate change.:neutral_face:")
			elif prediction == 1:
				st.success("This tweet suggests that this person believes in climate change.:earth_africa::fire:")
			else:
				st.success("This tweet suggests that this person believes in climate change and believes that it is an immediate threat. :earth_africa::fire::exclamation:")

			
	# Building out the "Information" page
	if selection == "How does it work?":
		st.info("Simple Explanation")

		st.image('resources/forsvm.png')

		# You can read a markdown file from supporting resources folder
		#what I wrote earlier
		#A machine learning model is a file that has been trained to recognize certain types of patterns. \
		#we have trained a model over a set of data, providing it an algorithm that it can use to reason over and learn from the tweets dataset. \
		#This trained model can reason over data that it hasn't seen before and make prediction about the data and these predictions are what you see. \

		expander = st.expander("See here for more info")
		expander.write("TechIntel Tweet Classifier app is a powerful tool that helps organizations gain insights into public opinion about climate change. \
		By analyzing tweets from given datasets, several models were trained and they're able to make adequate predictions when it given unknown data. \
		With this app, valuable information that can inform decision-making, shape marketing strategies, \
		and contribute to the development of sustainable practices is made easily available.")

		st.info("Complicated Explanation")
		expand = st.expander("See here for more info")

		# You can read a markdown file from supporting resources folder\
		#st.markdown(''' The Machine Learning process starts with inputting training data into the selected algorithm. 
#Training data being known or unknown helps to develop the final Machine Learning algorithm. 
#New input data is fed into the machine learning algorithm to test whether the algorithm works correctly.  
#The prediction and results are then checked against each other. If the prediction and results don\’t match, 
#the algorithm is re-trained multiple times until the data scientist gets the desired outcome. 
# This enables the machine learning algorithm to continually learn on its own and produce the optimal answer, 
#gradually increasing in accuracy over time.''')

		expand.write('''TechIntel Tweet Classifier app, is a machine learning model designed to analyze tweets

 and classify them based on the sentiment expressed towards climate change. 
It is built using advanced natural language processing (NLP) techniques and supervised learning algorithms.

The classifier's goal is to accurately identify whether a tweet expresses support or skepticism towards climate change. 
To achieve this, it leverages a large labeled dataset of climate change-related tweets, where each tweet is annotated 
with its corresponding sentiment label.

The model employs various NLP techniques, such as tokenization, stemming, and stop-word removal, to preprocess the tweet
text and convert it into a numerical representation that can be fed into the machine learning algorithms. 
Feature engineering is also applied to extract relevant features, such as n-grams, word embeddings, 
or syntactic patterns, which capture important aspects of the tweet's sentiment.
Several machine learning algorithms can be used for training the classifier, including logistic regression, support vector
machines, or even more advanced approaches like deep learning models such as recurrent neural networks (RNNs) or transformers.
The choice of algorithm depends on the dataset size, complexity, and desired performance.
During the training phase, the classifier learns the patterns and relationships between the tweet features and their 
corresponding sentiment labels. This is done through an iterative process of optimizing a chosen objective function, 
such as maximizing the F1 score.
Once the classifier is trained, it can be used to predict the sentiment of new, unseen tweets. 
These tweets undergo the same preprocessing steps as the training data and are then passed through the trained model, 
which assigns them a sentiment label of either supportive or skeptical.
For evaluation, the classifier's performance is typically assessed using standard metrics such as accuracy, precision, 

recall, and F1 score. Cross-validation or holdout validation techniques are commonly employed to estimate the classifier's 
generalization ability and avoid overfitting.

In summary, the Climate Change Tweet Sentiment Classifier is a sophisticated machine learning model that utilizes 
NLP techniques and supervised learning algorithms to accurately classify tweets based on their sentiment towards climate change. 
It offers data scientists a valuable tool for analyzing public opinion, conducting market research 
and informing decision-making processes related to climate change awareness and mitigation strategies.''')

		expand.image('resources/ml_train.png')




	if selection == "Statistics":
		# Adding wordclouds
		st.info("A word cloud generated from the tweets of people who don't believe in climate change")
		st.image('resources/wc_no.png')

		st.info("A word cloud generated from the tweets of people who believed in climate change")
		st.image('resources/wc.png')
	

		# Creating a pie chart
		st.info("A pie chart showing the proportions of different sentiments")

		mylabels = ["No Belief", "Belief", "Strong Belief", "Disbelief"] # labels
		mycolors = ["Yellow", "Blue", "Green", "Red"] # custom colours

		# pie chart can only have positive numbers, so changing -1 to 3
		df["sentiment"] = df["sentiment"].replace([-1], 3)
		# group the data
		sentiment_counts = df.groupby(['sentiment']).size() 
		# make the pie chart
		fig, ax = plt.subplots()
		ax.pie(sentiment_counts, labels = mylabels, colors = mycolors)
		st.pyplot(fig) # show the pie chart

		
		st.info("A bar chart showing the proportions of different sentiments")
		st.image('resources/bar_cc.png')	
	if selection == "About TechIntel":
		# You can read a markdown file from supporting resources folder
		expa = st.expander("About TechIntel")
		expa.write('''At TechIntel, we are a leading data science company that specializes in unlocking the power of data 

to drive intelligent solutions and empower businesses. 
With our expertise in advanced analytics, machine learning, and artificial intelligence, we help organizations harness 
the potential of their data to make informed decisions and gain a competitive edge in the digital landscape.
Our team of experienced data scientists is passionate about transforming raw data into actionable insights. 
We combine cutting-edge technologies with industry best practices to deliver tailored solutions that address complex business 
challenges across various sectors, including finance, healthcare, retail, and manufacturing.
What sets us apart is our commitment to excellence in data-driven innovation. 
We pride ourselves in our ability to leverage state-of-the-art algorithms, advanced statistical models 
and scalable computing infrastructure to extract meaningful patterns and predictions from diverse data sources.
Our solutions empower businesses to optimize operations, improve customer experience, and drive revenue growth.

At TechIntel, we believe in the power of collaboration. 
We work closely with our clients to understand their unique needs, goals, and desired outcome. 
Through a collaborative approach, we co-create data-driven solutions that align with their strategic objectives and 
provide measurable value. 
We believe that the best results are achieved when data science expertise is combined with domain knowledge and a deep understanding of business context.''')

		ex_pa = st.expander("Meet the Team")
		ex_pa.image('resources/pathway.png')
		#exp_a = st.expander("Share")
		
# Shareable link using Streamlit
		shareable_link = st.markdown("[Share on Streamlit Sharing](share://app/your-app-url)")

# Social sharing buttons using Streamlit's share function
		st.markdown("---")
		st.markdown("Share on Social Media:")
		if st.button("Twitter"):
			st.share.share_on_twitter("Check out this awesome app! [your-app-url]")
		if st.button("LinkedIn"):
			st.share.share_on_linkedin("Check out this awesome app! [your-app-url]")
# Add more social media buttons as needed

	if selection == "FAQs":
		st.info("FAQs")
		# You can read a markdown file from supporting resources folder
		lst = ['How does the model work?', 'What model was used in creating the app?']
		s = ''
		for i in lst:
			s += "- " + i + "\n"
		st.markdown(s)

	if selection == "Feedback":
# You can read a markdown file from supporting resources folder
		st.info("We would love to hear from you!")
# Feedback form
		feedback_text = st.text_area("Please provide your feedback", height=150)
		feedback_button = st.button("Submit Feedback")
		if feedback_button:
			if feedback_text:
        # Save the feedback to a file, database, or process it as needed
				st.success("Thank you for your feedback!")
			else:
				st.warning("Please enter your feedback before submitting.")        


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

# With thanks to 
# https://levelup.gitconnected.com/how-to-add-a-background-image-to-your-streamlit-app-96001e0377b2