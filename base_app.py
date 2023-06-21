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

# Pretty graphs
import matplotlib.pyplot as plt

# For background image
import base64

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
df = pd.read_csv("resources/train.csv")

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
	"""Tweet Classifier App with Streamlit """
	# Add background image
	add_bg_from_local('resources/TechIntelCrop_o30.png')
	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifier")
	st.subheader("Climate Change Tweet Sentiment Classifier")

	# Adds logo to sidebar
	st.sidebar.image('resources/TechIntelCrop.png')
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Classifier", "How does it work?","Statistics","About TechIntel"]
	selection = st.sidebar.selectbox("Choose Page", options)

	# Building out the predication page
	if selection == "Classifier":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter tweet here")

		if st.button("Click here for result"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
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
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.info("Complicated Explanation")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")



	if selection == "Statistics":
		# Adding wordclouds
		st.info("A word cloud generated from the tweets of people who don't believe in climate change")
		st.image('resources/wc_no.png')

		st.info("A word cloud generated from the tweets of people who believed in climate change")
		st.image('resources/wc.png')
	

		# Creating a pie chart
		st.info("A pie chart showing the proportions of different sentiments")
		mylabels = ["Neutral", "Belief", "Strong Belief", "Conspiracy"] # labels
		mycolors = ["Aqua", "Azure", "DarkBlue", "DeepSkyBlue"] # custom colours
		# pie chart can only have positive numbers, so changing -1 to 3
		df["sentiment"] = df["sentiment"].replace([-1], 3)
		# group the data
		sentiment_counts = df.groupby(['sentiment']).size() 
		# make the pie chart
		fig, ax = plt.subplots()
		ax.pie(sentiment_counts, labels = mylabels, colors = mycolors)
		st.pyplot(fig) # show the pie chart

	if selection == "About TechIntel":
		st.info("About TechIntel")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")



# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

# With thanks to 
# https://levelup.gitconnected.com/how-to-add-a-background-image-to-your-streamlit-app-96001e0377b2