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
import matplotlib.pyplot as plt
from PIL import Image

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")
sentiment = ["1", "2", "0", "-1"]

# The main function where we will build the actual app


def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	
	options = ["Home", "About", "Exploratory Data Analysis", "Model", "Contact Us"]
	selection = st.sidebar.selectbox("",options)

	# Building out the "About" page
	if selection == "About":
		st.title("About")

		logo = Image.open('logo.jpg')
		st.sidebar.image(logo, use_column_width=True)

		# You can read a markdown file from supporting resources folder
		st.subheader("Background")
		st.markdown("Many companies are built around lessening one’s environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received. ")
		st.subheader("How to Make a Prediction")
		st.markdown("The following steps will provide you with a seamless interaction with our tweet classifiction app.")

	# Building out the "Exploratory Data Analysis" page
	if selection == "Exploratory Data Analysis":
		st.title("Exploratory Data Analysis")

		logo = Image.open('logo.jpg')
		st.sidebar.image(logo, use_column_width=True)

		st.subheader("Sentiments")
		# You can read a markdown file from supporting resources folder
		st.markdown("2 = News : Tweets linked to factual news about climate change.")
		st.markdown("1 = Pro : Tweets that support the belief of man-made climate change.")
		st.markdown("0 = Neutral : Tweets that neither support nor refuse beliefs of climate change.")
		st.markdown("-1 = Anti : Tweets that do not support the belief of man-made climate change.")

		st.subheader("Pie chart distribution of sentiments in percentage")
        
		labels = 'Pro', 'News', 'Neutral', 'Anti'
		sizes = [8530, 3640, 2353, 1296]
		explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

		fig1, ax1 = plt.subplots()
		ax1.pie(sizes, explode=explode, labels=labels, autopct='%0.1f%%',
        shadow=True, startangle=90)
		ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

		st.pyplot(fig1)

	# Building out the "Model" page
	if selection == "Model":
		st.title("Model")

		logo = Image.open('logo.jpg')
		st.sidebar.image(logo, use_column_width=True)

		st.subheader("What is Logistic Regression?")
		# You can read a markdown file from supporting resources folder
		st.markdown("Logistic regression estimates the probability of an event occurring, such as voted or didn’t vote, based on a given dataset of independent variables. It is often used for classification and predictive analytics. ")
		st.markdown("There are algebraically equivalent ways to write the logistic regression model:")
		st.markdown("The first is")
		st.latex(r'''
		P(X) = \displaystyle \frac{e^{\beta_0 + \beta_1 X}}{1+e^{\beta_0 + \beta_1 X}}
     	''')
		st.markdown("which is an equation that describes the odds of being in the current category of interest. By definition, the odds for an event is π / (1 - π) such that P is the probability of the event.")
		st.markdown("The second is")
		st.latex(r'''
		\begin{align}
		1 - P(X) &= \displaystyle \frac{1}{1+e^{\beta_0 + \beta_1 X}} \\
		\therefore \log \left( \frac{P(X)}{1-P(X)} \right) &= {\beta_0 + \beta_1 X}
		\end{align}
     	''')
		st.markdown("which states that the (natural) logarithm of the odds is a linear function of the X variables (and is often called the log odds). This is also referred to as the logit transformation of the probability of success, π.")
		st.subheader("Types of Logistic Regression")
		st.markdown("- Binary logistic regression")
		st.markdown("- Multinomial logistic regression")
		st.markdown("- Ordinal logistic regression")
		st.subheader("Advantages")
		st.markdown("- Logistic regression is much easier to implement than other methods, especially in the context of machine learning.")
		st.markdown("- Logistic regression works well for cases where the dataset is linearly separable.")
		st.markdown("- Logistic regression provides useful insights.")
		st.subheader("Disadvantages")
		st.markdown("- Logistic regression fails to predict a continuous outcome.")
		st.markdown("- Logistic regression assumes linearity between the predicted (dependent) variable and the predictor (independent) variables.")
		st.markdown("- Logistic regression may not be accurate if the sample size is too small.")

	# Building out the "Contact Us" page
	if selection == "Contact Us":
		st.title("Contact Us")

		logo = Image.open('logo.jpg')
		st.sidebar.image(logo, use_column_width=True)
		
		st.subheader("Our Company")
		st.markdown("Solid Solutions is an innovation tech company with a key focus on creating up to date technological products designed to make light of any problem thrown our way. We are extremely passionate about giving back to the community. Strengthening Today for a Stronger Tomorrow!")
		# You can read a markdown file from supporting resources folder
		col1, col2, col3, col4, col5, col6 = st.columns(6)
		img1 = Image.open("Robyn1.jpg")
		img2 = Image.open("Hendrick.jpg")
		img3 = Image.open("Robyn4.jpg")
		img4 = Image.open("Morema.jpg")
		img5 = Image.open("Robyn6.jpg")
		img6 = Image.open("Robyn1.jpg")
		
		with col1:
			st.caption("Market Technologist")
			st.image(img1)
			st.caption("Elizabeth Pata Matlala")

		with col2:
			st.caption("Software Developer")
			st.image(img2)
			st.caption("Hendrick Makau")

		with col3:
			st.caption("CEO")
			st.image(img3)
			st.caption("Mokgadi Precious Makgothoma")

		with col4:
			st.caption("Full-stack Developer")
			st.image(img4)
			st.caption("Morema Moloisi")

		with col5:
			st.caption("Information Architect")
			st.image(img5)
			st.caption("Njabulo Mudau")

		with col6:
			st.caption("UI/UX Designer")
			st.image(img6)
			st.caption("Robyn van der Merwe")
			

		st.subheader("Message Us")
		with st.form("form", clear_on_submit=True):
			name = st.text_input("Enter Full Name")
			email = st.text_input("Enter Email Address")
			message = st.text_area("Message")

			submit = st.form_submit_button("Submit")

		col1, col2, col3 = st.columns(3)
		with col1:
			st.subheader("Address")
			img = Image.open("map.png")
			st.image(img)
			st.markdown("1004 Otto du Plesis")
			st.markdown("Cape Town")
			st.markdown("8001")

		with col2:
			st.subheader("Phone")
			st.markdown("Monday - Friday")
			st.markdown("08h00 - 17h00 GMT+2")
			st.markdown("(+27) 021 554 1091")
			st.markdown("(+27) 084 553 4721")

		with col3:
			st.subheader("Email")
			st.markdown("robynvandermerwe@gmail.com")
			st.markdown("robynvandermerwe@yahoo.com")


	# Building out the Home page
	if selection == "Home":
		st.title("Tweet Classifer")

		option = st.sidebar.multiselect(
     	'Sentiment:',
     	('Pro', 'News', 'Neutral', 'Anti'))

		tweet = st.sidebar.radio(
     	"Tweet:",
     	('All', 'Original', 'Re-Tweet'))
		if tweet == 'All':
			st.write('You selected All tweets.')

		if tweet == 'Original':
			st.write('You selected Original tweets.')

		if tweet == 'Re-Tweet':
			st.write('You selected Re-Tweets.')

		container = st.container()
		container.write("")
		
		logo = Image.open('logo.jpg')
		st.sidebar.image(logo, use_column_width=True)
		
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
