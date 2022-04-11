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
import joblib
import os
import base64

# Data dependencies
import pandas as pd

st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous" >', unsafe_allow_html=True)

st.markdown("""
<nav class="d-flex navbar fixed-top navbar-expand-lg navbar-dark py-5 " style="background-color: #000000; z-index: 1050; height: 10px">
	<div class="d-flex justify-content-around container" >
        <div class="logo_holder">
		</div>
		<div class="collapse navbar-collapse" id="navbarNav">
			<a class="navbar-brand mx-auto  " href="" target="_blank">Home</a>
		</div>
		<div class="collapse navbar-collapse" id="navbarNav" >
			<a class="navbar-brand mx-auto  " href="" target="_blank">Application</a>
		</div>
		<div class="collapse navbar-collapse" id="navbarNav">
			<a class="navbar-brand mx-auto  " href="" target="_blank">Contact</a>
		</div>
	</div>
</nav>
""", unsafe_allow_html=True)


# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
		<style>

		header{
			background-color: black !important;
            z-index: -1 !important ;
		}
        .block-container{
            border: 2px ;
        }

		.logo_holder {
        height: 110px;
        width: 110px ;
		background-image: url("data:image/png;base64,%s");
		background-size: contain;
		background-repeat: no-repeat;
		}

       

		</style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

set_png_as_page_bg('logo_2.png')


def main():
    """Tweet Classifier App with Streamlit"""

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Tweet Classifier App")
    # st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            # will write the df to the page
            st.write(raw[['sentiment', 'message']])

    # Building out the predication page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter tweet below", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(
                open(os.path.join("resources/Logistic_regression.pkl"), "rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
