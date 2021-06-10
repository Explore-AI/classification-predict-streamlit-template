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
import os

import joblib
# Data dependencies
import pandas as pd
# Streamlit dependencies
import streamlit as st

# Vectorizer
# @st.cache
def load_vectorizer(vectorizer):
    news_vectorizer = open(vectorizer, "rb")
    tweet_cv = joblib.load(news_vectorizer)
    return news_vectorizer, tweet_cv

# loading your vectorizer from the pkl file


news_vectorizer, tweet_cv = load_vectorizer("resources/tfidfvect.pkl")

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app


def main():
    """Tweet Classifier App with Streamlit """
    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Tweet Classifer")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Tutorial"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Defining model descriptions
    model_paths = {"Random Forest": "resources/Logistic_regression.pkl",
                   "LinearSVC":"resources/Logistic_regression.pkl", 
                   "Logistic Regression":"resources/Logistic_regression.pkl"}

    model_descriptions = {"Random Forest": "A random forest is a meta estimator that fits a \
						   					number of decision tree classifiers on various sub-samples of the \
						   					dataset and uses averaging to improve the predictive accuracy and control over-fitting.",

                          "LinearSVC":  "The implementation of C-Support Vector Classification is based on libsvm. The fit time scales at least quadratically with the \
							  		 	 number of samples and may be impractical beyond tens of thousands of samples. \
										 For large datasets consider using LinearSVC or SGDClassifier instead, \
										 possibly after a Nystroem transformer. \
										 The multiclass support is handled according to a one-vs-one scheme.\n\
										 Linear Support Vector Classification is similar to SVC (C-Support Vector Classification) with parameter kernel=’linear’,\
										 but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss \
										 functions and should scale better to large numbers of samples.",

                          "Logistic Regression": "Logistic Regression (aka logit, MaxEnt) classifier. In the multiclass case, the training algorithm uses the\
							  					  one-vs-rest (OvR) scheme if the ‘multi_class’ option is set to ‘ovr’, and uses the cross-entropy loss if the\
												  ‘multi_class’ option is set to ‘multinomial’. (Currently the ‘multinomial’ option is supported only by the\
												  ‘lbfgs’, ‘sag’, ‘saga’ and ‘newton-cg’ solvers.) This class implements regularized logistic regression using the\
												  ‘liblinear’ library, ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ solvers. Note that regularization is applied by default.\
												  It can handle both dense and sparse input. Use C-ordered arrays or CSR matrices containing 64-bit floats for optimal performance;\
												  any other input format will be converted (and copied)."
                          }

    # Building out the "Information" page
    if selection == "Tutorial":
        st.info("Tutorial")
        # You can read a markdown file from supporting resources folder
        st.markdown("Watch the video below to learn how to use the Tweet Classifier")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            # will write the df to the page
            st.write(raw[['sentiment', 'message']])

    # Building out the predication page
    if selection == "Prediction":
            st.subheader("Climate change tweet classification")
            st.info("Prediction with a selection of ML Models")
            chosen = st.radio('Model Selection', ("Random Forest", "LinearSVC", "Logistic Regression"))
			# Creating a text box for user input
            tweet_text = st.text_area("Type or paste your tweet here:", "Type Here")
            if st.button("Classify"):
                # Transforming user input with vectorizer
                vect_text = tweet_cv.transform([tweet_text]).toarray()
                # Load your .pkl file with the model of your choice + make predictions
                # Try loading in multiple models to give the user a choice
                predictor = joblib.load(open(os.path.join(model_paths[f"{chosen}"]), "rb"))
                prediction = predictor.predict(vect_text)

                # When model has successfully run, will print prediction
                # You can use a dictionary or similar structure to make this output
                # more human interpretable.
                if prediction > 0:
                    st.success("This tweet was written by a believer of global warming")
                else:
                    st.success("This tweet was written by a disbeliever of global warming")
            st.subheader(f"{chosen}")
            st.markdown(f"Description from documentation: \n\n {model_descriptions[f'{chosen}']}")






# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
