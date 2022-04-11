
# # Streamlit dependencies
import streamlit as st
import joblib
import os
import base64
import re
import string

from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk import SnowballStemmer
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)
# Load your raw data
df_train = pd.read_csv("resources/train.csv")
setinments_dict = {2: 'Info', 1: "Pro", 0: "Neutral", -1: "Anti"}

# Load externsl css file
with open('base_app_css.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_company_logo(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
		<style>
		.logo_holder{
            background-image: url("data:image/png;base64,%s");
            background-size: contain;
            background-repeat: no-repeat;
		}

		</style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_company_logo('logo_2.png')


def set_company_image(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
		<style>
		.image{
            background-image: url("data:image/png;base64,%s");
            background-size: contain;
            background-repeat: no-repeat;
		}

		</style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_company_image('data_science.jpeg')


st.sidebar.markdown('''
# Sections
- [Home](#home)
- [M.L Application](#applications)
- [Data Insights](#insights)
''', unsafe_allow_html=True)

st.header('home')
st.markdown("""
<section class="home_page">
    <div class="content" >
        <div class=" logo_and_name ">
                <div class="company_logo" >
                    <div class="logo_holder" > </div >
                </div>
                <div class="name">
                    <p class="name_tag">SWAT Analytics Consulting</p>
                </div>
        </div>
        <div class="vision_statement">
            <h2 class ="statement">Vision Statement</h2>
            <p class="statement_words">
                To build data analytic and business insight capabilities to empower associates  
                with the right information at the right time  to drive data-driven decisions. 
            </p>
        <div class="vision_statement">
            <h2 class ="statement">Mission Statement</h2>
            <p class="statement_words">
                Our mission is to make it easier for more people to use powerful analytics every day, 
                to shorten the path from data to insight – and to inspire bold new discoveries that 
                drive progress. The result is analytics that breaks down barriers, fuels ambitions and 
                gets results.
            </p>
        </div>
        </div>
    </div>
    <div class="image_holder">
        <div class="image"></div>
    </div>
</section>
""", unsafe_allow_html=True)


# APPLICAATION
st.header('applications')
with st.container():
    col1, col2, col3 = st.columns(3)
    col1.markdown("""
        <div class="model_board">
            <h3 class="app_intro"> Tweet Classifier App</h3>
            <p class="app_intro"> with </p>
            <h5 class="app_intro"> Logistic Regression</h5>
        </div>
    """, unsafe_allow_html=True)
    tweet_text_1 = col1.text_area("Enter tweet below", placeholder="Type Here")
    if col1.button("Classify"):
        # Transforming user input with vectorizer
        vect_text_1 = tweet_cv.transform([tweet_text_1]).toarray()
        # Load your .pkl file with the model of your choice + make predictions
        # Try loading in multiple models to give the user a choice
        predictor = joblib.load(
            open(os.path.join("resources/Logistic_regression.pkl"), "rb"))
        prediction_1 = predictor.predict(vect_text_1)

        # When model has successfully run, will print prediction
        # You can use a dictionary or similar structure to make this output
        # more human interpretable.
        col1.success("Text Categorized as: {}".format(prediction_1))

    # Random Forest
    col2.markdown("""
        <div class="model_board">
            <h3 class="app_intro"> Tweet Classifier App</h3>
            <p class="app_intro"> with </p>
            <h5 class="app_intro"> MultinomialNB </h5>
            <div class="model_summary"></div>
        </div>
    """, unsafe_allow_html=True)
    tweet_text_2 = col2.text_area(
        "Enter tweet below", placeholder="Type Here", key="random")
    if col2.button("Classify", key="random"):
        # Transforming user input with vectorizer
        vect_text_2 = tweet_cv.transform([tweet_text_2]).toarray()
        # Load your .pkl file with the model of your choice + make predictions
        # Try loading in multiple models to give the user a choice
        predictor_2 = joblib.load(
            open(os.path.join("resources/MultinomialNB.pkl"), "rb"))
        prediction_2 = predictor_2.predict(vect_text_2)

        # When model has successfully run, will print prediction
        # You can use a dictionary or similar structure to make this output
        # more human interpretable.
        col2.success("Text Categorized as: {}".format(prediction_2))


# SUPPORT VECTOR
    col3.markdown("""
        <div class="model_board">
            <h3 class="app_intro"> Tweet Classifier App</h3>
            <p class="app_intro"> with </p>
            <h5 class="app_intro"> Support Vector Machine</h5>
            <div class="model_summary"></div>
        </div>
    """, unsafe_allow_html=True)
    tweet_text_3 = col3.text_area(
        "Enter tweet below", placeholder="Type Here", key="svc")
    if col3.button("Classify", key="svc"):
        # Transforming user input with vectorizer
        vect_text_3 = tweet_cv.transform([tweet_text_3]).toarray()
        # Load your .pkl file with the model of your choice + make predictions
        # Try loading in multiple models to give the user a choice
        predictor_3 = joblib.load(
            open(os.path.join("resources/LinearSVC.pkl"), "rb"))
        prediction_3 = predictor_3.predict(vect_text_3)

        # When model has successfully run, will print prediction
        # You can use a dictionary or similar structure to make this output
        # more human interpretable.
        col3.success("Text Categorized as: {}".format(prediction_3))


def delete_url(data, col):
    """
        Accepts a dataframe and col., removes web urls from the col.
        returns a new dataframe
    """
    df = data.copy()
    pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    subs_url = ''
    df[col] = df[col].replace(to_replace=pattern_url,
                              value=subs_url, regex=True)
    return df


def delete_tags(data, col):
    """
        This function takes in a dataframe and a col, removes all words started with '#' and '@' in the column,
        and returns a new dataframe
    """
    df = data.copy()
    pattern_tags = r'#\w+[#?]'
    pattern_2 = r'@\w+'
    pattern_3 = r'[0-9]+'
    pattern_4 = r'[^\x00-\x7f]'  # Pattern for unicode
    subs_tag = ''
    df[col] = df[col].replace(to_replace=pattern_tags,
                              value=subs_tag, regex=True)
    df[col] = df[col].replace(to_replace=pattern_2, value=subs_tag, regex=True)
    df[col] = df[col].replace(to_replace=pattern_3, value=subs_tag, regex=True)
    # Where it is being removed
    df[col] = df[col].replace(to_replace=pattern_4, value=subs_tag, regex=True)
    return df


def word_converter(data, col):
    """
        This function takes in a dataframe and col, converts all capitalized words in the column to lowercase,
        and returns a new dataframe.
    """
    df = data.copy()
    df[col] = df[col].str.lower()
    return df


def remove_punc(data, col):
    """
        This function takes in a dataframe and a column, uses python string package to identify and remove all
        punctions in the column. It returns a new dataframe
    """
    def operation(post):
        return ''.join([l for l in post if l not in string.punctuation])

    df = data.copy()

    df[col] = df[col].apply(operation)
    return df


def remove_new_line(data, col):
    """
        Takes in a dataframe and a column, returns a new dataframe with a new column void of new line command
    """
    def operation(text):
        result = re.sub("\n", "", text)
        result = re.sub("rt", "", result)
        return result

    df = data.copy()
    df[col] = df[col].apply(operation)
    return df


def tokenizer(data, col):
    """
        This function takes in a dataframe and a col, creates a new column to store the tokenized words
        in the inputed column, and returns a new dataframe.
    """
    df = data.copy()
    tokeniser = TreebankWordTokenizer()
    df['message_tok'] = df[col].apply(tokeniser.tokenize)
    return df


def lam_words(data, col):
    """
        Takes in a dataframe and a column, converts the words in the column to it root form,
        with the aid of WordNetLemmatizer class from the nltk package.
        Returns a new dataframe with an additional column "message_lam"
    """
    lemmatizer = WordNetLemmatizer()

    def operation(words, lemmatizer):
        return [lemmatizer.lemmatize(word) for word in words]
    df = data.copy()
    df["message_lam"] = df[col].apply(operation, args=(lemmatizer, ))

    return df


def remove_stop_words(data, col):
    """
        Takes a dataframe and a column, creates a new dataframe with a new column no_stop_word from the input
        dataframe and column, returns the new column
    """
    def operation(toks):
        new_toks = [
            tok for tok in toks if tok not in stopwords.words('english')]
        new_toks = [tok for tok in new_toks if len(tok) > 1]
        return new_toks

    df = data.copy()
    df['no_stop_word'] = df[col].apply(operation)

    return df


def create_doc_list(df, filt_col, text_col):
    """
        This function takes a dataframe, the column with repeated values ,and a tokenized text column.
        it returns a list of documents for all repeated values in the 'filt_col'.
    """

    # Generate unique values for repeated values column(sentiments)
    sentiments = df[filt_col].unique().tolist()

    # Create empty document list for all sentiment types
    doc_list = []

    # Generate documents by sentiment from no_stop_word column in the work_df dataframe
    for sentiment in sentiments:

        # Create a document for each sentiment
        doc = []

        # Filter dataframe by sentiment
        sentiment_df = df[df[filt_col] == sentiment]

        # Loop through each row in the dataframe and add the text value to the document
        for list_ in sentiment_df[text_col]:
            doc = doc + list_

        # Add the sentiment document to the central document list
        doc_list.append(doc)
     # return the general sentiment list
    return doc_list


new_df_train = delete_url(df_train, 'message')
# Create a new dataframe with message colun void of url links
new_df_train = delete_tags(new_df_train, 'message')
# Create a new dataframe with all words in the message column converted to its lowercase form
new_df_train = word_converter(new_df_train, 'message')
# Create a new dataframe with the message colmn void of punctuations
new_df_train = remove_punc(new_df_train, 'message')
# Create a new dataframe with the message colmn void of punctuations
new_df_train = remove_new_line(new_df_train, 'message')
# Create a new column to hold the tokens from message column
new_df_train = tokenizer(new_df_train, 'message')
# Create a new column to hold root words from stemmer
new_df_train = lam_words(new_df_train, 'message_tok')
# Create a new column from message_lam void of stop words
new_df_train = remove_stop_words(new_df_train, 'message_lam')
# Collect only intrested columns for visualization
work_df = new_df_train[['sentiment', 'no_stop_word']]
unique_vals = work_df['sentiment'].unique()
# Generate documents by sentiments
doc_list = create_doc_list(work_df, 'sentiment', 'no_stop_word')
all_fdist_1 = FreqDist(doc_list[0]).most_common(20)
row_1 = []
col_1 = []
for data in all_fdist_1:
    row_1.append(data[0])
    col_1.append(data[1])

all_fdist_2 = FreqDist(doc_list[1]).most_common(20)

row_2 = []
col_2 = []
for data in all_fdist_2:
    row_2.append(data[0])
    col_2.append(data[1])

all_fdist_3 = FreqDist(doc_list[2]).most_common(20)
row_3 = []
col_3 = []
for data in all_fdist_3:
    row_3.append(data[0])
    col_3.append(data[1])

all_fdist_4 = FreqDist(doc_list[3]).most_common(20)
row_4 = []
col_4 = []
for data in all_fdist_4:
    row_4.append(data[0])
    col_4.append(data[1])

downsampled_df = pd.DataFrame({
    'PRO_Tweet': col_1
}, index=row_1)

downsampled_df_2 = pd.DataFrame({
    'NEUTRAL_Tweet': col_2
}, index=row_2)

downsampled_df_3 = pd.DataFrame({
    'INFO_Tweet': col_3
}, index=row_3)

downsampled_df_4 = pd.DataFrame({
    'Anti_Tweet': col_4
}, index=row_4)


fig, ax = plt.subplots()

data = df_train['sentiment'].value_counts()

#my_labels = '1','2','0','-1'
my_labels = 'Pro-climate change', 'Actual News', 'Neutral', 'Anti-climate change'
my_colors = ['green', 'lightgreen', 'grey', 'lightgrey']
ax.pie(data, labels=my_labels, autopct='%1.1f%%',
       startangle=15, shadow=False, colors=my_colors)
# ax.title('Tweet Class Distribution')
ax.axis('equal')

# Adding legend
ax.legend(my_labels,
          title="Class",
          loc="upper left",
          bbox_to_anchor=(1, 0, 0.5, 1))
# plt.show()

# DATA INSIGHT
st.header('insights')
with st.container():
    st.markdown("""
        <div class="model_board">
            <h3 class="app_intro"> Most Common Twenty Words among all categorries</h3>
        </div>
    """, unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    col4.bar_chart(data=downsampled_df, width=0,
                   height=0, use_container_width=True)
    col5.bar_chart(data=downsampled_df_2, width=0,
                   height=0, use_container_width=True)
    col6, col7 = st.columns(2)
    col6.bar_chart(data=downsampled_df_3, width=0,
                   height=0, use_container_width=True)
    col7.bar_chart(data=downsampled_df_3, width=0,
                   height=0, use_container_width=True)
    st.pyplot(fig)


# st.markdown("""
# <div class=" application_section " >
# APPLICATION SECTION
# </div>
# """, unsafe_allow_html=True)
