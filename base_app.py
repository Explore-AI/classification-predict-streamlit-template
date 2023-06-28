# Streamlit dependencies
import re
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from streamlit import session_state
import requests
import streamlit.components.v1 as components


# Set up NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the news_vectorizer
with open("C:\\Users\\HP\\Downloads\\Weather Hub\\Team-BM-4-Streamlit-files\\news_vectorizer.pkl", "rb") as f:
    news_vectorizer = pickle.load(f)

# Load the trained model
with open("C:\\Users\\HP\\Downloads\\Weather Hub\\Team-BM-4-Streamlit-files\\my_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define Preprocessing
# Convert text to lowercase
def preprocess_lower(text):
    return text.lower()

# Function to remove stopwords from text
def remove_stopwords(text):
    stopword_list = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word.lower() not in stopword_list])
    return text

# Function to remove punctuation from text
def remove_punctuation(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Function to remove repeating characters from text
def remove_repeating_characters(text):
    text = re.sub(r'(.)\1+', r'\1', text)
    return text

# Function to remove URLs from text
def remove_urls(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    return text

# Function to preprocess special characters
def preprocess_special_chars(text):
    text = re.sub(r'(\W|â|Â|Ã)', ' ', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'RT', '', text)
    text = re.sub(r'https?\S+|www\S+|co\S+', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text

# Function to remove leading and trailing whitespaces
def preprocess_strip(text):
    return text.strip()

# Tokenize the text
def preprocess_tokenization(text):
    tokens = word_tokenize(text)
    return tokens

# Lemmatize the tokens
def preprocess_lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    text = ' '.join(lemmatized_tokens)
    return text

# Classification algorithm
def classify_message(message):
    # Preprocess the message
    message = preprocess_lower(message)
    message = remove_stopwords(message)
    message = remove_punctuation(message)
    message = remove_repeating_characters(message)
    message = remove_urls(message)
    message = preprocess_special_chars(message)
    message = preprocess_strip(message)
    tokens = preprocess_tokenization(message)
    message = preprocess_lemmatization(tokens)

    # Vectorize the message
    vectorized_message = news_vectorizer.transform([message])

    # Make predictions
    predicted_class = model.predict(vectorized_message)[0]

    return predicted_class



class _SessionState:
    def __init__(self):
        self.redirect_to_homepage = False

# Create an instance of the session state
session_state = _SessionState()

# Custom CSS styles
custom_styles = """
<style>
body {
    color: white;
    background-color: black;
}

.stButton button {
    color: white !important;
    background-color: purple !important;
}

.stTextInput > div > div > input {
  color: white !important;
  background-color: #333333 !important;
  z-index: 1;
}

.stDataframe>div>div>div>table {
    color: black !important;
    background-color: white !important;
}

.sidebar {
    width: 180px !important;
}

.sidebar-content {
    background-color: purple !important;
    width: 160px !important;
}

.sidebar .sidebar-content .block-container {
    color: white !important;
    background-color: purple !important;
}

.sidebar .sidebar-content .block-container .block {
    border-color: white !important;
    background-color: inherit !important;
}

.sidebar .sidebar-content .stButton button {
    color: white !important;
    background-color: purple !important;
    border-color: white !important;
}

.sidebar .sidebar-content .stButton button:hover {
    background-color: white !important;
    color: purple !important;
}

.sidebar .sidebar-content .stButton button:active {
    background-color: purple !important;
    color: white !important;
}

.streamlit-button.button-small {
    background-color: purple !important;
    color: white !important;
    border-color: white !important;
}

.streamlit-button.button-small:hover {
    background-color: white !important;
    color: purple !important;
}

.streamlit-button.button-small:active {
    background-color: purple !important;
    color: white !important;
}

.header-button-content {
    background-color: #ffcc00;
    padding: 10px;
}



/* Custom styles for feedback input field */
div.stTextInput.feedback-input>div>div>textarea {
    background-color: silver !important;
}

/* Custom styles for CSV upload area */
div.stFileUploader.css-1qgjklv>div>div.css-1vcdhb2 {
    background-color: silver !important;
}
</style>
"""

# Display custom CSS styles
st.markdown(custom_styles, unsafe_allow_html=True)

# Add a catchy headline
st.header("WeatherHub")

# Create a horizontal layout for the buttons
col1, col2, col3, col4, col5 = st.columns(5)

# Create a button for "WeatherHub" that reloads the homepage
if col1.button("WeatherHub"):
    # Rerun the app to go back to the homepage
    st.experimental_rerun()

# Display expander for "Help"
if col2.button("Help"):
    with st.expander("Categories Explanation"):
        st.write("- Belief: Messages expressing belief in climate change.")
        st.write("- No Belief: Messages expressing no belief in climate change.")
        st.write("- Neutral: Messages that are neutral or do not express a clear stance on climate change.")
        st.write("- News: Messages related to news articles or factual information about climate change.")

    with st.expander("Instructions"):
        st.write("To classify a message, enter the text in the input box and click the 'Classify Message' button.")

    with st.expander("Additional Information"):
        st.write("For further assistance or questions, please refer to the user manual or contact our support team.")

    with st.expander("User Manual"):
        st.write("Download the User Manual for detailed instructions on using the classification system.")
        download_link = '[Download User Manual](https://www.manual@weatherhub.com/user_manual.pdf)'
        st.markdown(download_link, unsafe_allow_html=True)

# Display button for "About Us"
if col3.button("About Us"):
    with st.expander("Here at WeatherHub"):
        st.write("At WeatherHub, we are a passionate team dedicated to providing cutting-edge solutions for companies seeking to unlock future success through climate intelligence. With a deep understanding of the impact weather patterns have on consumer behavior, market trends, and strategic decision-making, we are here to help businesses make informed choices that drive growth and profitability.")

# Display button for "FAQs"
if col4.button("FAQs"):
    faqs = [
        {
            "question": "What is the purpose of this website?",
            "answer": "The purpose of this website is to help users classify messages into different categories such as belief, no belief, neutral, and news fact. It uses machine learning algorithms to analyze the input message and provide a classification based on its content."
        },
        {
            "question": "How accurate is the classification of messages?",
            "answer": "The accuracy of message classification can vary depending on various factors, including the quality and diversity of the training data, the performance of the classification algorithm, and the complexity of the messages themselves. Continuous improvement and refinement of the algorithm based on user feedback are important for enhancing classification accuracy."
        },
        {
            "question": "Can I trust the classification results without any doubts?",
            "answer": "While the classification algorithm is designed to provide accurate results, it's important to remember that no classification system is perfect. There might be cases where the algorithm misclassifies messages due to the complexity of language and the nuances of individual messages. Users are encouraged to review the classification results and provide feedback to further enhance the accuracy of the system."
        },
        {
            "question": "How can I provide feedback on the classification results?",
            "answer": "You can provide feedback on the classification results by using the feedback mechanism provided on the website. There will be an option to provide feedback on whether you agree or disagree with the classification assigned to a specific message. Your feedback will be valuable in improving the system's performance over time."
        },
        {
            "question": "What happens to the messages and data that I enter into the website?",
            "answer": "The messages and data you enter into the website are used solely for the purpose of classification and improving the performance of the system. The data is treated with strict confidentiality and privacy measures. It is not shared with any third parties without your consent and is handled in accordance with privacy laws and regulations."
        },
        {
            "question": "Can I use my own pre-trained model or customize the classification algorithm?",
            "answer": "Currently, the system is designed to use a specific classification algorithm. However, if you have your own pre-trained model or would like to customize the algorithm, you can reach out to the website administrators or developers to discuss the possibility of integrating your model into the system. Customization options may vary depending on the specific implementation and requirements."
        }
    ]

    for faq in faqs:
        with st.expander(faq["question"]):
            st.write(faq["answer"])

# Display dropdown for "Privacy and Security"
if col5.button("Privacy"):
    with st.expander("Priority"):
        st.write("At WeatherHub, we prioritize the privacy and security of our users' data. This privacy statement outlines how we handle and protect the information collected through this application.")
        
    with st.expander("Information Collection and Usage"):
        st.write("User Input:")
        st.write("When you use our application, we may collect and store the messages you enter for the purpose of classification and improving our system's performance. This data is handled with strict confidentiality and is not shared with third parties without your consent.")
        
        st.write("Usage Data:")
        st.write("We may collect usage data, such as the number of messages processed, the time spent on the application, and other similar metrics. This information is used for analyzing and improving the performance of our application.")
        
        st.write("Cookies:")
        st.write("We may use cookies to enhance your browsing experience and ensure the proper functioning of our application. These cookies do not contain personally identifiable information and can be disabled through your browser settings.")
        
        st.write("Analytics:")
        st.write("We may utilize analytics tools to gather aggregated and anonymized information about user interactions with our application. This information helps us understand user behavior and improve our services.")
        
        st.write("Please refer to our full Privacy Policy for more detailed information.")

    with st.expander("Data Security"):
        st.write("Data Encryption:")
        st.write("We employ industry-standard encryption techniques to protect your data from unauthorized access and ensure its confidentiality.")
        
        st.write("Access Control:")
        st.write("Access to user data is restricted to authorized personnel only, and strict security measures are in place to prevent unauthorized access or disclosure.")
        
        st.write("Data Retention:")
        st.write("We retain user data only for as long as necessary to fulfill the purposes for which it was collected and to comply with legal requirements.")
        
        st.write("Third-Party Services:")
        st.write("We carefully evaluate and select third-party services that we integrate with our application to ensure they meet the highest standards of security and privacy.")
        
        st.write("If you have any concerns or questions regarding the privacy and security of your data, please contact us for assistance.")


# Add an engaging visual element as the background
st.image("https://drive.google.com/uc?id=1EZ9SK1aBZzS6r7drXdjmoAvfwE4NcHmz", use_column_width=True)

# Option to enter a message
message_input = st.empty()
default_text = "Enter a message"
message_value = message_input.text_input(label='Enter a message', value='', key='message_input')


# Initialize classification result variable
classification_result = ""

# Classify a single message
if st.button("Classify Message"):
    if message_value and message_value != default_text:
        # Predict the classification
        prediction = classify_message(message_value)

        # Map the prediction to the corresponding class label
        if prediction == 1:
            classification_result = "Belief"
        elif prediction == -1:
            classification_result = "No Belief"
        elif prediction == 0:
            classification_result = "Neutral"
        elif prediction == 2:
            classification_result = "News"

        st.write("Prediction:", classification_result)

# Display the classification result
st.subheader("Classification Result")
st.write("The message is classified as:", classification_result)

# Option to upload a CSV file
csv_file = st.file_uploader("Upload a CSV file")

# Classify messages in a CSV file
if st.button("Classify CSV"):
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        messages = df["message"]

        st.write("Classifying messages...")
        # Preprocess the messages if needed
        preprocessed_messages = [classify_message(msg) for msg in messages]

        # Map the predictions to the corresponding class labels
        class_labels = []
        for prediction in preprocessed_messages:
            if prediction == 1:
                class_labels.append("Belief")
            elif prediction == -1:
                class_labels.append("No Belief")
            elif prediction == 0:
                class_labels.append("Neutral")
            elif prediction == 2:
                class_labels.append("News")

        df["classification"] = class_labels

        st.write("Classification results:")
        st.dataframe(df)

        # Display a bar chart of the classification results if there are multiple messages
        if len(df) > 1:
            st.subheader("Classification Distribution")
            count_values = df["classification"].value_counts()
            plt.bar(count_values.index, count_values.values)
            plt.xlabel("Classification")
            plt.ylabel("Count")
            plt.title("Distribution of Message Classifications")
            st.pyplot()

# Display the classification result
st.subheader("Classification Result")
st.write("The message is classified as:", classification_result)

# Option to export classification results
if st.button("Export Results"):
    if csv_file is not None:
        # Export the classification results to a CSV file
        df.to_csv("classification_results.csv", index=False)
        
        # Generate a download button for the exported file
        with open("classification_results.csv", "rb") as file:
            data = file.read()
        b64 = base64.b64encode(data).decode('utf-8')
        href = f'<a href="data:file/csv;base64,{b64}" download="classification_results.csv"><button>Download Results</button></a>'
        st.markdown(href, unsafe_allow_html=True)
        st.success("Classification results exported successfully!")

# Display expander for "Our Team"
if st.sidebar.button("Our Team", key="our_team_button"):
    # Display our team section content
    team_members = [
        {
            "name": "Abiodun Adeagbo - CEO",
            "image_url": "https://drive.google.com/uc?id=1gOmV3fVHgmZ1UfRxbiuGeKovnB87Ud5d",
        },
        {
            "name": "Sandisiwe Mtsha - CTO",
            "image_url": "https://drive.google.com/uc?id=1y1cBIbUcCaTl7hm6zXMz5FYvJ5Sp0sUK"
        },
        {
            "name": "Andisiwe Jafta - Climate Visionary",
            "image_url": "https://drive.google.com/uc?id=1NdID53xKtPdpdO6db9M0_oytuZhdH3PR"
        },
        {
            "name": "Pere Ganranwei - Climate Research Expert",
            "image_url": "https://drive.google.com/uc?id=1glHkFxTxuwaOBQwzRqB0WoZYQsv15_Qx",
        },
        {
            "name": "Lerato Manana - Data Specialist",
            "image_url": "https://drive.google.com/uc?id=196SLPJi3jTSQe32LlXjTm-kAJ3-dshFH",
        },
        {
            "name": "Kolawole Olawale - Software Engineer",
            "image_url": "https://drive.google.com/uc?id=1ptZ3j5p_Ps1VcpOrtBaTVE6THpvvosEU",
        }
    ]

    for member in team_members:
        st.sidebar.write(
            "<div style='display: flex; flex-direction: column; align-items: center; text-align: center; margin-bottom: 20px;'>"
            "<div style='margin-bottom: 10px;'>"
            f"<img src='{member['image_url']}' style='width: 150px;'>"
            "</div>"
            f"<div style='font-size: 14px; font-weight: bold; margin-top: 0;'>{member['name']}</div>"
            "</div>",
            unsafe_allow_html=True
        )

# Add a feedback form
feedback = st.text_area("Leave your feedback", height=100)

# Display contact information
st.subheader("Contact Us")
st.write("Email: info@weatherhub.com")
st.write("Phone: +2348012345678")

# Add social media icons
st.subheader("Follow Us")

# Create columns for the social media icons and buttons
col1, col2, col3, col4 = st.columns(4)

# Display the social media icons and buttons in each column
with col1:
    st.image("https://drive.google.com/uc?id=1TgLX4zdLZVzpSzW5AmvEv1TE6HlJJXhu", width=30)
    st.markdown('<button class="silver-button" style="color: purple;">@weatherhub</button>', unsafe_allow_html=True)

with col2:
    st.image("https://drive.google.com/uc?id=1z3gXVYw8OQFewzCuu7B0TZINQcVsIdc7", width=30)
    st.markdown('<button class="silver-button" style="color: purple;">@weatherhub</button>', unsafe_allow_html=True)

with col3:
    st.image("https://drive.google.com/uc?id=1jJKoDcncPF0p6TULf16qv1p--qeRd-eu", width=30)
    st.markdown('<button class="silver-button" style="color: purple;">@weatherhub</button>', unsafe_allow_html=True)

with col4:
    st.image("https://drive.google.com/uc?id=1rsl60zUj9SsTmAbHN-vUUe9wV95sLAbG", width=30)
    st.markdown('<button class="silver-button" style="color: purple;">@weatherhub</button>', unsafe_allow_html=True)

# Add copyright information and links
st.write("© 2023 Weather H
