# Twitter Climate Sentiment Classification Project
## Project Description
In this project, we built a classification model to predict climate-related sentiment in tweets. The aim is to help companies determine how people perceive climate change based on their tweets. This information can assist companies in understanding how their product/service may be received in the context of climate sentiment. 

We explored various Supervised Statistical Learning models including Logistic Regression Classifier, Support Vector Machine, Naive Bayes model, and Random Forest to identify the best classifier. GridSearchCV was utilized to select the best parameters for our final model.

## Getting Started Guide
Follow these steps to get started with the project:
### Step 1: Install Python
Ensure that you have the latest version of Python installed, preferably Python 3.10.11. 
If you haven't already installed it, you can do so by running the following command:

```python
pip install ipython
```

### Step 2: Download Necessary Corpora and Model
You need to download the required corpora and model to aid with stopword removal and tokenization. 
Open a Python environment and execute the following commands:

```python
import nltk
nltk.download(['punkt', 'stopwords'])
```

### Step 3: Install Dependencies

Install the project dependencies including pandas, numpy, matplotlib, and scikit-learn using the following command:

```python
pip install -U matplotlib numpy pandas scikit-learn
```

## Usage
- Open your preferred Python environment or notebook.
- Import the necessary libraries.
- Load the data onto the notebook or import the "clean_train_csv" file directly to skip the cleaning process.
- Fit the data into the selected model. The model used for this project is the Support Vector Machine (SVM). You can experiment with different model types and tweak the parameters to suit your requirements.

## Project Structure
The project repository consists of the following folders/files:

- train.csv: Contains raw tweets and sentiments used for training the model.
- test_with_no_labels.csv: Contains raw tweets without labels, which can be used as a testing dataset.
- clean_train.csv: Contains the clean training data. You can load this file directly to skip the cleaning process.
- clean_test.csv: Contains the clean test data. You can load this file directly to skip the cleaning process.

## Development
We also developed a web application using Streamlit for easy interaction with our model. You can navigate to our app repository by following this link: [https://github.com/TheZeitgeist-RR12/Streamlit-App.git]
Feel free to explore, experiment, and contribute to the project.

## Conclusion
By developing this machine learning model, we provide companies in the environmentally friendly and sustainable sector with valuable insights into consumer sentiment. This, in turn, can inform their marketing strategies, help them better understand their target market, and help them improve their products.

