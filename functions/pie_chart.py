import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
# Number of Messages Per Sentiment
def plot_pie_chart():
    st.write('Distribution of the sentiments')
    # Labeling the target
    data = pd.read_csv('https://raw.githubusercontent.com/TEAM-CW3/classification-predict-streamlit-data/main/train.csv')
    data['sentiment'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1] for x in data['sentiment']]
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # checking the distribution
    st.write('The numerical proportion of the sentiments')
    values = data['sentiment'].value_counts()/data.shape[0]
    labels = (data['sentiment'].value_counts()/data.shape[0]).index
    colors = ['red', 'green', 'yellow', 'blue']
    plt.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=120, explode= (0.03, 0, 0, 0), colors=colors)
    st.pyplot()