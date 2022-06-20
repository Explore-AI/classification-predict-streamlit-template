import streamlit as st
import plotly.figure_factory as ff
import pandas as pd


def plot_line_and_bar():
    # Add histogram data
    data = pd.read_csv('https://raw.githubusercontent.com/TEAM-CW3/classification-predict-streamlit-data/main/train.csv')
    # display here
    st.bar_chart(data)