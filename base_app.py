
# Streamlit dependencies
import streamlit as st
import joblib
import os
import base64

# Data dependencies
import pandas as pd


st.markdown("""
<div class=" my_navbar " >
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class=" intro " >
</div>
""", unsafe_allow_html=True)

# Load externsl css file
with open('base_app_css.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
