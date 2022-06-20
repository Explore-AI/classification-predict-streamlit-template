# streamlit dependencies
import streamlit as st

# data dependencies
import pandas as pd
from PIL import Image
from nlppreprocess import NLP
nlp = NLP()

st.info('Explained, Gathered, Analyzed & Deployed by Multi-Skilled Data Professionals')

# define Github links for each team member
link1 = "https://github.com/ThulaniNyama"
link2 = "https://github.com/KganyagoE"
link3 = "https://github.com/Immaculate180"
link4 = "https://github.com/ngwenyanv"
link5 = "https://github.com/GiftNhlenyama"
link6 = "https://github.com/ZanderM14"
# define Pandas data frame with team members that developed the models, and the app
df = pd.DataFrame(
    {
        "Team-CW3 Members": [
            f'<a target="_blank" href="{link1}">Thulani Nyama</a>',
            f'<a target="_blank" href="{link2}">Ephraim Kganyago</a>',
            f'<a target="_blank" href="{link3}">Immaculate Makokga</a>',
            f'<a target="_blank" href="{link4}">Nhlanhla Ngwenya</a>',
            f'<a target="_blank" href="{link5}">Sqiniseko Sizwe Gift Nhlenyama</a>',
            f'<a target="_blank" href="{link6}">Zander Maré</a>'
        ],
        "Profession": ["Data Scientist", "Data Analyst", "Data Engineer", "Data Scientist", "Data Scientist", "Data Engineer"]
    }
)
st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
st.write("")
# footer display image with caption 
image = Image.open('./resources/imgs/EDSA_logo.png')
st.image(image, caption='© TEAM-CW3', use_column_width=True)