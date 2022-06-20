import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

df_train = pd.read_csv('https://raw.githubusercontent.com/TEAM-CW3/classification-predict-streamlit-data/main/train.csv') # load train data set

def missing_vals():
    # plot missing values in train set
    df_train['message'] = df_train.message.str.replace(r"(http[^\s]+)", 'link', regex=True)
    ax = df_train.isna().sum().sort_values().plot(kind = 'barh', figsize = (9, 10), color='tab:blue')
    ff.create_distplot('Percentage of Missing Values Per Column in Train Set')
    for p in ax.patches:
        percentage ='{:,.0f}%'.format((p.get_width()/df_train.shape[0])*100, fontdict={'size':18})
        width, height =p.get_width(),p.get_height()
        x=p.get_x()+width+0.02
        y=p.get_y()+height/2
        ax.annotate(percentage,(x,y))