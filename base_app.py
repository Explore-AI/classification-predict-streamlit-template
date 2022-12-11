
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import plotly.express as px

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url('https://anhdepfree.com/wp-content/uploads/2019/05/50-anh-background-dep-nhat-4.jpg');
background-size: cover;
}
[data-testid="stHeader"] {
background-color: rgba(0,0,0,0);

}

[data-testid="stToolbar"] {
right: 2rem;
}

[data-testid="stSidebar"] {
background-image: url('https://images.pexels.com/photos/2088203/pexels-photo-2088203.jpeg?auto=compress&cs=tinysrgb&w=600');
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


	st.info("This application is all about tweet sentiment analysis of climate change. It is able to classify whether" 
			 "or not a person believes in climate change, based on their novel tweet data.")


	# Building out the "Tweet Sentitment analysis " page		
	if selection == "Tweet analysis":
		st.info("This app analyses sentiments on climate change based on tweet data")
		#top level filters
		#message_filter = st.selectbox("Select the message", pd.unique(raw['sentiment']))
		# dataframe filter
		#df = raw[raw['sentiment']== message_filter] 
		st.markdown("### Tweet distribution")
		sentiment = raw['sentiment'].value_counts()
		sentiment = pd.DataFrame({'Sentiment':sentiment.index, 'Tweets':sentiment.values})
	
		# create two columns for charts
		fig_col1, fig_col2 = st.columns(2)
		
		with fig_col1:
			fig = fig = px.bar(sentiment, x='Sentiment', y='Tweets', color = 'Tweets', height= 500)
			st.plotly_chart(fig)
	
       	#
		with fig_col2:
			fig = px.pie(sentiment, values= 'Tweets', names= 'Sentiment')
			st.plotly_chart(fig)
			
		


