import streamlit as st

# from Log_reg import run_log_reg

st.header("Fake Product Review Detection and Removal System Using Sentiment Analysis")
review = st.text_area("Enter the Review")
button = st.button('Check Review')

with st.sidebar:
    st.