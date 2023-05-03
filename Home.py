<<<<<<< HEAD
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
=======
import streamlit as st
>>>>>>> 217d03578a76234dcc4b642f8a869f42a394f047

# from Log_reg import run_log_reg

st.header("Fake Product Review Detection and Removal System Using Sentiment Analysis")
review = st.text_area("Enter the Review")
button = st.button('Check Review')

<<<<<<< HEAD
# if button:
#     st.write(review)


if button:
    with st.spinner("Please wait while review is analyzing..."):
        df = pd.read_csv('data.csv')
        review_train, review_test, label_train, label_test = train_test_split(df['text_'], df['label'], test_size=0.35)
        pipeline = Pipeline([
            ('bag_of_words', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultinomialNB())
        ])
        pipeline.fit(review_train, label_train)
        pred = pipeline.predict([review])

        st.success(pred[0])
=======

# Set Background Image


# with st.sidebar():
#     st.selectbox("Select Model")
>>>>>>> 217d03578a76234dcc4b642f8a869f42a394f047
