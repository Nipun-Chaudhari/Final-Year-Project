import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# from Log_reg import run_log_reg

st.header("Fake Product Review Detection and Removal System Using Sentiment Analysis")
review = st.text_area("Enter the Review")
button = st.button('Check Review')

# if button:
#     st.write(review)


if button:

    with st.spinner("Please wait while review is analyzing..."):
        if len(review) < 1:
            st.warning('Review cannot be null!')

        else:
            df = pd.read_csv('data.csv')
            review_train, review_test, label_train, label_test = train_test_split(df['text_'], df['label'],
                                                                                  test_size=0.35)
            pipeline = Pipeline([
                ('bag_of_words', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('classifier', MultinomialNB())
            ])
            pipeline.fit(review_train, label_train)
            pred = pipeline.predict([review])

            st.success(f'This review is {pred[0]}')


st.title("Description")
homepage = open("homepage.txt","r")
st.write(homepage.read())
