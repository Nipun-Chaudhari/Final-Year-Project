import pandas as pd
import matplotlib.pyplot as plt
import string
import nltk
import warnings
import streamlit as st
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')


st.set_option('deprecation.showPyplotGlobalUse', False)


# Page Title
st.title("Data Cleaning")

# File Chooser
file = st.file_uploader("Choose a file")

nltk.download('wordnet')
nltk.download('omw-1.4')

if file is not None:
    df = pd.read_csv(file)
    with st.spinner('Please wait, your data is being cleaned...'):

        print("REVIEWS DATA\n", df.head(), "\n")

        print("NULL VALUES IN DATA\n", df.isnull().sum(), "\n")
        print("DATA INFORMATION\n", df.info(), "\n")
        print("DATA DESCRIPTION\n", df.describe(), "\n")
        print(df['rating'].value_counts())

        # Pie chart for Rating Proportion
        plt.figure(figsize=(3, 3))
        plt.figure(figsize=(5, 5))
        labels = df['rating'].value_counts().keys()
        values = df['rating'].value_counts().values
        exp = (0.1, 0, 0, 0, 0)
        plt.pie(values, labels=labels, explode=(0.1, 0, 0, 0, 0), shadow=True, autopct='%1.1f%%')
        plt.title('Rating Proportions', fontweight='bold', fontsize=40, pad=20, color='black')
        st.pyplot(plt.show())

        print(df['text_'][0])

        # Text Cleaning
        def clean_text(text):
            # Remove all punctuation marks using the `translate` method
            table = str.maketrans('', '', string.punctuation)
            nopunctuation = text.translate(table)

            # Remove stopwords and convert all words to lowercase
            filtered_words = [word.lower() for word in nopunctuation.split(' ') if
                              word.lower() not in stopwords.words('english')]

            # Join the filtered words back into a string
            preprocessed_text = ' '.join(filtered_words)

            return preprocessed_text


        df['text_'] = df['text_'].apply(clean_text)
        df['text_'] = df['text_'].astype(str)

        def preprocess(text):
            # Preprocesses the input text by tokenizing it, removing stopwords, and non-alphabetic characters.
            tokens = word_tokenize(text)

            stop_words = set(stopwords.words('english'))

            filtered_tokens = [word.lower() for word in tokens if
                               word.isalpha() and len(word) > 1 and word.lower() not in stop_words]

            preprocessed_text = ' '.join(filtered_tokens)

            return preprocessed_text


        df['text_'] = df['text_'].apply(preprocess)
        # Sample output of preprocessed data
        preprocess_out = preprocess(df['text_'][2540])
        print("SAMPLE PREPROCESSED DATA OUTPUT: \n", preprocess_out, "\n")

        # Stemming the words to their base form
        stemmer = PorterStemmer()


        def word_stemming(text):
            return ' '.join([stemmer.stem(word) for word in text.split()])


        df['text_'] = df['text_'].apply(lambda x: word_stemming(x))

        # lemmatizing the words to reduce them to their dictionary form
        lemmatizer = WordNetLemmatizer()


        def lemmatize_words(text):
            return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


        df["text_"] = df["text_"].apply(lambda text: lemmatize_words(text))
        print("REVIEWS FROM PREPROCESSED DATASET: \n", df['text_'], "\n")

        # # Change labels
        df['label'].replace('CG', 'fake', inplace=True)
        df['label'].replace('OR', 'original', inplace=True)

        # Saving preprocessed reviews in new dataset
        df.to_csv('data.csv')

    st.success('Done')
