import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import streamlit as st

df = pd.read_csv('data.csv')
data = pd.read_csv('datafile1.csv')

test_size = [0.30, 0.35, 0.40, 0.45]
# random = np.random.randint(43000)


def runRF():
    def run_random_forest():

        # Initializing random review
        random = np.random.randint(40000)

        # Splitting the data into training and testing data
        # state = int(size * 100)
        review_train, review_test, label_train, label_test = train_test_split(df['text_'], df['label'], test_size=size,
                                                                              random_state=35)

        classifier = RandomForestClassifier()
        pipeline = Pipeline([
            ('bag_of_words', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', classifier)
        ])

        pipeline.fit(review_train, label_train)
        predictions = pipeline.predict(review_test)
        print(predictions)

        # Confusion matrix of Multinomial Naive Bayes
        con_mat_rf = confusion_matrix(label_test, predictions)
        disp_rf = ConfusionMatrixDisplay(confusion_matrix=con_mat_rf, display_labels=['Fake', 'Original'])
        disp_rf.plot(cmap=plt.cm.Blues)
        # plt.title('Confusion matrix for test size ',size)
        plt.show()

        # Accuracy score
        accuracy_rf = str(np.round(accuracy_score(label_test, predictions) * 100, 2))
        print("\nACCURACY OF RANDOM FOREST CLASSIFIER MODEL FOR TEST SIZE ", size, " = \n", accuracy_rf + '%')

        # Classification report
        clf_report_rf = classification_report(label_test, predictions)
        print("\nCLASSIFICATION REPORT FOR TEST SIZE ", size, " = \n", clf_report_rf)

        # Prediction
        review = df['text_'][random]
        print('Review : ', data['text_'][random])
        print('\nReview is classified as : ', df['label'][random])
        pred = pipeline.predict([review])
        print('\nRandom Forest Classifier result : ', pred)

        res = {'Test Size': [size],
               'Random Review': [data['text_'][random]],
               'Dataset Label': [df['label'][random]],
               'Predicted Label': [pred]}

        st.write('ITERATION ', i)

        st.dataframe(res)

        st.write("\nACCURACY OF RANDOM FOREST MODEL FOR TEST SIZE ", size, " = \n", accuracy_rf + '%')


        # Splitting the data into training and testing data
        # state = int(size * 100)
        review_train, review_test, label_train, label_test = train_test_split(df['text_'], df['label'], test_size=size,
                                                                              random_state=35)

        classifier = RandomForestClassifier()
        pipeline = Pipeline([
            ('bag_of_words', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', classifier)
        ])

        pipeline.fit(review_train, label_train)
        predictions = pipeline.predict(review_test)
        print(predictions)

        # Confusion matrix of Multinomial Naive Bayes
        con_mat_rf = confusion_matrix(label_test, predictions)
        disp_rf = ConfusionMatrixDisplay(confusion_matrix=con_mat_rf, display_labels=['Fake', 'Original'])
        disp_rf.plot(cmap=plt.cm.Blues)
        # plt.title('Confusion matrix for test size ',size)
        plt.show()

        # Accuracy score
        accuracy_rf = str(np.round(accuracy_score(label_test, predictions) * 100, 2))
        print("\nACCURACY OF RANDOM FOREST CLASSIFIER MODEL FOR TEST SIZE ", size, " = \n", accuracy_rf + '%')

        # Classification report
        clf_report_rf = classification_report(label_test, predictions)
        # print("\nCLASSIFICATION REPORT FOR TEST SIZE ", size, " = \n", clf_report_rf)

        # Prediction
        review = df['text_'][random]
        print('Review : ', data['text_'][random])
        print('\nReview is classified as : ', df['label'][random])
        pred = pipeline.predict([review])
        print('\nRandom Forest Classifier result : ', pred)


    i = 1
    for size in test_size:
        # print('----------------ITERATION ', i, '-----------------\n\n')
        run_random_forest()
        # print('\n\n')
        i += 1
