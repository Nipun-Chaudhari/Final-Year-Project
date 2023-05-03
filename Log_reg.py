import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import streamlit as st

df = pd.read_csv('data.csv')
data = pd.read_csv('datafile1.csv')

test_size = [0.30, 0.35, 0.40, 0.45]
<<<<<<< HEAD
# random = np.random.randint(40000)

df['text_'].fillna('')
=======
random = np.random.randint(40000)
>>>>>>> 217d03578a76234dcc4b642f8a869f42a394f047

df['text_'].fillna('')

def runLR():
    def run_log_reg():
<<<<<<< HEAD
        # Initializing random review
        random = np.random.randint(40000)

        # Splitting the data into training and testing data
        # state = int(size * 100)
        review_train, review_test, label_train, label_test = train_test_split(df['text_'], df['label'], test_size=size,
                                                                              random_state=35)

        classifier = LogisticRegression(max_iter=100)
        pipeline = Pipeline([
            ('bag_of_words', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', classifier)
        ])

        pipeline.fit(review_train, label_train)
        predictions = pipeline.predict(review_test)
        print(predictions)

        # Confusion matrix of Multinomial Naive Bayes
        con_mat_logistic = confusion_matrix(label_test, predictions)
        disp_logreg = ConfusionMatrixDisplay(confusion_matrix=con_mat_logistic, display_labels=['Fake', 'Original'])
        disp_logreg.plot(cmap=plt.cm.Blues)
        # plt.title('Confusion matrix for test size ',size)
        plt.show()

        # Accuracy score
        accuracy_logreg = str(np.round(accuracy_score(label_test, predictions) * 100, 2))
        # print("\nACCURACY OF LOGISTIC REGRESSION MODEL FOR TEST SIZE ", size, " = \n", accuracy_logreg + '%')

        # Classification report
        clf_report_logistic = classification_report(label_test, predictions)
        print("\nCLASSIFICATION REPORT FOR TEST SIZE ", size, " = \n", clf_report_logistic)

        # Prediction
        review = df['text_'][random]
        # print('Review : ', data['text_'][random])
        # print('\nReview is classified as : ', df['label'][random])
        pred = pipeline.predict([review])
        # print('\nLogistic Regression result : ', pred)

        res = {'Test Size': [size],
               'Random Review': [data['text_'][random]],
               'Dataset Label': [df['label'][random]],
               'Predicted Label': [pred]}

        st.write('ITERATION ', i)

        st.dataframe(res)

        st.write("\nACCURACY OF LOGISTIC REGRESSION FOR TEST SIZE ", size, " = \n", accuracy_logreg + '%')

    i = 1
    for size in test_size:
        print('---------------ITERATION ', i, '-----------------\n\n')
        run_log_reg()
        print('\n\n')
        i += 1


def lr_home(text):
    pred = runLR().run_log_reg().pipeline.predict([text])
    st.success(pred)
=======
        # Splitting the data into training and testing data
        # state = int(size * 100)
        review_train, review_test, label_train, label_test = train_test_split(df['text_'], df['label'], test_size=size,
                                                                              random_state=35)

        classifier = LogisticRegression(max_iter=100)
        pipeline = Pipeline([
            ('bag_of_words', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', classifier)
        ])

        pipeline.fit(review_train, label_train)
        predictions = pipeline.predict(review_test)
        print(predictions)

        # Confusion matrix of Multinomial Naive Bayes
        con_mat_logistic = confusion_matrix(label_test, predictions)
        disp_logreg = ConfusionMatrixDisplay(confusion_matrix=con_mat_logistic, display_labels=['Fake', 'Original'])
        disp_logreg.plot(cmap=plt.cm.Blues)
        # plt.title('Confusion matrix for test size ',size)
        plt.show()

        # Accuracy score
        accuracy_logreg = str(np.round(accuracy_score(label_test, predictions) * 100, 2))
        print("\nACCURACY OF LOGISTIC REGRESSION MODEL FOR TEST SIZE ", size, " = \n", accuracy_logreg + '%')

        # Classification report
        clf_report_logistic = classification_report(label_test, predictions)
        print("\nCLASSIFICATION REPORT FOR TEST SIZE ", size, " = \n", clf_report_logistic)

        # Prediction
        review = df['text_'][random]
        print('Review : ', data['text_'][random])
        print('\nReview is classified as : ', df['label'][random])
        pred = pipeline.predict([review])
        print('\nLogistic Regression result : ', pred)

    i = 1
    for size in test_size:
        print('---------------ITERATION ', i, '-----------------\n\n')
        run_log_reg()
        print('\n\n')
        i += 1
>>>>>>> 217d03578a76234dcc4b642f8a869f42a394f047
