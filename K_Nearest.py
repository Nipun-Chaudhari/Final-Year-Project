import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import streamlit as st

df = pd.read_csv('data.csv')
data = pd.read_csv('datafile1.csv')

test_size = [0.45, 0.40, 0.35, 0.30]


def runKNN():
    def run_knn():
        random = np.random.randint(40000)
        # Splitting the data into training and testing data
        # state = int(size * 100)
        review_train, review_test, label_train, label_test = train_test_split(df['text_'], df['label'], test_size=size,
                                                                              random_state=35)

        classifier = KNeighborsClassifier()
        pipeline = Pipeline([
            ('bag_of_words', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', classifier)
        ])

        pipeline.fit(review_train, label_train)
        predictions = pipeline.predict(review_test)
        print(predictions)

        # Confusion matrix of Multinomial Naive Bayes
        con_mat_knn = confusion_matrix(label_test, predictions)
        disp_knn = ConfusionMatrixDisplay(confusion_matrix=con_mat_knn, display_labels=['Fake', 'Original'])
        disp_knn.plot(cmap=plt.cm.Blues)
        # plt.title('Confusion matrix for test size ',size)
        plt.show()

        # Accuracy score
        accuracy_knn = str(np.round(accuracy_score(label_test, predictions) * 100, 2))
        print("\nACCURACY OF KNN MODEL FOR TEST SIZE ", size, " = \n", accuracy_knn + '%')

        # Classification report
        clf_report_knn = classification_report(label_test, predictions)
        print("\nCLASSIFICATION REPORT FOR TEST SIZE ", size, " = \n", clf_report_knn)

        # Prediction
        review = df['text_'][random]

        # print('Review : ', data['text_'][random])
        # print('\nReview is classified as : ', df['label'][random])
        pred = pipeline.predict([review])
        # print('\nKNN result : ', pred)

        res = {'Test Size': [size],
               'Random Review': [data['text_'][random]],
               'Dataset Label': [df['label'][random]],
               'Predicted Label': [pred]}

        st.write('ITERATION ', i)

        st.dataframe(res)

        st.write("\nACCURACY OF KNN MODEL FOR TEST SIZE ", size, " = \n", accuracy_knn + '%')
        print('Review : ', data['text_'][random])
        print('\nReview is classified as : ', df['label'][random])
        pred = pipeline.predict([review])
        print('\nKNN result : ', pred)

    i = 1

    for size in test_size:
        print('---------------ITERATION ', i, '-----------------\n\n')
        run_knn()
        print('\n\n')
        i += 1


def removeKNN():
    data1 = pd.read_csv('datafile1.csv')
    data1 = data1[data1['label'] != 'OR']
    st.info("Data After Removing Fake Reviews")
    st.dataframe(data1.head())
