import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

df = pd.read_csv('data.csv')
data = pd.read_csv('datafile1.csv')

test_size = [0.30, 0.35, 0.40, 0.45]
random = np.random.randint(43000)


def run_dtree():
    # Splitting the data into training and testing data
    # state = int(size * 100)
    review_train, review_test, label_train, label_test = train_test_split(df['text_'], df['label'], test_size=size,
                                                                          random_state=35)

    classifier = DecisionTreeClassifier()
    pipeline = Pipeline([
        ('bag_of_words', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', classifier)
    ])

    pipeline.fit(review_train, label_train)
    predictions = pipeline.predict(review_test)
    print(predictions)

    # Confusion matrix of Multinomial Naive Bayes
    con_mat_dtree = confusion_matrix(label_test, predictions)
    disp_dtree = ConfusionMatrixDisplay(confusion_matrix=con_mat_dtree, display_labels=['Fake', 'Original'])
    disp_dtree.plot(cmap=plt.cm.Blues)
    # plt.title('Confusion matrix for test size ',size)
    plt.show()

    # Accuracy score
    accuracy_dtree = str(np.round(accuracy_score(label_test, predictions) * 100, 2))
    print("\nACCURACY OF DECISION TREE MODEL FOR TEST SIZE ", size, " = \n", accuracy_dtree + '%')

    # Classification report
    clf_report_dtree = classification_report(label_test, predictions)
    print("\nCLASSIFICATION REPORT FOR TEST SIZE ", size, " = \n", clf_report_dtree)

    # Prediction
    review = df['text_'][random]
    print('Review : ', data['text_'][random])
    print('\nReview is classified as : ', df['label'][random])
    pred = pipeline.predict([review])
    print('\nDecision Tree result : ', pred)


i = 1
for size in test_size:
    print('---------------ITERATION ', i, '-----------------\n\n')
    run_dtree()
    print('\n\n')
    i += 1
