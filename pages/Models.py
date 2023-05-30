import streamlit as st
from SVMM import runSVC, removeSVM
from Random_forest import runRF, removeRF
from Log_reg import runLR, removeLR
from Decision_tree import runDTREE, removeDTC
from Naive_B import runNB, removeNB
from K_Nearest import runKNN, removeKNN

st.title('Models')
st.text('Train Machine Learning model on entire data')

lr = 'Logistic regression is a linear classification algorithm that is used to model the probability \
     of a binary outcome. In the case of sentiment analysis, the binary outcome is positive or negative sentiment.'

nb = 'Naive Bayes is a probabilistic algorithm that uses Bayes theorem to classify text into different sentiment \
    categories. It works by calculating the likelihood of a particular word occurring in a given sentiment class and \
    combining it with the prior probability of the class to determine the final classification.'

knn = 'KNN is a simple, non-parametric algorithm that works by finding the K nearest neighbors to a new data point and \
      assigning it the label that is most common among those neighbors.'

dtree = 'Decision trees are a popular machine learning algorithm for classification tasks, including sentiment  \
        analysis.A decision tree is a tree-like model where each internal node represents a test on a feature, and each\
         leaf node represents a class label.'

rf = 'Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions. \
     In sentiment analysis, a Random Forest model can be trained to classify text into different sentiment  \
     categories by combining the predictions of individual decision trees.'

svc = 'SVM is a classification algorithm that works by finding the optimal hyperplane that separates the data points  \
      into different classes. In sentiment analysis, SVM can be used to classify text into positive, negative, \
      or neutral sentiment categories.'
#
# with st.expander('Select Model to train'):
#
#     if st.button('Support Vector Classifier'):
#         runSVC()
#     elif st.button('Random Forest Classifier'):
#         runRF()
#     elif st.button('Logistic Regression'):
#         runLR()
#     elif st.button('Naive Bayes Classifier'):
#         runNB()
#     elif st.button('Decision Tree Model'):
#         runDTREE()
#     elif st.button('KNN Model'):
#         runKNN()


with st.expander('Logistic Regression'):
    st.write(lr)
    button = st.button('Train LR model')
    if button:
        runLR()
    remove1 = st.button('Remove Fake Reviews Detected by LR')
    if remove1:
        removeLR()

with st.expander('Naive Bayes Classifier'):
    st.write(nb)
    button1 = st.button('Train NB model')
    if button1:
        runNB()
    remove2 = st.button('Remove Fake Reviews Detected by NB')
    if remove2:
        removeNB()

with st.expander('KNN Classifier'):
    st.write(knn)
    button2 = st.button('Train KNN model')
    if button2:
        runKNN()
    remove3 = st.button('Remove Fake Reviews Detected by KNN')
    if remove3:
        removeKNN()

with st.expander('Decision Tree Classifier'):
    st.write(dtree)
    button3 = st.button('Train DTC model')
    if button3:
        runDTREE()
    remove4 = st.button('Remove Fake Reviews Detected by DTC')
    if remove4:
        removeDTC()

with st.expander('Random Forest Classifier'):
    st.write(rf)
    button4 = st.button('Train RF model')
    if button4:
        runRF()
    remove5 = st.button('Remove Fake Reviews Detected by RF')
    if remove5:
        removeRF()

with st.expander('Support Vector Machine'):
    st.write(svc)
    button5 = st.button('Train SVC model')
    if button5:
        runSVC()
    remove6 = st.button('Remove Fake Reviews Detected by SVM')
    if remove6:
        removeSVM()


# tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Logistic Regression', 'Naive Bayes Classifier', 'KNN Model',
#                                               'Random Forest Classifier', 'Decision Tree Model',
#                                               'Support Vector Classifier'])
#
# button = st.button('Train Model')
#
# with tab1:
#     if button:
#         runLR()
