import streamlit as st
from SVMM import runSVC
from Random_forest import runRF
from Log_reg import runLR
from Decision_tree import runDTREE
from Naive_B import runNB
from K_Nearest import runKNN

# import

with st.expander('Select Model'):
    if st.button('Support Vector Classifier'):
        runSVC()
    elif st.button('Random Forest Classifier'):
        runRF()
    elif st.button('Logistic Regression'):
        runLR()
    elif st.button('Naive Bayes Classifier'):
        runNB()
    elif st.button('Decision Tree Model'):
        runDTREE()
    elif st.button('KNN Model'):
        runKNN()
