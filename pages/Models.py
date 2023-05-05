import streamlit as st
from SVMM import runSVC
from Random_forest import runRF
from Log_reg import runLR
from Decision_tree import runDTREE
from Naive_B import runNB
from K_Nearest import runKNN

st.title('Models')
st.text('Train Machine Learning model on entire data')

with st.expander('Select Model to train'):

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

# tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Logistic Regression', 'Naive Bayes Classifier', 'KNN Model',
#                                               'Random Forest Classifier', 'Decision Tree Model',
#                                               'Support Vector Classifier'])
#
# button = st.button('Train Model')
#
# with tab1:
#     if button:
#         runLR()
