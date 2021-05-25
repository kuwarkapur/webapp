import pandas as pd
import streamlit as st
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
medi=pd.read_csv('meditations.csv')
medi=medi.drop('id',axis=1)
X=medi.drop('playlist',axis=1)
y=medi['playlist']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
neigh = KNeighborsRegressor(n_neighbors=1)
neigh.fit(X_train,y_train)
predict_KNN=neigh.predict(X_test)
print(predict_KNN.round())
score=neigh.score(X_train,y_train)
print(score)
pickle.dump(neigh, open("webaapp.pkl", "wb"))
print(medi.columns)


dataset_name = st.sidebar.selectbox(
    'Select yes for procedding further and no for not',
    ('yes', 'no')
)
if dataset_name == 'yes':
    st.write("""
    #    Welcome!
        """)
    st.header("webapp name")
    st.write(" this webapp helps you to remove the uneccesary negative aora in your life")
    st.info('IN THE below ans choose 0 for no and 1 for yes')
    c = st.number_input('DO you some times feel anxiety?', 0, 1)
    v = st.number_input('DO you tend to do overthinking alot?', 0, 1)
    k = st.number_input('DO you tend to sleep alot?', 0, 1)
    z = st.number_input('DO you sometimes feel that you are confused about the trivial things?', 0, 1)
    l = st.number_input('DO you practice meditation ofently ?', 0, 1)
    n = st.number_input('DO you feel positive all the time ?', 0, 1)
    o = st.number_input('CAN you say you are feel_happy person ?', 0, 1)
    x = np.array([c, v, k, z, l, n, o]).reshape(1, -1)
    loaded_model = pickle.load(open("webaapp.pkl", "rb"))
    pred = loaded_model.predict(x)
    x[:] = np.zeros([1, 7])
    # 4:- soothing,2:- happy,3:-relaxing,1:-focused,0:- calm
    if pred == 0:
        st.title('you should look into our calm music playlist')
    elif pred == 1:
        st.title('you should look into our focused music playlist')
    elif pred == 2:
        st.title('you should look into our happy music playlist')
    elif pred == 3:
        st.title('you should look into our relaxing  music playlist')
    elif pred == 4:
            st.title('you should look into our soothing music playlist')


else:
    st.title('you can also check our other options in webapp')





