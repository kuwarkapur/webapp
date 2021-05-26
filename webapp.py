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
    'Select yes for proceeding further and no for not',
    ('yes', 'no')
)
if dataset_name == 'yes':
    st.write("""
    #    Welcome!
        """)
    st.header("webapp name")
    st.write(" This webapp helps you to remove the uneccesary negative aura in your life")
    c=st.selectbox('Do you some times feel anxiety?',('yes','no'))
    v = st.selectbox('Do you tend to do overthinking alot?', ('yes', 'no'))
    k=st.selectbox('Do you tend to sleep alot?',('yes','no'))
    z = st.selectbox('Do you sometimes feel that you are confused about the trivial things?', ('yes', 'no'))
    l = st.selectbox('Do you practice meditation ofently ?', ('yes', 'no'))
    n = st.selectbox('Do you feel positive all the time ?', ('yes', 'no'))
    o = st.selectbox('Can you say you always feel happy all the time  ?', ('yes', 'no'))
    list=[c, v, k, z, l, n, o]
    for i in range(len(list)):
        if list[i] == "yes":
            list[i]=1
        else:
            list[i]=0

    x = np.array(list,dtype='float').reshape(1, -1)

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
