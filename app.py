import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("rf.pkl", "rb"))

st.title("Glass Classification")
ri = st.number_input("RI")
na = st.number_input("Na")
mg = st.number_input("Mg")
al = st.number_input("Al")
si = st.number_input("Si")
k = st.number_input("K")
ca = st.number_input("Ca")
ba = st.number_input("Ba")
fe = st.number_input("Fe")

if st.button("Predict"):
    test = np.array([[ri, na, mg, al, si, k, ca, ba, fe]])
    res = model.predict(test)
    print(res)
    st.success("Type: " + str(res[0]))
