import streamlit as st

st.title("Airborne Microbiome Prediction App")

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload microbiome/environment data", type=["csv", "txt"])

if uploaded_file:
    st.write("File uploaded:", uploaded_file.name)

st.write("This is where model predictions and visualizations will go.")
import streamlit as st

st.title("Airborne Microbiome Prediction App")

import streamlit as st

st.title("Airborne Microbiome Prediction App")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload microbiome/environment data", type=["csv"])

if uploaded_file:
    st.write(" File uploaded:", uploaded_file.name)

st.write("This is where model predictions and visualizations will go.")


