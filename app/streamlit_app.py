# app/streamlit_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project folder to Python path so we can import ML code
sys.path.append(str(Path(__file__).resolve().parents[1]))
import Group1_ML_code

st.set_page_config(page_title="Airborne Microbial Predictor")
st.title(" Airborne Microbial Predictor — Demo")

# File upload / sample data
use_sample = st.checkbox("Use sample test data")

if use_sample:
    sample_path = Path(__file__).resolve().parents[1] / "data" / "test" / "sample_upload.csv"
    df = pd.read_csv(sample_path)
    st.success("Using sample test data")
elif uploaded_file := st.file_uploader("Upload your CSV file", type=["csv"]):
    df = pd.read_csv(uploaded_file)
else:
    st.info("Upload a CSV file or select 'Use sample test data'.")
    st.stop()

st.subheader("Preview Data")
st.dataframe(df.head())

# Run prediction
if st.button("Run Prediction"):
    st.info("Running ML prediction...")
    # Call your classmate's predict function
    Group1_ML_code.predict()
    st.success("Done! Check terminal for prediction results.")
