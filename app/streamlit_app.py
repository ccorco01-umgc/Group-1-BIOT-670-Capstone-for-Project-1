# app/streamlit_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import io

# Add project folder to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import Group1_ML_code

st.set_page_config(page_title="Airborne Microbial Predictor")
st.title("Airborne Microbial Predictor — Demo")

# Upload CSV or use sample data
use_sample = st.checkbox("Use sample test data")
upload = st.file_uploader("Upload CSV", type=["csv"])

if use_sample:
    sample_path = Path(__file__).resolve().parents[1] / "data" / "test" / "sample_upload.csv"
    df = pd.read_csv(sample_path)
    st.success("Using sample test data")
elif upload:
    df = pd.read_csv(upload)
else:
    st.info("Upload a CSV or check 'Use sample test data'")
    st.stop()

# Preview data
st.subheader("Preview data")
st.dataframe(df.head())

# Run prediction and capture printed output
if st.button("Run prediction"):
    st.write("Running predictions...")
    buffer = io.StringIO()  # capture print output
    try:
        # Redirect stdout to buffer
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer

        # Call your classmate's function
        Group1_ML_code.predict()

        # Reset stdout
        sys.stdout = old_stdout

        # Show results in Streamlit
        st.subheader("Prediction results")
        st.text(buffer.getvalue())

        st.success("Prediction complete!")

    except Exception as e:
        sys.stdout = old_stdout
        st.error(f"Error running prediction: {e}")
