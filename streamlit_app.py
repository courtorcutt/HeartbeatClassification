import math
import pandas as pd
import streamlit as st
import numpy as np

# Creating sections to organize site
data = st.beta_container()
theModel = st.beta_container()
theGraphs = st.beta_container()
modelTraining = st.beta_container()
result = st.beta_container()

"""
# Abnormal Heartbeat Classification

Detection of normal vs abnormal heartbeats in humans. Then algorithm will be applied to canines based on accuracy.

There are two columns with the last column being either a 0 (normal) or a 1 (abnormal). 

# Arrhythmia Dataset
Number of Samples: 109446
Number of Categories: 5
Sampling Frequency: 125Hz
Data Source: Physionet's MIT-BIH Arrhythmia Dataset
Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]

# The PTB Diagnostic ECG Database
Number of Samples: 14552
Number of Categories: 2
Sampling Frequency: 125Hz
Data Source: Physionet's PTB Diagnostic Database

All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 188.


Using a Recurrent Neural Network...

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:
"""

#section for reading in the csv file
with data:
  st.title('Data')
  st.write('The client has the ability to change the forecast dates or use a different csv file with  different data so that the model can be used in the future.')  
  uploaded_file = st.file_uploader("Then please upload the CSV file to start:", type = ["csv"])
  if uploaded_file is not None:
    with st.beta_expander("Details about upload"):
        st.write(type(uploaded_file))
        file_details = {"filename":uploaded_file.name, "filetype":uploaded_file.type, "filesize":uploaded_file.size}
        st.write(file_details)
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data:")
    st.write(df)
