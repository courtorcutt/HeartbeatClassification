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

# Arrhythmia Dataset
Abnormal Heartbeat Classification
Detection of normal vs abnormal heartbeats in humans. Then algorithm will be applied to canines based on accuracy.

There are two columns with the last column being either a 0 (normal) or a 1 (abnormal).

#section for reading in the csv file
"""

with data:
  st.title('Data')
  st.write('Upload heart rate data.')  
  uploaded_file = st.file_uploader("Upload in CSV format:", type = ["csv"])
  if uploaded_file is not None:
    with st.beta_expander("Details about upload"):
        st.write(type(uploaded_file))
        file_details = {"filename":uploaded_file.name, "filetype":uploaded_file.type, "filesize":uploaded_file.size}
        st.write(file_details)
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data:")
    st.write(df)


# list of data frames, loading in both normal and abnormal data
# x changes between the data paths (2 different data frames)
dfs = [pd.read_csv('ptbdb_' + x + '.csv') for x in ['normal', 'abnormal']]
'''
# Normal Dataset:
Final values are all 0
'''
dfs[0]

'''
# Abnormal Dataset:
Final values are all 1
'''
dfs[1]
