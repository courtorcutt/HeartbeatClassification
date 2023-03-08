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



# to merge the data into one, must reaname columns so they align
for df in dfs :
    df.columns = list(range(len(df.columns)))
    
# both datasets will now have an integer column identifiers so now they can be merged into one dataset
# dfs[0]

# concat will merge and to shuffle data use sample (last column shuffle zeros and ones)
# the indices get shuffled too in sample so they must be reset (new index column created as a result is dropped)
data = pd.concat(dfs, axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)

# name the last column Label instead of 187 (this row contains the ones or zeros)
data = data.rename({187: 'Label'}, axis=1)

'''
# New Merged Shuffled Dataset:
Includes both Normal and Abnormal ECG data

'''
data

