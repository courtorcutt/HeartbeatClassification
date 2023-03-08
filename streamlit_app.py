import math
import pandas as pd
import streamlit as st
import numpy as np

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
