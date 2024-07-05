"""Visualize HMM-GLM results from Faeze
"""

import streamlit as st
import numpy as numpy
import s3fs

import streamlit_nested_layout

s3_HMM_GLM_folder = "s3://aind-behavior-data/faeze/HMM-GLM/"
fs = s3fs.S3FileSystem(anon=False)

# Get data folder

data_folders = fs.glob(s3_HMM_GLM_folder)

data_name = [folder.split('/')[-1] for folder in data_folders]

data_name