import streamlit as st

import importlib
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

from streamlit_util import filter_dataframe, aggrid_interactive_table_session, add_session_filter, data_selector
from population_util import _draw_variable_trial_back, _draw_variable_trial_back_linear_reg
import seaborn as sns

import s3fs

st.session_state.use_s3 = True
fs = s3fs.S3FileSystem(anon=False)

video_root = 'aind-behavior-data/Han/video/raw'

subfolders = fs.ls(video_root)
mice = [subfolder.split('/')[-1] for subfolder in subfolders]
mice = [mouse for mouse in mice if any(m in mouse for m in ['HH', 'KH', 'new'])]
mouse = st.selectbox('mouse', mice)

sessions = [sub.split('/')[-1] for sub in fs.ls(f'{video_root}/{mouse}')]
session = st.selectbox('session', sessions)

videos = [file.split('/')[-1] for file in fs.glob(f'{video_root}/{mouse}/{session}/*.mp4')]
video = st.selectbox('video', videos)

file_name = f'{video_root}/{mouse}/{session}/{video}'

cols = st.columns([1, 2])
with fs.open(file_name, 'rb') as f:
    cols[0].video(f.read())