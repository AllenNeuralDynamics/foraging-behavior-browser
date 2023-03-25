import streamlit as st

import importlib
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import os

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from streamlit_util import filter_dataframe, aggrid_interactive_table_session, add_session_filter, data_selector
from streamlit_plotly_events import plotly_events
from population_util import _draw_variable_trial_back, _draw_variable_trial_back_linear_reg
import seaborn as sns

import s3fs

st.session_state.use_s3 = True
fs = s3fs.S3FileSystem(anon=False)

import os, glob
from pynwb import NWBFile, TimeSeries, NWBHDF5IO

video_root = 'aind-behavior-data/Han/video/raw'

# --- load raw tracks from nwb ---
nwb_folder = '/root/capsule/data/s3/export/nwb/'  # Use s3 drive mounted in the capsule for now

nwb_files = [f.split('/')[-1] for f in glob.glob(nwb_folder + '/*.nwb')]

cols = st.columns([1, 1, 2])
nwb_file = cols[0].selectbox('Select nwb file', nwb_files, label_visibility='collapsed')
button_load = cols[1].button('Load nwb file!')

def init():
    if 'df_trials' not in st.session_state:
        st.session_state.df_trials = []

@st.cache_data(ttl=3600*24)
def load_nwb(file_name):
    with NWBHDF5IO(file_name, mode='r') as io:
        nwb = io.read()
        df_trials = nwb.trials.to_dataframe()
        
        behav_events = nwb.acquisition['BehavioralEvents']
        dict_behav_events = {}
        for event_name in behav_events.fields['time_series'].keys():
            dict_behav_events[event_name] = behav_events[event_name].timestamps[:]
        
        dlc = nwb.acquisition['BehavioralTimeSeries']
        dict_dlc = {}
        for feature in dlc.fields['time_series'].keys():
            dict_dlc[feature] = {'x': dlc[feature].data[:, 0],
                                 'y': dlc[feature].data[:, 1],
                                 'likelihood': dlc[feature].data[:, 2],
                                 't': dlc[feature].timestamps[:]}
      
    return df_trials, dict_behav_events, dict_dlc

init()

if button_load or len(st.session_state.df_trials):
    st.session_state.df_trials, dict_behav_events, dict_dlc = load_nwb(nwb_folder + nwb_file)

    dlc_features = list(dict_dlc.keys())
    feature = st.selectbox('DLC feature', dlc_features, index=dlc_features.index('Camera1_bottom_tongue'))
    likelihood_threshold = st.slider('Likeshold threshold', 0.000, 1.000, 0.950, step=0.001)

    bottom_tongue = dict_dlc[feature]

    event_color_map = {'lickportready': 'yellow', 'go': 'green', 'choice': 'magenta', 'right_lick': 'blue', 'left_lick': 'red', 'reward': 'cyan', 'trialend': 'black'}

    x, y, likelihood, t = bottom_tongue['x'], bottom_tongue['y'], bottom_tongue['likelihood'], bottom_tongue['t']

    invalid = likelihood < likelihood_threshold
    # likelihood[invalid] = np.nan
    x[invalid] = np.nan
    y[invalid] = np.nan

    cols = st.columns([1, 1, 2])
    t_range = cols[0].slider('time_range', 0.0, 200.0, value=50.0, step=5.0)
    t_start = cols[1].slider('time_start', 0.0, max(t), value=0.0, step=5.0)
    t_end = t_start + t_range

    valid_time = (t_start <= t) & (t < t_end)
    x = x[valid_time]
    y = y[valid_time]
    likelihood = likelihood[valid_time]
    t = t[valid_time]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[2, 1, 3, 3], vertical_spacing=0.01)

    for i, (event_name, color) in enumerate(event_color_map.items()):
        times = dict_behav_events[event_name]
        fig.add_trace(go.Scattergl(x=times, y=[i*2]*len(times), 
                                mode='markers', 
                                marker=dict(symbol='line-ns-open', color=color, size=7), 
                                showlegend=True, 
                                name=event_name), row=1, col=1)

    fig.add_trace(go.Scattergl(x=t, y=likelihood, mode='markers', name=f'likelihood'), row=2, col=1)
    fig.add_shape(type='line', x0=t_start, x1=t_end, y0=likelihood_threshold, y1=likelihood_threshold, 
                yref='y2', xref='x2', line=dict(color='red', dash='dash'))

    fig.add_trace(go.Scattergl(x=t, y=x, mode='markers', name='X'), row=3, col=1)
    fig.add_trace(go.Scattergl(x=t, y=y, mode='markers', name='Y'), row=4, col=1)

    fig.update_yaxes(title_text='likelihood', row=2, col=1)
    fig.update_yaxes(title_text='X', row=3, col=1)
    fig.update_yaxes(title_text='Y', row=4, col=1)

    fig.update_xaxes(range=[t_start, t_end], row=4, col=1)
    fig.update_layout(height=600, width=1500, title_text=feature, 
                    #   plot_bgcolor='white', paper_bgcolor='white', 
                    font=dict(color='black'))

    # Show the plot
    st.plotly_chart(fig)
    # plotly_events(fig)


    # # --- navigate raw annotated videos ---
    # subfolders = fs.ls(video_root)
    # mice = [subfolder.split('/')[-1] for subfolder in subfolders]
    # mice = [mouse for mouse in mice if any(m in mouse for m in ['HH', 'KH', 'new'])]
    # mouse = st.selectbox('mouse', mice)

    # sessions = [sub.split('/')[-1] for sub in fs.ls(f'{video_root}/{mouse}')]
    # session = st.selectbox('session', sessions)

    # videos = [file.split('/')[-1] for file in fs.glob(f'{video_root}/{mouse}/{session}/*.mp4')]
    # video = st.selectbox('video', videos)

    # file_name = f'{video_root}/{mouse}/{session}/{video}'

    # cols = st.columns([1, 2])
    # with fs.open(file_name, 'rb') as f:
    #     cols[0].video(f.read())
        
        
