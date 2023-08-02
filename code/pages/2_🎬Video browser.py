import streamlit as st

import importlib
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import os

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from util.streamlit import filter_dataframe, aggrid_interactive_table_session, add_session_filter, data_selector
from streamlit_plotly_events import plotly_events
from util.population import _draw_variable_trial_back, _draw_variable_trial_back_linear_reg
import seaborn as sns
import pandas as pd

import s3fs, h5py

    
try:
    st.set_page_config(layout="wide", 
                page_title='Foraging behavior browser',
                page_icon=':mouse2:',
                menu_items={
                'Report a bug': "https://github.com/hanhou/foraging-behavior-browser/issues",
                'About': "Github repo: https://github.com/hanhou/foraging-behavior-browser/"
                }
                )
except:
    pass


st.session_state.use_s3 = True
fs = s3fs.S3FileSystem(anon=False)

video_root = 'aind-behavior-data/Han/video/raw'
nwb_folder = 'aind-behavior-data/Han/ephys/export/nwb/' 

# --- load raw tracks from nwb ---

nwb_files = [f.split('/')[-1] for f in fs.glob(nwb_folder + '/*.nwb')]

cols = st.columns([1, 1, 2])
nwb_file = cols[0].selectbox('Select nwb file', nwb_files, label_visibility='collapsed')
button_load = cols[1].button('Load nwb file!')

def init():
    if 'df_trials' not in st.session_state:
        st.session_state.df_trials = []

def hdf_group_to_df(group):
    '''
    Convert h5py group to dataframe
    '''
    
    tmp_dict = {}
    
    # Handle spike_times
    # see https://nwb-schema.readthedocs.io/en/latest/format_description.html#tables-and-ragged-arrays
    if 'spike_times' in group.keys():
        spike_times, spike_times_index = group['spike_times'], group['spike_times_index']
        unit_spike_times = []
        for i, idx in enumerate(spike_times_index):
            start_idx = 0 if i == 0 else spike_times_index[i - 1]
            unit_spike_times.append(spike_times[start_idx : idx])

        tmp_dict['spike_times'] = unit_spike_times

    # Other keys
    for key in group:
        if 'spike_times' in key: continue

        if isinstance(group[key][0], bytes):
            tmp_dict[key] = list(map(lambda x: x.decode(), group[key][:]))
        else:
            tmp_dict[key] = list(group[key][:])
        
    return pd.DataFrame(tmp_dict)

def import_behavior_from_nwb(nwb_hdf):
    g_trials = nwb_hdf['intervals']['trials']
    g_events = nwb_hdf['acquisition']['BehavioralEvents']
    
    df_trials = hdf_group_to_df(g_trials)
    dict_behav_events = {key: g_events[key]['timestamps'][:] for key in g_events}
    
    return df_trials, dict_behav_events

def import_dlc_from_nwb(file_name, features=None, time_idx_mask=None):
    with fs.open(file_name) as f:        
        nwb_hdf = h5py.File(f, mode='r')    # Can only use h5py (not NWBHDF5IO) for s3fs!
        
        g_dlc = nwb_hdf['acquisition']['BehavioralTimeSeries']
        dict_dlc = {}
        if features is None:
            features = g_dlc.keys()
            
        for feature in features:
            if time_idx_mask is None:
                dict_dlc[feature] = {'x': g_dlc[feature]['data'][:, 0],
                                    'y': g_dlc[feature]['data'][:, 1],
                                    'likelihood': g_dlc[feature]['data'][:, 2],
                                    }
            else:
                dict_dlc[feature] = {'x': g_dlc[feature]['data'][time_idx_mask, 0],
                                    'y': g_dlc[feature]['data'][time_idx_mask, 1],
                                    'likelihood': g_dlc[feature]['data'][time_idx_mask, 2],
                                    }
    return dict_dlc

def import_ephys_from_nwb(nwb_hdf):
    g_units = nwb_hdf['units']        
    df_units = hdf_group_to_df(g_units)
    g_electrodes = nwb_hdf['general']['extracellular_ephys']['electrodes']
    df_electrodes = hdf_group_to_df(g_electrodes)
        
    # Add electrode info to units table
    df_units = df_units.merge(df_electrodes, left_on='electrodes', right_on='id')

    # Add hemisphere
    df_units['hemisphere'] = np.where(df_units['z'] <= 5739, 'left', 'right')
    return df_units

def get_dlc_features_and_times(nwb_hdf):
    dlc_features = list(nwb_hdf['acquisition']['BehavioralTimeSeries'].keys())
    
    _frame_idx = nwb_hdf['scratch']['video_frame_mapping']['frame_time_index']
    _device = nwb_hdf['scratch']['video_frame_mapping']['tracking_device'][:]
    _camera_1_start_idx = _frame_idx[_device == b'Camera 0'][-1]
    dlc_times = {'Camera0': nwb_hdf['scratch']['video_frame_mapping']['frame_time'][:_camera_1_start_idx],
                 'Camera1': nwb_hdf['scratch']['video_frame_mapping']['frame_time'][_camera_1_start_idx:]}
    return dlc_features, dlc_times

@st.cache_data(max_entries=100)
def load_nwb(file_name):
    with fs.open(file_name) as f:        
        nwb_hdf = h5py.File(f, mode='r')    # Can only use h5py (not NWBHDF5IO) for s3fs!
        
        # Get behavioral trials and events
        df_trials, dict_behav_events = import_behavior_from_nwb(nwb_hdf)
        
        # Get dlc features and times (not data for this moment)
        dlc_features, dlc_times = get_dlc_features_and_times(nwb_hdf)

    return df_trials, dict_behav_events, dlc_features, dlc_times

init()

if button_load or len(st.session_state.df_trials):
    with st.spinner('load nwb'):
        st.session_state.df_trials, dict_behav_events, dlc_features, dlc_times = load_nwb(nwb_folder + nwb_file)

    feature = st.selectbox('DLC feature', dlc_features, index=dlc_features.index('Camera1_bottom_tongue'))
    camera = feature[:7]
    camera_times = dlc_times[camera]
    likelihood_threshold = st.slider('Likeshold threshold', 0.000, 1.000, 0.950, step=0.001)

    # Time slider
    cols = st.columns([1, 1, 2])
    t_range = cols[0].slider('time_range', 0.0, 200.0, value=10.0, step=5.0)
    t_start = cols[1].slider('time_start', 0.0, max(camera_times), value=0.0, step=5.0)
    t_end = t_start + t_range
    time_idx_mask = (t_start < camera_times) & (camera_times < t_end)

    with st.spinner('load dlc for this epoch'):
        dict_dlc_this = import_dlc_from_nwb(nwb_folder + nwb_file, features=[feature], time_idx_mask=time_idx_mask)

    event_color_map = {'lickportready': 'yellow', 'go': 'green', 'choice': 'magenta', 'right_lick': 'blue', 'left_lick': 'red', 'reward': 'cyan', 'trialend': 'gray'}

    x, y, likelihood = dict_dlc_this[feature]['x'], dict_dlc_this[feature]['y'], dict_dlc_this[feature]['likelihood']
    t = camera_times[time_idx_mask]

    invalid = likelihood < likelihood_threshold
    # likelihood[invalid] = np.nan
    x[invalid] = np.nan
    y[invalid] = np.nan

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[2, 1, 3, 3], vertical_spacing=0.01)

    for i, (event_name, color) in enumerate(event_color_map.items()):
        if event_name == 'choice': continue
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
        
        
