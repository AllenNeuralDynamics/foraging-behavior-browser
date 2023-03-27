import streamlit as st

import importlib
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import os
import itertools

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

def add_behav_events(dict_behav_events, range=None, fig=None, **kwarg):
    for i, (event_name, color) in enumerate(event_color_map.items()):
        times = dict_behav_events[event_name]
        if range is not None:
            times = times[(range[0] <= times) & (times < range[1])]
        fig.add_trace(go.Scattergl(x=times, y=[20 - i*2]*len(times), 
                        mode='markers', 
                        marker=dict(symbol='line-ns-open', color=color, size=7, line=dict(width=2)), 
                        hovertemplate='%s' % (event_name),
                        showlegend=True, 
                        name=event_name), **kwarg)
    return fig
    
def add_dlc_feature_x_y(dict_dlc, feature, likelihood_threshold, range=None, fig=None, start_row=2, col=1, if_plot_likelihood=True):
    
    this_feature = dict_dlc[feature]

    x, y, likelihood, t = this_feature['x'], this_feature['y'], this_feature['likelihood'], this_feature['t']
    invalid = likelihood < likelihood_threshold
    x[invalid] = np.nan
    y[invalid] = np.nan
    
    valid_time = (t_start <= t) & (t < t_end)
    x = x[valid_time]
    y = y[valid_time]
    likelihood = likelihood[valid_time]
    t = t[valid_time]
    
    this_row = start_row
    if if_plot_likelihood:
        fig.add_trace(go.Scattergl(x=t, y=likelihood, 
                                mode='markers',
                                marker_color='#FFC2D2', 
                                name=f'likelihood',
                                showlegend=False,
                                hovertemplate=
                                    '%s<br>' % (f'Likelihood ({feature})')+
                                    '%{x}, %{y}<br><extra></extra>',), row=this_row, col=1)
        
        fig.add_shape(type='line', x0=t_start, x1=t_end, y0=likelihood_threshold, y1=likelihood_threshold, 
                    yref='y2', xref='x2', line=dict(color='red', dash='dash'))
        fig.update_yaxes(title_text='likelihood', range=[0, 1.05], row=this_row, col=1)
        this_row += 1

    fig.add_trace(go.Scattergl(x=t, y=x, mode='markers+lines', name='X',
                               showlegend=False,
                               hovertemplate=
                                '%s<br>' % (f'X ({feature})')+
                                '%{x}, %{y}<br><extra></extra>',
                               ), row=this_row, col=1)
    fig.update_yaxes(title_text='X', row=this_row, col=1)
    this_row += 1
    
    fig.add_trace(go.Scattergl(x=t, y=y, mode='markers+lines', name='Y',
                               showlegend=False,
                               hovertemplate=
                                '%s<br>' % (f'Y ({feature})')+
                                '%{x}, %{y}<br><extra></extra>',
                               ), row=this_row, col=1)
    fig.update_yaxes(title_text='Y', row=this_row, col=1)
    fig.update_xaxes(range=[t_start, t_end], row=this_row, col=1)
    
    
def plot_dlc_time_course(if_plot_likelihood=True):
    row_heights = [1.5] + [0.7, 2, 2] * len(features_to_plot) if if_plot_likelihood else ([1.5] + [2, 2] * len(features_to_plot))
    fig = make_subplots(rows=len(row_heights), cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.025,
                        subplot_titles=[""] + list(itertools.chain(*[[feature, "", ""] for feature in features_to_plot]))
                                        if if_plot_likelihood else [""] + list(itertools.chain(*[[feature, ""] for feature in features_to_plot])))
    
    add_behav_events(dict_behav_events, range=[t_start, t_end],
                     fig=fig, row=1, col=1)
    
    for n, feature in enumerate(features_to_plot):
        add_dlc_feature_x_y(dict_dlc, feature, likelihood_threshold, range=[t_start, t_end], fig=fig, start_row=2+n*(3 if if_plot_likelihood else 2), col=1, if_plot_likelihood=if_plot_likelihood)

    fig.update_layout(height=300+row_height*len(features_to_plot), width=1700, 
                      hovermode='closest',
                    #   plot_bgcolor='white', paper_bgcolor='white', 
                     font=dict(color='black'))

    # Show the plot
    # st.plotly_chart(fig)
    events_on_time_course = plotly_events(fig, override_width=fig.layout.width, override_height=fig.layout.height)
    return events_on_time_course

init()

with st.sidebar:
    with st.expander('DLC settings', expanded=True):
        likelihood_threshold = st.slider('Likeshold threshold', 0.000, 1.000, 0.950, step=0.001)
        if_plot_likelihood = st.checkbox('Plot likelihood', value=True)
        row_height = st.slider('row height', 100, 1000, 200)
        

if button_load or len(st.session_state.df_trials):
    st.session_state.df_trials, dict_behav_events, dict_dlc = load_nwb(nwb_folder + nwb_file)

    dlc_features = list(dict_dlc.keys())
    
    st.markdown(
    """
    <style>
        .stMultiSelect [data-baseweb=select] span{
            max-width: 1000px;
        }
    </style>""",
    unsafe_allow_html=True,
    )
    features_to_plot = st.multiselect('DLC feature', dlc_features, ['Camera1_bottom_tongue'],)
    event_color_map = {'lickportready': 'greenyellow', 'go': 'green', 'choice': 'magenta', 'right_lick': 'blue', 'left_lick': 'red', 'reward': 'cyan', 'trialend': 'black'}

    max_t = list(dict_dlc.values())[0]['t'][-1]
    cols = st.columns([1, 1, 2])
    t_range = cols[0].slider('time_range', 0.0, 200.0, value=50.0, step=5.0)
    t_start = cols[1].slider('time_start', 0.0, max_t, value=0.0, step=5.0)
    t_end = t_start + t_range
    
    events_on_time_course = plot_dlc_time_course(if_plot_likelihood)



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
        
        
