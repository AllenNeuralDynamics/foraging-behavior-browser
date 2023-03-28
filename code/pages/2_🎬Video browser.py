import streamlit as st

import importlib
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import os
import itertools

import plotly.graph_objs as go
import plotly.express as px
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
camera_mapping = {'Camera0': 'side', 'Camera1': 'bottom'}

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
    
def add_dlc_feature_x_y(dict_dlc, feature, likelihood_threshold, color='black', range=None, fig=None, start_row=2, col=1, if_plot_likelihood=True, t_range_2d=None):
    
    this_feature = dict_dlc[feature]

    x, y, likelihood, t = this_feature['x'], this_feature['y'], this_feature['likelihood'], this_feature['t']
    valid_time = (t_start <= t) & (t < t_end)
    x = x[valid_time]
    y = y[valid_time]
    likelihood = likelihood[valid_time]
    t = t[valid_time]
    
    invalid = likelihood < likelihood_threshold
    x[invalid] = np.nan
    y[invalid] = np.nan

    this_row = start_row
    if if_plot_likelihood:
        fig.add_trace(go.Scattergl(x=t, y=likelihood, 
                                mode='markers',
                                marker_color='black', 
                                name=f'likelihood',
                                showlegend=False,
                                hovertemplate=
                                    '%s<br>' % (f'Likelihood ({feature})')+
                                    '%{x}, %{y}<br><extra></extra>',), row=this_row, col=1)
        
        fig.add_shape(type='line', x0=t_start, x1=t_end, y0=likelihood_threshold, y1=likelihood_threshold, 
                      yref='y2', xref='x2', line=dict(color='red', dash='dash'), row=this_row, col=1)
        
        if t_range_2d is not None:
            fig.add_vrect(x0=t_range_2d[0], x1=t_range_2d[1], fillcolor='grey', opacity=0.1, line_width=0, row=this_row, col=1)

        fig.update_yaxes(title_text='likelihood', range=[0, 1.05], row=this_row, col=1)
        this_row += 1

    fig.add_trace(go.Scattergl(x=t, y=x, mode='markers+lines', name='X',
                               showlegend=False,
                               marker_color=color,
                               hovertemplate=
                                '%s<br>' % (f'X ({feature})')+
                                '%{x}, %{y}<br><extra></extra>',
                               ), row=this_row, col=1)
    if t_range_2d is not None:
        fig.add_vrect(x0=t_range_2d[0], x1=t_range_2d[1], fillcolor='grey', opacity=0.1, line_width=0, row=this_row, col=1)

    fig.update_yaxes(title_text='X', row=this_row, col=1)
    this_row += 1
    
    fig.add_trace(go.Scattergl(x=t, y=y, mode='markers+lines', name='Y',
                               showlegend=False,
                               marker_color=color,
                               hovertemplate=
                                '%s<br>' % (f'Y ({feature})')+
                                '%{x}, %{y}<br><extra></extra>',
                               ), row=this_row, col=1)
    if t_range_2d is not None:
        fig.add_vrect(x0=t_range_2d[0], x1=t_range_2d[1], fillcolor='grey', opacity=0.1, line_width=0, row=this_row, col=1)
    fig.update_yaxes(title_text='Y', row=this_row, col=1)
    fig.update_xaxes(range=[t_start, t_end], row=this_row, col=1)
    
    
@st.cache_data(max_entries=100)
def plot_dlc_time_course(dict_dlc, features_to_plot, t_range, if_plot_likelihood=True, t_range_2d=None):
    row_heights = [1.5] + [0.7, 2, 2] * len(features_to_plot) if if_plot_likelihood else ([1.5] + [2, 2] * len(features_to_plot))
    fig = make_subplots(rows=len(row_heights), cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.025,
                        subplot_titles=[""] + list(itertools.chain(*[[feature, "", ""] for feature in features_to_plot]))
                                        if if_plot_likelihood else [""] + list(itertools.chain(*[[feature, ""] for feature in features_to_plot])))
    
    add_behav_events(dict_behav_events, range=t_range,
                     fig=fig, row=1, col=1)
    
    col_map = px.colors.qualitative.Plotly
    
    for n, (feature, color) in enumerate(features_to_plot.items()):
        add_dlc_feature_x_y(dict_dlc, feature, likelihood_threshold, range=t_range, fig=fig, 
                            start_row=2+n*(3 if if_plot_likelihood else 2), col=1, if_plot_likelihood=if_plot_likelihood,
                            color=color, t_range_2d=t_range_2d)

    fig.update_layout(height=300+row_height*len(features_to_plot), width=1700, 
                      hovermode='closest',
                    #   plot_bgcolor='white', paper_bgcolor='white', 
                     font=dict(color='black'))

    # Show the plot
    # st.plotly_chart(fig)
    return fig

def plot_dlc_2d(dict_dlc, features_to_plot, t_range_2d, if_lines_in_2d=False):
    
    fig_2ds = {}
    
    for camera in ['Camera0', 'Camera1']:
        features_to_plot_this = {feature: features_to_plot[feature] for feature in features_to_plot if camera in feature}
        fig_2d = go.Figure()
        
        for n, (feature, color) in enumerate(features_to_plot_this.items()):
            
            this_feature = dict_dlc[feature]

            x, y, likelihood, t = this_feature['x'], this_feature['y'], this_feature['likelihood'], this_feature['t']
            
            valid_time = (t_range_2d[0] <= t) & (t < t_range_2d[1])
            x = x[valid_time]
            y = y[valid_time]
            likelihood = likelihood[valid_time]
            t = t[valid_time]
            
            invalid = likelihood < likelihood_threshold
            x[invalid] = np.nan
            y[invalid] = np.nan

            fig_2d.add_trace(
                go.Scattergl(x=x,
                            y=y,
                            mode=f'markers{"+lines" if if_lines_in_2d else ""}',
                            marker=dict(
                                    size=5,
                                    opacity=(t - np.min(t)) / (np.max(t) - np.min(t)), 
                                    color=color,
                                ),
                            name=feature
                            )
            )
            
        fig_2d.update_layout(
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
                autorange = "reversed"
            ),
            height=700,
            width=900,
            title=f'{camera}, {camera_mapping[camera]}',
            showlegend=True,
            )
        
        fig_2ds[camera] = fig_2d
        
    return fig_2ds

init()

with st.sidebar:
    with st.expander('DLC settings', expanded=True):
        likelihood_threshold = st.slider('Likeshold threshold', 0.000, 1.000, 0.950, step=0.001)
        if_plot_likelihood = st.checkbox('Plot likelihood', value=True)
        row_height = st.slider('row height', 100, 1000, 200)
        
        if_2d = st.checkbox('If draw 2d', value=True)
        if_lines_in_2d = st.checkbox('If show lines', value=False, disabled=not if_2d)
        

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
    
    features = st.multiselect('DLC feature', dlc_features, ['Camera1_bottom_tongue'],)
    event_color_map = {'lickportready': 'greenyellow', 'go': 'green', 'choice': 'magenta', 'right_lick': 'blue', 'left_lick': 'red', 'reward': 'cyan', 'trialend': 'black'}
    
    col_map = px.colors.qualitative.Plotly
    features_to_plot = {}
    for n, feature in enumerate(features):
        features_to_plot[feature] = col_map[n % len(col_map)]

    max_t = list(dict_dlc.values())[0]['t'][-1]
    cols = st.columns([1, 1, 2])
    t_range = cols[0].slider('Time course range', 0.0, 500.0, value=20.0, step=1.0)
    t_center = cols[1].slider('Time course center', t_range / 2, max_t - t_range / 2, value=50.0, step=t_range / 10)
    t_start, t_end = t_center - t_range / 2, t_center + t_range / 2   
    
    # --- Time course ---
    h_time_course = st.empty()
    
    if if_2d:
        cols = st.columns([1, 1, 2])
        t_range_2d = cols[0].slider('time_range', 0.0, t_range, value=t_range / 2, step=t_range / 50)
        t_center_2d = cols[1].slider('time_start', t_start + t_range_2d / 2, t_end - t_range_2d / 2, value=t_center, step=t_range_2d / 10)
        t_start_2d, t_end_2d = t_center_2d - t_range_2d / 2, t_center_2d + t_range_2d / 2  
    
    with h_time_course:
        fig = plot_dlc_time_course(dict_dlc, features_to_plot, t_range=[t_start, t_end], if_plot_likelihood=if_plot_likelihood, t_range_2d=[t_start_2d, t_end_2d] if if_2d else None)
        plotly_events(fig, override_width=fig.layout.width, override_height=fig.layout.height)

    # --- 2-d view ---
    if if_2d:
        col_2d = st.columns([1, 1])
        fig_2ds = plot_dlc_2d(dict_dlc, features_to_plot, t_range_2d=[t_start_2d, t_end_2d], if_lines_in_2d=if_lines_in_2d)
        
        for n, camera in enumerate(['Camera0', 'Camera1']):
            with col_2d[n]:
                plotly_events(fig_2ds[camera], override_height=fig_2ds[camera].layout.height, override_width=fig_2ds[camera].layout.width)
    
    #  --- navigate raw annotated videos ---
    mouse, date= nwb_file.split('_')[0], nwb_file.split('_')[1]
    sessions = [sub.split('/')[-1] for sub in fs.ls(f'{video_root}/{mouse}')]
    session = [s for s in sessions if date in s][0]
    videos = [file.split('/')[-1] for file in fs.glob(f'{video_root}/{mouse}/{session}/*.mp4')]

    for n, camera in enumerate(['Camera0', 'Camera1']):
        with col_2d[n]:
            video = st.selectbox('video', [v for v in videos if camera_mapping[camera] in v])
            file_name = f'{video_root}/{mouse}/{session}/{video}'

            cols = st.columns([1, 2])
            with fs.open(file_name, 'rb') as f:
                cols[0].video(f.read())
        
        
