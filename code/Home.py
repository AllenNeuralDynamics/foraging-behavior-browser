#%%
import pandas as pd
import streamlit as st
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import s3fs
import os

from PIL import Image, ImageColor
import streamlit.components.v1 as components
import streamlit_nested_layout

from streamlit_util import filter_dataframe, aggrid_interactive_table_units


if_profile = False

if if_profile:
    from streamlit_profiler import Profiler
    p = Profiler()
    p.start()


# from pipeline import experiment, ephys, lab, psth_foraging, report, foraging_analysis
# from pipeline.plot import foraging_model_plot

cache_folder = 'xxx'  #'/root/capsule/data/s3/report/st_cache/'
cache_fig_folder = 'xxx' #'/root/capsule/data/s3/report/all_units/'  # 

if os.path.exists(cache_folder):
    use_s3 = False
else:
    cache_folder = 'aind-behavior-data/Han/ephys/report/st_cache/'
    cache_fig_folder = 'aind-behavior-data/Han/ephys/report/all_sessions/'
    
    fs = s3fs.S3FileSystem(anon=False)
    use_s3 = True

st.set_page_config(layout="wide", page_title='Foraging behavior navigator')

if 'selected_points' not in st.session_state:
    st.session_state['selected_points'] = []

    
@st.experimental_memo(ttl=24*3600)
def load_data(tables=['sessions']):
    df = {}
    for table in tables:
        file_name = cache_folder + f'{table}.pkl'
        if use_s3:
            with fs.open(file_name) as f:
                df[table] = pd.read_pickle(f)
        else:
            df[table] = pd.read_pickle(file_name)
        
    return df

# @st.experimental_memo(ttl=24*3600)
def get_session_logistic_regression(key):
    sess_date_str = datetime.strftime(datetime.strptime(key['session_date'], '%Y-%m-%dT%H:%M:%S'), '%Y%m%d')
     
    fn = f'/{key["h2o"]}_{sess_date_str}_*'
    glob_str = cache_fig_folder + 'logistic_regression/' + key["h2o"] + fn
    
    if use_s3:
        file = fs.glob(glob_str)
        if len(file) == 1:
            with fs.open(file[0]) as f:
                img = Image.open(f)
                img = img.crop((0, 0, 5400, 3000))           
    else:
        file = glob.glob(glob_str)
        if len(file) == 1:
            img = Image.open(file[0])
            img = img.crop((500, 140, 5400, 3000))
            
    return img

# table_mapping = {
#     'sessions': fetch_sessions,
#     'ephys_units': fetch_ephys_units,
# }


@st.experimental_memo(ttl=24*3600)
def get_fig(key):
    fig = plt.figure(figsize=(8, 3), constrained_layout=True)
    ax = fig.subplots(1,1)
    
    foraging_model_plot.plot_session_fitted_choice(key, ax=ax, remove_ignored=False, first_n=2)
    return fig   

def add_session_filter():
    with st.expander("Unit filter", expanded=True):   
        st.session_state.df_session_filtered = filter_dataframe(df=st.session_state.df['sessions'])
    st.markdown(f"### {len(st.session_state.df_session_filtered)} sessions filtered (use_s3 = {use_s3})")


# ------- Layout starts here -------- #    
def init():
    df = load_data(['sessions'])
    st.session_state.df = df
    
    # Some global variables


def app():
    st.markdown('## Foraging Behavior Browser')
       
          
    with st.container():
        # col1, col2 = st.columns([1.5, 1], gap='small')
        # with col1:
        # -- 1. unit dataframe --
        st.markdown(f'### Filtered sessions')
        
        # aggrid_outputs = aggrid_interactive_table_units(df=df['ephys_units'])
        # st.session_state.df_session_filtered = aggrid_outputs['data']
        
        container_filtered_frame = st.container()
        
    with st.sidebar:
        add_session_filter()
        
    st.session_state.aggrid_outputs = aggrid_interactive_table_units(df=st.session_state.df_session_filtered)

    # st.dataframe(st.session_state.df_session_filtered, use_container_width=True, height=1000)

    container_unit_all_in_one = st.container()
    
    with container_unit_all_in_one:
        # with st.expander("Expand to see all-in-one plot for selected unit", expanded=True):
        if len(st.session_state.aggrid_outputs['selected_rows']) == 1:
            unit_fig = get_session_logistic_regression(st.session_state.aggrid_outputs['selected_rows'][0])
            st.image(unit_fig, output_format='PNG', width=3000)

if 'df' not in st.session_state: 
    init()
    
app()

            
if if_profile:    
    p.stop()