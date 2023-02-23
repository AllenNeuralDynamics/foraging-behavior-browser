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

from streamlit_util import filter_dataframe, aggrid_interactive_table_session

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

    
@st.cache_data(ttl=24*3600)
def load_data(tables=['sessions']):
    df = {}
    for table in tables:
        file_name = cache_folder + f'df_{table}.pkl'
        if use_s3:
            with fs.open(file_name) as f:
                df[table] = pd.read_pickle(f)
        else:
            df[table] = pd.read_pickle(file_name)
        
    return df

def fetch_img_from_s3(glob_patterns):
    for pattern in glob_patterns:
        file = fs.glob(pattern)
        if len(file): break
        
    if not len(file): return None
    with fs.open(file[0]) as f:
        img = Image.open(f)
        img = img.crop() 
    
    return img

# @st.cache_data(ttl=24*3600)
def get_session_logistic_regression(key):
    sess_date_str = datetime.strftime(datetime.strptime(key['session_date'], '%Y-%m-%dT%H:%M:%S'), '%Y%m%d')
     
    fn = f'/{key["h2o"]}_{sess_date_str}_*'
    glob_patterns = [cache_fig_folder + 'logistic_regression/' + key["h2o"] + fn]
    
    if use_s3:
        img = fetch_img_from_s3(glob_patterns)
        # img = img.crop((0, 0, 5400, 3000))     
    else:
        file = glob.glob(glob_str)
        if len(file) == 1:
            img = Image.open(file[0])
            img = img.crop((500, 140, 5400, 3000))
            
    return img


# @st.cache_data(ttl=24*3600)
def get_session_fitted_choice(key):
    sess_date_str = datetime.strftime(datetime.strptime(key['session_date'], '%Y-%m-%dT%H:%M:%S'), '%Y%m%d')
     
    fns = [f'/{key["h2o"]}_{sess_date_str}_*model_best*',
           f'/{key["h2o"]}_{sess_date_str}_*model_None*']
    
    glob_patterns = [cache_fig_folder + 'fitted_choice/' + key["h2o"] + fn for fn in fns]
    
    if use_s3:
        img = fetch_img_from_s3(glob_patterns)
        img = img.crop((0, 0, img.size[0], img.size[1])) 
    else:
        file = glob.glob(glob_patterns)
        if len(file) == 1:
            img = Image.open(file[0])
            img = img.crop((500, 140, 5400, 3000))
            
    return img


# table_mapping = {
#     'sessions': fetch_sessions,
#     'ephys_units': fetch_ephys_units,
# }


def add_session_filter():
    with st.expander("Behavioral session filter", expanded=True):   
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
        
    st.session_state.aggrid_outputs = aggrid_interactive_table_session(df=st.session_state.df_session_filtered)

    # st.dataframe(st.session_state.df_session_filtered, use_container_width=True, height=1000)

    container_unit_all_in_one = st.container()
    
    with container_unit_all_in_one:
        # with st.expander("Expand to see all-in-one plot for selected unit", expanded=True):
        if len(st.session_state.aggrid_outputs['selected_rows']) == 1:
            fig_fitted_choice = get_session_fitted_choice(st.session_state.aggrid_outputs['selected_rows'][0])
            st.image(fig_fitted_choice, output_format='PNG', width=1500, caption='')  # use_column_width='always', 

            fig_logistic_regression = get_session_logistic_regression(st.session_state.aggrid_outputs['selected_rows'][0])
            st.image(fig_logistic_regression, output_format='PNG', width=500)




if 'df' not in st.session_state: 
    init()
    
app()

            
if if_profile:    
    p.stop()