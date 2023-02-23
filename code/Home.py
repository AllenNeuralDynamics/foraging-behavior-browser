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

def _fetch_img(glob_patterns, crop=None):
    # Fetch the img that first matches the patterns
    for pattern in glob_patterns:
        file = fs.glob(pattern) if use_s3 else glob.glob(pattern)
        if len(file): break
        
    if not len(file): 
        return None

    if use_s3:
        with fs.open(file[0]) as f:
            img = Image.open(f)
            img = img.crop(crop) 
    else:
        img = Image.open(file[0])
        img = img.crop(crop)         
    
    return img

# @st.cache_data(ttl=24*3600, max_entries=20)
def get_img_by_key(key, prefix, other_patterns=[''], crop=None):
    sess_date_str = datetime.strftime(datetime.strptime(key['session_date'], '%Y-%m-%dT%H:%M:%S'), '%Y%m%d')
     
    fns = [f'/{key["h2o"]}_{sess_date_str}_*{other_pattern}*' for other_pattern in other_patterns]
    glob_patterns = [cache_fig_folder + f'{prefix}/' + key["h2o"] + fn for fn in fns]
    
    img = _fetch_img(glob_patterns, crop)

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

        selected_keys = st.session_state.aggrid_outputs['selected_rows']
        if len(selected_keys):
            for key in selected_keys:
                fig_fitted_choice = get_img_by_key(key, prefix='fitted_choice', other_patterns=['model_best', 'model_None'])
                st.image(fig_fitted_choice, output_format='PNG', width=1500, caption='')  # use_column_width='always', 

                fig_logistic_regression = get_img_by_key(key, prefix='logistic_regression')
                st.image(fig_logistic_regression, output_format='PNG', width=500)

                fig_lick_psth = get_img_by_key(key, prefix='lick_psth')
                st.image(fig_lick_psth, output_format='PNG', width=None)



if 'df' not in st.session_state: 
    init()
    
app()

            
if if_profile:    
    p.stop()