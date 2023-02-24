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
        return None, None

    if use_s3:
        with fs.open(file[0]) as f:
            img = Image.open(f)
            img = img.crop(crop) 
    else:
        img = Image.open(file[0])
        img = img.crop(crop)         
    
    return img, file[0]

# @st.cache_data(ttl=24*3600, max_entries=20)
def show_img_by_key_and_prefix(key, prefix, column=None, other_patterns=[''], crop=None, caption=True, **kwargs):
    sess_date_str = datetime.strftime(datetime.strptime(key['session_date'], '%Y-%m-%dT%H:%M:%S'), '%Y%m%d')
     
    fns = [f'/{key["h2o"]}_{sess_date_str}_*{other_pattern}*' for other_pattern in other_patterns]
    glob_patterns = [cache_fig_folder + f'{prefix}/' + key["h2o"] + fn for fn in fns]
    
    img, f_name = _fetch_img(glob_patterns, crop)

    _f = st if column is None else column
    
    _f.image(img if img is not None else "https://cdn-icons-png.flaticon.com/512/3585/3585596.png", 
                output_format='PNG', 
                caption=f_name.split('/')[-1] if caption else '',
                use_column_width='always',
                **kwargs)

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
        
        cols = st.columns([1, 2, 2])
        cols[0].markdown(f'### Filtered sessions')
        if cols[1].button('Press this and then Ctrl + R to reload from S3'):
            st.cache_data.clear()
            st.experimental_rerun()
             
        # aggrid_outputs = aggrid_interactive_table_units(df=df['ephys_units'])
        # st.session_state.df_session_filtered = aggrid_outputs['data']
        
        container_filtered_frame = st.container()
        
    with st.sidebar:
        add_session_filter()
        
    st.session_state.aggrid_outputs = aggrid_interactive_table_session(df=st.session_state.df_session_filtered)

    # st.dataframe(st.session_state.df_session_filtered, use_container_width=True, height=1000)
    
    # Setting up layout for each session
    layout_definition = [[1],   # columns in the first row
                         [2, 1],  # columns in the second row
                         ]  
    
    draw_type_mapper = {'1. Choice history': ('fitted_choice',   # prefix
                                           (0, 0),     # location (row_idx, column_idx)
                                           dict(other_patterns=['model_best', 'model_None'])),
                        '2. Lick times': ('lick_psth', 
                                       (1, 0), 
                                       {}),                         
                        '3. Logistic regression on choice': ('logistic_regression', 
                                                          (1, 1), 
                                                          dict(crop=(0, 0, 1200, 2000))),
                        '4. Win-stay-lose-shift prob.': ('wsls', 
                                                      (1, 1), 
                                                      dict(crop=(0, 0, 1200, 600))), 
                        }
    
    st.markdown('### Select session(s) above to draw')
    cols_option = st.columns([3, 0.5, 1])
    selected_draw_types = cols_option[0].multiselect('Which plot(s) to draw?', draw_type_mapper.keys(), default=draw_type_mapper.keys())
    num_cols = cols_option[1].number_input('Number of columns', 1, 10)
    container_session_all_in_one = st.container()
    
    with container_session_all_in_one:
        # with st.expander("Expand to see all-in-one plot for selected unit", expanded=True):

        selected_keys = st.session_state.aggrid_outputs['selected_rows']
        
        if len(selected_keys):
            st.write(f'Loading selected {len(selected_keys)} sessions...')
            my_bar = st.columns((1, 7))[0].progress(0)
             
            major_cols = st.columns([1] * num_cols)

            for i, key in enumerate(selected_keys):
                this_major_col = major_cols[i % num_cols]
                
                # setting up layout for each session
                rows = []
                with this_major_col:
                    st.markdown(f'''<h3 style='text-align: center; color: blue;'>{key["h2o"]}, Session {key["session"]}, {key["session_date"].split("T")[0]}''',
                              unsafe_allow_html=True)
                    if len(selected_draw_types) > 1:  # more than one types, use the pre-defined layout
                        for row, column_setting in enumerate(layout_definition):
                            rows.append(this_major_col.columns(column_setting))
                    else:    # else, put it in the whole column
                        rows = this_major_col.columns([1])
                    st.markdown("---")

                for draw_type in selected_draw_types:
                    prefix, position, setting = draw_type_mapper[draw_type]
                    this_col = rows[position[0]][position[1]] if len(selected_draw_types) > 1 else rows[0]
                    show_img_by_key_and_prefix(key, 
                                                column=this_col,
                                                prefix=prefix, 
                                                **setting)
                    
                my_bar.progress(int((i + 1) / len(selected_keys) * 100))



if 'df' not in st.session_state: 
    init()
    
app()

            
if if_profile:    
    p.stop()