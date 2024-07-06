"""Visualize HMM-GLM results from Faeze

Han, ChatGPT
"""
import os
import re
from PIL import Image

import streamlit as st
import s3fs
import streamlit_nested_layout

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

# Set up the S3 bucket and prefix
bucket_name = "s3://aind-behavior-data/faeze/HMM-GLM"

# Initialize s3fs
s3 = s3fs.S3FileSystem(anon=False)

# Function to open image from S3
@st.cache_data(ttl=24*3600)
def open_image_from_s3(file_key, crop=None, **kwargs):
    with s3.open(f'{file_key}', 'rb') as f:
        img = Image.open(f)
        img = img.crop(crop)
        
    st.image(img, **kwargs)
    
# Function to extract the session number from file name
def extract_session_number(file_name):
    match = re.search(r'sess_(\d+)', file_name)
    return int(match.group(1)) if match else None


# --------------- Main app -------------------
def app():
    # Get the list of data folders
    data_folders = [os.path.basename(f) for f in 
                    s3.glob(f'{bucket_name}/*')
                    ]

    # Data folder selection dropdown
    with st.sidebar:
        widget_data_folder = st.container()
        widget_mouse = st.container()
        widget_n_states = st.container()
        widget_model_comparison = st.container()
        
        if st.button('Reload data from S3'):
            st.cache_data.clear()
            st.rerun()
        
    data_folder_selected = widget_data_folder.selectbox('Select Data Folder', data_folders)

    if data_folder_selected:
        # Get the list of mice folders
        mice = [os.path.basename(f) for f in
                        s3.glob(f'{bucket_name}/{data_folder_selected}/*')
                        if not os.path.basename(f).startswith('.')
                        ]
        # Mouse selection dropdown
        mouse_selected = widget_mouse.selectbox('Select Mouse', mice)

        # Show mouse-wise figures
        if mouse_selected:
            mouse_folder = f'{bucket_name}/{data_folder_selected}/{mouse_selected}'
            
            with widget_model_comparison:
                fig_model_comparisons = ['AIC.png', 'BIC.png', 'LL.png']

                for i, fig_model_comparison in enumerate(fig_model_comparisons):
                    img = open_image_from_s3(
                        f'{mouse_folder}/{fig_model_comparison}',
                        caption=fig_model_comparison,
                        )

            # Number of states selection
            num_states = widget_n_states.selectbox('Select Number of States', 
                                                    ['two_states', 'three_states', 'four_states'],
                                                    index=2, # Default shows four states
                                                    )

            if num_states:
                num_states_folder = f'{mouse_folder}/{num_states}'
                fig_states = ['GLM_Weights.png', 'GLM_TM.png', 'frac_occupancy.png', 'frac_occupancy_of_sessions.png']

                cols = st.columns([1, 0.2, 1])
                with cols[0]:
                    open_image_from_s3(f'{num_states_folder}/GLM_Weights.png', 
                                    caption='GLM weights')
                with cols[2]:
                    open_image_from_s3(f'{num_states_folder}/frac_occupancy_of_sessions.png', 
                                    caption='Fraction occupancy over sessions')
                with cols[1]:
                    open_image_from_s3(f'{num_states_folder}/frac_occupancy.png', 
                                    caption='Fraction occupancy (all)')
                    open_image_from_s3(f'{num_states_folder}/GLM_TM.png', 
                                    caption='Fraction occupancy (all)')    
                                                
                # Grouped by selection
                with st.container(height=700):
                    cols = st.columns([1.5, 1, 4])
                    grouped_by = cols[0].selectbox('Grouped By', ['grouped_by_sessions', 'grouped_by_sessions_conventional_view', 'grouped_by_states'])
                    num_cols = cols[1].number_input(
                        label='number of columns',
                        min_value=1,
                        max_value=10,
                        value=2,
                    )
                    
                    if grouped_by:
                        fig_sessions = s3.glob(f'{num_states_folder}/{grouped_by}/*.png')
                        fig_sessions = sorted(fig_sessions, key=extract_session_number)
                        cols = st.columns(num_cols)
                        
                        for i, fig_session in enumerate(fig_sessions):
                            with cols[i % num_cols]:
                                img = open_image_from_s3(fig_session, caption=fig_session)


app()