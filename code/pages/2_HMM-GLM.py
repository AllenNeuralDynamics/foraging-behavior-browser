"""Visualize HMM-GLM results from Faeze
"""
import os
from PIL import Image

import streamlit as st
import s3fs
import streamlit_nested_layout

# Set up the S3 bucket and prefix
bucket_name = "s3://aind-behavior-data/faeze/HMM-GLM"

# Initialize s3fs
s3 = s3fs.S3FileSystem(anon=False)

# Function to open image from S3
@st.cache_data(ttl=24*3600, max_entries=20)
def open_image_from_s3(file_key):
    with s3.open(f'{file_key}', 'rb') as f:
        img = Image.open(f)
        img = img.crop()
        return img

# Function to get image file names in a folder
def get_image_files(bucket_name, prefix):
    all_objects = s3.ls(f'{bucket_name}/{prefix}', detail=True)
    image_files = [obj['Key'] for obj in all_objects if obj['type'] == 'file' and obj['Key'].endswith('.png')]
    return image_files

# Get the list of data folders
data_folders = [os.path.basename(f) for f in 
                s3.glob(f'{bucket_name}/*')
                ]

# Data folder selection dropdown
data_folder_selected = st.selectbox('Select Data Folder', data_folders)

if data_folder_selected:

    # Get the list of mice folders
    mice = [os.path.basename(f) for f in
                     s3.glob(f'{bucket_name}/{data_folder_selected}/*')
                     if not os.path.basename(f).startswith('.')
                    ]
    # Mouse selection dropdown
    mouse_selected = st.selectbox('Select Mouse', mice)

    # Show mouse-wise figures
    if mouse_selected:
        mouse_prefix = f'{data_folder_selected}/{mouse_selected}/'
        mouse_figures = ['AIC.png', 'BIC.png', 'LL.png']
        mouse_figures_keys = [f'{mouse_prefix}{fig}' for fig in mouse_figures]

        cols = st.columns(len(mouse_figures_keys))
        for i, img_key in enumerate(mouse_figures_keys):
            with cols[i]:
                img = open_image_from_s3(f'{bucket_name}/{img_key}')
                st.image(img, caption=os.path.basename(img_key))

        # Number of states selection
        num_states = st.selectbox('Select Number of States', ['two_states', 'three_states', 'four_states'])

        if num_states:
            states_prefix = f'{mouse_prefix}{num_states}/'
            glm_images = ['GLM_Weights.png', 'GLM_TM.png', 'frac_occupancy.png', 'frac_occupancy_of_sessions.png']
            glm_images_keys = [f'{states_prefix}{img}' for img in glm_images]

            cols = st.columns(len(glm_images_keys))
            for i, img_key in enumerate(glm_images_keys):
                with cols[i]:
                    img = open_image_from_s3(f'{bucket_name}/{img_key}')
                    st.image(img, caption=os.path.basename(img_key))

            # Grouped by selection
            grouped_by = st.selectbox('Grouped By', ['grouped_by_sessions', 'grouped_by_sessions_conventional_view', 'grouped_by_states'])

            if grouped_by:
                grouped_prefix = f'{states_prefix}{grouped_by}/'
                session_images_keys = get_image_files(bucket_name, grouped_prefix)
                
                for img_key in session_images_keys:
                    img = open_image_from_s3(img_key)
                    st.image(img, caption=os.path.basename(img_key))
