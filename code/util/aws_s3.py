import json

import pandas as pd
import s3fs
import streamlit as st
from PIL import Image

from .settings import (draw_type_layout_definition,
                       draw_type_mapper_session_level,
                       draw_types_quick_preview)

# --------------------------------------
data_sources = ['bonsai', 'bpod']

s3_nwb_folder = {data: f'aind-behavior-data/foraging_nwb_{data}/' for data in data_sources}
s3_processed_nwb_folder = {data: f'aind-behavior-data/foraging_nwb_{data}_processed/' for data in data_sources}
# --------------------------------------

fs = s3fs.S3FileSystem(anon=False)


if 'selected_points' not in st.session_state:
    st.session_state['selected_points'] = []

def load_data(tables=['sessions'], data_source = 'bonsai'):
    df = {}
    for table in tables:
        file_name = s3_processed_nwb_folder[data_source] + f'df_{table}.pkl'
        try:
            with fs.open(file_name) as f:
                df[table + '_bonsai'] = pd.read_pickle(f)
        except FileNotFoundError as e:
            st.markdown(f'''### df_{table}.pkl is missing on S3. \n'''
                        f'''## It is very likely that Han is [rerunning the whole pipeline](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-trigger-pipeline?tab=readme-ov-file#notes-on-manually-re-process-all-nwbs-and-overwrite-s3-database-and-thus-the-streamlit-app). Please come back after an hour.''')
            st.markdown('')
            st.image("https://github.com/user-attachments/assets/bea2c7a9-c561-4c5f-afa5-92d63c040be6",
                     width=500)
    return df


def draw_session_plots_quick_preview(df_to_draw_session):

    container_session_all_in_one = st.container()

    key = df_to_draw_session.to_dict(orient='records')[0]

    with container_session_all_in_one:
        try:
            date_str = key["session_date"].strftime('%Y-%m-%d')
        except:
            date_str = key["session_date"].split("T")[0]

        st.markdown(f'''<h5 style='text-align: center; color: orange;'>{key["h2o"]}, Session {int(key["session"])}, {date_str} '''
                    f'''({key["user_name"]}@{key["data_source"]})''',
                    unsafe_allow_html=True)

        rows = []
        for row, column_setting in enumerate(draw_type_layout_definition):
            rows.append(st.columns(column_setting))

        for draw_type in draw_types_quick_preview:
            prefix, position, setting = draw_type_mapper_session_level[draw_type]
            this_col = rows[position[0]][position[1]] if len(draw_types_quick_preview) > 1 else rows[0]
            show_session_level_img_by_key_and_prefix(
                key,
                column=this_col,
                prefix=prefix,
                data_source=key["hardware"],
                **setting,
            )


# @st.cache_data(ttl=24*3600, max_entries=20)
def show_session_level_img_by_key_and_prefix(key, prefix, column=None, other_patterns=[''], crop=None, caption=True, data_source='bonsai', **kwargs):
    try:
        date_str = key["session_date"].strftime(r'%Y-%m-%d')
    except:
        date_str = key["session_date"].split("T")[0]
    
    # Convert session_date to 2024-04-01 format
    subject_session_date_str = f"{key['subject_id']}_{date_str}_{key['nwb_suffix']}".split('_0')[0]
    glob_patterns = [s3_processed_nwb_folder[data_source] + f"{subject_session_date_str}/{subject_session_date_str}_{prefix}*"]
    
    img, f_name = _fetch_img(glob_patterns, crop)

    _f = st if column is None else column
    
    _f.image(img if img is not None else "https://cdn-icons-png.flaticon.com/512/3585/3585596.png", 
                output_format='PNG', 
                caption=f_name.split('/')[-1] if caption and f_name else '',
                use_column_width='always',
                **kwargs)

    return img


def _fetch_img(glob_patterns, crop=None): 
    # Fetch the img that first matches the patterns
    for pattern in glob_patterns:
        file = fs.glob(pattern)
        if len(file): break
        
    if not len(file):
        return None, None

    try:
        with fs.open(file[0]) as f:
            img = Image.open(f)
            img = img.crop(crop) 
    except:
        st.write('File found on S3 but failed to load...')
        return None, None
    
    return img, file[0]


def show_debug_info():
    with st.expander('CO processing NWB errors', expanded=False):
        error_file = s3_processed_nwb_folder['bonsai'] + 'error_files.json'
        if fs.exists(error_file):
            with fs.open(error_file) as file:
                st.json(json.load(file))
        else:
            st.write('No NWB error files')
            
    with st.expander('CO Pipeline log', expanded=False):
        with fs.open(s3_processed_nwb_folder['bonsai'] + 'pipeline.log') as file:
            log_content = file.read().decode('utf-8')
        log_content = log_content.replace('\\n', '\n')
        st.text(log_content)
        
    with st.expander('NWB convertion and upload log', expanded=False):
        with fs.open(s3_nwb_folder['bonsai'] + 'bonsai_pipeline.log') as file:
            log_content = file.read().decode('utf-8')
        st.text(log_content)