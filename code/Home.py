"""
Streamlit app for visualizing behavior data
https://foraging-behavior-browser.streamlit.app/

Note the url is now queryable, e.g. https://foraging-behavior-browser.streamlit.app/?subject_id=41392

Example queries:
 /?subject_id=699982   # only show one subject
 /?session=10&session=20  # show sessions between 10 and 20
 /?tab_id=tab_1  # Show specific tab
 /?if_aggr_all=false

"""

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
import plotly.express as px
import plotly
import plotly.graph_objects as go
import statsmodels.api as sm
import json

from PIL import Image, ImageColor
import streamlit.components.v1 as components
import streamlit_nested_layout
from streamlit_plotly_events import plotly_events
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm

# To suppress the warning that I set the default value of a widget and also set it in the session state
from streamlit.elements.utils import _shown_default_value_warning
_shown_default_value_warning = False

from util.streamlit import (filter_dataframe, aggrid_interactive_table_session,
                            aggrid_interactive_table_curriculum, add_session_filter, data_selector,
                            add_xy_selector, add_xy_setting, add_auto_train_manager,
                            _plot_population_x_y)
from util.url_query_helper import sync_widget_with_query

import extra_streamlit_components as stx

from aind_auto_train.curriculum_manager import CurriculumManager
from aind_auto_train.auto_train_manager import DynamicForagingAutoTrainManager


# Sync widgets with URL query params
# https://blog.streamlit.io/how-streamlit-uses-streamlit-sharing-contextual-apps/
# dict of "key": default pairs
# Note: When creating the widget, add argument "value"/"index" as well as "key" for all widgets you want to sync with URL
to_sync_with_url_query = {
    'filter_subject_id': '',
    'filter_session': [0.0, None],
    'filter_finished_trials': [0.0, None],
    'filter_foraging_eff': [0.0, None],
    'filter_task': ['all'],
    
    'tab_id': 'tab_session_x_y',
    'x_y_plot_xname': 'session',
    'x_y_plot_yname': 'foraging_eff',
    'x_y_plot_group_by': 'h2o',
    'x_y_plot_if_show_dots': True,
    'x_y_plot_if_aggr_each_group': True,
    'x_y_plot_aggr_method_group': 'lowess',
    'x_y_plot_if_aggr_all': True,
    'x_y_plot_aggr_method_all': 'mean +/- sem',
    'x_y_plot_smooth_factor': 5,
    'x_y_plot_if_use_x_quantile_group': False,
    'x_y_plot_q_quantiles_group': 20,
    'x_y_plot_if_use_x_quantile_all': False,
    'x_y_plot_q_quantiles_all': 20,
    'x_y_plot_dot_size': 10,
    'x_y_plot_dot_opacity': 0.5,
    'x_y_plot_line_width': 2.0,
    
    'session_plot_mode': 'sessions selected from table or plot',

    'auto_training_history_x_axis': 'date',
    'auto_training_history_sort_by': 'subject_id',
    'auto_training_history_sort_order': 'descending',
    'auto_training_curriculum_name': 'Uncoupled Baiting',
    'auto_training_curriculum_version': '1.0',
    'auto_training_curriculum_schema_version': '1.0',
    }


raw_nwb_folder = 'aind-behavior-data/foraging_nwb_bonsai/'
cache_folder = 'aind-behavior-data/foraging_nwb_bonsai_processed/'
# cache_session_level_fig_folder = 'aind-behavior-data/Han/ephys/report/all_sessions/'
# cache_mouse_level_fig_folder = 'aind-behavior-data/Han/ephys/report/all_subjects/'

fs = s3fs.S3FileSystem(anon=False)
st.session_state.use_s3 = True

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

if 'selected_points' not in st.session_state:
    st.session_state['selected_points'] = []

    

def _get_urls():
    cache_folder = 'aind-behavior-data/Han/ephys/report/st_cache/'
    cache_session_level_fig_folder = 'aind-behavior-data/Han/ephys/report/all_sessions/'
    cache_mouse_level_fig_folder = 'aind-behavior-data/Han/ephys/report/all_subjects/'
    
    fs = s3fs.S3FileSystem(anon=False)
   
    with fs.open('aind-behavior-data/Han/streamlit_CO_url.json', 'r') as f:
        data = json.load(f)
    
    return data['behavior'], data['ephys']
                    
@st.cache_data(ttl=24*3600)
def load_data(tables=['sessions']):
    df = {}
    for table in tables:
        file_name = cache_folder + f'df_{table}.pkl'
        if st.session_state.use_s3:
            with fs.open(file_name) as f:
                df[table + '_bonsai'] = pd.read_pickle(f)
        else:
            df[table + '_bonsai'] = pd.read_pickle(file_name)
    return df

def _fetch_img(glob_patterns, crop=None):
    # Fetch the img that first matches the patterns
    for pattern in glob_patterns:
        file = fs.glob(pattern) if st.session_state.use_s3 else glob.glob(pattern)
        if len(file): break
        
    if not len(file):
        return None, None

    try:
        if st.session_state.use_s3:
            with fs.open(file[0]) as f:
                img = Image.open(f)
                img = img.crop(crop) 
        else:
            img = Image.open(file[0])
            img = img.crop(crop)         
    except:
        st.write('File found on S3 but failed to load...')
        return None, None
    
    return img, file[0]

def _user_name_mapper(user_name):
    user_mapper = {  # tuple of key words --> user name
        ('Avalon',): 'Avalon Amaya',
        ('Ella',): 'Ella Hilton',
        ('Katrina',): 'Katrina Nguyen',
        ('Lucas',): 'Lucas Kinsey',
        ('Travis',): 'Travis Ramirez',
        ('Xinxin', 'the ghost'): 'Xinxin Yin',
        }
    for key_words, name in user_mapper.items():
        for key_word in key_words:
            if key_word in user_name:
                return name
    else:
        return user_name

# @st.cache_data(ttl=24*3600, max_entries=20)
def show_session_level_img_by_key_and_prefix(key, prefix, column=None, other_patterns=[''], crop=None, caption=True, **kwargs):
    try:
        date_str = key["session_date"].strftime(r'%Y-%m-%d')
    except:
        date_str = key["session_date"].split("T")[0]
    
    # Convert session_date to 2024-04-01 format
    subject_session_date_str = f"{key['subject_id']}_{date_str}_{key['nwb_suffix']}".split('_0')[0]
    glob_patterns = [cache_folder + f"{subject_session_date_str}/{subject_session_date_str}_{prefix}*"]
    
    img, f_name = _fetch_img(glob_patterns, crop)

    _f = st if column is None else column
    
    _f.image(img if img is not None else "https://cdn-icons-png.flaticon.com/512/3585/3585596.png", 
                output_format='PNG', 
                caption=f_name.split('/')[-1] if caption and f_name else '',
                use_column_width='always',
                **kwargs)

    return img

def show_mouse_level_img_by_key_and_prefix(key, prefix, column=None, other_patterns=[''], crop=None, caption=True, **kwargs):
     
    fns = [f'/{key["h2o"]}_*{other_pattern}*' for other_pattern in other_patterns]
    glob_patterns = [cache_mouse_level_fig_folder + f'{prefix}/' + fn for fn in fns]
    
    img, f_name = _fetch_img(glob_patterns, crop)
    
    if img is None:  # Use "not_found" image
        glob_patterns = [cache_mouse_level_fig_folder + f'{prefix}/not_found_*{other_pattern}**' for other_pattern in other_patterns]
        img, f_name = _fetch_img(glob_patterns, crop)
        
    _f = st if column is None else column
    
    _f.stream(img if img is not None else "https://cdn-icons-png.flaticon.com/512/3585/3585596.png", 
                output_format='PNG', 
                #caption=f_name.split('/')[-1] if caption and f_name else '',
                use_column_width='always',
                **kwargs)

    return img

# table_mapping = {
#     'sessions_bonsai': fetch_sessions,
#     'ephys_units': fetch_ephys_units,
# }

@st.cache_resource(ttl=24*3600)
def get_pyg_renderer(df, spec="./gw_config.json", **kwargs) -> "StreamlitRenderer":
    return StreamlitRenderer(df, spec=spec, debug=False, **kwargs)

    
def draw_session_plots(df_to_draw_session):
    
    # Setting up layout for each session
    layout_definition = [[1],   # columns in the first row
                         [1, 1],  # columns in the second row
                         [1, 1],
                         ]  
    
    # cols_option = st.columns([3, 0.5, 1])
    container_session_all_in_one = st.container()
    
    with container_session_all_in_one:
        # with st.expander("Expand to see all-in-one plot for selected unit", expanded=True):
        
        if len(df_to_draw_session):
            st.write(f'Loading selected {len(df_to_draw_session)} sessions...')
            my_bar = st.columns((1, 7))[0].progress(0)
             
            major_cols = st.columns([1] * st.session_state.num_cols)
            
            for i, key in enumerate(df_to_draw_session.to_dict(orient='records')):
                this_major_col = major_cols[i % st.session_state.num_cols]
                
                # setting up layout for each session
                rows = []
                with this_major_col:
                    
                    try:
                        date_str = key["session_date"].strftime('%Y-%m-%d')
                    except:
                        date_str = key["session_date"].split("T")[0]
                    
                    st.markdown(f'''<h4 style='text-align: center; color: orange;'>{key["h2o"]}, Session {int(key["session"])}, {date_str}''',
                              unsafe_allow_html=True)
                    if len(st.session_state.selected_draw_types) > 1:  # more than one types, use the pre-defined layout
                        for row, column_setting in enumerate(layout_definition):
                            rows.append(this_major_col.columns(column_setting))
                    else:    # else, put it in the whole column
                        rows = this_major_col.columns([1])
                    st.markdown("---")

                for draw_type in st.session_state.draw_type_mapper_session_level:
                    if draw_type not in st.session_state.selected_draw_types: continue  # To keep the draw order defined by st.session_state.draw_type_mapper_session_level
                    prefix, position, setting = st.session_state.draw_type_mapper_session_level[draw_type]
                    this_col = rows[position[0]][position[1]] if len(st.session_state.selected_draw_types) > 1 else rows[0]
                    show_session_level_img_by_key_and_prefix(key, 
                                                column=this_col,
                                                prefix=prefix, 
                                                **setting)
                    
                my_bar.progress(int((i + 1) / len(df_to_draw_session) * 100))
                
                
                
def draw_mice_plots(df_to_draw_mice):
    
    # Setting up layout for each session
    layout_definition = [[1],   # columns in the first row
                         ]  
    
    # cols_option = st.columns([3, 0.5, 1])
    container_session_all_in_one = st.container()
    
    with container_session_all_in_one:
        # with st.expander("Expand to see all-in-one plot for selected unit", expanded=True):
        
        if len(df_to_draw_mice):
            st.write(f'Loading selected {len(df_to_draw_mice)} mice...')
            my_bar = st.columns((1, 7))[0].progress(0)
             
            major_cols = st.columns([1] * st.session_state.num_cols_mice)
            
            for i, key in enumerate(df_to_draw_mice.to_dict(orient='records')):
                this_major_col = major_cols[i % st.session_state.num_cols_mice]
                
                # setting up layout for each session
                rows = []
                with this_major_col:
                    st.markdown(f'''<h3 style='text-align: center; color: orange;'>{key["h2o"]}''',
                              unsafe_allow_html=True)
                    if len(st.session_state.selected_draw_types_mice) > 1:  # more than one types, use the pre-defined layout
                        for row, column_setting in enumerate(layout_definition):
                            rows.append(this_major_col.columns(column_setting))
                    else:    # else, put it in the whole column
                        rows = this_major_col.columns([1])
                    st.markdown("---")

                for draw_type in st.session_state.draw_type_mapper_mouse_level:
                    if draw_type not in st.session_state.selected_draw_types_mice: continue
                    prefix, position, setting = st.session_state.draw_type_mapper_mouse_level[draw_type]
                    this_col = rows[position[0]][position[1]] if len(st.session_state.selected_draw_types_mice) > 1 else rows[0]
                    show_mouse_level_img_by_key_and_prefix(key, 
                                                        column=this_col,
                                                        prefix=prefix, 
                                                        **setting)
                    
                my_bar.progress(int((i + 1) / len(df_to_draw_mice) * 100))
                



  

def session_plot_settings(need_click=True):
    st.markdown('##### Show plots for individual sessions ')
    cols = st.columns([2, 1])
    
    session_plot_modes = [f'sessions selected from table or plot', f'all sessions filtered from sidebar']
    st.session_state.selected_draw_sessions = cols[0].selectbox(f'Which session(s) to draw?', 
                                                                session_plot_modes,
                                                                index=session_plot_modes.index(st.session_state['session_plot_mode'])
                                                                    if 'session_plot_mode' in st.session_state else 
                                                                    session_plot_modes.index(st.query_params['session_plot_mode'])
                                                                    if 'session_plot_mode' in st.query_params 
                                                                    else 0, 
                                                                key='session_plot_mode',
                                                               )
    
    n_session_to_draw = len(st.session_state.df_selected_from_plotly) \
        if 'selected from table or plot' in st.session_state.selected_draw_sessions \
        else len(st.session_state.df_session_filtered) 
    st.markdown(f'{n_session_to_draw} sessions to draw')
    
    st.session_state.num_cols = cols[1].number_input('number of columns', 1, 10, 
                                                     3 if 'num_cols' not in st.session_state else st.session_state.num_cols)
    
    st.markdown(
    """
    <style>
        .stMultiSelect [data-baseweb=select] span{
            max-width: 1000px;
        }
    </style>""",
    unsafe_allow_html=True,
    )
    st.session_state.selected_draw_types = st.multiselect('Which plot(s) to draw?', 
                                                          st.session_state.draw_type_mapper_session_level.keys(), 
                                                          default=st.session_state.draw_type_mapper_session_level.keys()
                                                          if 'selected_draw_types' not in st.session_state else 
                                                          st.session_state.selected_draw_types)
    if need_click:
        draw_it = st.button(f'Show me all {n_session_to_draw} sessions!', use_container_width=True)
    else:
        draw_it = True
    return draw_it

def mouse_plot_settings(need_click=True):
    st.markdown('##### Show plots for individual mice ')
    cols = st.columns([2, 1])
    st.session_state.selected_draw_mice = cols[0].selectbox('Which mice to draw?', 
                                                           [f'selected from table/plot ({len(st.session_state.df_selected_from_plotly.h2o.unique())} mice)', 
                                                            f'filtered from sidebar ({len(st.session_state.df_session_filtered.h2o.unique())} mice)'], 
                                                           index=0
                                                           )
    st.session_state.num_cols_mice = cols[1].number_input('Number of columns', 1, 10, 
                                                          3 if 'num_cols_mice' not in st.session_state else st.session_state.num_cols_mice)
    st.markdown(
        """
        <style>
            .stMultiSelect [data-baseweb=select] span{
                max-width: 1000px;
            }
        </style>""",
        unsafe_allow_html=True,
        )
    st.session_state.selected_draw_types_mice = st.multiselect('Which plot(s) to draw?', 
                                                          st.session_state.draw_type_mapper_mouse_level.keys(), 
                                                          default=st.session_state.draw_type_mapper_mouse_level.keys()
                                                          if 'selected_draw_types_mice' not in st.session_state else
                                                          st.session_state.selected_draw_types_mice)
    if need_click:
        draw_it = st.button('Show me all mice!', use_container_width=True)
    else:
        draw_it = True
    return draw_it


def plot_x_y_session():
            
    cols = st.columns([4, 10])
    
    with cols[0]:

        x_name, y_name, group_by, size_mapper = add_xy_selector(if_bonsai=True)

        (if_show_dots, if_aggr_each_group, aggr_method_group, if_use_x_quantile_group, q_quantiles_group,
        if_aggr_all, aggr_method_all, if_use_x_quantile_all, q_quantiles_all, smooth_factor,
        dot_size, dot_opacity, line_width) = add_xy_setting()
        

    
    # If no sessions are selected, use all filtered entries
    # df_x_y_session = st.session_state.df_selected_from_dataframe if if_plot_only_selected_from_dataframe else st.session_state.df_session_filtered
    df_x_y_session = st.session_state.df_session_filtered
    
    names = {('session', 'foraging_eff'): 'Foraging efficiency',
             ('session', 'finished'):   'Finished trials', 
             }

    df_selected_from_plotly = pd.DataFrame()
    # for i, (title, (x_name, y_name)) in enumerate(names.items()):
        # with cols[i]:
    with cols[1]:
        fig = _plot_population_x_y(df=df_x_y_session, 
                                        x_name=x_name, y_name=y_name, 
                                        group_by=group_by,
                                        smooth_factor=smooth_factor, 
                                        if_show_dots=if_show_dots,
                                        if_aggr_each_group=if_aggr_each_group,
                                        if_aggr_all=if_aggr_all,
                                        aggr_method_group=aggr_method_group,
                                        aggr_method_all=aggr_method_all,
                                        if_use_x_quantile_group=if_use_x_quantile_group,
                                        q_quantiles_group=q_quantiles_group,
                                        if_use_x_quantile_all=if_use_x_quantile_all,
                                        q_quantiles_all=q_quantiles_all,
                                        title=names[(x_name, y_name)] if (x_name, y_name) in names else y_name,
                                        states = st.session_state.df_selected_from_plotly,
                                        dot_size_base=dot_size,
                                        dot_size_mapping_name='session',
                                        dot_opacity=dot_opacity,
                                        line_width=line_width)
        
        # st.plotly_chart(fig)
        selected = plotly_events(fig, click_event=True, hover_event=False, select_event=True, 
                                 override_height=fig.layout.height * 1.1, override_width=fig.layout.width)
      
    if len(selected):
        df_selected_from_plotly = df_x_y_session.merge(pd.DataFrame(selected).rename({'x': x_name, 'y': y_name}, axis=1), 
                                                    on=[x_name, y_name], how='inner')

    return df_selected_from_plotly, cols



def show_curriculums():
    pass

# ------- Layout starts here -------- #    
def init():
    
    # Clear specific session state and all filters
    for key in st.session_state:
        if key in ['selected_draw_types'] or '_changed' in key:
            del st.session_state[key]
            
    # Set session state from URL
    for key, default in to_sync_with_url_query.items():
        sync_widget_with_query(key, default)

    df = load_data(['sessions', 
                   ])
                
    st.session_state.df = df
    st.session_state.df_selected_from_plotly = pd.DataFrame(columns=['h2o', 'session'])
    st.session_state.df_selected_from_dataframe = pd.DataFrame(columns=['h2o', 'session'])
            
    # Init auto training database
    st.session_state.curriculum_manager = CurriculumManager(
        saved_curriculums_on_s3=dict(
            bucket='aind-behavior-data',
            root='foraging_auto_training/saved_curriculums/'
        ),
        saved_curriculums_local=os.path.expanduser('~/curriculum_manager/'),
    )
    st.session_state.auto_train_manager = DynamicForagingAutoTrainManager(
        manager_name='447_demo',
        df_behavior_on_s3=dict(bucket='aind-behavior-data',
                                root='foraging_nwb_bonsai_processed/',
                                file_name='df_sessions.pkl'),
        df_manager_root_on_s3=dict(bucket='aind-behavior-data',
                                root='foraging_auto_training/')
    )
    
    logistic_regression_models = ['Su2022', 'Bari2019', 'Hattori2019', 'Miller2021']
    
    
    st.session_state.draw_type_mapper_session_level = {'1. Choice history': ('choice_history',   # prefix
                                                            (0, 0),     # location (row_idx, column_idx)
                                                            dict()),
                                                       **{f'{n + 2}. Logistic regression ({model})': (f'logistic_regression_{model}',   # prefix
                                                            (1 + int(n/2), n%2),     # location (row_idx, column_idx)
                                                            dict()) for n, model in enumerate(logistic_regression_models)},
        
                                        # '1. Choice history': ('fitted_choice',   # prefix
                                        #                     (0, 0),     # location (row_idx, column_idx)
                                        #                     dict(other_patterns=['model_best', 'model_None'])),
                                        # '2. Lick times': ('lick_psth',  
                                        #                 (1, 0), 
                                        #                 {}),            
                                        # '3. Win-stay-lose-shift prob.': ('wsls', 
                                        #                                 (1, 1), 
                                        #                                 dict(crop=(0, 0, 1200, 600))),
                                        # '4. Linear regression on RT': ('linear_regression_rt', 
                                        #                                 (1, 1), 
                                        #                                 dict()),
                                        # '5. Logistic regression on choice (Hattori)': ('logistic_regression_hattori', 
                                        #                                                 (2, 0), 
                                        #                                                 dict(crop=(0, 0, 1200, 2000))),
                                        # '6. Logistic regression on choice (Su)': ('logistic_regression_su', 
                                        #                                                 (2, 1), 
                                        #                                                 dict(crop=(0, 0, 1200, 2000))),
                    }
    
    # st.session_state.draw_type_mapper_mouse_level = {'1. Model comparison': ('model_all_sessions',   # prefix
    #                                                                          (0, 0),     # location (row_idx, column_idx)
    #                                                                          dict(other_patterns=['comparison'], 
    #                                                                               crop=(0, #900, 
    #                                                                                     100, 2800, 2200))),
    #                                                 '2. Model prediction accuracy': ('model_all_sessions',
    #                                                                                  (0, 0), 
    #                                                                                  dict(other_patterns=['pred_acc'])),            
    #                                                 '3. Model fitted parameters': ('model_all_sessions', 
    #                                                                                (0, 0), 
    #                                                                                dict(other_patterns=['fitted_para'])),
    #                 }
   
   
    # Some ad-hoc modifications on df_sessions
    st.session_state.df['sessions_bonsai'].columns = st.session_state.df['sessions_bonsai'].columns.get_level_values(1)
    st.session_state.df['sessions_bonsai'].sort_values(['session_end_time'], ascending=False, inplace=True)
    st.session_state.df['sessions_bonsai'] = st.session_state.df['sessions_bonsai'].reset_index().query('subject_id != "0"')
    st.session_state.df['sessions_bonsai']['h2o'] = st.session_state.df['sessions_bonsai']['subject_id']
    st.session_state.df['sessions_bonsai'].dropna(subset=['session'], inplace=True) # Remove rows with no session number (only leave the nwb file with the largest finished_trials for now)
    st.session_state.df['sessions_bonsai'].drop(st.session_state.df['sessions_bonsai'].query('session < 1').index, inplace=True)
    
    # # add something else
    # add abs(bais) to all terms that have 'bias' in name
    for col in st.session_state.df['sessions_bonsai'].columns:
        if 'bias' in col:
            st.session_state.df['sessions_bonsai'][f'abs({col})'] = np.abs(st.session_state.df['sessions_bonsai'][col])
        
    # # delta weight
    # diff_relative_weight_next_day = st.session_state.df['sessions_bonsai'].set_index(
    #     ['session']).sort_values('session', ascending=True).groupby('h2o').apply(
    #         lambda x: - x.relative_weight.diff(periods=-1)).rename("diff_relative_weight_next_day")
        
    # weekday
    st.session_state.df['sessions_bonsai'].session_date = pd.to_datetime(st.session_state.df['sessions_bonsai'].session_date)
    st.session_state.df['sessions_bonsai']['weekday'] = st.session_state.df['sessions_bonsai'].session_date.dt.dayofweek + 1
    
    # map user_name
    st.session_state.df['sessions_bonsai']['user_name'] = st.session_state.df['sessions_bonsai']['user_name'].apply(_user_name_mapper)
    
    # foraging performance = foraing_eff * finished_rate
    if 'foraging_performance' not in st.session_state.df['sessions_bonsai'].columns:
        st.session_state.df['sessions_bonsai']['foraging_performance'] = \
            st.session_state.df['sessions_bonsai']['foraging_eff'] \
            * st.session_state.df['sessions_bonsai']['finished_rate']
        st.session_state.df['sessions_bonsai']['foraging_performance_random_seed'] = \
            st.session_state.df['sessions_bonsai']['foraging_eff_random_seed'] \
            * st.session_state.df['sessions_bonsai']['finished_rate']

    # st.session_state.df['sessions_bonsai'] = st.session_state.df['sessions_bonsai'].merge(
    #     diff_relative_weight_next_day, how='left', on=['h2o', 'session'])

    st.session_state.session_stats_names = [keys for keys in st.session_state.df['sessions_bonsai'].keys()]
       
    # Establish communication between pygwalker and streamlit
    init_streamlit_comm()
    

def app():
    
    cols = st.columns([1, 1.2])
    with cols[0]:
        st.markdown('## 🌳🪴 Foraging sessions from Bonsai 🌳🪴')

    with st.sidebar:
        
        # === Get query from url ===
        url_query = st.query_params
        
        add_session_filter(if_bonsai=True,
                           url_query=url_query)
        data_selector()
    
        st.markdown('---')
        st.markdown('#### Han Hou @ 2024 v2.0.0')
        st.markdown('[bug report / feature request](https://github.com/AllenNeuralDynamics/foraging-behavior-browser/issues)')
        
        with st.expander('Debug', expanded=False):
            if st.button('Reload data from AWS S3'):
                st.cache_data.clear()
                init()
                st.rerun()
        
    

    with st.container():
        # col1, col2 = st.columns([1.5, 1], gap='small')
        # with col1:
        # -- 1. unit dataframe --
        
        cols = st.columns([2, 1, 4, 1])
        cols[0].markdown(f'### Filter the sessions on the sidebar\n'
                         f'#####  {len(st.session_state.df_session_filtered)} sessions, '
                         f'{len(st.session_state.df_session_filtered.h2o.unique())} mice filtered')
        with cols[1]:        
            st.markdown('# ')
            if st.button('  Reload data  ', type='primary'):
                st.cache_data.clear()
                init()
                st.rerun()  
              
        table_height = cols[3].slider('Table height', 100, 2000, 400, 50, key='table_height')
    
        # aggrid_outputs = aggrid_interactive_table_units(df=df['ephys_units'])
        # st.session_state.df_session_filtered = aggrid_outputs['data']
        
        container_filtered_frame = st.container()

        
    if len(st.session_state.df_session_filtered) == 0:
        st.markdown('## No filtered results!')
        return
    
    aggrid_outputs = aggrid_interactive_table_session(df=st.session_state.df_session_filtered, table_height=table_height)
    
    if len(aggrid_outputs['selected_rows']) and not set(pd.DataFrame(aggrid_outputs['selected_rows']
                                                                 ).set_index(['h2o', 'session']).index
                                                        ) == set(st.session_state.df_selected_from_dataframe.set_index(['h2o', 'session']).index):
        st.session_state.df_selected_from_dataframe = pd.DataFrame(aggrid_outputs['selected_rows'])
        st.session_state.df_selected_from_plotly = st.session_state.df_selected_from_dataframe  # Sync selected on plotly
        # if st.session_state.tab_id == "tab_session_x_y":
        st.rerun()

    chosen_id = stx.tab_bar(data=[
        stx.TabBarItemData(id="tab_session_x_y", title="📈 Session X-Y plot", description="Interactive session-wise scatter plot"),
        stx.TabBarItemData(id="tab_session_inspector", title="👀 Session Inspector", description="Select sessions from the table and show plots"),
        stx.TabBarItemData(id="tab_pygwalker", title="📊 PyGWalker (Tableau)", description="Interactive dataframe explorer"),
        stx.TabBarItemData(id="tab_auto_train_history", title="🎓 Automatic Training History", description="Track progress"),
        stx.TabBarItemData(id="tab_auto_train_curriculum", title="📚 Automatic Training Curriculums", description="Collection of curriculums"),
        # stx.TabBarItemData(id="tab_mouse_inspector", title="🐭 Mouse Inspector", description="Mouse-level summary"),
        ], default=st.query_params['tab_id'] if 'tab_id' in st.query_params
                   else st.session_state.tab_id)

    placeholder = st.container()
    st.session_state.tab_id = chosen_id

    if chosen_id == "tab_session_x_y":
        with placeholder:
            df_selected_from_plotly, x_y_cols = plot_x_y_session()
            
            with x_y_cols[0]:
                for i in range(7): st.write('\n')
                st.markdown("***")
                if_draw_all_sessions = session_plot_settings()

            df_to_draw_sessions = st.session_state.df_selected_from_plotly if 'selected' in st.session_state.selected_draw_sessions else st.session_state.df_session_filtered

            if if_draw_all_sessions and len(df_to_draw_sessions):
                draw_session_plots(df_to_draw_sessions)
                
            if len(df_selected_from_plotly) and not set(df_selected_from_plotly.set_index(['h2o', 'session']).index) == set(
                                                st.session_state.df_selected_from_plotly.set_index(['h2o', 'session']).index):
                st.session_state.df_selected_from_plotly = df_selected_from_plotly
                st.session_state.df_selected_from_dataframe = df_selected_from_plotly  # Sync selected on dataframe
                st.rerun()
                
    elif chosen_id == "tab_pygwalker":
        with placeholder:
            cols = st.columns([1, 4])
            cols[0].markdown('##### Exploring data using [PyGWalker](https://docs.kanaries.net/pygwalker)')
            with cols[1]:
                with st.expander('Specify PyGWalker json'):
                    # Load json from ./gw_config.json
                    pyg_user_json = st.text_area("Export your plot settings to json by clicking `export_code` "
                                                 "button below and then paste your json here to reproduce your plots", 
                                                key='pyg_walker', height=100)
            
            # If pyg_user_json is not empty, use it; otherwise, use the default gw_config.json
            if pyg_user_json:
                try:
                    pygwalker_renderer = get_pyg_renderer(
                        df=st.session_state.df_session_filtered,
                        spec=pyg_user_json,
                        )
                except:
                    pygwalker_renderer = get_pyg_renderer(
                        df=st.session_state.df_session_filtered,
                        spec="./gw_config.json",
                        )
            else:
                pygwalker_renderer = get_pyg_renderer(
                    df=st.session_state.df_session_filtered,
                    spec="./gw_config.json",
                    )
                            
            pygwalker_renderer.render_explore(height=1010, scrolling=False)
        
    elif chosen_id == "tab_session_inspector":
        with placeholder:
            cols = st.columns([6, 3, 7])
            with cols[0]:
                if_draw_all_sessions = session_plot_settings(need_click=False)
                df_to_draw_sessions = st.session_state.df_selected_from_plotly if 'selected' in st.session_state.selected_draw_sessions else st.session_state.df_session_filtered

            if if_draw_all_sessions and len(df_to_draw_sessions):
                draw_session_plots(df_to_draw_sessions)
                
    elif chosen_id == "tab_mouse_inspector":
        with placeholder:
            selected_subject_id = st.columns([1, 3])[0].selectbox('Select a mouse', options=st.session_state.df_session_filtered['subject_id'].unique())
            st.markdown(f"### [Go to WaterLog](http://eng-tools:8004/water_weight_log/?external_donor_name={selected_subject_id})")
            
    elif chosen_id == "tab_auto_train_history":  # Automatic training history
        with placeholder:
            add_auto_train_manager()

    elif chosen_id == "tab_auto_train_curriculum":  # Automatic training curriculums
        df_curriculums = st.session_state.curriculum_manager.df_curriculums().sort_values(
            by=['curriculum_schema_version', 'curriculum_name', 'curriculum_version']).reset_index().drop(columns='index')
        with placeholder:
            # Show curriculum manager dataframe
            st.markdown("#### Select auto training curriculums")

            # Curriculum drop down selector
            cols = st.columns([0.8, 0.5, 0.8, 4])
            options = list(df_curriculums['curriculum_name'].unique())
            selected_curriculum_name = cols[0].selectbox(
                'Curriculum name', 
                options=options,
                index=options.index(st.session_state['auto_training_curriculum_name'])
                    if ('auto_training_curriculum_name' in st.session_state) and (st.session_state['auto_training_curriculum_name'] != '') else 
                    options.index(st.query_params['auto_training_curriculum_name'])
                    if 'auto_training_curriculum_name' in st.query_params and st.query_params['auto_training_curriculum_name'] != ''
                    else 0, 
                key='auto_training_curriculum_name'
                )
            
            options = list(df_curriculums[
                df_curriculums['curriculum_name'] == selected_curriculum_name
                ]['curriculum_version'].unique())
            if ('auto_training_curriculum_version' in st.session_state) and (st.session_state['auto_training_curriculum_version'] in options):
                default = options.index(st.session_state['auto_training_curriculum_version'])
            elif 'auto_training_curriculum_version' in st.query_params and st.query_params['auto_training_curriculum_version'] in options:
                default = options.index(st.query_params['auto_training_curriculum_version'])
            else:
                default = 0
            selected_curriculum_version = cols[1].selectbox(
                'Curriculum version', 
                options=options, 
                index=default, 
                key='auto_training_curriculum_version'
            )
            
            options = list(df_curriculums[
                (df_curriculums['curriculum_name'] == selected_curriculum_name) 
                & (df_curriculums['curriculum_version'] == selected_curriculum_version)
                ]['curriculum_schema_version'].unique())
            if ('auto_training_curriculum_schema_version' in st.session_state) and (st.session_state['auto_training_curriculum_schema_version'] in options):
                default = options.index(st.session_state['auto_training_curriculum_schema_version'])
            elif 'auto_training_curriculum_schema_version' in st.query_params and st.query_params['auto_training_curriculum_schema_version'] in options:
                default = options.index(st.query_params['auto_training_curriculum_schema_version'])
            else:
                default = 0
            selected_curriculum_schema_version = cols[2].selectbox(
                'Curriculum schema version', 
                options=options,
                index=default,
                key='auto_training_curriculum_schema_version'
                )
                        
            selected_curriculum = st.session_state.curriculum_manager.get_curriculum(
                curriculum_name=selected_curriculum_name,
                curriculum_schema_version=selected_curriculum_schema_version,
                curriculum_version=selected_curriculum_version,
                )
            
            # Get selected curriculum from previous selected or the URL
            if 'auto_training_curriculum_name' in st.session_state:
                selected_row = {'curriculum_name': st.session_state['auto_training_curriculum_name'],
                                'curriculum_schema_version': st.session_state['auto_training_curriculum_schema_version'],
                                'curriculum_version': st.session_state['auto_training_curriculum_version']}
                matched_curriculum = df_curriculums[(df_curriculums[list(selected_row)] == pd.Series(selected_row)).all(axis=1)]
                
                if len(matched_curriculum):
                    pre_selected_rows = matched_curriculum.index.to_list() 
                else:
                    selected_row = None # Clear selected row if not found
                    pre_selected_rows = None
            
            # Show df_curriculum       
            aggrid_interactive_table_curriculum(df=df_curriculums,
                                                pre_selected_rows=pre_selected_rows)        

            
            if selected_curriculum is not None:
                curriculum = selected_curriculum['curriculum']
                # Show diagrams
                cols = st.columns([1.3, 1.5, 1])
                with cols[0]:
                    st.graphviz_chart(curriculum.diagram_rules(render_file_format=''),
                                      use_container_width=True)
                with cols[1]:
                    st.graphviz_chart(curriculum.diagram_paras(render_file_format=''),
                                    use_container_width=True)
            else:
                st.write('load curriculum failed')

    # Add debug info
    if chosen_id != "tab_auto_train_curriculum":
        with st.expander('CO processing NWB errors', expanded=False):
            error_file = cache_folder + 'error_files.json'
            if fs.exists(error_file):
                with fs.open(error_file) as file:
                    st.json(json.load(file))
            else:
                st.write('No NWB error files')
                
        with st.expander('CO Pipeline log', expanded=False):
            with fs.open(cache_folder + 'pipeline.log') as file:
                log_content = file.read().decode('utf-8')
            log_content = log_content.replace('\\n', '\n')
            st.text(log_content)
            
        with st.expander('NWB convertion and upload log', expanded=False):
            with fs.open(raw_nwb_folder + 'bonsai_pipeline.log') as file:
                log_content = file.read().decode('utf-8')
            st.text(log_content)

    
    # Update back to URL
    for key in to_sync_with_url_query:
        try:
            st.query_params.update({key: st.session_state[key]})
        except:
            print(f'Failed to update {key} to URL query')
    
    # st.dataframe(st.session_state.df_session_filtered, use_container_width=True, height=1000)


if 'df' not in st.session_state or 'sessions_bonsai' not in st.session_state.df.keys(): 
    init()
    
app()

