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

from PIL import Image, ImageColor
import streamlit.components.v1 as components
import streamlit_nested_layout
from streamlit_plotly_events import plotly_events

from util.streamlit import (filter_dataframe, aggrid_interactive_table_session, add_session_filter, data_selector, 
                            add_xy_selector, _sync_widget_with_query, add_xy_setting, add_auto_train_manager,
                            _plot_population_x_y)
import extra_streamlit_components as stx

from aind_auto_train.auto_train_manager import DynamicForagingAutoTrainManager


# Sync widgets with URL query params
# https://blog.streamlit.io/how-streamlit-uses-streamlit-sharing-contextual-apps/
# dict of "key": default pairs
# Note: When creating the widget, add argument "value"/"index" as well as "key" for all widgets you want to sync with URL
to_sync_with_url_query = {
    'filter_h2o': '',
    'filter_session': [0.0, None],
    'filter_finished_trials': [0.0, None],
    'filter_foraging_eff': [0.0, None],
    'filter_task': ['all'],
    'filter_photostim_location': ['all'],
    
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
    'x_y_plot_dot_size': 7,
    'x_y_plot_dot_opacity': 0.2,
    'x_y_plot_line_width': 2.0,
    
    'auto_training_history_x_axis': 'session',
    'auto_training_history_sort_by': 'progress_to_graduated',
    'auto_training_history_sort_order': 'descending',
    }


if_profile = False

if if_profile:
    from streamlit_profiler import Profiler
    p = Profiler()
    p.start()


# from pipeline import experiment, ephys, lab, psth_foraging, report, foraging_analysis
# from pipeline.plot import foraging_model_plot

cache_folder = 'xxx'  #'/root/capsule/data/s3/report/st_cache/'
cache_session_level_fig_folder = 'xxx' #'/root/capsule/data/s3/report/all_units/'  # 

if os.path.exists(cache_folder):
    st.session_state.st.session_state.use_s3 = False
else:
    cache_folder = 'aind-behavior-data/Han/ephys/report/st_cache/'
    cache_session_level_fig_folder = 'aind-behavior-data/Han/ephys/report/all_sessions/'
    cache_mouse_level_fig_folder = 'aind-behavior-data/Han/ephys/report/all_subjects/'
    
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

    
@st.cache_data(ttl=24*3600)
def load_data(tables=['sessions']):
    df = {}
    for table in tables:
        file_name = cache_folder + f'df_{table}.pkl'
        if st.session_state.use_s3:
            with fs.open(file_name) as f:
                df[table] = pd.read_pickle(f)
        else:
            df[table] = pd.read_pickle(file_name)
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


# @st.cache_data(ttl=24*3600, max_entries=20)
def show_session_level_img_by_key_and_prefix(key, prefix, column=None, other_patterns=[''], crop=None, caption=True, **kwargs):
    try:
        sess_date_str = datetime.strftime(datetime.strptime(key['session_date'], '%Y-%m-%dT%H:%M:%S'), '%Y%m%d')
    except:
        sess_date_str = datetime.strftime(key['session_date'], '%Y%m%d')
     
    fns = [f'/{key["h2o"]}_{sess_date_str}_*{other_pattern}*' for other_pattern in other_patterns]
    glob_patterns = [cache_session_level_fig_folder + f'{prefix}/' + key["h2o"] + fn for fn in fns]
    
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
    
    _f.image(img if img is not None else "https://cdn-icons-png.flaticon.com/512/3585/3585596.png", 
                output_format='PNG', 
                #caption=f_name.split('/')[-1] if caption and f_name else '',
                use_column_width='always',
                **kwargs)

    return img

# table_mapping = {
#     'sessions': fetch_sessions,
#     'ephys_units': fetch_ephys_units,
# }

    
def draw_session_plots(df_to_draw_session):
    
    # Setting up layout for each session
    layout_definition = [[1],   # columns in the first row
                         [1.5, 1],  # columns in the second row
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
                    
                    st.markdown(f'''<h3 style='text-align: center; color: orange;'>{key["h2o"]}, Session {key["session"]}, {date_str}''',
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

def session_plot_settings(need_click=True):
    st.markdown('##### Show plots for individual sessions ')
    cols = st.columns([2, 1])
    st.session_state.selected_draw_sessions = cols[0].selectbox('Which session(s) to draw?', 
                                                           [f'selected from table/plot ({len(st.session_state.df_selected_from_plotly)} sessions)', 
                                                            f'filtered from sidebar ({len(st.session_state.df_session_filtered)} sessions)'], 
                                                           index=0
                                                           )
    st.session_state.num_cols = cols[1].number_input('Number of columns', 1, 10, 
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
        draw_it = st.button('Show me all sessions!', use_container_width=True)
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
        x_name, y_name, group_by = add_xy_selector(if_bonsai=False)
        
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
                                        dot_size=dot_size,
                                        dot_opacity=dot_opacity,
                                        line_width=line_width)
        
        # st.plotly_chart(fig)
        selected = plotly_events(fig, click_event=True, hover_event=False, select_event=True, 
                                 override_height=fig.layout.height * 1.1, override_width=fig.layout.width)
      
    if len(selected):
        df_selected_from_plotly = df_x_y_session.merge(pd.DataFrame(selected).rename({'x': x_name, 'y': y_name}, axis=1), 
                                                    on=[x_name, y_name], how='inner')

    return df_selected_from_plotly, cols


# ------- Layout starts here -------- #    
def init():
    
    # Clear specific session state and all filters
    for key in st.session_state:
        if key in ['selected_draw_types'] or '_changed' in key:
            del st.session_state[key]

    # Set session state from URL
    for key, default in to_sync_with_url_query.items():
        _sync_widget_with_query(key, default)

    df = load_data(['sessions', 
                    'logistic_regression_hattori', 
                    'logistic_regression_su',
                    'linear_regression_rt',
                    'model_fitting_params'])
    
    # Try to convert datetimes into a standard format (datetime, no timezone)
    df['sessions']['session_date'] = pd.to_datetime(df['sessions']['session_date'])
    # if is_datetime64_any_dtype(df[col]):
    df['sessions']['session_date'] = df['sessions']['session_date'].dt.tz_localize(None)
    df['sessions']['photostim_location'].fillna('None', inplace=True)
    
    st.session_state.df = df
    st.session_state.df_selected_from_plotly = pd.DataFrame(columns=['h2o', 'session'])
    st.session_state.df_selected_from_dataframe = pd.DataFrame(columns=['h2o', 'session'])
    
    # Init auto training database
    st.session_state.auto_train_manager = DynamicForagingAutoTrainManager(
        manager_name='Janelia_demo',
        df_behavior_on_s3=dict(bucket='aind-behavior-data',
                                root='Han/ephys/report/all_sessions/export_all_nwb/',
                                file_name='df_sessions.pkl'),
        df_manager_root_on_s3=dict(bucket='aind-behavior-data',
                                root='foraging_auto_training/')
    )
    
    # Init session states
    to_init = [
               ['model_id', 21],   # add some model fitting params to session
               ]
    
    for name, default in to_init:
        if name not in st.session_state:
            st.session_state[name] = default
        
    selected_id = st.session_state.model_id 
    
    st.session_state.draw_type_mapper_session_level = {'1. Choice history': ('fitted_choice',   # prefix
                                                            (0, 0),     # location (row_idx, column_idx)
                                                            dict(other_patterns=['model_best', 'model_None'])),
                                        '2. Lick times': ('lick_psth', 
                                                        (1, 0), 
                                                        {}),            
                                        '3. Win-stay-lose-shift prob.': ('wsls', 
                                                                        (1, 1), 
                                                                        dict(crop=(0, 0, 1200, 600))),
                                        '4. Linear regression on RT': ('linear_regression_rt', 
                                                                        (1, 1), 
                                                                        dict()),
                                        '5. Logistic regression on choice (Hattori)': ('logistic_regression_hattori', 
                                                                                        (2, 0), 
                                                                                        dict(crop=(0, 0, 1200, 2000))),
                                        '6. Logistic regression on choice (Su)': ('logistic_regression_su', 
                                                                                        (2, 1), 
                                                                                        dict(crop=(0, 0, 1200, 2000))),
                    }
    
    st.session_state.draw_type_mapper_mouse_level = {'1. Model comparison': ('model_all_sessions',   # prefix
                                                                             (0, 0),     # location (row_idx, column_idx)
                                                                             dict(other_patterns=['comparison'], 
                                                                                  crop=(0, #900, 
                                                                                        100, 2800, 2200))),
                                                    '2. Model prediction accuracy': ('model_all_sessions',
                                                                                     (0, 0), 
                                                                                     dict(other_patterns=['pred_acc'])),            
                                                    '3. Model fitted parameters': ('model_all_sessions', 
                                                                                   (0, 0), 
                                                                                   dict(other_patterns=['fitted_para'])),
                    }
   
    
    # process dfs
    df_this_model = st.session_state.df['model_fitting_params'].query(f'model_id == {selected_id}')
    valid_field = df_this_model.columns[~np.all(~df_this_model.notna(), axis=0)]
    to_add_model = st.session_state.df['model_fitting_params'].query(f'model_id == {selected_id}')[valid_field]
    
    st.session_state.df['sessions'] = st.session_state.df['sessions'].merge(to_add_model, on=('subject_id', 'session'), how='left')

    # add something else
    st.session_state.df['sessions']['abs(bias)'] = np.abs(st.session_state.df['sessions'].biasL)
    
    # delta weight
    diff_relative_weight_next_day = st.session_state.df['sessions'].set_index(
        ['session']).sort_values('session', ascending=True).groupby('h2o').apply(
            lambda x: - x.relative_weight.diff(periods=-1)).rename("diff_relative_weight_next_day")
        
    # weekday
    st.session_state.df['sessions']['weekday'] =  st.session_state.df['sessions'].session_date.dt.dayofweek + 1

    st.session_state.df['sessions'] = st.session_state.df['sessions'].merge(
        diff_relative_weight_next_day, how='left', on=['h2o', 'session'])

    st.session_state.session_stats_names = [keys for keys in st.session_state.df['sessions'].keys()]
   
   
    

def app():
    st.markdown('## Foraging Behavior Browser')
    
    with st.sidebar:
        add_session_filter()
        data_selector()
    
        st.markdown('---')
        st.markdown('#### Han Hou @ 2024 v2.0.0')
        st.markdown('[bug report / feature request](https://github.com/AllenNeuralDynamics/foraging-behavior-browser/issues)')
        
        with st.expander('Debug', expanded=False):
            st.session_state.model_id = st.selectbox('model_id', st.session_state.df['model_fitting_params'].model_id.unique())
            if st.button('Reload data from AWS S3'):
                st.cache_data.clear()
                init()
                st.rerun()
        
    

    with st.container():
        # col1, col2 = st.columns([1.5, 1], gap='small')
        # with col1:
        # -- 1. unit dataframe --
        
        cols = st.columns([2, 2, 2])
        cols[0].markdown(f'### Filter the sessions on the sidebar ({len(st.session_state.df_session_filtered)} filtered)')
        # if cols[1].button('Press this and then Ctrl + R to reload from S3'):
        #     st.rerun()
        if cols[1].button('Reload data '):
            st.cache_data.clear()
            init()
            st.rerun()
    
        # aggrid_outputs = aggrid_interactive_table_units(df=df['ephys_units'])
        # st.session_state.df_session_filtered = aggrid_outputs['data']
        
        container_filtered_frame = st.container()

        
    if len(st.session_state.df_session_filtered) == 0:
        st.markdown('## No filtered results!')
        return
    
    aggrid_outputs = aggrid_interactive_table_session(df=st.session_state.df_session_filtered)
    
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
        stx.TabBarItemData(id="tab_auto_train_history", title="🎓 Automatic Training History", description="Track progress"),
        stx.TabBarItemData(id="tab_mouse_inspector", title="🐭 Mouse Model Fitting", description="Mouse-level model fitting results"),
        ], default="tab_session_inspector" if 'tab_id' not in st.session_state else st.session_state.tab_id)
    # chosen_id = "tab_session_x_y"

    placeholder = st.container()

    if chosen_id == "tab_session_x_y":
        st.session_state.tab_id = chosen_id
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
            
    elif chosen_id == "tab_session_inspector":
        st.session_state.tab_id = chosen_id
        with placeholder:
            with st.columns([4, 10])[0]:
                if_draw_all_sessions = session_plot_settings(need_click=False)
                df_to_draw_sessions = st.session_state.df_selected_from_plotly if 'selected' in st.session_state.selected_draw_sessions else st.session_state.df_session_filtered
                
            if if_draw_all_sessions and len(df_to_draw_sessions):
                draw_session_plots(df_to_draw_sessions)
                
    elif chosen_id == "tab_auto_train_history":  # Automatic training history
        st.session_state.tab_id = chosen_id
        with placeholder:
            add_auto_train_manager()
                
    elif chosen_id == "tab_mouse_inspector":
        st.session_state.tab_id = chosen_id
        with placeholder:
            with st.columns([4, 10])[0]:
                if_draw_all_mice = mouse_plot_settings(need_click=False)
                df_selected = st.session_state.df_selected_from_plotly if 'selected' in st.session_state.selected_draw_mice else st.session_state.df_session_filtered
                df_to_draw_mice = df_selected.groupby('h2o').count().reset_index()
                
            if if_draw_all_mice and len(df_to_draw_mice):
                draw_mice_plots(df_to_draw_mice)
        
    

    # st.dataframe(st.session_state.df_session_filtered, use_container_width=True, height=1000)

    # Update back to URL
    for key in to_sync_with_url_query:
        try:
            st.query_params.update({key: st.session_state[key]})
        except:
            print(f'Failed to update {key} to URL query')


if 'df' not in st.session_state or 'sessions' not in st.session_state.df.keys(): 
    init()
    
app()

            
if if_profile:    
    p.stop()