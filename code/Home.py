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

import os

import extra_streamlit_components as stx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_nested_layout
from aind_auto_train import __version__ as auto_train_version
from aind_auto_train.auto_train_manager import DynamicForagingAutoTrainManager
from aind_auto_train.curriculum_manager import CurriculumManager
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
from streamlit_plotly_events import plotly_events
from util.aws_s3 import (draw_session_plots_quick_preview, load_data,
                         show_debug_info,
                         show_session_level_img_by_key_and_prefix)
from util.fetch_data_docDB import load_data_from_docDB
from util.settings import (draw_type_layout_definition,
                           draw_type_mapper_session_level)
from util.streamlit import (_plot_population_x_y, add_auto_train_manager,
                            add_dot_property_mapper, add_session_filter,
                            add_xy_selector, add_xy_setting,
                            aggrid_interactive_table_curriculum,
                            aggrid_interactive_table_session, data_selector,
                            add_footnote)
from util.url_query_helper import (checkbox_wrapper_for_url_query,
                                   multiselect_wrapper_for_url_query,
                                   number_input_wrapper_for_url_query,
                                   slider_wrapper_for_url_query,
                                   sync_session_state_to_URL,
                                   sync_URL_to_session_state)

try:
    st.set_page_config(layout="wide", 
                    page_title='Foraging behavior browser',
                    page_icon=':mouse2:',
                        menu_items={
                        'Report a bug': "https://github.com/AllenNeuralDynamics/foraging-behavior-browser/issues",
                        'About': "Github repo: https://github.com/AllenNeuralDynamics/foraging-behavior-browser"
                        }
                    )
except:
    pass


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
    

@st.cache_resource(ttl=24*3600)
def get_pyg_renderer(df, spec="./gw_config.json", **kwargs) -> "StreamlitRenderer":
    return StreamlitRenderer(df, spec=spec, debug=False, **kwargs)


def draw_session_plots(df_to_draw_session):
    
    # cols_option = st.columns([3, 0.5, 1])
    container_session_all_in_one = st.container()
    
    with container_session_all_in_one:
        # with st.expander("Expand to see all-in-one plot for selected unit", expanded=True):
        
        if len(df_to_draw_session):
            st.write(f'Loading selected {len(df_to_draw_session)} sessions...')
            my_bar = st.columns((1, 7))[0].progress(0)
             
            major_cols = st.columns([1] * st.session_state['session_plot_number_cols'])
            
            for i, key in enumerate(df_to_draw_session.to_dict(orient='records')):
                this_major_col = major_cols[i % st.session_state['session_plot_number_cols']]
                
                # setting up layout for each session
                rows = []
                with this_major_col:
                    
                    try:
                        date_str = key["session_date"].strftime('%Y-%m-%d')
                    except:
                        date_str = key["session_date"].split("T")[0]
                    
                    st.markdown(f'''<h5 style='text-align: center; color: orange;'>{key["h2o"]}, Session {int(key["session"])}, {date_str} '''
                                f'''({key["user_name"]}@{key["data_source"]})''',
                                unsafe_allow_html=True)
                    if len(st.session_state.session_plot_selected_draw_types) > 1:  # more than one types, use the pre-defined layout
                        for row, column_setting in enumerate(draw_type_layout_definition):
                            rows.append(this_major_col.columns(column_setting))
                    else:    # else, put it in the whole column
                        rows = this_major_col.columns([1])
                    st.markdown("---")

                for draw_type in draw_type_mapper_session_level:
                    if draw_type not in st.session_state.session_plot_selected_draw_types: continue  # To keep the draw order defined by draw_type_mapper_session_level
                    prefix, position, setting = draw_type_mapper_session_level[draw_type]
                    this_col = rows[position[0]][position[1]] if len(st.session_state.session_plot_selected_draw_types) > 1 else rows[0]
                    show_session_level_img_by_key_and_prefix(key, 
                                                            column=this_col,
                                                            prefix=prefix,
                                                            data_source=key['hardware'],
                                                            **setting)
                    
                my_bar.progress(int((i + 1) / len(df_to_draw_session) * 100))


def session_plot_settings(need_click=True):
    with st.form(key='session_plot_settings'):
        st.markdown('##### Show plots for individual sessions ')
        cols = st.columns([2, 6, 1])
        
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
                
        _ = number_input_wrapper_for_url_query(
            st_prefix=cols[2],
            label='number of columns',
            min_value=1,
            max_value=10,
            default=3,
            key='session_plot_number_cols',
        )
        
        st.markdown(
        """
        <style>
            .stMultiSelect [data-baseweb=select] span{
                max-width: 1000px;
            }
        </style>""",
        unsafe_allow_html=True,
        )
        _ = multiselect_wrapper_for_url_query(
            cols[1],
            label='Which plot(s) to draw?',
            options=draw_type_mapper_session_level.keys(),
            default=draw_type_mapper_session_level.keys(),
            key='session_plot_selected_draw_types',
        )
                
        cols[0].markdown(f'{n_session_to_draw} sessions to draw')
        draw_it_now_override = cols[2].checkbox('Auto show', value=not need_click, disabled=not need_click)
        submitted = cols[0].form_submit_button("Update settings", type='primary')
        
        
    if not need_click:
        return True
        
    if draw_it_now_override:
        return True
    
    draw_it = st.button(f'Show {n_session_to_draw} sessions!', use_container_width=False, type="primary")
    return draw_it


def plot_x_y_session():
    with st.expander("X-Y plot settings", expanded=True):            
        with st.form(key='x_y_plot_settings', border=False):
            cols = st.columns([1, 1, 1])
            
            with cols[0]:
                x_name, y_name, group_by = add_xy_selector(if_bonsai=True)

            with cols[1]:
                (if_show_dots, if_aggr_each_group, aggr_method_group, if_use_x_quantile_group, q_quantiles_group,
                if_aggr_all, aggr_method_all, if_use_x_quantile_all, q_quantiles_all, smooth_factor, if_show_diagonal,
                dot_size, dot_opacity, line_width, x_y_plot_figure_width, x_y_plot_figure_height, 
                font_size_scale, color_map) = add_xy_setting()
            
            if st.session_state.x_y_plot_if_show_dots:
                with cols[2]:
                    size_mapper, size_mapper_range, size_mapper_gamma = add_dot_property_mapper()
            else:
                size_mapper = 'None'
                size_mapper_range, size_mapper_gamma = None, None
            
            submitted = st.form_submit_button("👉 Update X-Y settings 👈", type='primary')
    
    # If no sessions are selected, use all filtered entries
    # df_x_y_session = st.session_state.df_selected_from_dataframe if if_plot_only_selected_from_dataframe else st.session_state.df_session_filtered
    df_x_y_session = st.session_state.df_session_filtered
    
    names = {('session', 'foraging_eff'): 'Foraging efficiency',
             ('session', 'finished'):   'Finished trials', 
             }

    df_selected_from_plotly = pd.DataFrame()
    # for i, (title, (x_name, y_name)) in enumerate(names.items()):
        # with cols[i]:
    
    if hasattr(st.session_state, 'x_y_plot_figure_width'):
        _x_y_plot_scale = st.session_state.x_y_plot_figure_width / 1300
        cols = st.columns([1 * _x_y_plot_scale, 0.7])
    else:
        cols = st.columns([1, 0.7])
    with cols[0]:
        fig = _plot_population_x_y(df=df_x_y_session.copy(), 
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
                                    if_show_diagonal=if_show_diagonal,
                                    dot_size_base=dot_size,
                                    dot_size_mapping_name=size_mapper,
                                    dot_size_mapping_range=size_mapper_range,
                                    dot_size_mapping_gamma=size_mapper_gamma,
                                    dot_opacity=dot_opacity,
                                    line_width=line_width,
                                    x_y_plot_figure_width=x_y_plot_figure_width,
                                    x_y_plot_figure_height=x_y_plot_figure_height,
                                    font_size_scale=font_size_scale,
                                    color_map=color_map,
                                    )
        
        # st.plotly_chart(fig)
        selected = plotly_events(fig, click_event=True, hover_event=False, select_event=True, 
                                 override_height=fig.layout.height * 1.1, override_width=fig.layout.width)
    
    with cols[1]:
        st.markdown('#### 👀 Quick preview')
        st.markdown('###### Click on one session to preview here, or Box/Lasso select multiple sessions to draw them in the section below')
        st.markdown('(sometimes you have to click twice...)')
      
    if len(selected):
        df_selected_from_plotly = df_x_y_session.merge(pd.DataFrame(selected).rename({'x': x_name, 'y': y_name}, axis=1), 
                                                    on=[x_name, y_name], how='inner')
    if len(st.session_state.df_selected_from_plotly) == 1:
        with cols[1]:
            draw_session_plots_quick_preview(st.session_state.df_selected_from_plotly)

    return df_selected_from_plotly, cols


def show_curriculums():
    pass

# ------- Layout starts here -------- #
def init(if_load_docDB=True):
    
    # Clear specific session state and all filters
    for key in st.session_state:
        if key in ['selected_draw_types'] or '_changed' in key:
            del st.session_state[key]
            
    df = load_data(['sessions'], data_source='bonsai')
    
    if not len(df):
        return False
    
    # --- Perform any data source-dependent preprocessing here ---
    # Because sync_URL_to_session_state() needs df to be loaded (for dynamic column filtering),
    # 'if_load_bpod_sessions' has not been synced from URL to session state yet.
    # So here we need to manually get it from URL or session state.
    if (st.query_params['if_load_bpod_sessions'].lower() == 'true'
        if 'if_load_bpod_sessions' in st.query_params
        else st.session_state.if_load_bpod_sessions 
        if 'if_load_bpod_sessions' in st.session_state
        else False):
        df_bpod = load_data(['sessions'], data_source='bpod')
        
        # For historial reason, the suffix of df['sessions_bonsai'] just mean the data of the Home.py page
        df['sessions_bonsai'] = pd.concat([df['sessions_bonsai'], df_bpod['sessions_bonsai']], axis=0)
        
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
  
   
    # Some ad-hoc modifications on df_sessions
    _df = st.session_state.df['sessions_bonsai']  # temporary df alias
    
    _df.columns = _df.columns.get_level_values(1)
    _df.sort_values(['session_start_time'], ascending=False, inplace=True)
    _df['session_start_time'] = _df['session_start_time'].astype(str)  # Turn to string
    _df = _df.reset_index().query('subject_id != "0"')
 
    # Handle mouse and user name
    if 'bpod_backup_h2o' in _df.columns:
        _df['h2o'] = np.where(_df['bpod_backup_h2o'].notnull(), _df['bpod_backup_h2o'], _df['subject_id'])
        _df['user_name'] = np.where(_df['bpod_backup_user_name'].notnull(), _df['bpod_backup_user_name'], _df['user_name'])
    else:
        _df['h2o'] = _df['subject_id']
        
        
    def _get_data_source(rig):
        """From rig string, return "{institute}_{rig_type}_{room}_{hardware}"
        """
        institute = 'Janelia' if ('bpod' in rig) and not ('AIND' in rig) else 'AIND'
        hardware = 'bpod' if ('bpod' in rig) else 'bonsai'
        rig_type = 'ephys' if ('ephys' in rig.lower()) else 'training'
        
        # This is a mess...
        if institute == 'Janelia':
            room = 'NA'
        elif 'Ephys-Han' in rig:
            room = '321'
        elif hardware == 'bpod':
            room = '347'
        elif '447' in rig:
            room = '447'
        elif '446' in rig:
            room = '446'
        elif '323' in rig:
            room = '323'
        elif rig_type == 'ephys':
            room = '323'
        else:
            room = '447'
        return institute, rig_type, room, hardware, '_'.join([institute, rig_type, room, hardware])
        
    # Add data source (Room + Hardware etc)
    _df[['institute', 'rig_type', 'room', 'hardware', 'data_source']] = _df['rig'].apply(lambda x: pd.Series(_get_data_source(x)))
    
    # Handle session number
    _df.dropna(subset=['session'], inplace=True) # Remove rows with no session number (only leave the nwb file with the largest finished_trials for now)
    _df.drop(_df.query('session < 1').index, inplace=True)
    
    # Remove abnormal values
    _df.loc[_df['weight_after'] > 100, 
            ['weight_after', 'weight_after_ratio', 'water_in_session_total', 'water_after_session', 'water_day_total']
            ] = np.nan

    _df.loc[_df['water_in_session_manual'] > 100, 
            ['water_in_session_manual', 'water_in_session_total', 'water_after_session']] = np.nan

    _df.loc[(_df['duration_iti_median'] < 0) | (_df['duration_iti_mean'] < 0),
            ['duration_iti_median', 'duration_iti_mean', 'duration_iti_std', 'duration_iti_min', 'duration_iti_max']] = np.nan
    
    _df.loc[_df['invalid_lick_ratio'] < 0, 
            ['invalid_lick_ratio']]= np.nan
    
    # # add something else
    # add abs(bais) to all terms that have 'bias' in name
    for col in _df.columns:
        if 'bias' in col:
            _df[f'abs({col})'] = np.abs(_df[col])
        
    # # delta weight
    # diff_relative_weight_next_day = _df.set_index(
    #     ['session']).sort_values('session', ascending=True).groupby('h2o').apply(
    #         lambda x: - x.relative_weight.diff(periods=-1)).rename("diff_relative_weight_next_day")
        
    # weekday
    _df.session_date = pd.to_datetime(_df.session_date)
    _df['weekday'] = _df.session_date.dt.dayofweek + 1
    
    # map user_name
    _df['user_name'] = _df['user_name'].apply(_user_name_mapper)
    
    # trial stats
    _df['avg_trial_length_in_seconds'] = _df['session_run_time_in_min'] / _df['total_trials_with_autowater'] * 60
    
    # last day's total water
    _df['water_day_total_last_session'] = _df.groupby('h2o')['water_day_total'].shift(1)
    _df['water_after_session_last_session'] = _df.groupby('h2o')['water_after_session'].shift(1)    
    
    # fill nan for autotrain fields
    filled_values = {'curriculum_name': 'None', 
                     'curriculum_version': 'None',
                     'curriculum_schema_version': 'None',
                     'current_stage_actual': 'None',
                     'has_video': False,
                     'has_ephys': False,
                     'if_overriden_by_trainer': False,
                     }
    _df.fillna(filled_values, inplace=True)
        
    # foraging performance = foraing_eff * finished_rate
    if 'foraging_performance' not in _df.columns:
        _df['foraging_performance'] = \
            _df['foraging_eff'] \
            * _df['finished_rate']
        _df['foraging_performance_random_seed'] = \
            _df['foraging_eff_random_seed'] \
            * _df['finished_rate']

    # drop 'bpod_backup_' columns
    _df.drop([col for col in _df.columns if 'bpod_backup_' in col], axis=1, inplace=True)
    
    # fix if_overriden_by_trainer
    _df['if_overriden_by_trainer'] = _df['if_overriden_by_trainer'].astype(bool)
    
    # _df = _df.merge(
    #     diff_relative_weight_next_day, how='left', on=['h2o', 'session'])
    
    # Recorder columns so that autotrain info is easier to see
    first_several_cols = ['subject_id', 'session_date', 'nwb_suffix', 'session', 'rig', 
                          'user_name', 'curriculum_name', 'curriculum_version', 'current_stage_actual', 
                          'task', 'notes']
    new_order = first_several_cols + [col for col in _df.columns if col not in first_several_cols]
    _df = _df[new_order]
    

    # --- Load data from docDB ---
    if if_load_docDB:
        _df = merge_in_df_docDB(_df)
        
        # add docDB_status column
        _df["docDB_status"] = _df.apply(
            lambda row: (
                "0_not uploaded"
                if pd.isnull(row["session_loc"])
                else (
                    "1_uploaded but not processed"
                    if pd.isnull(row["processed_session_loc"])
                    else "2_uploaded and processed"
                )
            ),
            axis=1,
        )

    st.session_state.df['sessions_bonsai'] = _df  # Somehow _df loses the reference to the original dataframe
    st.session_state.session_stats_names = [keys for keys in _df.keys()]

    # Set session state from URL
    sync_URL_to_session_state()
       
    # Establish communication between pygwalker and streamlit
    init_streamlit_comm()
    
    return True

def merge_in_df_docDB(_df):
    # Fetch df_docDB
    df = load_data_from_docDB()

    # Parse session and subject_id from session_name
    df['session_date'] = pd.to_datetime(df['session_name'].str.split('_').str[2])
    # Extract the session_time. remove the '-' and remove the leading zero. 
    df['session_time'] = df['session_name'].str.split('_').str[-1]
    df['nwb_suffix'] = df['session_time'].str.replace('-', '').str.lstrip('0').astype('int64')    
    
    # Merge with _df. left merged to keep everything on han's side 

    left_merged = pd.merge(_df, df, how='left', on=['subject_id', 'session_date', 'nwb_suffix'])

    return left_merged

def app():
    
    cols = st.columns([1, 1.2])
    with cols[0]:
        st.markdown('## 🌳🪴 Dynamic Foraging Sessions 🌳🪴')

    with st.sidebar:
        
        # === Get query from url ===
        url_query = st.query_params
        
        add_session_filter(if_bonsai=True,
                           url_query=url_query)
        data_selector()
        add_footnote()
        
        with st.expander('Debug', expanded=False):
            if st.button('Clear session state and reload data'):
                st.cache_data.clear()
                init()
                st.rerun()
        
    with st.container():
        # col1, col2 = st.columns([1.5, 1], gap='small')
        # with col1:
        # -- 1. unit dataframe --
        
        cols = st.columns([2, 2, 4, 1])
        cols[0].markdown(f'### Filter the sessions on the sidebar\n'
                         f'#####  {len(st.session_state.df_session_filtered)} sessions, '
                         f'{len(st.session_state.df_session_filtered.h2o.unique())} mice filtered')
    
        if_load_bpod_sessions = checkbox_wrapper_for_url_query(
            st_prefix=cols[1],
            label='Include old Bpod sessions (reload after change)',
            key='if_load_bpod_sessions',
            default=False,
        )
                                                                   
        with cols[1]:        
            if st.button('  Reload data  ', type='primary'):
                st.cache_data.clear()
                init()
                st.rerun()  
              
        table_height = slider_wrapper_for_url_query(st_prefix=cols[3],
                                                    label='Table height',
                                                    min_value=0,
                                                    max_value=2000,
                                                    default=300,
                                                    step=50,
                                                    key='table_height',
        )
            
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
        stx.TabBarItemData(id="tab_auto_train_history", title="🎓 Automatic Training History", description="Track progress"),
        stx.TabBarItemData(id="tab_session_inspector", title="👀 Session Inspector", description="Select sessions from the table and show plots"),
        stx.TabBarItemData(id="tab_session_x_y", title="📈 Session X-Y plot", description="Interactive session-wise scatter plot"),
        stx.TabBarItemData(id="tab_pygwalker", title="📊 PyGWalker (Tableau)", description="Interactive dataframe explorer"),
        stx.TabBarItemData(id="tab_auto_train_curriculum", title="📚 Automatic Training Curriculums", description="Collection of curriculums"),
        # stx.TabBarItemData(id="tab_mouse_inspector", title="🐭 Mouse Inspector", description="Mouse-level summary"),
        ], default=st.query_params['tab_id'] if 'tab_id' in st.query_params
                   else st.session_state.tab_id)

    placeholder = st.container()
    st.session_state.tab_id = chosen_id

    if chosen_id == "tab_session_x_y":
        with placeholder:
            df_selected_from_plotly, x_y_cols = plot_x_y_session()
            
            # Add session_plot_setting
            with st.columns([1, 0.5])[0]:
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
            cols = st.columns([1, 0.5])
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
            cols[3].markdown(f"(aind_auto_train lib version = {auto_train_version})")
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
        for _ in range(10): st.write('\n')
        st.markdown('---\n##### Debug zone')
        show_debug_info()

    
    # Update back to URL
    sync_session_state_to_URL()
    
    # st.dataframe(st.session_state.df_session_filtered, use_container_width=True, height=1000)

if __name__ == "__main__":
    ok = True
    if 'df' not in st.session_state or 'sessions_bonsai' not in st.session_state.df.keys(): 
        ok = init()

    if ok:
        app()
