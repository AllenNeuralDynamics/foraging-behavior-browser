import streamlit as st
from pandas.api.types import (is_categorical_dtype, is_datetime64_any_dtype,
                              is_numeric_dtype)

from .settings import draw_type_mapper_session_level

# Sync widgets with URL query params
# https://blog.streamlit.io/how-streamlit-uses-streamlit-sharing-contextual-apps/
# dict of "key": default pairs
# Note: When creating the widget, add argument "value"/"index" as well as "key" for all widgets you want to sync with URL
to_sync_with_url_query_default = {
    'if_load_bpod_sessions': False,
    
    'to_filter_columns': ['subject_id', 'task', 'session', 'finished_trials', 'foraging_eff'],
    'filter_subject_id': '',
    'filter_session': [0.0, None],
    'filter_finished_trials': [0.0, None],
    'filter_foraging_eff': [0.0, None],
    'filter_task': ['all'],
    
    'table_height': 300,
    
    'tab_id': 'tab_auto_train_history',
    'x_y_plot_xname': 'session',
    'x_y_plot_yname': 'foraging_performance_random_seed',
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
    'x_y_plot_if_show_diagonal': False,
    'x_y_plot_dot_size': 10,
    'x_y_plot_dot_opacity': 0.3,
    'x_y_plot_line_width': 2.0,
    'x_y_plot_figure_width': 1300,
    'x_y_plot_figure_height': 900,
    'x_y_plot_font_size_scale': 1.0,
    'x_y_plot_selected_color_map': 'Plotly',
    
    'x_y_plot_size_mapper': 'finished_trials',
    'x_y_plot_size_mapper_gamma': 1.0,
    'x_y_plot_size_mapper_range': [3, 20],
    
    'session_plot_mode': 'sessions selected from table or plot',
    'session_plot_selected_draw_types': list(draw_type_mapper_session_level.keys()),
    'session_plot_number_cols': 3,

    'auto_training_history_x_axis': 'date',
    'auto_training_history_sort_by': 'first_date',
    'auto_training_history_sort_order': 'descending',
    'auto_training_curriculum_name': 'Uncoupled Baiting',
    'auto_training_curriculum_version': '1.0',
    'auto_training_curriculum_schema_version': '1.0',
    'auto_training_history_recent_weeks': 8,
    
    'tab_id_learning_trajectory': 'tab_PCA',
    }

def checkbox_wrapper_for_url_query(st_prefix, label, key, default, **kwargs):
    return st_prefix.checkbox(
        label,
        value=st.session_state[key]
                if key in st.session_state else 
                st.query_params[key].lower()=='true' 
                if key in st.query_params 
                else default,
        key=key,
        **kwargs,
    )

def selectbox_wrapper_for_url_query(st_prefix, label, options, key, default, **kwargs):
    return st_prefix.selectbox(
        label,
        options=options,
        index=(
            options.index(st.session_state[key])
            if key in st.session_state
            else options.index(st.query_params[key]) 
            if key in st.query_params 
            else options.index(default)
        ),
        key=key,
        **kwargs,
    )
    
def multiselect_wrapper_for_url_query(st_prefix, label, options, key, default, **kwargs):
    return st_prefix.multiselect(
        label,
        options=options,
        default=(
            st.session_state[key]
            if key in st.session_state
            else st.query_params[key]
            if key in st.query_params 
            else default
        ),
        key=key,
        **kwargs,
    )

def slider_wrapper_for_url_query(st_prefix, label, min_value, max_value, key, default, **kwargs):
    
    # Parse range from URL, compatible with both one or two values
    if key in st.query_params:
        parse_range_from_url = [type(min_value)(v) for v in st.query_params.get_all(key)]
        if len(parse_range_from_url) == 1:
            parse_range_from_url = parse_range_from_url[0]
    
    return st_prefix.slider(
        label,
        min_value,
        max_value,
        value=(
            st.session_state[key]
            if key in st.session_state
            else parse_range_from_url
            if key in st.query_params 
            else default
        ),
        key=key,
        **kwargs,
    )
    
    
def number_input_wrapper_for_url_query(st_prefix, label, min_value, max_value, key, default, **kwargs):
    return st_prefix.number_input(
        label=label,
        min_value=min_value,
        max_value=max_value,
        value=(
            st.session_state[key]
            if key in st.session_state
            else type(min_value)(st.query_params[key])
            if key in st.query_params 
            else default
        ),
        key=key,
        **kwargs,
    )
    
    
def sync_URL_to_session_state():
    """Assign session_state to sync with URL"""
    
    to_sync_with_session_state_dynamic_filter_added = list(
        set(
            list(st.query_params.keys())
            + list(to_sync_with_url_query_default.keys())
        )
    )
    
    for key in to_sync_with_session_state_dynamic_filter_added:
        
        if key in to_sync_with_url_query_default:
            # If in default list, get default value there
            default = to_sync_with_url_query_default[key]
        else:
            # Else, the user should have added a new filter column
            # let's get the type from the dataframe directly.
            # 
            # Also, in this case, the key must be from st.query_params, 
            # so the only purpose of getting default is to get the correct type,
            # not its value per se.
            filter_type = get_filter_type(st.session_state.df['sessions_bonsai'],
                                          key.replace('filter_', ''))
            if filter_type == 'slider_range_float':
                default = [0.0, 1.0]
            elif filter_type == 'reg_ex':
                default = ''
            elif filter_type == 'multiselect':
                default = ['a', 'b']
            else:
                print('sync_URL_to_session_state: Unrecognized filter type')
                continue
            
        if key in st.query_params:
            # always get all query params as a list
            q_all = st.query_params.get_all(key)
            
            # convert type according to default
            list_default = default if isinstance(default, list) else [default]
            for d in list_default:
                _type = type(d)
                if _type: break  # The first non-None type
                
            if _type == bool:
                q_all_correct_type = [q.lower() == 'true' for q in q_all]
            else:
                q_all_correct_type = [_type(q) 
                                    if q.lower() != 'none'
                                    else None
                                    for q in q_all]
            
            # flatten list if only one element
            if not isinstance(default, list):
                q_all_correct_type = q_all_correct_type[0]
            
            try:
                st.session_state[key] = q_all_correct_type
                # print(f'sync_URL_to_session_state: Set {key} to {q_all_correct_type}')
            except:
                print(f'sync_URL_to_session_state: Failed to set {key} to {q_all_correct_type}')
        else:
            try:
                st.session_state[key] = default
            except:
                print(f'Failed to set {key} to {default}')
                        

def sync_session_state_to_URL():
    # Add all 'filter_' fields to the default list 
    # so that all dynamic filters are synced with URL
    to_sync_with_url_query_dynamic_filter_added = list(
        set(
            list(to_sync_with_url_query_default.keys()) + 
                [
                    filter_name for filter_name in st.session_state 
                    if (
                        filter_name.startswith('filter_') 
                        and not (filter_name.endswith('_changed'))
                    )
                ]
                )
        )
    for key in to_sync_with_url_query_dynamic_filter_added:
        try:
            st.query_params.update({key: st.session_state[key]})
        except:
            print(f'Failed to update {key} to URL query')
            
            
def get_filter_type(df, column):
    if is_numeric_dtype(df[column]):
        return 'slider_range_float'
    
    if (is_categorical_dtype(df[column]) 
        or df[column].nunique() < 10
        or column in ('user_name') # pin to multiselect
        ):
        return 'multiselect'

    if is_datetime64_any_dtype(df[column]):
        return 'slider_range_date'
    
    return 'reg_ex'  # Default
