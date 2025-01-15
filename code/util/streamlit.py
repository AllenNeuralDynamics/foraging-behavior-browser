import json
from collections import OrderedDict
from datetime import datetime

from __init__ import __ver__

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.json.config.default_engine = "orjson"
import statsmodels.api as sm
import streamlit as st
import streamlit.components.v1 as components
from streamlit_bokeh3_events import streamlit_bokeh3_events
from pandas.api.types import (is_categorical_dtype, is_numeric_dtype,
                              is_string_dtype)
from scipy.stats import linregress
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import (ColumnsAutoSizeMode, DataReturnMode,
                              GridUpdateMode)
from streamlit_plotly_events import plotly_events

from .aws_s3 import draw_session_plots_quick_preview
from .plot_autotrain_manager import plot_manager_all_progress_bokeh
from .url_query_helper import (checkbox_wrapper_for_url_query, get_filter_type,
                               multiselect_wrapper_for_url_query,
                               selectbox_wrapper_for_url_query,
                               slider_wrapper_for_url_query)
from.settings import override_plotly_theme

custom_css = {
".ag-root.ag-unselectable.ag-layout-normal": {"font-size": "15px !important",
"font-family": "Roboto, sans-serif !important;"},
".ag-header-cell-text": {"color": "#495057 !important;"},
".ag-theme-alpine .ag-ltr .ag-cell": {"color": "#444 !important;"},
".ag-theme-alpine .ag-row-odd": {"background": "rgba(243, 247, 249, 0.3) !important;",
"border": "1px solid #eee !important;"},
".ag-theme-alpine .ag-row-even": {"border-bottom": "1px solid #eee !important;"},
".ag-theme-light button": {"font-size": "0 !important;", "width": "auto !important;", "height": "24px !important;",
"border": "1px solid #eee !important;", "margin": "4px 2px !important;",
"background": "#3162bd !important;", "color": "#fff !important;",
"border-radius": "3px !important;"},
".ag-theme-light button:before": {"content": "'Confirm' !important", "position": "relative !important",
"z-index": "1000 !important", "top": "0 !important",
"font-size": "10px !important", "left": "0 !important",
"padding": "4px !important"},
}

def aggrid_interactive_table_session(df: pd.DataFrame, table_height: int = 400):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True,
    )

    options.configure_side_bar()
    
    if 'session_end_time' in df.columns:
        df = df.sort_values('session_start_time', ascending=False)
    else:
        df = df.sort_values('session_date', ascending=False)
    
    # preselect
    if len(st.session_state.get("df_selected_from_dataframe", [])) \
       and ('tab_id' in st.session_state) and (st.session_state.tab_id == "tab_session_x_y"):
        try:
            indexer = st.session_state.df_selected_from_dataframe.set_index(['h2o', 'session']
                                                                ).index.get_indexer(df.set_index(['h2o', 'session']).index)
            pre_selected_rows = np.where(indexer != -1)[0].tolist()
        except:
            pre_selected_rows = []
    else:
        pre_selected_rows = []
        
    options.configure_selection(selection_mode="multiple",
                                pre_selected_rows=pre_selected_rows)  # , use_checkbox=True, header_checkbox=True)
    
    # options.configure_column(field="session_date", sort="desc")
    
    # options.configure_column(field="h2o", hide=True, rowGroup=True)
    # options.configure_column(field='subject_id', hide=True)
    options.configure_column(field="session_date", type=["customDateTimeFormat"], custom_format_string='yyyy-MM-dd')
    options.configure_column(field="ephys_ins", dateType="DateType")
    
    # options.configure_column(field="water_restriction_number", header_name="subject", 
    #                          children=[dict(field="water_restriction_number", rowGroup=True),
    #                                    dict(field="session")])
    
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="balham",
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        height=table_height,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        custom_css=custom_css,
    )
    
    return selection

def aggrid_interactive_table_basic(df: pd.DataFrame,
                                   height: int = 200,
                                   pre_selected_rows: list = None,
                                   configure_columns: list = None,):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True,
    )

    options.configure_side_bar()
    options.configure_selection(selection_mode=None,
                                pre_selected_rows=pre_selected_rows)

    if configure_columns:
        for col in configure_columns:
            options.configure_column(**col)

    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="balham",
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
        height=height,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        custom_css=custom_css,
    )
    return selection

def aggrid_interactive_table_units(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True,
    )

    options.configure_selection(selection_mode="multiple", use_checkbox=False, header_checkbox=True)
    options.configure_side_bar()
    options.configure_selection("single")
    options.configure_columns(column_names=['subject_id', 'electrode'], hide=True)
 

    # options.configure_column(field="water_restriction_number", header_name="subject", 
    #                          children=[dict(field="water_restriction_number", rowGroup=True),
    #                                    dict(field="session")])
    
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        theme="balham",
        update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.FILTERING_CHANGED,
        allow_unsafe_jscode=True,
        height=500,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        custom_css=custom_css,
    )

    return selection

def cache_widget(field, clear=None):
    st.session_state[f'{field}_changed'] = st.session_state[field]
    
    # Clear cache if needed
    if clear:
        if clear in st.session_state: del st.session_state[clear]

# def dec_cache_widget_state(widget, ):


def filter_dataframe(df: pd.DataFrame, 
                     default_filters=['h2o', 'task', 'finished_trials', 'photostim_location'],
                     url_query={}) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe
        url_query: a dictionary of filters from the url query to apply to the dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    
    # modify = st.checkbox("Add filters")

    # if not modify:
    #     return df

    df = df.copy()


    modification_container = st.container()
    
    with modification_container:
        cols = st.columns([1, 1.5])
        cols[0].markdown(f"Add filters")
        if_reset_filters = cols[1].button(label="Reset filters")
        
        to_filter_columns = multiselect_wrapper_for_url_query(
            st_prefix=st,
            label="Filter dataframe on",
            options=df.columns,
            default=['subject_id', 'session', 'finished_trials', 'foraging_eff', 'task'],
            key='to_filter_columns',
            label_visibility='collapsed',
        )

        for column in to_filter_columns:
            if not len(df): break
            
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            
            filter_type = get_filter_type(df, column)
            
            if filter_type == 'multiselect':
                right.markdown(f"Filter for :red[**{column}**]")
                
                if if_reset_filters:
                    default_value = list(df[column].unique())
                    st.session_state[f'filter_{column}_changed'] = default_value
                elif f'filter_{column}_changed' in st.session_state:
                    default_value = st.session_state[f'filter_{column}_changed']
                elif f'filter_{column}' in st.session_state:  # Set by URL or default
                    if st.session_state[f'filter_{column}'] == ['all']:
                        default_value = list(df[column].unique())
                    else:
                        default_value = [i for i in st.session_state[f'filter_{column}'] if i in list(df[column].unique())]
                else:
                    default_value = list(df[column].unique())  
                
                default_value = [v for v in default_value if v in list(df[column].unique())]
                st.session_state[f'filter_{column}'] = default_value           
                
                selected = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    label_visibility='collapsed',
                    default=default_value,
                    key=f'filter_{column}',
                    on_change=cache_widget,
                    args=[f'filter_{column}']
                )
                df = df[df[column].isin(selected)]
                
            elif filter_type == 'slider_range_float':
                
                # fig = px.histogram(df[column], nbins=100, )
                # fig.update_layout(showlegend=False, height=50)
                # st.plotly_chart(fig)
                # counts, bins = np.histogram(df[column], bins=100)
                # st.bar_chart(pd.DataFrame(
                #                 {'x': bins[1:], 'y': counts},
                #                 ),
                #                 x='x', y='y')
                
                with right:
                    col1, col2 = st.columns((2, 1))
                    col1.markdown(f"Filter for :red[**{column}**]")
                    if float(df[column].min()) >= 0: 
                        show_log = col2.checkbox('log 10', 
                                                 value=st.session_state[f'if_log_{column}_changed']
                                                       if f'if_log_{column}_changed' in st.session_state
                                                       else False,
                                                 key=f'if_log_{column}',
                                                 on_change=cache_widget,
                                                 args=[f'if_log_{column}'],
                                                 kwargs={'clear': f'filter_{column}_changed'}  # If show_log is changed, clear select cache
                                                 )
                    else:
                        show_log = 0
                        
                    if show_log:
                        x = np.log10(df[column] + 1e-6)  # Cutoff at 1e-5
                    else:
                        x = df[column]               
                        
                    _min = float(x.min() - 0.05 * abs(x.min()))  # Avoid collapsing the range
                    _max = float(x.max() + 0.05 * abs(x.max()))
                    step = (_max - _min) / min(100, len(np.unique(x)) + 1) # Avoid too many steps

                    c_hist = st.container()  # Histogram
                    
                    if if_reset_filters:
                        default_value = (_min, _max)
                        st.session_state[f'filter_{column}_changed'] = default_value
                    elif f'filter_{column}_changed' in st.session_state:
                        default_value = st.session_state[f'filter_{column}_changed']
                    elif f'filter_{column}' in st.session_state and st.session_state[f'filter_{column}'] != []:
                        # If session_state was preset by a query, use that
                        # Handle None value 
                        default_value = [a or b for a, b in 
                                        zip(st.session_state[f'filter_{column}'], (_min, _max))]
                    else:
                        default_value = (_min, _max)
                    
                    default_value = list(default_value)
                    default_value[0] = max(_min, default_value[0])
                    default_value[1] = min(_max, default_value[1])
                    st.session_state[f'filter_{column}'] = default_value

                    user_num_input = st.slider(
                        f"Values for {column}",
                        label_visibility='collapsed',
                        min_value=_min,
                        max_value=_max,
                        value=default_value,
                        step=step,
                        key=f'filter_{column}',
                        on_change=cache_widget,
                        args=[f'filter_{column}']
                    )
                    
                    with c_hist:
                        
                        counts, bins = np.histogram(x[~np.isnan(x)], bins=100)
                        
                        fig = px.bar(x=bins[1:], y=counts)
                        fig.add_vrect(x0=user_num_input[0], x1=user_num_input[1], fillcolor='red', opacity=0.1, line_width=0)
                        fig.update_layout(showlegend=False, height=100, 
                                          yaxis=dict(visible=False),
                                          xaxis=dict(title=f'log 10 ({column})' if show_log else column,
                                                     range=(_min, _max)),
                                          margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig, use_container_width=True)

                    df = df[x.between(*user_num_input)]
                
            elif filter_type == 'slider_range_date':
                user_date_input = right.date_input(
                    f"Values for :red[**{column}**]",
                    value=st.session_state[f'filter_{column}_changed']
                                if f'filter_{column}_changed' in st.session_state
                                else (df[column].min(), df[column].max()),
                    key=f'filter_{column}',
                    on_change=cache_widget,
                    args=[f'filter_{column}']
                )
                
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
                    
            elif filter_type == 'reg_ex':
                if if_reset_filters:
                    default_value = ''
                    st.session_state[f'filter_{column}_changed'] = default_value
                elif f'filter_{column}_changed' in st.session_state:
                    default_value = st.session_state[f'filter_{column}_changed']
                elif f'filter_{column}' in st.session_state:
                    default_value = st.session_state[f'filter_{column}']
                    if column == 'subject_id' and default_value != '':
                        try:
                            if float(default_value) == 0:
                                default_value = ''
                        except:
                            pass
                else:
                    default_value = ''
                
                st.session_state[f'filter_{column}'] = default_value
                
                user_text_input = right.text_input(
                    f"Substring or regex in :red[**{column}**]",
                    value=default_value,
                    key=f'filter_{column}',
                    on_change=cache_widget,
                    args=[f'filter_{column}']
                    )
                
                if user_text_input:
                    try:
                        df = df[df[column].astype(str).str.contains(user_text_input, regex=True)]
                    except:
                        st.warning('Wrong regular expression!')

    return df

def add_session_filter(if_bonsai=False, url_query={}):
    with st.expander("Behavioral session filter", expanded=True):   
        if not if_bonsai:
            st.session_state.df_session_filtered = filter_dataframe(df=st.session_state.df['sessions'],
                                                                    default_filters=['h2o', 'task', 'session', 'finished_trials', 'foraging_eff', 'photostim_location'],
                                                                    url_query=url_query)
        else:
            st.session_state.df_session_filtered = filter_dataframe(
                df=st.session_state.df["sessions_bonsai"],
                default_filters=[
                    "subject_id",
                    "task",
                    "session",
                    "finished_trials",
                    "foraging_eff",
                ],
                url_query=url_query,
            )


@st.cache_data(ttl=3600 * 24)
def _get_grouped_by_fields(if_bonsai):
    if if_bonsai:
        options = ["h2o", "task", "user_name", "rig", "data_source", "weekday"]

        for col in st.session_state.df_session_filtered.columns:
            if any(
                [
                    exclude in col
                    for exclude in (
                        "date",
                        "time",
                        "session",
                        "finished",
                        "foraging_eff",
                    )
                ]
            ):
                continue
            this_col = st.session_state.df_session_filtered[col]
            if is_categorical_dtype(this_col):
                options += [col]
                continue
            try:
                if len(this_col.unique()) < 30:
                    options += [col]
                    continue
            except:
                print(f"column {col} is unhashable")

        options = list(list(OrderedDict.fromkeys(options)))  # Remove duplicates
    else:
        options = [
            "h2o",
            "task",
            "photostim_location",
            "weekday",
            "headbar",
            "user_name",
            "sex",
            "rig",
        ]
    return options


def add_xy_selector(if_bonsai):
    with st.expander("Select axes", expanded=True):
        # with st.form("axis_selection"):
        cols = st.columns([1])
        x_name = cols[0].selectbox("x axis", 
                                   st.session_state.session_stats_names, 
                                   index=st.session_state.session_stats_names.index(st.session_state['x_y_plot_xname'])
                                         if 'x_y_plot_xname' in st.session_state else 
                                         st.session_state.session_stats_names.index(st.query_params['x_y_plot_xname'])
                                         if 'x_y_plot_xname' in st.query_params 
                                         else st.session_state.session_stats_names.index('session'), 
                                   key='x_y_plot_xname'
                                   )
        y_name = selectbox_wrapper_for_url_query(
            cols[0],
            label="y axis",
            options=st.session_state.session_stats_names,
            key="x_y_plot_yname",
            default=st.session_state.session_stats_names.index('foraging_eff')
        )


        group_by = selectbox_wrapper_for_url_query(
            cols[0],
            label="grouped by",
            options=_get_grouped_by_fields(if_bonsai),
            key="x_y_plot_group_by",
            default=0,
        )

        # st.form_submit_button("update axes")
    return x_name, y_name, group_by

def add_xy_setting():
    with st.expander('Plot settings', expanded=True):            
        s_cols = st.columns([1, 1, 1])
        # if_plot_only_selected_from_dataframe = s_cols[0].checkbox('Only selected', False)
        if_show_dots = checkbox_wrapper_for_url_query(s_cols[0], 
                                                      label='Show data points', 
                                                      key='x_y_plot_if_show_dots', 
                                                      default=True)        

        if_aggr_each_group = checkbox_wrapper_for_url_query(s_cols[1],
                                                            label='Aggr each group', 
                                                            key='x_y_plot_if_aggr_each_group', 
                                                            default=True)
        
        aggr_methods =  ['mean', 'mean +/- sem', 'lowess', 'running average', 'linear fit']
        aggr_method_group = selectbox_wrapper_for_url_query(s_cols[1],
                                                            label='aggr method group', 
                                                            options=aggr_methods, 
                                                            key='x_y_plot_aggr_method_group', 
                                                            default=2,
                                                            disabled=not if_aggr_each_group)
        
        if_use_x_quantile_group = checkbox_wrapper_for_url_query(s_cols[1],
                                                                  label='Use quantiles of x', 
                                                                  key='x_y_plot_if_use_x_quantile_group', 
                                                                  default=False,
                                                                  disabled='mean' not in aggr_method_group)
            
        q_quantiles_group = slider_wrapper_for_url_query(s_cols[1],
                                                        label='Number of quantiles ', 
                                                        min_value=1, 
                                                        max_value=100, 
                                                        key='x_y_plot_q_quantiles_group', 
                                                        default=20,
                                                        disabled=not if_use_x_quantile_group
                                                        )
        
        if_aggr_all = checkbox_wrapper_for_url_query(s_cols[2],
                                                    label='Aggr all', 
                                                    key='x_y_plot_if_aggr_all', 
                                                    default=True)
        
        # st.session_state.if_aggr_all_cache = if_aggr_all  # Have to use another variable to store this explicitly (my cache_widget somehow doesn't work with checkbox)
        aggr_method_all = selectbox_wrapper_for_url_query(s_cols[2],
                                                            label='aggr method all', 
                                                            options=aggr_methods, 
                                                            key='x_y_plot_aggr_method_all', 
                                                            default=1,
                                                            disabled=not if_aggr_all)

        if_use_x_quantile_all = checkbox_wrapper_for_url_query(s_cols[2],
                                                                label='Use quantiles of x', 
                                                                key='x_y_plot_if_use_x_quantile_all', 
                                                                default=False,
                                                                disabled='mean' not in aggr_method_all)

        
        q_quantiles_all = slider_wrapper_for_url_query(s_cols[2],
                                                        label='Number of quantiles', 
                                                        min_value=1, 
                                                        max_value=100, 
                                                        key='x_y_plot_q_quantiles_all', 
                                                        default=20,
                                                        disabled=not if_use_x_quantile_all
                                                        )

        smooth_factor = slider_wrapper_for_url_query(s_cols[0],
                                                    label='smooth factor',
                                                    min_value=1,
                                                    max_value=20,
                                                    key='x_y_plot_smooth_factor',
                                                    default=5,
                                                    disabled= not ((if_aggr_each_group and aggr_method_group in ('running average', 'lowess'))
                                                                    or (if_aggr_all and aggr_method_all in ('running average', 'lowess')))
                                                    )
        
        if_show_diagonal = checkbox_wrapper_for_url_query(s_cols[0],
                                                       label='Show diagonal line', 
                                                       key='x_y_plot_if_show_diagonal', 
                                                       default=False)
        
        with st.expander('Misc', expanded=False):
            c = st.columns([1, 1, 1])
            dot_size = slider_wrapper_for_url_query(c[0],
                                                    label='dot size', 
                                                    min_value=1, 
                                                    max_value=30, 
                                                    key='x_y_plot_dot_size', 
                                                    default=10)
            
            dot_opacity = slider_wrapper_for_url_query(c[1],
                                                        label='opacity',
                                                        min_value=0.0,
                                                        max_value=1.0,
                                                        step=0.05,
                                                        key='x_y_plot_dot_opacity',
                                                        default=0.5)

            line_width = slider_wrapper_for_url_query(c[2],
                                                        label='line width',
                                                        min_value=0.0,
                                                        max_value=5.0,
                                                        step=0.25,
                                                        key='x_y_plot_line_width',
                                                        default=2.0)
            
            figure_width = slider_wrapper_for_url_query(c[0],
                                                        label='figure width',
                                                        min_value=500,
                                                        max_value=2500,
                                                        key='x_y_plot_figure_width',
                                                        default=1300)
            
            figure_height = slider_wrapper_for_url_query(c[1],
                                                        label='figure height',
                                                        min_value=500,
                                                        max_value=2500,
                                                        key='x_y_plot_figure_height',
                                                        default=900)
            
            font_size_scale = slider_wrapper_for_url_query(c[2],
                                                    label='font size',
                                                    min_value=0.0,
                                                    max_value=2.0,
                                                    step=0.1,
                                                    key='x_y_plot_font_size_scale',
                                                    default=1.0)
            
            available_color_maps = list(px.colors.qualitative.__dict__.keys())
            available_color_maps = [c for c in available_color_maps if not c.startswith("_") and c != 'swatches']
            color_map = selectbox_wrapper_for_url_query(c[0],
                                                        label='Color map',
                                                        options=available_color_maps,
                                                        key='x_y_plot_selected_color_map',
                                                        default=available_color_maps.index('Plotly'))

    return  (if_show_dots, if_aggr_each_group, aggr_method_group, if_use_x_quantile_group, q_quantiles_group,
            if_aggr_all, aggr_method_all, if_use_x_quantile_all, q_quantiles_all, smooth_factor, if_show_diagonal,
            dot_size, dot_opacity, line_width, figure_width, figure_height, font_size_scale, color_map)

@st.cache_data(ttl=24*3600)
def _get_min_max(x, size_mapper_gamma):
    x_gamma_all = x ** size_mapper_gamma
    return np.percentile(x_gamma_all, 5), np.percentile(x_gamma_all, 95)

def _size_mapping(x, size_mapper_range, size_mapper_gamma):
    x = x / np.quantile(x[~np.isnan(x)], 0.95)
    x_gamma = x**size_mapper_gamma
    min_x, max_x = _get_min_max(x, size_mapper_gamma)
    sizes = size_mapper_range[0] + x_gamma / (max_x - min_x) * (size_mapper_range[1] - size_mapper_range[0])
    sizes[np.isnan(sizes)] = 0
    return sizes

def add_dot_property_mapper():
    with st.expander('Data point property mapper', expanded=True):            
        cols = st.columns([2, 1, 1])
        
        # Get all columns that are numeric
        available_size_cols = ['None'] + [
            col
            for col in st.session_state.session_stats_names
            if is_numeric_dtype(st.session_state.df_session_filtered[col])
        ]
        
        size_mapper = selectbox_wrapper_for_url_query(
                cols[0],
                label="dot size mapper",
                options=available_size_cols,
                key='x_y_plot_size_mapper',
                default=0,
            )
        
        if st.session_state.x_y_plot_size_mapper != 'None':
            size_mapper_range = slider_wrapper_for_url_query(cols[1],
                                                    label="size range",
                                                    min_value=0,
                                                    max_value=50,
                                                    key='x_y_plot_size_mapper_range',
                                                    default=(0, 10),
                                                    )
            
            size_mapper_gamma = slider_wrapper_for_url_query(cols[2],
                                                    label="size gamma",
                                                    min_value=0.0,
                                                    max_value=2.0,
                                                    key='x_y_plot_size_mapper_gamma',
                                                    default=1.0,
                                                    step=0.1)
        else:
            size_mapper_range, size_mapper_gamma = None, None
        
    return size_mapper, size_mapper_range, size_mapper_gamma

@st.fragment(run_every=5)
def data_selector():
            
    with st.expander(f'Session selector', expanded=True):        
        with st.expander(f"Filtered: {len(st.session_state.df_session_filtered)} sessions, "
                         f"{len(st.session_state.df_session_filtered.h2o.unique())} mice", expanded=False):
            st.dataframe(st.session_state.df_session_filtered)
            
        # --- add a download button ---
        with st.columns([1, 10])[1]:
            _add_download_filtered_session()
        
        # cols = st.columns([4, 1])
        # with cols[0].expander(f"From dataframe: {len(st.session_state.df_selected_from_dataframe)} sessions", expanded=False):
        #     st.dataframe(st.session_state.df_selected_from_dataframe)
        
        # if cols[1].button('❌'):
        #     st.session_state.df_selected_from_dataframe = pd.DataFrame()
        #     st.rerun()

        # Separate selection from table or streamlit
        def _show_selected(source="dataframe"):
            df_this = st.session_state['df_selected_from_' + source]
            with st.expander(f"Selected from {source}: {len(df_this)} sessions, "
                                f"{len(df_this.h2o.unique())} mice", expanded=False):
                st.dataframe(df_this)
            cols = st.columns([1, 1, 1])
            
            if source == "plotly": 
                return  # Don't allow select all or clear all for plotly
            
            if cols[1].button('select all', key=f'select_all_from_{source}'):
                st.session_state['df_selected_from_' + source] = st.session_state.df_session_filtered
                st.session_state['df_selected_from_' + source + '_just_overriden'] = True
                st.rerun()
            
            if cols[2].button('clear all', key=f'clear_all_from_{source}'):
                st.session_state['df_selected_from_' + source] = pd.DataFrame(columns=['h2o', 'session'])
                st.session_state['df_selected_from_' + source + '_just_overriden'] = True
                st.rerun()

        for source in ['dataframe', 'plotly']:
            _show_selected(source)
        

def _add_download_filtered_session():
    """Download the master table of the filtered session"""
    # Convert DataFrame to CSV format
    csv = st.session_state.df_session_filtered.to_csv(index=False)
    
    # Get the queries from URL for reproducibility
    filters = {key: st.query_params.get_all(key)
               for key in st.query_params.to_dict().keys()
               if 'filter' in key}
    query = json.dumps(filters, indent=4) 
    
    current_time = datetime.now().strftime("%Y%m%d")
    
    # Create download buttons
    st.download_button(
        label="Download filtered df as CSV",
        data=csv,
        file_name=f'filtered_data_{current_time}.csv',
        mime='text/csv'
    )
    
    # Create a download button for the JSON file
    st.download_button(
        label="Download filters as JSON",
        data=query,
        file_name=f'query_params_{current_time}.json',
        mime='application/json'
    )

def add_auto_train_manager():

    df_training_manager = st.session_state.auto_train_manager.df_manager

    # -- Show plotly chart --
    cols = st.columns([1, 1, 1, 0.7, 0.7, 1, 2])
    options = ["date", "session", "relative_date"]
    x_axis = selectbox_wrapper_for_url_query(
        st_prefix=cols[0],
        label="X axis",
        options=options,
        default="date",
        key="auto_training_history_x_axis",
    )

    options = ["first_date", "last_date", "subject_id", "progress_to_graduated"]
    sort_by = selectbox_wrapper_for_url_query(
        st_prefix=cols[1],
        label="Sort by",
        options=options,
        default="first_date",
        key="auto_training_history_sort_by",
    )

    options = ["descending", "ascending"]
    sort_order = selectbox_wrapper_for_url_query(
        st_prefix=cols[2],
        label="Sort order",
        options=options,
        default="descending",
        key="auto_training_history_sort_order",
    )

    marker_size = cols[3].number_input('Marker size', value=12, step=1)
    marker_edge_width = cols[4].number_input('Marker edge width', value=3, step=1)
    
    recent_months = slider_wrapper_for_url_query(cols[5],
                                                label="only recent months",
                                                min_value=1,
                                                max_value=12*3,
                                                step=1,
                                                key='auto_training_history_recent_months',
                                                default=8,
                                                disabled=x_axis != 'date',
                                                )

    # Get highlighted subjects
    if ('filter_subject_id' in st.session_state and st.session_state['filter_subject_id']) or\
       ('filter_h2o' in st.session_state and st.session_state['filter_h2o']):   
        # If subject_id is manually filtered or through URL query
        highlight_subjects = list(st.session_state.df_session_filtered['subject_id'].unique())
        highlight_subjects = [str(x) for x in highlight_subjects]
    else:
        highlight_subjects = []
        
    # --- Bokeh ---
    fig_auto_train, data_df = plot_manager_all_progress_bokeh(
        x_axis=x_axis,
        recent_days=recent_months*30.437,  # Turn months into days
        sort_by=sort_by,
        sort_order=sort_order,
        marker_size=marker_size,
        marker_edge_width=marker_edge_width,
        highlight_subjects=highlight_subjects,
        if_show_fig=False
    )
    
    event_result = streamlit_bokeh3_events(
        events="TestSelectEvent",
        bokeh_plot=fig_auto_train,
        key="autotrain_manager",
        debounce_time=100,
        refresh_on_update=True,
        override_height=fig_auto_train.height,
    )

    # some event was thrown
    if event_result is not None:
        # TestSelectEvent was thrown
        if "TestSelectEvent" in event_result:
            indices = event_result["TestSelectEvent"].get("indices", [])
            st.write(data_df.iloc[indices])

    # --- Plotly ---
    # fig_auto_train = plot_manager_all_progress(
    #     x_axis=x_axis,
    #     recent_days=recent_months*7,
    #     sort_by=sort_by,
    #     sort_order=sort_order,
    #     marker_size=marker_size,
    #     marker_edge_width=marker_edge_width,
    #     highlight_subjects=highlight_subjects,
    #     if_show_fig=False
    # )

    # fig_auto_train.update_layout(
    #     hoverlabel=dict(
    #         font_size=20,
    #     ),
    #     font=dict(size=18),
    #     height=30 * len(df_training_manager.subject_id.unique()),
    #     xaxis_side='top',
    #     title='',
    # )            

    # cols = st.columns([2, 1])
    # with cols[0]:
    #     selected_ = plotly_events(fig_auto_train,
    #                                 override_height=fig_auto_train.layout.height * 1.1, 
    #                                 override_width=fig_auto_train.layout.width,
    #                                 click_event=True,
    #                                 select_event=True,
    #                                 )
    # with cols[1]:
    #     st.markdown('#### 👀 Quick preview')
    #     st.markdown('###### Click on one session to preview here')
    #     if selected_:
    #         # Some hacks to get back selected data
    #         curve_number = selected_[0]['curveNumber']
    #         point_number = selected_[0]['pointNumber']
    #         this_subject = fig_auto_train['data'][curve_number]
    #         session_date = datetime.strptime(this_subject['customdata'][point_number][1], "%Y-%m-%d")
    #         subject_id = fig_auto_train['data'][curve_number]['name'].split(' ')[1]

    #         df_selected = (st.session_state.df['sessions_bonsai'].query(
    #             f'''subject_id == "{subject_id}" and session_date == "{session_date}"'''))
    #         draw_session_plots_quick_preview(df_selected)

    # -- Show dataframe --
    # only show filtered subject
    df_training_manager = df_training_manager[df_training_manager['subject_id'].isin(
        st.session_state.df_session_filtered['subject_id'].unique().astype(str))]

    # reorder columns
    df_training_manager = df_training_manager[['subject_id', 'session_date', 'session', 
                                                'curriculum_name', 'curriculum_version', 'curriculum_schema_version',
                                                'current_stage_suggested', 'current_stage_actual',
                                                'session_at_current_stage',
                                                'if_closed_loop', 'if_overriden_by_trainer',
                                                'foraging_efficiency', 'finished_trials', 
                                                'decision', 'next_stage_suggested'
                                                ]]

    with st.expander('Automatic training manager', expanded=False):
        st.dataframe(df_training_manager, height=3000)

@st.cache_data(ttl=3600*24)                
def _plot_population_x_y(df, x_name='session', y_name='foraging_eff', group_by='h2o',
                         smooth_factor=5, 
                         if_show_dots=True, 
                         if_aggr_each_group=True,
                         if_aggr_all=True, 
                         aggr_method_group='smooth',
                         aggr_method_all='mean +/ sem',
                         if_use_x_quantile_group=False,
                         q_quantiles_group=10,
                         if_use_x_quantile_all=False,
                         q_quantiles_all=20,
                         title='',
                         if_show_diagonal=False,
                         dot_size_base=10,
                         dot_size_mapping_name='None',
                         dot_size_mapping_range=None,
                         dot_size_mapping_gamma=None,
                         dot_opacity=0.4,
                         line_width=2,
                         x_y_plot_figure_width=1300,
                         x_y_plot_figure_height=900,
                         font_size_scale=1.0,
                         color_map='Plotly',
                         **kwarg):

    def _add_agg(df_this, x_name, y_name, group, aggr_method, if_use_x_quantile, q_quantiles, col, line_width, hoverinfo='skip', **kwarg):
        x = df_this.sort_values(x_name)[x_name].astype(float)
        y = df_this.sort_values(x_name)[y_name].astype(float)

        n_mice = len(df_this['h2o'].unique())
        n_sessions = len(df_this.groupby(['h2o', 'session']).count())
        n_str = f' ({n_mice} mice, {n_sessions} sessions)' if group_by !='h2o' else f' ({n_sessions} sessions)' 

        if aggr_method == 'running average':
            fig.add_trace(go.Scatter(    
                        x=x, 
                        y=y.rolling(window=smooth_factor, center=True, min_periods=1).mean(), 
                        name=group + n_str,
                        legendgroup=f'group_{group}',
                        mode="lines",
                        marker_color=col,
                        line_width=line_width,
                        opacity=1,
                        hoveron='points+fills',   # Scattergl doesn't support this
                        hoverinfo=hoverinfo,
                        **kwarg,
                        ))

        elif aggr_method == 'lowess':
            x_new = np.linspace(x.min(), x.max(), 200)
            lowess = sm.nonparametric.lowess(y, x, frac=smooth_factor/20)

            fig.add_trace(go.Scatter(    
                        x=lowess[:, 0], 
                        y=lowess[:, 1], 
                        name=group + n_str,
                        legendgroup=f'group_{group}',
                        mode="lines",
                        line_width=line_width,
                        marker_color=col,
                        opacity=1,
                        hoveron='points+fills',   # Scattergl doesn't support this
                        hoverinfo=hoverinfo,
                        **kwarg,
                        ))

        elif aggr_method in ('mean +/- sem', 'mean'):

            # Re-bin x if use quantiles of x
            if if_use_x_quantile:
                df_this[f'{x_name}_quantile'] = pd.qcut(df_this[x_name], q=q_quantiles, labels=False, duplicates='drop')

                mean = df_this.groupby(f'{x_name}_quantile')[y_name].mean()
                sem = df_this.groupby(f'{x_name}_quantile')[y_name].sem()
                valid_y = mean.notna()
                mean = mean[valid_y]
                sem = sem[valid_y]
                sem[~sem.notna()] = 0

                x = df_this.groupby(f'{x_name}_quantile')[x_name].median()  # Use median of x in each quantile as x
                y_upper = mean + sem
                y_lower = mean - sem

            else:    
                # mean and sem groupby x_name
                mean = df_this.groupby(x_name)[y_name].mean()
                sem = df_this.groupby(x_name)[y_name].sem()
                valid_y = mean.notna()
                mean = mean[valid_y]
                sem = sem[valid_y]
                sem[~sem.notna()] = 0

                x = mean.index
                y_upper = mean + sem
                y_lower = mean - sem

            fig.add_trace(go.Scatter(    
                        x=x, 
                        y=mean, 
                        # error_y=dict(type='data',
                        #             symmetric=True,
                        #             array=sem) if 'sem' in aggr_method else None,
                        name=group + n_str,
                        legendgroup=f'group_{group}',
                        mode="lines",
                        line_width=line_width,
                        marker_color=col,
                        opacity=1,
                        hoveron='points+fills',   # Scattergl doesn't support this
                        hoverinfo=hoverinfo,
                        **kwarg,                        
                        ))

            if 'sem' in aggr_method:
                fig.add_trace(go.Scatter(
                                # name='Upper Bound',
                                x=x,
                                y=y_upper,
                                mode='lines',
                                marker=dict(color=col),
                                line=dict(width=0),
                                legendgroup=f'group_{group}',
                                showlegend=False,
                                hoverinfo=hoverinfo,
                            ))
                fig.add_trace(go.Scatter(
                                # name='Upper Bound',
                                x=x,
                                y=y_lower,
                                mode='lines',
                                marker=dict(color=col),
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor=f'rgba({plotly.colors.convert_colors_to_same_type(col)[0][0].split("(")[-1][:-1]}, 0.2)',
                                legendgroup=f'group_{group}',
                                showlegend=False,
                                hoverinfo=hoverinfo
                            ))                                            

        elif aggr_method == 'linear fit':
            # perform linear regression
            mask = ~np.isnan(x) & ~np.isnan(y)
            try:
                slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
                sig = lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                fig.add_trace(go.Scatter(x=x, 
                                        y=intercept + slope*x, 
                                        mode='lines',
                                        name=f"{group} {n_str}<br>  {sig(p_value):<5}p={p_value:.1e}, r={r_value:.3f}",
                                        marker_color=col,
                                        line=dict(dash='dot' if p_value > 0.05 else 'solid',
                                                  width=line_width if p_value > 0.05 else line_width*1.5),
                                        legendgroup=f'group_{group}',
                                        hoverinfo=hoverinfo
                                        )
                )            
            except:
                pass

    fig = go.Figure()
    col_map = px.colors.qualitative.__dict__[color_map]

    # Add some more columns
    if dot_size_mapping_name !='None' and dot_size_mapping_name in df.columns:
        df['dot_size'] = _size_mapping(df[dot_size_mapping_name], dot_size_mapping_range, dot_size_mapping_gamma)
    else:
        df['dot_size'] = dot_size_base

    # Always turn group_by column to str
    df[group_by] = df[group_by].astype(str)
        
    # Add a diagonal line first
    if if_show_diagonal:
        _min = np.nanmin(df[x_name].values.ravel())
        _max = np.nanmax(df[y_name].values.ravel())
        fig.add_trace(go.Scattergl(x=[_min, _max], 
                                   y=[_min, _max], 
                                   mode='lines',
                                   line=dict(dash='dash', color='black', width=2),
                                   name='x=y',
                                   showlegend=True)
                      )

    for i, group in enumerate(df.sort_values(group_by)[group_by].unique()):
        this_session = df.query(f'{group_by} == "{group}"').sort_values('session')
        col = col_map[i%len(col_map)]

        if if_show_dots:
            # if not len(st.session_state.df_selected_from_plotly):   
            this_session['colors'] = col  # all use normal colors
            # else:
            #     merged = pd.merge(this_session, st.session_state.df_selected_from_plotly, on=['h2o', 'session'], how='left')
            #     merged['colors'] = 'lightgrey'  # default, grey
            #     merged.loc[merged.subject_id_y.notna(), 'colors'] = col   # only use normal colors for the selected dots 
            #     this_session['colors'] = merged.colors.values
            #     this_session = pd.concat([this_session.query('colors != "lightgrey"'), this_session.query('colors == "lightgrey"')])  # make sure the real color goes first

            fig.add_trace(go.Scattergl(
                            x=this_session[x_name], 
                            y=this_session[y_name], 
                            name=group,
                            legendgroup=f'group_{group}',
                            showlegend=not if_aggr_each_group,
                            mode="markers",
                            line_width=line_width,
                            marker_size=this_session['dot_size'],
                            marker_color=this_session['colors'],
                            opacity=dot_opacity,
                            hovertemplate =  '<b>%{customdata[0]}, %{customdata[1]}, Session %{customdata[2]}'
                                             '<br>%{customdata[4]} @ %{customdata[9]}'
                                             '<br>Rig: %{customdata[3]}'
                                             '<br>Task: %{customdata[5]}'
                                             '<br>AutoTrain: %{customdata[7]} @ %{customdata[6]}</b>'
                                             f'<br>{"-"*10}<br><b>X: </b>{x_name} = %{{x}}'
                                             f'<br><b>Y: </b>{y_name} = %{{y}}' 
                                             + (f'<br><b>Size: </b>{dot_size_mapping_name} = %{{customdata[8]}}' 
                                                   if dot_size_mapping_name !='None' 
                                                   else '') 
                                             + '<extra></extra>',
                            customdata=np.stack((this_session.h2o, # 0
                                                 this_session.session_date.dt.strftime('%Y-%m-%d'), # 1
                                                 this_session.session, # 2
                                                 this_session.rig, # 3
                                                 this_session.user_name, # 4
                                                 this_session.task, # 5
                                                 this_session.curriculum_name
                                                    if 'curriculum_name' in this_session.columns
                                                    else ['None'] * len(this_session.h2o), # 6
                                                 this_session.current_stage_actual
                                                    if 'current_stage_actual' in this_session.columns
                                                    else ['None'] * len(this_session.h2o), # 7
                                                 this_session[dot_size_mapping_name] 
                                                    if dot_size_mapping_name !='None' 
                                                    else [np.nan] * len(this_session.h2o), # 8
                                                 this_session.data_source if 'data_source' in this_session else [''] * len(this_session.h2o), # 9
                                                 ), axis=-1),
                            unselected=dict(marker_color='lightgrey')
                            ))

        if if_aggr_each_group:
            _add_agg(this_session, x_name, y_name, group, aggr_method_group, 
                     if_use_x_quantile_group, q_quantiles_group, col, line_width=line_width,
                     hoverinfo='all' if not if_show_dots else 'skip')

    if if_aggr_all:
        _add_agg(df, x_name, y_name, 'all', aggr_method_all, if_use_x_quantile_all, q_quantiles_all, 'rgb(0, 0, 0)', line_width=line_width*1.5,
                 hoverinfo='all' if not if_show_dots else 'skip')

    n_mice = len(df['h2o'].unique())
    n_sessions = len(df.groupby(['h2o', 'session']).count())

    override_plotly_theme(fig, theme="simple_white", font_size_scale=font_size_scale)

    fig.update_layout(
        width=x_y_plot_figure_width,
        height=x_y_plot_figure_height,
        xaxis_title_text=x_name,
        yaxis_title_text=y_name,
        hovermode="closest",
        legend={"traceorder": "reversed"},
        title=f"{title}, {n_mice} mice, {n_sessions} sessions",
        dragmode="zoom",  # 'select',
    ) 
    return fig


def add_footnote():
    st.markdown('---')
    st.markdown(f'#### AIND Behavior Team @ 2025 {__ver__}')
    st.markdown('[bug report / feature request](https://github.com/AllenNeuralDynamics/foraging-behavior-browser/issues)')


def download_df(df, label="Download filtered df as CSV", file_name="df.csv"):
    """ Add a button to download df as csv """
    csv = df.to_csv(index=True)
    
    # Create download buttons
    st.download_button(
        label=label,
        data=csv,
        file_name=file_name,
        mime='text/csv'
    )