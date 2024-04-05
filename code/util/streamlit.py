from collections import OrderedDict
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, ColumnsAutoSizeMode, DataReturnMode
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
    is_object_dtype,
)
import streamlit.components.v1 as components
from streamlit_plotly_events import plotly_events

import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy.stats import linregress

from .url_query_helper import checkbox_wrapper_for_url_query, selectbox_wrapper_for_url_query, slider_wrapper_for_url_query


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
        df = df.sort_values('session_end_time', ascending=False)
    else:
        df = df.sort_values('session_date', ascending=False)
    
    # preselect
    if (('df_selected_from_dataframe' in st.session_state and len(st.session_state.df_selected_from_dataframe)) 
       and ('tab_id' in st.session_state and st.session_state.tab_id == "tab_session_x_y")):
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

def aggrid_interactive_table_curriculum(df: pd.DataFrame,
                                        pre_selected_rows: list = None):
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
        
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="balham",
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
        height=200,
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
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns,
                                                            label_visibility='collapsed',
                                                            default=st.session_state.to_filter_columns_changed 
                                                                    if 'to_filter_columns_changed' in st.session_state
                                                                    else default_filters,
                                                            key='to_filter_columns',
                                                            on_change=cache_widget,
                                                            args=['to_filter_columns'])
        for column in to_filter_columns:
            if not len(df): break
            
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10 and column not in ('finished', 'foraging_eff', 'session', 'finished_trials'):
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
                
            elif is_numeric_dtype(df[column]):
                
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
                        
                    _min = float(x.min())
                    _max = float(x.max())
                    step = (_max - _min) / 100

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
                
            elif is_datetime64_any_dtype(df[column]):
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
            else:  # Regular string
                    
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
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def add_session_filter(if_bonsai=False, url_query={}):
    with st.expander("Behavioral session filter", expanded=True):   
        if not if_bonsai:
            st.session_state.df_session_filtered = filter_dataframe(df=st.session_state.df['sessions'],
                                                                    default_filters=['h2o', 'task', 'session', 'finished_trials', 'foraging_eff', 'photostim_location'],
                                                                    url_query=url_query)
        else:
            st.session_state.df_session_filtered = filter_dataframe(df=st.session_state.df['sessions_bonsai'],
                                                                    default_filters=['subject_id', 'task', 'session', 'finished_trials', 'foraging_eff'],
                                                                    url_query=url_query)

@st.cache_data(ttl=3600*24)
def _get_grouped_by_fields(if_bonsai):
    if if_bonsai:
        options = ['h2o', 'task', 'user_name', 'rig', 'weekday']
        options += [col 
                for col in st.session_state.df_session_filtered.columns
                if is_categorical_dtype(st.session_state.df_session_filtered[col]) 
                or st.session_state.df_session_filtered[col].nunique() < 20
                and not any([exclude in col for exclude in 
                             ('date', 'time', 'session', 'finished', 'foraging_eff')])
        ]
        options = list(list(OrderedDict.fromkeys(options))) # Remove duplicates
    else:
        options = ['h2o', 'task', 'photostim_location', 'weekday',
                   'headbar', 'user_name', 'sex', 'rig']
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

        # st.form_submit_button("update axes")
    return x_name, y_name, group_by, size_mapper

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

    return  (if_show_dots, if_aggr_each_group, aggr_method_group, if_use_x_quantile_group, q_quantiles_group,
            if_aggr_all, aggr_method_all, if_use_x_quantile_all, q_quantiles_all, smooth_factor,
            dot_size, dot_opacity, line_width, figure_width, figure_height, font_size_scale)

def data_selector():
            
    with st.expander(f'Session selector', expanded=True):
        
        with st.expander(f"Filtered: {len(st.session_state.df_session_filtered)} sessions, "
                         f"{len(st.session_state.df_session_filtered.h2o.unique())} mice", expanded=False):
            st.dataframe(st.session_state.df_session_filtered)
        
        # cols = st.columns([4, 1])
        # with cols[0].expander(f"From dataframe: {len(st.session_state.df_selected_from_dataframe)} sessions", expanded=False):
        #     st.dataframe(st.session_state.df_selected_from_dataframe)
        
        # if cols[1].button('❌'):
        #     st.session_state.df_selected_from_dataframe = pd.DataFrame()
        #     st.rerun()

        cols = st.columns([5, 1, 1])
        with cols[0].expander(f"Selected: {len(st.session_state.df_selected_from_plotly)} sessions, "
                              f"{len(st.session_state.df_selected_from_plotly.h2o.unique())} mice", expanded=False):
            st.dataframe(st.session_state.df_selected_from_plotly)
        if cols[1].button('all'):
            st.session_state.df_selected_from_plotly = st.session_state.df_session_filtered
            st.rerun()
            
        
        if cols[2].button('❌ '):
            st.session_state.df_selected_from_plotly = pd.DataFrame(columns=['h2o', 'session'])
            st.session_state.df_selected_from_dataframe = pd.DataFrame(columns=['h2o', 'session'])
            st.rerun()

def add_auto_train_manager():
    
    st.session_state.auto_train_manager.df_manager = st.session_state.auto_train_manager.df_manager[
        st.session_state.auto_train_manager.df_manager.subject_id.astype(float) > 0]  # Remove dummy mouse 0
    df_training_manager = st.session_state.auto_train_manager.df_manager
    
    # -- Show plotly chart --
    cols = st.columns([1, 1, 1, 0.7, 0.7, 3])
    options=['session', 'date', 'relative_date']
    x_axis = cols[0].selectbox('X axis', options=options, 
                                index=options.index(st.session_state['auto_training_history_x_axis']),
                                key="auto_training_history_x_axis")
    
    options=['subject_id', 
            'first_date',
            'last_date',
            'progress_to_graduated']
    sort_by = cols[1].selectbox('Sort by', 
                                options=options,
                                index=options.index(st.session_state['auto_training_history_sort_by']),
                                key="auto_training_history_sort_by")
    
    options=['descending', 'ascending']
    sort_order = cols[2].selectbox('Sort order', 
                                    options=options,
                                    index=options.index(st.session_state['auto_training_history_sort_order']),
                                    key='auto_training_history_sort_order'
                                    )
    
    marker_size = cols[3].number_input('Marker size', value=15, step=1)
    marker_edge_width = cols[4].number_input('Marker edge width', value=3, step=1)
    
    # Get highlighted subjects
    if ('filter_subject_id' in st.session_state and st.session_state['filter_subject_id']) or\
       ('filter_h2o' in st.session_state and st.session_state['filter_h2o']):   
        # If subject_id is manually filtered or through URL query
        highlight_subjects = list(st.session_state.df_session_filtered['subject_id'].unique())
        highlight_subjects = [str(x) for x in highlight_subjects]
    else:
        highlight_subjects = []
                                
    fig_auto_train = st.session_state.auto_train_manager.plot_all_progress(
        x_axis=x_axis,
        sort_by=sort_by,
        sort_order=sort_order,
        marker_size=marker_size,
        marker_edge_width=marker_edge_width,
        highlight_subjects=highlight_subjects,
        if_show_fig=False
    )
    
    fig_auto_train.update_layout(
        hoverlabel=dict(
            font_size=20,
        ),
        font=dict(size=18),
        height=30 * len(df_training_manager.subject_id.unique()),
    )            
    
    selected_ = plotly_events(fig_auto_train,
                                override_height=fig_auto_train.layout.height * 1.1, 
                                override_width=fig_auto_train.layout.width,
                                click_event=False,
                                select_event=False,
                                )
    
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
    
    with st.expander('Automatic training manager', expanded=True):
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
                         dot_size_base=10,
                         dot_size_mapping_name='None',
                         dot_opacity=0.4,
                         line_width=2,
                         x_y_plot_figure_width=1300,
                         x_y_plot_figure_height=900,
                         font_size_scale=1.0,
                         **kwarg):

    def _add_agg(df_this, x_name, y_name, group, aggr_method, if_use_x_quantile, q_quantiles, col, line_width, **kwarg):
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
                        hoverinfo='skip',
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
                        hoverinfo='skip',
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
                        hoverinfo='skip',
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
                                hoverinfo='skip',
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
                                hoverinfo='skip'
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
                                        hoverinfo='skip'
                                        )
                )            
            except:
                pass

    fig = go.Figure()
    col_map = px.colors.qualitative.Plotly

    # Add some more columns
    if dot_size_mapping_name !='None' and dot_size_mapping_name in df.columns:
        df['dot_size'] = df[dot_size_mapping_name]
    else:
        df['dot_size'] = dot_size_base
        
    # Turn column of group_by to string if it's not
    if not is_string_dtype(df[group_by]):
        df[group_by] = df[group_by].astype(str)
        
    for i, group in enumerate(df.sort_values(group_by)[group_by].unique()):
        this_session = df.query(f'{group_by} == "{group}"').sort_values('session')
        col = col_map[i%len(col_map)]
        
        if if_show_dots:
            if not len(st.session_state.df_selected_from_plotly):   
                this_session['colors'] = col  # all use normal colors
            else:
                merged = pd.merge(this_session, st.session_state.df_selected_from_plotly, on=['h2o', 'session'], how='left')
                merged['colors'] = 'lightgrey'  # default, grey
                merged.loc[merged.subject_id_y.notna(), 'colors'] = col   # only use normal colors for the selected dots 
                this_session['colors'] = merged.colors.values
                this_session = pd.concat([this_session.query('colors != "lightgrey"'), this_session.query('colors == "lightgrey"')])  # make sure the real color goes first

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
                                             '<br>%{customdata[3]}, %{customdata[4]}'
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
                                                 ), axis=-1),
                            unselected=dict(marker_color='lightgrey')
                            ))

        if if_aggr_each_group:
            _add_agg(this_session, x_name, y_name, group, aggr_method_group, if_use_x_quantile_group, q_quantiles_group, col, line_width=line_width)

    if if_aggr_all:
        _add_agg(df, x_name, y_name, 'all', aggr_method_all, if_use_x_quantile_all, q_quantiles_all, 'rgb(0, 0, 0)', line_width=line_width*1.5)

    n_mice = len(df['h2o'].unique())
    n_sessions = len(df.groupby(['h2o', 'session']).count())

    fig.update_layout(
        width=x_y_plot_figure_width,
        height=x_y_plot_figure_height,
        xaxis_title=x_name,
        yaxis_title=y_name,
        font=dict(size=24 * font_size_scale),
        hovermode="closest",
        hoverlabel=dict(font_size=17 * font_size_scale),
        legend={"traceorder": "reversed"},
        legend_font_size=20 * font_size_scale,
        title=f"{title}, {n_mice} mice, {n_sessions} sessions",
        dragmode="zoom",  # 'select',
        margin=dict(l=130 * font_size_scale, 
                    r=50 * font_size_scale, 
                    b=130 * font_size_scale, 
                    t=100 * font_size_scale,
                    ),
    )
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', 
                    #  range=[1, min(100, df[x_name].max())],
                     ticks = "outside", tickcolor='black', ticklen=10, tickwidth=2, ticksuffix=' ')

    fig.update_yaxes(showline=True, linewidth=2, linecolor='black',
                     title_standoff=40,
                     ticks = "outside", tickcolor='black', ticklen=10, tickwidth=2, ticksuffix=' ')
    return fig
