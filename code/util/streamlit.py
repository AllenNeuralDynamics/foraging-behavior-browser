from email import header
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, ColumnsAutoSizeMode, DataReturnMode
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import streamlit.components.v1 as components

import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

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

def aggrid_interactive_table_session(df: pd.DataFrame):
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
        height=400,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        custom_css=custom_css,
    )
    
    return selection

def aggrid_interactive_table_curriculum(df: pd.DataFrame):
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
    
    options.configure_selection(selection_mode="single")
        
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="balham",
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        height=300,
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
        st.markdown(f"Add filters")
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
                
                if f'filter_{column}_changed' in st.session_state:
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
                    
                    if f'filter_{column}_changed' in st.session_state:
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
                    
                if f'filter_{column}_changed' in st.session_state:
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

def add_xy_selector(if_bonsai):
    with st.expander("Select axes", expanded=True):
        # with st.form("axis_selection"):
        cols = st.columns([1, 1, 1])
        x_name = cols[0].selectbox("x axis", 
                                   st.session_state.session_stats_names, 
                                   index=st.session_state.session_stats_names.index(st.session_state['x_y_plot_xname']),
                                   key='x_y_plot_xname'
                                   )
        y_name = cols[1].selectbox("y axis", 
                                   st.session_state.session_stats_names, 
                                   index=st.session_state.session_stats_names.index(st.session_state['x_y_plot_xname']),
                                   key='x_y_plot_yname')
        
        if if_bonsai:
            options = ['h2o', 'task', 'user_name', 'rig']
        else:
            options = ['h2o', 'task', 'photostim_location', 'weekday',
                       'headbar', 'user_name', 'sex', 'rig']
        
        group_by = cols[2].selectbox("grouped by", 
                                     options=options, 
                                     index=options.index(st.session_state['x_y_plot_group_by']),
                                     key='x_y_plot_group_by')
        
            # st.form_submit_button("update axes")
    return x_name, y_name, group_by
    

def add_xy_setting():
    with st.expander('Plot settings', expanded=True):            
        s_cols = st.columns([1, 1, 1])
        # if_plot_only_selected_from_dataframe = s_cols[0].checkbox('Only selected', False)
        if_show_dots = s_cols[0].checkbox('Show data points', 
                                            value=st.session_state['x_y_plot_if_show_dots'],
                                            key='x_y_plot_if_show_dots')
        

        if_aggr_each_group = s_cols[1].checkbox('Aggr each group', 
                                                value=st.session_state['x_y_plot_if_aggr_each_group'],
                                                key='x_y_plot_if_aggr_each_group')
        
        aggr_methods =  ['mean', 'mean +/- sem', 'lowess', 'running average', 'linear fit']
        aggr_method_group = s_cols[1].selectbox('aggr method group', 
                                                options=aggr_methods, 
                                                index=aggr_methods.index(st.session_state['x_y_plot_aggr_method_group']),
                                                key='x_y_plot_aggr_method_group', 
                                                disabled=not if_aggr_each_group)
        
        if_use_x_quantile_group = s_cols[1].checkbox('Use quantiles of x ', 
                                                     value=st.session_state['x_y_plot_if_use_x_quantile_group'],
                                                     key='x_y_plot_if_use_x_quantile_group',
                                                     disabled='mean' not in aggr_method_group) 
            
        q_quantiles_group = s_cols[1].slider('Number of quantiles ', 1, 100,
                                             value=st.session_state['x_y_plot_q_quantiles_group'],
                                             key='x_y_plot_q_quantiles_group',
                                             disabled=not if_use_x_quantile_group
                                             )
        
        if_aggr_all = s_cols[2].checkbox('Aggr all',
                                            value=st.session_state['x_y_plot_if_aggr_all'],
                                            key='x_y_plot_if_aggr_all',
                                        )
        
        # st.session_state.if_aggr_all_cache = if_aggr_all  # Have to use another variable to store this explicitly (my cache_widget somehow doesn't work with checkbox)
        aggr_method_all = s_cols[2].selectbox('aggr method all', aggr_methods, 
                                                index=aggr_methods.index(st.session_state['x_y_plot_aggr_method_all']), 
                                                key='x_y_plot_aggr_method_all',
                                                disabled=not if_aggr_all)

        if_use_x_quantile_all = s_cols[2].checkbox('Use quantiles of x', 
                                                   value=st.session_state['x_y_plot_if_use_x_quantile_all'],
                                                   key='x_y_plot_if_use_x_quantile_all',
                                                   disabled='mean' not in aggr_method_all,
                                                   )
        q_quantiles_all = s_cols[2].slider('number of quantiles', 1, 100, 
                                           value=st.session_state['x_y_plot_q_quantiles_all'],
                                           key='x_y_plot_q_quantiles_all',
                                           disabled=not if_use_x_quantile_all
                                           )

        smooth_factor = s_cols[0].slider('smooth factor', 1, 20,
                                            value=st.session_state['x_y_plot_smooth_factor'],
                                            key='x_y_plot_smooth_factor',
                                            disabled= not ((if_aggr_each_group and aggr_method_group in ('running average', 'lowess'))
                                                                    or (if_aggr_all and aggr_method_all in ('running average', 'lowess'))) 
        )
        
        c = st.columns([1, 1])
        dot_size = c[0].slider('dot size', 1, 30, 
                                step=1,
                                value=st.session_state['x_y_plot_dot_size'],
                                key='x_y_plot_dot_size'
                                )
        dot_opacity = c[1].slider('opacity', 0.0, 1.0, 
                                    step=0.05, 
                                    value=st.session_state['x_y_plot_dot_opacity'],
                                    key='x_y_plot_dot_opacity')

    return  (if_show_dots, if_aggr_each_group, aggr_method_group, if_use_x_quantile_group, q_quantiles_group,
        if_aggr_all, aggr_method_all, if_use_x_quantile_all, q_quantiles_all, smooth_factor,
        dot_size, dot_opacity)

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
        #     st.experimental_rerun()

        cols = st.columns([5, 1, 1])
        with cols[0].expander(f"Selected: {len(st.session_state.df_selected_from_plotly)} sessions, "
                              f"{len(st.session_state.df_selected_from_plotly.h2o.unique())} mice", expanded=False):
            st.dataframe(st.session_state.df_selected_from_plotly)
        if cols[1].button('all'):
            st.session_state.df_selected_from_plotly = st.session_state.df_session_filtered
            st.experimental_rerun()
            
        
        if cols[2].button('❌ '):
            st.session_state.df_selected_from_plotly = pd.DataFrame(columns=['h2o', 'session'])
            st.session_state.df_selected_from_dataframe = pd.DataFrame(columns=['h2o', 'session'])
            st.experimental_rerun()
        

def _sync_widget_with_query(key, default):
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
            q_all_correct_type = [_type(q) for q in q_all]
        
        # flatten list if only one element
        if not isinstance(default, list):
            q_all_correct_type = q_all_correct_type[0]
        
        try:
            st.session_state[key] = q_all_correct_type
        except:
            print(f'Failed to set {key} to {q_all_correct_type}')
    else:
        try:
            st.session_state[key] = default
        except:
            print(f'Failed to set {key} to {default}')
