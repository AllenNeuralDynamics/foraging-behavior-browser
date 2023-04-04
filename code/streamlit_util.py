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
       and ('tab_id' in st.session_state and st.session_state.tab_id == "tab1")):
        indexer = st.session_state.df_selected_from_dataframe.set_index(['h2o', 'session']
                                                              ).index.get_indexer(df.set_index(['h2o', 'session']).index)
        pre_selected_rows = np.where(indexer != -1)[0].tolist()
    else:
        pre_selected_rows = []
        
    options.configure_selection(selection_mode="multiple",
                                pre_selected_rows=pre_selected_rows)  # , use_checkbox=True, header_checkbox=True)
    
    # options.configure_column(field="session_date", sort="desc")
    
    # options.configure_column(field="h2o", hide=True, rowGroup=True)
    options.configure_column(field='subject_id', hide=True)
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
    st.session_state[f'{field}_cache'] = st.session_state[field]
    
    # Clear cache if needed
    if clear:
        if clear in st.session_state: del st.session_state[clear]
        
# def dec_cache_widget_state(widget, ):


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

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
                                                            default=st.session_state.to_filter_columns_cache 
                                                                    if 'to_filter_columns_cache' in st.session_state
                                                                    else ['h2o', 'task', 'finished_trials', 'photostim_location'],
                                                            key='to_filter_columns',
                                                            on_change=cache_widget,
                                                            args=['to_filter_columns'])
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10 and column not in ('finished', 'foraging_eff'):
                right.markdown(f"Filter for :red[**{column}**]")
                selected = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    label_visibility='collapsed',
                    default=[i for i in st.session_state[f'select_{column}_cache'] if i in list(df[column].unique())]
                            if f'select_{column}_cache' in st.session_state
                            else list(df[column].unique()),
                    key=f'select_{column}',
                    on_change=cache_widget,
                    args=[f'select_{column}']
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
                                                 value=st.session_state[f'if_log_{column}_cache']
                                                       if f'if_log_{column}_cache' in st.session_state
                                                       else False,
                                                 key=f'if_log_{column}',
                                                 on_change=cache_widget,
                                                 args=[f'if_log_{column}'],
                                                 kwargs={'clear': f'select_{column}_cache'}  # If show_log is changed, clear select cache
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
                    
                    user_num_input = st.slider(
                        f"Values for {column}",
                        label_visibility='collapsed',
                        min_value=_min,
                        max_value=_max,
                        value= st.session_state[f'select_{column}_cache']
                                if f'select_{column}_cache' in st.session_state
                                else (_min, _max),
                        step=step,
                        key=f'select_{column}',
                        on_change=cache_widget,
                        args=[f'select_{column}']
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
                    value=st.session_state[f'select_{column}_cache']
                                if f'select_{column}_cache' in st.session_state
                                else (df[column].min(), df[column].max()),
                    key=f'select_{column}',
                    on_change=cache_widget,
                    args=[f'select_{column}']
                )
                
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in :red[**{column}**]",
                    value=st.session_state[f'select_{column}_cache']
                                if f'select_{column}_cache' in st.session_state
                                else '',
                    key=f'select_{column}',
                    on_change=cache_widget,
                    args=[f'select_{column}']
                    )
                
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def add_session_filter():
    with st.expander("Behavioral session filter", expanded=True):   
        st.session_state.df_session_filtered = filter_dataframe(df=st.session_state.df['sessions'])
    
    
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
        
