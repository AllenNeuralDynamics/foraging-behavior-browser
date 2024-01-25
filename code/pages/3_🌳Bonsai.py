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
from scipy.stats import linregress
import statsmodels.api as sm
import json

from PIL import Image, ImageColor
import streamlit.components.v1 as components
import streamlit_nested_layout
from streamlit_plotly_events import plotly_events

from util.streamlit import (filter_dataframe, aggrid_interactive_table_session,
                            aggrid_interactive_table_curriculum, add_session_filter, data_selector)
from Home import add_caution
import extra_streamlit_components as stx

from aind_auto_train.curriculum_manager import CurriculumManager
from aind_auto_train.auto_train_manager import DynamicForagingAutoTrainManager
from aind_auto_train.schema.task import TrainingStage

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


# @st.cache_data(ttl=24*3600, max_entries=20)
def show_session_level_img_by_key_and_prefix(key, prefix, column=None, other_patterns=[''], crop=None, caption=True, **kwargs):
    
    subject_session_date_str = f"{key['subject_id']}_{key['session_date']}_{key['nwb_suffix']}".split('_0')[0]
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
    
    _f.image(img if img is not None else "https://cdn-icons-png.flaticon.com/512/3585/3585596.png", 
                output_format='PNG', 
                #caption=f_name.split('/')[-1] if caption and f_name else '',
                use_column_width='always',
                **kwargs)

    return img

# table_mapping = {
#     'sessions_bonsai': fetch_sessions,
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
                    
                my_bar.progress(int((i + 1) / len(df_to_draw_mice) * 100))
                


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
                         dot_size=10,
                         dot_opacity=0.4,
                         **kwarg):
    
    def _add_agg(df_this, x_name, y_name, group, aggr_method, if_use_x_quantile, q_quantiles, col):
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
                        opacity=1,
                        hoveron='points+fills',   # Scattergl doesn't support this
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
                        marker_color=col,
                        opacity=1,
                        hoveron='points+fills',   # Scattergl doesn't support this
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
                        marker_color=col,
                        opacity=1,
                        hoveron='points+fills',   # Scattergl doesn't support this
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
                                                  width=2 if p_value > 0.05 else 3),
                                        legendgroup=f'group_{group}',
                                        # hoverinfo='skip'
                                        )
                )            
            except:
                pass
                
    fig = go.Figure()
    col_map = px.colors.qualitative.Plotly
    
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
                            marker_size=dot_size,
                            marker_color=this_session['colors'],
                            opacity=dot_opacity, # 0.5 if if_aggr_each_group else 0.8,
                            text=this_session['session'],
                            hovertemplate =   '<br>%{customdata[0]}, Session %{text}' +
                                            '<br>%s = %%{x}' % (x_name) +
                                            '<br>%s = %%{y}' % (y_name),
                                            #   '<extra>%{name}</extra>',
                            customdata=np.stack((this_session.h2o, this_session.session), axis=-1),
                            unselected=dict(marker_color='lightgrey')
                            ))
            
        if if_aggr_each_group:
            _add_agg(this_session, x_name, y_name, group, aggr_method_group, if_use_x_quantile_group, q_quantiles_group, col)
        

    if if_aggr_all:
        _add_agg(df, x_name, y_name, 'all', aggr_method_all, if_use_x_quantile_all, q_quantiles_all, 'rgb(0, 0, 0)')
        

    n_mice = len(df['h2o'].unique())
    n_sessions = len(df.groupby(['h2o', 'session']).count())
    
    fig.update_layout(
                    width=1300, 
                    height=900,
                    xaxis_title=x_name,
                    yaxis_title=y_name,
                    font=dict(size=25),
                    hovermode='closest',
                    legend={'traceorder':'reversed'},
                    legend_font_size=15,
                    title=f'{title}, {n_mice} mice, {n_sessions} sessions',
                    dragmode='select', # 'zoom',
                    margin=dict(l=130, r=50, b=130, t=100),
                    )
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', 
                    #  range=[1, min(100, df[x_name].max())],
                     ticks = "outside", tickcolor='black', ticklen=10, tickwidth=2, ticksuffix=' ')
    
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black',
                     title_standoff=40,
                     ticks = "outside", tickcolor='black', ticklen=10, tickwidth=2, ticksuffix=' ')
    return fig
  

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

        x_name, y_name, group_by = add_xy_selector()

        with st.expander('Plot settings', expanded=True):            
            s_cols = st.columns([1, 1, 1])
            # if_plot_only_selected_from_dataframe = s_cols[0].checkbox('Only selected', False)
            if_show_dots = s_cols[0].checkbox('Show data points', True)
            
            aggr_methods =  ['mean', 'mean +/- sem', 'lowess', 'running average', 'linear fit']

            if_aggr_each_group = s_cols[1].checkbox('Aggr each group', 
                                                    value=st.session_state.if_aggr_each_group_cache 
                                                        if 'if_aggr_each_group_cache' in st.session_state
                                                        else True, )
            
            st.session_state.if_aggr_each_group_cache = if_aggr_each_group  # Have to use another variable to store this explicitly (my cache_widget somehow doesn't work with checkbox)
            aggr_method_group = s_cols[1].selectbox('aggr method group', aggr_methods, index=aggr_methods.index('lowess'), disabled=not if_aggr_each_group)
            
            if_use_x_quantile_group = s_cols[1].checkbox('Use quantiles of x ', False) if 'mean' in aggr_method_group else False
            q_quantiles_group = s_cols[1].slider('Number of quantiles ', 1, 100, 20, disabled=not if_use_x_quantile_group) if if_use_x_quantile_group else None
            
            if_aggr_all = s_cols[2].checkbox('Aggr all', 
                                            value=st.session_state.if_aggr_all_cache
                                                if 'if_aggr_all_cache' in st.session_state
                                                else True,
                                            )
            
            st.session_state.if_aggr_all_cache = if_aggr_all  # Have to use another variable to store this explicitly (my cache_widget somehow doesn't work with checkbox)
            aggr_method_all = s_cols[2].selectbox('aggr method all', aggr_methods, index=aggr_methods.index('mean +/- sem'), disabled=not if_aggr_all)

            if_use_x_quantile_all = s_cols[2].checkbox('Use quantiles of x', False) if 'mean' in aggr_method_all else False
            q_quantiles_all = s_cols[2].slider('number of quantiles', 1, 100, 20, disabled=not if_use_x_quantile_all) if if_use_x_quantile_all else None

            smooth_factor = s_cols[0].slider('smooth factor', 1, 20, 5) if ((if_aggr_each_group and aggr_method_group in ('running average', 'lowess'))
                                                                        or (if_aggr_all and aggr_method_all in ('running average', 'lowess'))) else None
            
            c = st.columns([1, 1])
            dot_size = c[0].slider('dot size', 1, 30, step=1, value=10)
            dot_opacity = c[1].slider('opacity', 0.0, 1.0, step=0.05, value=0.5)

    
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
                                        dot_opacity=dot_opacity,)
        
        # st.plotly_chart(fig)
        selected = plotly_events(fig, click_event=True, hover_event=False, select_event=True, 
                                 override_height=fig.layout.height * 1.1, override_width=fig.layout.width)
      
    if len(selected):
        df_selected_from_plotly = df_x_y_session.merge(pd.DataFrame(selected).rename({'x': x_name, 'y': y_name}, axis=1), 
                                                    on=[x_name, y_name], how='inner')

    return df_selected_from_plotly, cols

def add_xy_selector():
    with st.expander("Select axes", expanded=True):
        # with st.form("axis_selection"):
        cols = st.columns([1, 1, 1])
        x_name = cols[0].selectbox("x axis", st.session_state.session_stats_names, index=st.session_state.session_stats_names.index('session'))
        y_name = cols[1].selectbox("y axis", st.session_state.session_stats_names, index=st.session_state.session_stats_names.index('foraging_eff'))
        group_by = cols[2].selectbox("grouped by", ['h2o', 'task',
                                                    'user_name', 'rig'], index=['h2o', 'task'].index('h2o'))
            # st.form_submit_button("update axes")
    return x_name, y_name, group_by


def show_curriculums():
    pass

# ------- Layout starts here -------- #    
def init():
    
    # Clear Session state
    for key in ['selected_draw_types']:
        if key in st.session_state:
            del st.session_state[key]

    df = load_data(['sessions', 
                   ])
    
        
    st.session_state.df = df
    st.session_state.df_selected_from_plotly = pd.DataFrame(columns=['h2o', 'session'])
    st.session_state.df_selected_from_dataframe = pd.DataFrame(columns=['h2o', 'session'])
    
    # Init session states
    to_init = [
               ['tab_id', "tab2"],
               ]
    
    for name, default in to_init:
        if name not in st.session_state:
            st.session_state[name] = default
            
    # Init auto training database
    st.session_state.curriculum_manager = CurriculumManager(
        saved_curriculums_on_s3=dict(
            bucket='aind-behavior-data',
            root='foraging_auto_training/saved_curriculums/'
        ),
        saved_curriculums_local='/root/capsule/curriculum_manager/',
    )
    st.session_state.auto_train_manager = DynamicForagingAutoTrainManager(
        manager_name='447_demo',
        df_behavior_on_s3=dict(bucket='aind-behavior-data',
                                root='foraging_nwb_bonsai_processed/',
                                file_name='df_sessions.pkl'),
        df_manager_root_on_s3=dict(bucket='aind-behavior-data',
                                root='foraging_auto_training/')
    )
    
    
    st.session_state.draw_type_mapper_session_level = {'1. Choice history': ('choice_history',   # prefix
                                                            (0, 0),     # location (row_idx, column_idx)
                                                            dict()),
        
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
    st.session_state.df['sessions_bonsai'] = st.session_state.df['sessions_bonsai'].reset_index()
    st.session_state.df['sessions_bonsai']['h2o'] = st.session_state.df['sessions_bonsai']['subject_id']
    st.session_state.df['sessions_bonsai'].dropna(subset=['session'], inplace=True) # Remove rows with no session number (only leave the nwb file with the largest finished_trials for now)
    
    # # add something else
    # st.session_state.df['sessions_bonsai']['abs(bias)'] = np.abs(st.session_state.df['sessions_bonsai'].biasL)
    
    # # delta weight
    # diff_relative_weight_next_day = st.session_state.df['sessions_bonsai'].set_index(
    #     ['session']).sort_values('session', ascending=True).groupby('h2o').apply(
    #         lambda x: - x.relative_weight.diff(periods=-1)).rename("diff_relative_weight_next_day")
        
    # weekday
    # st.session_state.df['sessions_bonsai']['weekday'] =  st.session_state.df['sessions_bonsai'].session_date.dt.dayofweek + 1

    # st.session_state.df['sessions_bonsai'] = st.session_state.df['sessions_bonsai'].merge(
    #     diff_relative_weight_next_day, how='left', on=['h2o', 'session'])

    st.session_state.session_stats_names = [keys for keys in st.session_state.df['sessions_bonsai'].keys()]
   
   
    

def app():
    add_caution()
    
    st.markdown('## 🌳🪴 Foraging sessions from Bonsai 🌳🪴')
    st.markdown('##### (still using a temporary workaround until AIND behavior metadata and pipeline are set up)')

    with st.sidebar:
        add_session_filter(if_bonsai=True)
        data_selector()
    
        st.markdown('---')
        st.markdown('#### Han Hou @ 2023 v1.0.2')
        st.markdown('[bug report / feature request](https://github.com/AllenNeuralDynamics/foraging-behavior-browser/issues)')
        
        with st.expander('Debug', expanded=False):
            if st.button('Reload data from AWS S3'):
                st.cache_data.clear()
                init()
                st.experimental_rerun()
        
    

    with st.container():
        # col1, col2 = st.columns([1.5, 1], gap='small')
        # with col1:
        # -- 1. unit dataframe --
        
        cols = st.columns([2, 2, 2])
        cols[0].markdown(f'### Filter the sessions on the sidebar ({len(st.session_state.df_session_filtered)} filtered)')
        # if cols[1].button('Press this and then Ctrl + R to reload from S3'):
        #     st.experimental_rerun()
        if cols[1].button('Reload data '):
            st.cache_data.clear()
            init()
            st.experimental_rerun()
    
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
        # if st.session_state.tab_id == "tab1":
        st.experimental_rerun()
            
    chosen_id = stx.tab_bar(data=[
        stx.TabBarItemData(id="tab2", title="👀 Session Inspector", description="Select sessions from the table and show plots"),
        stx.TabBarItemData(id="tab1", title="📈 Session X-Y plot", description="Interactive session-wise scatter plot"),
        stx.TabBarItemData(id="tab3", title="🐭 Mouse Inspector", description="Mouse-level summary"),
        stx.TabBarItemData(id="tab4", title="🎓 Automatic Training History", description="Track progress"),
        stx.TabBarItemData(id="tab5", title="📚 Automatic Training Curriculums", description="Collection of curriculums"),
        ], default="tab2" if 'tab_id' not in st.session_state else st.session_state.tab_id)
    # chosen_id = "tab1"

    placeholder = st.container()

    if chosen_id == "tab1":
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
                st.experimental_rerun()
            
        # Add debug info
        with st.expander('NWB errors', expanded=False):
            with fs.open(cache_folder + 'error_files.json') as file:
                st.json(json.load(file))
                
        with st.expander('Pipeline log', expanded=False):
            with fs.open(cache_folder + 'pipeline.log') as file:
                log_content = file.read().decode('utf-8')
            log_content = log_content.replace('\\n', '\n')
            st.text(log_content)
        
    elif chosen_id == "tab2":
        st.session_state.tab_id = chosen_id
        with placeholder:
            with st.columns([4, 10])[0]:
                if_draw_all_sessions = session_plot_settings(need_click=False)
                df_to_draw_sessions = st.session_state.df_selected_from_plotly if 'selected' in st.session_state.selected_draw_sessions else st.session_state.df_session_filtered
                
            if if_draw_all_sessions and len(df_to_draw_sessions):
                draw_session_plots(df_to_draw_sessions)
                
    elif chosen_id == "tab3":
        st.session_state.tab_id = chosen_id
        with placeholder:
            selected_subject_id = st.columns([1, 3])[0].selectbox('Select a mouse', options=st.session_state.df_session_filtered['subject_id'].unique())
            st.markdown(f"### [Go to WaterLog](http://eng-tools:8004/water_weight_log/?external_donor_name={selected_subject_id})")
            
    elif chosen_id == "tab4":  # Automatic training history
        st.session_state.tab_id = chosen_id
        with placeholder:
            df_training_manager = st.session_state.auto_train_manager.df_manager
            st.write(df_training_manager)

    elif chosen_id == "tab5":  # Automatic training curriculums
        st.session_state.tab_id = chosen_id
        df_curriculums = st.session_state.curriculum_manager.df_curriculums().sort_values(by='curriculum_name')     
        with placeholder:
            # Show curriculum manager dataframe
            st.markdown("#### Available auto training curriculums")
            cols = st.columns([1, 1])
            with cols[0]:
                aggrid_curriculum_outputs = aggrid_interactive_table_curriculum(df=df_curriculums)
                
            # Get selected curriculum
            selected_row = aggrid_curriculum_outputs['selected_rows'][0]
            selected_curriculum = st.session_state.curriculum_manager.get_curriculum(
                curriculum_name=selected_row['curriculum_name'],
                curriculum_schema_version=selected_row['curriculum_schema_version'],
                curriculum_version=selected_row['curriculum_version'],
                )
            curriculum = selected_curriculum['curriculum']
        
            # Show diagrams
            cols = st.columns([1, 1.5])
            with cols[0]:
                st.graphviz_chart(curriculum.diagram_rules(render_file_format=''),
                                use_container_width=True)
            with cols[1]:
                st.graphviz_chart(curriculum.diagram_paras(render_file_format=''),
                                use_container_width=True)
                

    # Add debug info
    if chosen_id != "tab5":
        with st.expander('NWB errors', expanded=False):
            error_file = cache_folder + 'error_files.json'
            if fs.exists(error_file):
                with fs.open(error_file) as file:
                    st.json(json.load(file))
            else:
                st.write('No NWB error files')
                
        with st.expander('Pipeline log', expanded=False):
            with fs.open(cache_folder + 'pipeline.log') as file:
                log_content = file.read().decode('utf-8')
            log_content = log_content.replace('\\n', '\n')
            st.text(log_content)
    
    # st.dataframe(st.session_state.df_session_filtered, use_container_width=True, height=1000)


if 'df' not in st.session_state or 'sessions_bonsai' not in st.session_state.df.keys(): 
    init()
    
app()

