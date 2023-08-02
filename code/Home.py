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

from PIL import Image, ImageColor
import streamlit.components.v1 as components
import streamlit_nested_layout
from streamlit_plotly_events import plotly_events

from util.streamlit import filter_dataframe, aggrid_interactive_table_session, add_session_filter, data_selector
import extra_streamlit_components as stx

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
                            marker_size=10,
                            marker_color=this_session['colors'],
                            opacity=0.2 if if_aggr_each_group else 0.7,
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
        q_quantiles_all = s_cols[2].slider('Number of quantiles', 1, 100, 20, disabled=not if_use_x_quantile_all) if if_use_x_quantile_all else None

        smooth_factor = s_cols[0].slider('Smooth factor', 1, 20, 5) if ((if_aggr_each_group and aggr_method_group in ('running average', 'lowess'))
                                                                     or (if_aggr_all and aggr_method_all in ('running average', 'lowess'))) else None
        
    
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
                                        states = st.session_state.df_selected_from_plotly)
        
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
        group_by = cols[2].selectbox("grouped by", ['h2o', 'task', 'photostim_location', 'weekday',
                                                    'headbar', 'user_name', 'sex', 'rig'], index=['h2o', 'task'].index('h2o'))
            # st.form_submit_button("update axes")
    return x_name, y_name, group_by


# ------- Layout starts here -------- #    
def init():
    
    # Clear Session state
    for key in ['selected_draw_types']:
        if key in st.session_state:
            del st.session_state[key]

    
    df = load_data(['sessions', 
                    'logistic_regression_hattori', 
                    'logistic_regression_su',
                    'linear_regression_rt',
                    'model_fitting_params'])
    
    # Try to convert datetimes into a standard format (datetime, no timezone)
    df['sessions']['session_date'] = pd.to_datetime(df['sessions']['session_date'])
    # if is_datetime64_any_dtype(df[col]):
    df['sessions']['session_date'] = df['sessions']['session_date'].dt.tz_localize(None)
    
    st.session_state.df = df
    st.session_state.df_selected_from_plotly = pd.DataFrame(columns=['h2o', 'session'])
    st.session_state.df_selected_from_dataframe = pd.DataFrame(columns=['h2o', 'session'])
    
    # Init session states
    to_init = [
               ['model_id', 21],   # add some model fitting params to session
               ['tab_id', "tab2"],
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
        st.markdown('#### Han Hou @ 2023 v1.0.2')
        st.markdown('[bug report / feature request](https://github.com/AllenNeuralDynamics/foraging-behavior-browser/issues)')
        
        with st.expander('Debug', expanded=False):
            st.session_state.model_id = st.selectbox('model_id', st.session_state.df['model_fitting_params'].model_id.unique())
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
        stx.TabBarItemData(id="tab3", title="🐭 Mouse Model Fitting", description="Mouse-level model fitting results"),
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
            with st.columns([4, 10])[0]:
                if_draw_all_mice = mouse_plot_settings(need_click=False)
                df_selected = st.session_state.df_selected_from_plotly if 'selected' in st.session_state.selected_draw_mice else st.session_state.df_session_filtered
                df_to_draw_mice = df_selected.groupby('h2o').count().reset_index()
                
            if if_draw_all_mice and len(df_to_draw_mice):
                draw_mice_plots(df_to_draw_mice)
        
    

    # st.dataframe(st.session_state.df_session_filtered, use_container_width=True, height=1000)


if 'df' not in st.session_state or 'sessions' not in st.session_state.df.keys(): 
    init()
    
app()

            
if if_profile:    
    p.stop()