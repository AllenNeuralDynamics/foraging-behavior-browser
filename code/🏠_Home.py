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
import plotly.graph_objects as go

from PIL import Image, ImageColor
import streamlit.components.v1 as components
import streamlit_nested_layout
from streamlit_plotly_events import plotly_events

from streamlit_util import filter_dataframe, aggrid_interactive_table_session, add_session_filter
import extra_streamlit_components as stx

if_profile = False

if if_profile:
    from streamlit_profiler import Profiler
    p = Profiler()
    p.start()


# from pipeline import experiment, ephys, lab, psth_foraging, report, foraging_analysis
# from pipeline.plot import foraging_model_plot

cache_folder = 'xxx'  #'/root/capsule/data/s3/report/st_cache/'
cache_fig_folder = 'xxx' #'/root/capsule/data/s3/report/all_units/'  # 

if os.path.exists(cache_folder):
    st.session_state.st.session_state.use_s3 = False
else:
    cache_folder = 'aind-behavior-data/Han/ephys/report/st_cache/'
    cache_fig_folder = 'aind-behavior-data/Han/ephys/report/all_sessions/'
    
    fs = s3fs.S3FileSystem(anon=False)
    st.session_state.use_s3 = True

st.set_page_config(layout="wide", 
                   page_title='Foraging behavior browser',
                   page_icon=':mouse2:',
                    menu_items={
                    'Report a bug': "https://github.com/hanhou/foraging-behavior-browser/issues",
                    'About': "Github repo: https://github.com/hanhou/foraging-behavior-browser/"
                    }
                   )

if 'selected_points' not in st.session_state:
    st.session_state['selected_points'] = []

    
@st.experimental_memo(ttl=24*3600)
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

    if st.session_state.use_s3:
        with fs.open(file[0]) as f:
            img = Image.open(f)
            img = img.crop(crop) 
    else:
        img = Image.open(file[0])
        img = img.crop(crop)         
    
    return img, file[0]


# @st.experimental_memo(ttl=24*3600, max_entries=20)
def show_img_by_key_and_prefix(key, prefix, column=None, other_patterns=[''], crop=None, caption=True, **kwargs):
    sess_date_str = datetime.strftime(datetime.strptime(key['session_date'], '%Y-%m-%dT%H:%M:%S'), '%Y%m%d')
     
    fns = [f'/{key["h2o"]}_{sess_date_str}_*{other_pattern}*' for other_pattern in other_patterns]
    glob_patterns = [cache_fig_folder + f'{prefix}/' + key["h2o"] + fn for fn in fns]
    
    img, f_name = _fetch_img(glob_patterns, crop)

    _f = st if column is None else column
    
    _f.image(img if img is not None else "https://cdn-icons-png.flaticon.com/512/3585/3585596.png", 
                output_format='PNG', 
                caption=f_name.split('/')[-1] if caption and f_name else '',
                use_column_width='always',
                **kwargs)

    return img


# table_mapping = {
#     'sessions': fetch_sessions,
#     'ephys_units': fetch_ephys_units,
# }

    
def draw_session_plots():
    st.markdown('##### Select session(s) from the table above to draw')
    
    # Setting up layout for each session
    layout_definition = [[1],   # columns in the first row
                         [1.5, 1],  # columns in the second row
                         ]  
    
    draw_type_mapper = {'1. Choice history': ('fitted_choice',   # prefix
                                           (0, 0),     # location (row_idx, column_idx)
                                           dict(other_patterns=['model_best', 'model_None'])),
                        '2. Lick times': ('lick_psth', 
                                       (1, 0), 
                                       {}),            
                        '3. Logistic regression on choice': ('logistic_regression', 
                                                          (1, 1), 
                                                          dict(crop=(0, 0, 1200, 2000))),
                        '4. Win-stay-lose-shift prob.': ('wsls', 
                                                      (1, 1), 
                                                      dict(crop=(0, 0, 1200, 600))),
                        '5. Linear regression on RT': ('linear_regression_rt', 
                                                      (1, 0), 
                                                      dict()),
                        }
    
    cols_option = st.columns([3, 0.5, 1])
    selected_draw_types = cols_option[0].multiselect('Which plot(s) to draw?', draw_type_mapper.keys(), default=draw_type_mapper.keys())
    num_cols = cols_option[1].number_input('Number of columns', 1, 10, 2)
    container_session_all_in_one = st.container()
    
    with container_session_all_in_one:
        # with st.expander("Expand to see all-in-one plot for selected unit", expanded=True):

        selected_keys = st.session_state.aggrid_outputs['selected_rows']
        
        if len(selected_keys):
            st.write(f'Loading selected {len(selected_keys)} sessions...')
            my_bar = st.columns((1, 7))[0].progress(0)
             
            major_cols = st.columns([1] * num_cols)

            for i, key in enumerate(selected_keys):
                this_major_col = major_cols[i % num_cols]
                
                # setting up layout for each session
                rows = []
                with this_major_col:
                    st.markdown(f'''<h3 style='text-align: center; color: blue;'>{key["h2o"]}, Session {key["session"]}, {key["session_date"].split("T")[0]}''',
                              unsafe_allow_html=True)
                    if len(selected_draw_types) > 1:  # more than one types, use the pre-defined layout
                        for row, column_setting in enumerate(layout_definition):
                            rows.append(this_major_col.columns(column_setting))
                    else:    # else, put it in the whole column
                        rows = this_major_col.columns([1])
                    st.markdown("---")

                for draw_type in draw_type_mapper:
                    if draw_type not in selected_draw_types: continue  # To keep the draw order defined by draw_type_mapper
                    prefix, position, setting = draw_type_mapper[draw_type]
                    this_col = rows[position[0]][position[1]] if len(selected_draw_types) > 1 else rows[0]
                    show_img_by_key_and_prefix(key, 
                                                column=this_col,
                                                prefix=prefix, 
                                                **setting)
                    
                my_bar.progress(int((i + 1) / len(selected_keys) * 100))
                
                

def _plot_population_x_y(df, x_name='session', y_name='foraging_eff', 
                         smooth_factor=5, if_average=True, if_show_dots=True):
    
    fig = go.Figure()
    col_map = px.colors.qualitative.Plotly
    
    for i, h2o in enumerate(df.sort_values('h2o')['h2o'].unique()):
        this_session = df.query(f'h2o == "{h2o}"').sort_values('session')
        x = this_session[x_name]
        y = this_session[y_name]
        
        if if_show_dots:
            fig.add_trace(go.Scattergl(
                            x=x, 
                            y=y, 
                            name=h2o,
                            legendgroup=f'group_{h2o}',
                            showlegend=False,
                            mode="markers",
                            marker_color=col_map[i%len(col_map)],
                            opacity=0.5,
                            ))

        fig.add_trace(go.Scatter(    
                        x=x, 
                        y=y.rolling(window=smooth_factor, center=True, min_periods=1).mean(), 
                        name=h2o,
                        legendgroup=f'group_{h2o}',
                        mode="lines",
                        marker_color=col_map[i%len(col_map)],
                        opacity=0.5 if if_average else 1,
                        hoveron = 'points+fills',   # Scattergl doesn't support this
                        ))        

    if if_average:
        mean = df.groupby(x_name)[y_name].mean()
        sem = df.groupby(x_name)[y_name].sem()
        fig.add_trace(go.Scattergl(
            x=mean.index,
            y=mean,
            error_y=dict(type='data',
                         symmetric=True,
                         array=sem),
            name='mean +/- sem'
        )
    )

    fig.update_layout(width=1000, 
                    height=700,
                    xaxis_title=x_name,
                    yaxis_title=y_name,
                    xaxis_range=[0, min(100, df.session.max())],
                    font=dict(size=15),
                    hovermode='closest',
                    legend={'traceorder':'reversed'}
                    )
    
    # st.plotly_chart(fig)
    selected_sessions_from_plot = plotly_events(fig, click_event=True, hover_event=False, select_event=True, override_height=700)
        
    return selected_sessions_from_plot
                

def population_analysis():
    
    selected_keys = st.session_state.aggrid_outputs['selected_rows']
    
    # If no sessions are selected, use all filtered entries
    use_all_filtered = len(selected_keys) == 0  
    df_selected = pd.DataFrame(selected_keys) if not use_all_filtered else st.session_state.df_session_filtered
    st.markdown(f'to do population: {len(df_selected)} sessions')
    
    cols = st.columns([1, 1, 6])
    smooth_factor = cols[0].slider('Smooth factor', 1, 20, 5)
    if_show_dots = cols[1].checkbox('Show data points', True)
    if_average = cols[1].checkbox('Show population average', True)
    
    cols = st.columns([1, 1])
    with cols[0]:
       _plot_population_x_y(df=df_selected, x_name='session', y_name='foraging_eff', 
                            smooth_factor=smooth_factor, 
                            if_show_dots=if_show_dots,
                            if_average=if_average)
    with cols[1]:
        _plot_population_x_y(df=df_selected, x_name='session', y_name='finished', 
                             smooth_factor=smooth_factor, 
                             if_show_dots=if_show_dots,
                             if_average=if_average)
       


# ------- Layout starts here -------- #    
def init():
    df = load_data(['sessions', 'logistic_regression'])
    st.session_state.df = df
    
    # Some global variables
    

def app():
    st.markdown('## Foraging Behavior Browser')
       
    with st.container():
        # col1, col2 = st.columns([1.5, 1], gap='small')
        # with col1:
        # -- 1. unit dataframe --
        
        cols = st.columns([1, 2, 2])
        cols[0].markdown(f'### Filtered sessions')
        if cols[1].button('Press this and then Ctrl + R to reload from S3'):
            st.experimental_memo.clear()
            st.experimental_rerun()
             
        # aggrid_outputs = aggrid_interactive_table_units(df=df['ephys_units'])
        # st.session_state.df_session_filtered = aggrid_outputs['data']
        
        container_filtered_frame = st.container()
        
    with st.sidebar:
        add_session_filter()
        
    st.session_state.aggrid_outputs = aggrid_interactive_table_session(df=st.session_state.df_session_filtered)

    chosen_id = stx.tab_bar(data=[
        stx.TabBarItemData(id="tab1", title="📈Population plots", description="Do population analysis"),
        stx.TabBarItemData(id="tab2", title="📚Session plots", description="Generate plots for each session"),
        ], default="tab1")

    placeholder = st.container()
    

    if chosen_id == "tab1":
        with placeholder:
            selected_sessions_from_plot = population_analysis()
            selected_sessions_from_plot
    elif chosen_id == "tab2":
        with placeholder:
            draw_session_plots()

    # st.dataframe(st.session_state.df_session_filtered, use_container_width=True, height=1000)



if 'df' not in st.session_state: 
    init()
    
app()

            
if if_profile:    
    p.stop()