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

    if st.session_state.use_s3:
        with fs.open(file[0]) as f:
            img = Image.open(f)
            img = img.crop(crop) 
    else:
        img = Image.open(file[0])
        img = img.crop(crop)         
    
    return img, file[0]


# @st.cache_data(ttl=24*3600, max_entries=20)
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

    
def draw_session_plots(keys_to_draw_session):
    
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
        
        if len(keys_to_draw_session):
            st.write(f'Loading selected {len(keys_to_draw_session)} sessions...')
            my_bar = st.columns((1, 7))[0].progress(0)
             
            major_cols = st.columns([1] * num_cols)
            
            if not isinstance(keys_to_draw_session, list):  # Turn dataframe to list, if necessary
                keys_to_draw_session = keys_to_draw_session.to_dict(orient='records')

            for i, key in enumerate(keys_to_draw_session):
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
                    
                my_bar.progress(int((i + 1) / len(keys_to_draw_session) * 100))
                
                

def _plot_population_x_y(df, x_name='session', y_name='foraging_eff', group_by='h2o',
                         smooth_factor=5, if_average=True, if_show_dots=True, if_smooth=True,
                         title=''):
    
    fig = go.Figure()
    col_map = px.colors.qualitative.Plotly
    
    for i, group in enumerate(df.sort_values(group_by)[group_by].unique()):
        this_session = df.query(f'{group_by} == "{group}"').sort_values('session')
        x = this_session[x_name]
        y = this_session[y_name]
        
        if if_show_dots:
            fig.add_trace(go.Scattergl(
                            x=x, 
                            y=y, 
                            name=group,
                            legendgroup=f'group_{group}',
                            showlegend=not if_smooth,
                            mode="markers",
                            marker_color=col_map[i%len(col_map)],
                            opacity=0.5 if if_smooth else 1,
                            unselected=dict(marker_color='grey')
                            ))

        if if_smooth:
            fig.add_trace(go.Scatter(    
                        x=x, 
                        y=y.rolling(window=smooth_factor, center=True, min_periods=1).mean(), 
                        name=group,
                        legendgroup=f'group_{group}',
                        mode="lines",
                        marker_color=col_map[i%len(col_map)],
                        opacity=0.5 if if_average else 1,
                        hoveron='points+fills',   # Scattergl doesn't support this
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
            name='mean +/- sem',
            mode="lines",
        )
    )

    n_mice = len(df['h2o'].unique())
    n_sessions = len(df.groupby(['h2o', 'session']).count())
    
    fig.update_layout(width=1200, 
                    height=800,
                    xaxis_title=x_name,
                    yaxis_title=y_name,
                    # xaxis_range=[0, min(100, df[x_name].max())],
                    font=dict(size=15),
                    hovermode='closest',
                    legend={'traceorder':'reversed'},
                    title=f'{title}, {n_mice} mice, {n_sessions} sessions',
                    )
    
    # st.plotly_chart(fig)
    selected_sessions_from_plot = plotly_events(fig, click_event=True, hover_event=False, select_event=True, override_height=800)
        
    return selected_sessions_from_plot
                

def population_analysis():
    
    selected_keys = st.session_state.aggrid_outputs['selected_rows']
    
    # If no sessions are selected, use all filtered entries
    use_all_filtered = len(selected_keys) == 0  
    df_selected = pd.DataFrame(selected_keys) if not use_all_filtered else st.session_state.df_session_filtered
    # st.markdown(f'to do population: {len(df_selected)} sessions')
    
    cols = st.columns([5, 10])
    
    with cols[0]:
        x_name, y_name, group_by = add_xy_selector()
        
        s_cols = st.columns([1, 1])
        if_show_dots = s_cols[0].checkbox('Show data points', True)
        if_smooth = s_cols[0].checkbox('Show smoothed', True)
        smooth_factor = s_cols[0].slider('Smooth factor', 1, 20, 5, disabled=not if_smooth)
        if_average = s_cols[1].checkbox('Show population average', True)

        
    names = {('session', 'foraging_eff'): 'Foraging efficiency',
             ('session', 'finished'):   'Finished trials', 
             }

    df_selected_from_plotly = pd.DataFrame()
    # for i, (title, (x_name, y_name)) in enumerate(names.items()):
        # with cols[i]:
    with cols[1]:
        selected = _plot_population_x_y(df=df_selected, 
                                        x_name=x_name, y_name=y_name, 
                                        group_by=group_by,
                                        smooth_factor=smooth_factor, 
                                        if_show_dots=if_show_dots,
                                        if_average=if_average,
                                        if_smooth=if_smooth,
                                        title=names[(x_name, y_name)] if (x_name, y_name) in names else y_name)
    if len(selected):
        df_selected_from_plotly = df_selected.merge(pd.DataFrame(selected).rename({'x': x_name, 'y': y_name}, axis=1), 
                                                    on=[x_name, y_name], how='inner')

    return df_selected_from_plotly


def add_xy_selector():
    with st.expander("Select axes", expanded=True):
        # with st.form("axis_selection"):
        cols = st.columns([1, 1, 1])
        x_name = cols[0].selectbox("x axis", st.session_state.session_stats_names, index=st.session_state.session_stats_names.index('session'))
        y_name = cols[1].selectbox("y axis", st.session_state.session_stats_names, index=st.session_state.session_stats_names.index('foraging_eff'))
        group_by = cols[2].selectbox("grouped by", ['h2o', 'task'], index=['h2o', 'task'].index('h2o'))
            # st.form_submit_button("update axes")
    return x_name, y_name, group_by


# ------- Layout starts here -------- #    
def init():
    df = load_data(['sessions', 'logistic_regression', 'linear_regression_rt'])
    st.session_state.df = df
    
    st.session_state.df_selected_from_plotly = pd.DataFrame()
    st.session_state.session_stats_names = [keys for keys in st.session_state.df['sessions'].keys()]
    
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
            st.cache_data.clear()
            st.experimental_rerun()
             
        # aggrid_outputs = aggrid_interactive_table_units(df=df['ephys_units'])
        # st.session_state.df_session_filtered = aggrid_outputs['data']
        
        container_filtered_frame = st.container()
        
    with st.sidebar:
        add_session_filter()


        
    if len(st.session_state.df_session_filtered) == 0:
        st.markdown('## No filtered results!')
        return
    
    st.session_state.aggrid_outputs = aggrid_interactive_table_session(df=st.session_state.df_session_filtered)

    chosen_id = stx.tab_bar(data=[
        stx.TabBarItemData(id="tab1", title="📈Training summary", description="Plot training summary"),
        stx.TabBarItemData(id="tab2", title="📚Session inspection", description="Generate plots for each session"),
        ], default="tab1")

    placeholder = st.container()

    if chosen_id == "tab1":
        with placeholder:
            df_selected_from_plotly = population_analysis()
            st.markdown('##### Select session(s) from the plots above to draw')

            if len(st.session_state.df_selected_from_plotly):
                draw_session_plots(st.session_state.df_selected_from_plotly)
                
            if len(df_selected_from_plotly) and not df_selected_from_plotly.equals(st.session_state.df_selected_from_plotly):
                st.session_state.df_selected_from_plotly = df_selected_from_plotly
                st.experimental_rerun()
            
    elif chosen_id == "tab2":
        with placeholder:
            selected_keys_from_aggrid = st.session_state.aggrid_outputs['selected_rows']
            st.markdown('##### Select session(s) from the table above to draw')
            draw_session_plots(selected_keys_from_aggrid)

    # st.dataframe(st.session_state.df_session_filtered, use_container_width=True, height=1000)


if 'df' not in st.session_state: 
    init()
    
app()

            
if if_profile:    
    p.stop()