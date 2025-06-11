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
from aind_analysis_arch_result_access.han_pipeline import get_session_table
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
from util.aws_s3 import (draw_session_plots_quick_preview, 
                         load_data,
                         load_auto_train,
                         load_mouse_PI_mapping,
                         show_debug_info,
                         show_session_level_img_by_key_and_prefix)
from util.fetch_data_docDB import load_data_from_docDB
from util.settings import (draw_type_layout_definition,
                           draw_type_mapper_session_level)
from util.streamlit import (_plot_population_x_y, add_auto_train_manager,
                            add_dot_property_mapper, add_session_filter,
                            add_xy_selector, add_xy_setting,
                            aggrid_interactive_table_basic,
                            aggrid_interactive_table_session, data_selector,
                            add_footnote,
                            add_download_plotly_as_svg)
from util.url_query_helper import (checkbox_wrapper_for_url_query,
                                   multiselect_wrapper_for_url_query,
                                   selectbox_wrapper_for_url_query,
                                   number_input_wrapper_for_url_query,
                                   slider_wrapper_for_url_query,
                                   sync_session_state_to_URL,
                                   sync_URL_to_session_state)
from util.reformat import get_data_source

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


def _trainer_mapper(trainer):
    user_mapper = {
        'Avalon Amaya': ['Avalon'],
        'Ella Hilton': ['Ella'],
        'Katrina Nguyen': ['Katrina'],
        'Lucas Kinsey': ['Lucas'],
        'Travis Ramirez': ['Travis'],
        'Xinxin Yin': ['Xinxin', 'the ghost'],
        'Bowen Tan': ['Bowen'],
        'Henry Loeffler': ['Henry Loeffer'],
        'Margaret Lee': ['margaret lee'],
        'Madeline Tom': ['Madseline Tom'],
    }
    for canonical_name, alias in user_mapper.items():
        for key_word in alias:
            if key_word in trainer:
                return canonical_name
    else:
        return trainer


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
                    
                    st.markdown(f'''<h6 style='text-align: center; color: orange;'>{key["subject_id"]} ({key["PI"]}), {date_str}, Session {int(key["session"])}<br>'''
                                f'''{key["trainer"]} @ {key["rig"]} ({key["data_source"]})''',
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


def session_plot_settings(df_selected_from_plotly=None, need_click=True):
    with st.form(key='session_plot_settings'):
        st.markdown('##### Show plots for individual sessions ')
        cols = st.columns([2, 6, 1])

        session_plot_modes = [f'sessions selected from table or plot', f'all sessions filtered from sidebar']
        st.session_state.selected_draw_sessions = selectbox_wrapper_for_url_query(
            cols[0],
            label='Which session(s) to draw?',
            options=session_plot_modes,
            default=session_plot_modes[0],
            key='session_plot_mode',
        )
        
        if "selected" in st.session_state.selected_draw_sessions:
            if df_selected_from_plotly is None:  # Selected from dataframe
                df_to_draw_sessions = st.session_state.df_selected_from_dataframe
            else:
                df_to_draw_sessions = df_selected_from_plotly
        else:  # all sessions filtered from sidebar
            df_to_draw_sessions = st.session_state.df_session_filtered

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

        cols[0].markdown(f'{len(df_to_draw_sessions)} sessions to draw')
        draw_it_now_override = cols[2].checkbox('Auto show', value=not need_click, disabled=not need_click)
        submitted = cols[0].form_submit_button(
            "Update settings", type="primary"
        )

    if not need_click:
        return True, df_to_draw_sessions

    if draw_it_now_override:
        return True, df_to_draw_sessions

    draw_it = st.button(f'Show {len(df_to_draw_sessions)} sessions!', use_container_width=False, type="primary")
    return draw_it, df_to_draw_sessions


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

        selected = st.plotly_chart(fig, 
                                   key='x_y_plot',
                                   on_select="rerun",
                                   use_container_width=False,
                                   theme=None,  # full controlled by plotly chart itself
                        )
        add_download_plotly_as_svg(fig, 'x_y_plot.svg')

    with cols[1]:
        st.markdown('#### 👀 Quick preview')
        st.markdown('###### Click on one session to preview here, or Box/Lasso select multiple sessions to draw them in the section below')
        st.markdown('(sometimes you have to click twice...)')

    if len(selected.selection.points):  # Selected this time
        df_key_selected = pd.DataFrame(
            [data["customdata"][:2] for data in selected.selection.points],
            columns=["subject_id", "session_date"],
        )
        df_key_selected["session_date"] = pd.to_datetime(df_key_selected["session_date"])
        df_selected_from_plotly = df_x_y_session.merge(df_key_selected, on=["subject_id", "session_date"], how='inner')
        
        # Update session state
        st.session_state.df_selected_from_plotly = df_selected_from_plotly
        
    if len(df_selected_from_plotly) == 1:
        with cols[1]:
            draw_session_plots_quick_preview(df_selected_from_plotly)
    return df_selected_from_plotly, cols


def show_curriculums():
    pass


# ------- Layout starts here -------- #
def init(if_load_bpod_data_override=None, if_load_docDB_override=None, if_load_sessions_older_than_6_month_override=None):

    # Clear specific session state and all filters
    for key in st.session_state:
        if key in ['selected_draw_types'] or '_changed' in key:
            del st.session_state[key]

    # Because sync_URL_to_session_state() needs df to be loaded (for dynamic column filtering),
    # 'if_load_bpod_sessions' has not been synced from URL to session state yet.
    # So here we need to manually get it from URL or session state.
    _if_load_bpod = if_load_bpod_data_override if if_load_bpod_data_override is not None else (
        st.query_params['if_load_bpod_sessions'].lower() == 'true'
        if 'if_load_bpod_sessions' in st.query_params
        else st.session_state.if_load_bpod_sessions 
        if 'if_load_bpod_sessions' in st.session_state
        else False)
    st.session_state.bpod_loaded = _if_load_bpod

    _if_load_sessions_older_than_6_month = (
        if_load_sessions_older_than_6_month_override
        if if_load_sessions_older_than_6_month_override is not None
        else (
            st.query_params["if_load_sessions_older_than_6_month"].lower() == "true"
            if "if_load_sessions_older_than_6_month" in st.query_params
            else (
                st.session_state.if_load_sessions_older_than_6_month
                if "if_load_sessions_older_than_6_month" in st.session_state
                else False
            )
        )
    )

    # --- Load data using aind-analysis-arch-result-access ---
    # Convert boolean to months: if True, load all sessions (None), if False, load only recent 6 months
    only_recent_n_month = None if _if_load_sessions_older_than_6_month else 6
    df_han = get_session_table(if_load_bpod=_if_load_bpod, only_recent_n_month=only_recent_n_month)
    df = {'sessions_main': df_han}  # put it in df['session_main'] for backward compatibility

    if not len(df):
        return False

    st.session_state.df = df
    for source in ["dataframe", "plotly"]:
        st.session_state[f'df_selected_from_{source}'] = pd.DataFrame(columns=['subject_id', 'session'])

    # Load autotrain
    _, curriculum_manager = load_auto_train()
    st.session_state.curriculum_manager = curriculum_manager

    # Some ad-hoc modifications on df_sessions
    _df = st.session_state.df['sessions_main'].copy()

    # --- Load data from docDB ---
    if_load_docDb = if_load_docDB_override if if_load_docDB_override is not None else (
        st.query_params['if_load_docDB'].lower() == 'true'
        if 'if_load_docDB' in st.query_params
        else st.session_state.if_load_docDB 
        if 'if_load_docDB' in st.session_state
        else False)

    if if_load_docDb:
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

    st.session_state.df['sessions_main'] = _df  # Somehow _df loses the reference to the original dataframe
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
        
        
    with st.container():
        # col1, col2 = st.columns([1.5, 1], gap='small')
        # with col1:
        # -- 1. unit dataframe --
        
        cols = st.columns([4, 4, 4, 1])
        
        with cols[0].expander(':bulb: Get the master session table by code', expanded=False):
            st.code(f'''
                    #!pip install aind-analysis-arch-result-access
                    from aind_analysis_arch_result_access.han_pipeline import get_session_table
                    df = get_session_table(if_load_bpod={st.session_state.bpod_loaded})
                    ''')
            st.markdown("See [aind-analysis-arch-result-access](https://github.com/AllenNeuralDynamics/aind-analysis-arch-result-access) for details.")
        
        with cols[1]:
            with st.form(key='load_settings', clear_on_submit=False):
                if_load_sessions_older_than_6_month = checkbox_wrapper_for_url_query(
                    st_prefix=st,
                    label='Include sessions older than 6 months (reload after change)',
                    key='if_load_sessions_older_than_6_month',
                    default=False,
                )
                if_load_bpod_sessions = checkbox_wrapper_for_url_query(
                    st_prefix=st,
                    label='Include old Bpod sessions (reload after change)',
                    key='if_load_bpod_sessions',
                    default=False,
                )
                if_load_docDB = checkbox_wrapper_for_url_query(
                    st_prefix=st,
                    label='Load metadata from docDB (reload after change)',
                    key='if_load_docDB',
                    default=False,
                )
                                                                    
                submitted = st.form_submit_button("Reload data! 🔄", type='primary')
                if submitted:
                    st.cache_data.clear()
                    sync_session_state_to_URL()
                    init()
                    st.rerun()  # Reload the page to apply the changes
                    
        cols[0].markdown(f'### Filter the sessions on the sidebar\n' +
                    f'#####  {len(st.session_state.df_session_filtered)} sessions, ' +
                    f'{len(st.session_state.df_session_filtered.subject_id.unique())} mice filtered' +
                    (f' (recent 6 months only)' if not st.session_state.if_load_sessions_older_than_6_month else '')
                    )
              
        table_height = slider_wrapper_for_url_query(st_prefix=cols[-1],
                                                    label='Table height',
                                                    min_value=0,
                                                    max_value=2000,
                                                    default=300,
                                                    step=50,
                                                    key='table_height',
        )
        
        container_filtered_frame = st.container()

        
    if len(st.session_state.df_session_filtered) == 0:
        st.markdown('## No filtered results! :thinking_face:')
        st.markdown('### :bulb: Try clicking "Reset filters" and add your filters again!')
        return

    aggrid_outputs = aggrid_interactive_table_session(
        df=st.session_state.df_session_filtered.round(3),
        table_height=table_height,
    )

    if len(aggrid_outputs['selected_rows']) \
        and not set(pd.DataFrame(aggrid_outputs['selected_rows']).set_index(['subject_id', 'session']).index
            ) == set(st.session_state.df_selected_from_dataframe.set_index(['subject_id', 'session']).index) \
        and not st.session_state.get("df_selected_from_dataframe_just_overriden", False):  # so that if the user just overriden the df_selected_from_dataframe by pressing sidebar button, it won't sync selected rows in the table to session state
        st.session_state.df_selected_from_dataframe = pd.DataFrame(aggrid_outputs['selected_rows'])  # Use selected in dataframe to update "selected"
        st.rerun()
        
    st.session_state["df_selected_from_dataframe_just_overriden"] = False  # Reset the flag anyway

    add_main_tabs()

@st.fragment
def add_main_tabs():
    chosen_id = stx.tab_bar(data=[
        stx.TabBarItemData(id="tab_auto_train_history", title="🎓 Automatic Training History", description="Track progress"),
        stx.TabBarItemData(id="tab_session_inspector", title="👀 Session Inspector (table)", description="Select sessions from the table and show figures"),
        stx.TabBarItemData(id="tab_session_x_y", title="📈 Session X-Y plot", description="Select sessions from x-y plot and show figures"),
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
            with st.columns([1])[0]:
                st.markdown("***")
                if_draw_all_sessions, df_to_draw_sessions = session_plot_settings(df_selected_from_plotly=df_selected_from_plotly, need_click=True)

            if if_draw_all_sessions and len(df_to_draw_sessions):
                draw_session_plots(df_to_draw_sessions)

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

            pygwalker_renderer.render_explore()

    elif chosen_id == "tab_session_inspector":
        with placeholder:
            cols = st.columns([1])
            with cols[0]:
                if_draw_all_sessions, df_to_draw_sessions = session_plot_settings(
                    df_selected_from_plotly=None, need_click=False
                )

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
            by=['curriculum_version', 'curriculum_schema_version', 'curriculum_name'],
            ascending=[False, True, False], 
            ).reset_index().drop(columns='index').query("curriculum_name != 'Dummy task'")

        with placeholder:
            # Show curriculum manager dataframe
            st.markdown("#### Select auto training curriculums")

            # Curriculum drop down selector
            cols = st.columns([0.8, 0.8, 0.8, 3])
            cols[3].markdown(f"(aind_auto_train lib version = {auto_train_version})")

            options = list(df_curriculums['curriculum_name'].unique())
            selected_curriculum_name = selectbox_wrapper_for_url_query(
                st_prefix=cols[0],
                label='Curriculum name',
                options=options,
                default=options[0],
                default_override=True,
                key='auto_training_curriculum_name',
            )

            options = list(df_curriculums[
                df_curriculums['curriculum_name'] == selected_curriculum_name
                ]['curriculum_version'].unique())
            selected_curriculum_version = selectbox_wrapper_for_url_query(
                st_prefix=cols[1],
                label='Curriculum version',
                options=options,
                default=options[0],
                default_override=True,
                key='auto_training_curriculum_version',
            )

            options = list(df_curriculums[
                (df_curriculums['curriculum_name'] == selected_curriculum_name) 
                & (df_curriculums['curriculum_version'] == selected_curriculum_version)
                ]['curriculum_schema_version'].unique())

            selected_curriculum_schema_version = selectbox_wrapper_for_url_query(
                st_prefix=cols[2],
                label='Curriculum schema version',
                options=options,
                default=options[0],
                default_override=True,
                key='auto_training_curriculum_schema_version',
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
            aggrid_interactive_table_basic(df=df_curriculums,
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
    try:
        ok = True
        if 'df' not in st.session_state or 'sessions_main' not in st.session_state.df.keys(): 
            ok = init()

        if ok:
            app()
            pass
    except Exception as e:
        st.markdown('# Something went wrong! :scream: ')
        st.markdown('## :bulb: Please follow these steps to troubleshoot:')
        st.markdown('####  1. Reload the page')
        st.markdown('####  2. Click this original URL https://foraging-behavior-browser.allenneuraldynamics-test.org/')
        st.markdown('####  3. Report your bug here: https://github.com/AllenNeuralDynamics/foraging-behavior-browser/issues (paste your URL and screenshoots)')
        raise e
