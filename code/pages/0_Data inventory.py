import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json

from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_dynamic_filters import DynamicFilters
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events

import time
import streamlit_nested_layout

from util.streamlit import aggrid_interactive_table_basic
from util.fetch_data_docDB import load_data_from_docDB, load_client
from util.aws_s3 import load_raw_sessions_on_VAST
from util.reformat import split_nwb_name
from Home import init


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

st.markdown(
"""
<style>
    .stMultiSelect [data-baseweb=select] span{
        max-width: 1000px;
    }
</style>""",
unsafe_allow_html=True,
)

client = load_client()

# Load QUERY_PRESET from json
with open("data_inventory_QUERY_PRESET.json", "r") as f:
    QUERY_PRESET = json.load(f)

meta_columns = [
    "Han_temp_pipeline (bpod)",
    "Han_temp_pipeline (bonsai)",
    "VAST_raw_data_on_VAST",
] + [query["alias"] for query in QUERY_PRESET]

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

def _formatting_metadata_df(df, source_prefix="docDB"):
    """Formatting metadata dataframe
    Given a dataframe with a column of "name" that contains nwb names
    1. parse the nwb names into subject_id, session_date, nwb_suffix.
    2. remove invalid subject_id
    3. handle multiple sessions per day
    """

    df.rename(columns={col: f"{source_prefix}_{col}" for col in df.columns}, inplace=True)
    new_name_field = f"{source_prefix}_name"

    # Create index of subject_id, session_date, nwb_suffix by parsing nwb_name
    df[["subject_id", "session_date", "nwb_suffix"]] = df[new_name_field].apply(
        lambda x: pd.Series(split_nwb_name(x))
    )
    df["session_date"] = pd.to_datetime(df["session_date"])
    df = df.set_index(["subject_id", "session_date", "nwb_suffix"]).sort_index(
        level=["session_date", "subject_id", "nwb_suffix"],
        ascending=[False, False, False],
    )

    # Remove invalid subject_id
    df = df[(df.index.get_level_values("subject_id").astype(int) > 300000) 
            & (df.index.get_level_values("subject_id").astype(int) < 999999)]

    # --- Handle multiple sessions per day ---
    # Build a dataframe with unique mouse-dates.
    # If multiple sessions per day, combine them into a list of 'name'
    df_unique_mouse_date = (
        df.reset_index()
        .groupby(["subject_id", "session_date"])
        .agg({new_name_field: list, **{col: "first" for col in df.columns if col != new_name_field}})
    ).sort_index(
        level=["session_date", "subject_id"], # Restore order 
        ascending=[False, False],
    )
    # Add a new column to indicate multiple sessions per day
    df_unique_mouse_date[f"{source_prefix}_multiple_sessions_per_day"] = df_unique_mouse_date[
        new_name_field
    ].apply(lambda x: len(x) > 1)

    # Also return the dataframe with multiple sessions per day
    df_multi_sessions_per_day = df_unique_mouse_date[
        df_unique_mouse_date[f"{source_prefix}_multiple_sessions_per_day"]
    ]

    # Create a new column to mark duplicates in the original df
    df.loc[
        df.index.droplevel("nwb_suffix").isin(df_multi_sessions_per_day.index), 
        f"{source_prefix}_multiple_sessions_per_day"
    ] = True

    return df, df_unique_mouse_date, df_multi_sessions_per_day

@st.cache_data(ttl=3600*24)
def fetch_single_query(query, pagination, paginate_batch_size):
    """ Fetch a query from docDB and process the result into dataframe
    
    Return: 
    - df: dataframe of the original returned records
    - df_multi_sessions_per_day: the dataframe with multiple sessions per day
    - df_unique_mouse_date: the dataframe with multiple sessions per day combined
    """

    # --- Fetch data from docDB ---
    results = client.retrieve_docdb_records(
        filter_query=query["filter"],
        projection={
            "_id": 0,
            "name": 1,
            "rig.rig_id": 1,
            "session.experimenter_full_name": 1,
        },
        paginate=pagination,
        paginate_batch_size=paginate_batch_size,
    )
    print(f"Done querying {query['alias']}!")

    # --- Process data into dataframe ---
    df = pd.json_normalize(results)
    
    # Formatting dataframe
    df, df_unique_mouse_date, df_multi_sessions_per_day = _formatting_metadata_df(df)
    df_unique_mouse_date[query["alias"]] = True  # Add a column to prepare for merging

    return df, df_unique_mouse_date, df_multi_sessions_per_day


def fetch_all_queries_from_docDB(queries_to_merge, parallel=False, pagination=False, paginate_batch_size=5000):
    """ Get merged queries from selected queries """

    dfs = {}

    # Fetch data in parallel
    if parallel:
        with st.spinner("Querying docDB in parallel..."):
            with ThreadPoolExecutor(max_workers=len(queries_to_merge)) as executor:
                future_to_query = {
                    executor.submit(
                        fetch_single_query,
                        key,
                        pagination=pagination,
                        paginate_batch_size=paginate_batch_size,
                    ): key
                    for key in queries_to_merge
                }
                for i, future in enumerate(as_completed(future_to_query), 1):
                    key = future_to_query[future]
                    try:
                        df, df_unique_mouse_date, df_multi_sessions_per_day = future.result()
                        dfs[key["alias"]] = { 
                            "df": df,
                            "df_unique_mouse_date": df_unique_mouse_date,
                            "df_multi_sessions_per_day": df_multi_sessions_per_day,
                        }
                    except Exception as e:
                        print(f"Error querying {key}: {e}")
    else:
        # Fetch data in serial
        p_bar = st.progress(0, text="Querying docDB in serial...")
        for i, query in enumerate(queries_to_merge):
            df, df_unique_mouse_date, df_multi_sessions_per_day = fetch_single_query(
                query, pagination=pagination, paginate_batch_size=paginate_batch_size
            )
            dfs[query["alias"]] = {
                "df": df,
                "df_unique_mouse_date": df_unique_mouse_date,
                "df_multi_sessions_per_day": df_multi_sessions_per_day,
            }
            p_bar.progress((i+1) / len(queries_to_merge), text=f"Querying docDB... ({i+1}/{len(queries_to_merge)})")

    if not dfs:
        st.warning("Querying docDB error! Refresh the page to try again or ask Han.")
        return None, None

    # Combine queried dfs using df_unique_mouse_date (on index "subject_id", "session_date" only)
    df_merged = dfs[queries_to_merge[0]["alias"]]["df_unique_mouse_date"]
    for df in [dfs[query["alias"]]["df_unique_mouse_date"] for query in queries_to_merge[1:]]:
        df_merged = df_merged.combine_first(df)  # Combine nwb_names

    # Recover the column order of QUERY_PRESET
    query_cols = [query["alias"] for query in queries_to_merge]
    df_merged = df_merged.reindex(
        columns=[
            other_col for other_col in df_merged.columns if other_col not in query_cols
        ]
        + query_cols
    )
    return df_merged, dfs

def _filter_df_by_patch_ids(df, patch_ids):
    """ Filter df by patch_ids [110, 001] etc"""
    # Turn NAN to False
    df = df.fillna(False)
    conditions = []
    for patch_id in patch_ids:
        # Convert patch_id string to boolean conditions
        condition = True
        for i, col in enumerate(df.columns):
            if patch_id[i] == "1":  # Include rows where the column is True
                condition &= df[col]
            elif patch_id[i] == "0":  # Include rows where the column is not True
                condition &= ~df[col] 
        conditions.append(condition)
    
    # Combine all conditions with OR
    final_condition = pd.concat(conditions, axis=1).any(axis=1)
    return df[final_condition].index

@st.cache_data(ttl=3600*24)
def generate_venn(df, venn_preset):
    """ Show venn diagram """
    circle_settings = venn_preset["circle_settings"]
    
    fig, ax = plt.subplots()
    if len(circle_settings) == 2:
        v_func = venn2
        c_func = venn2_circles
    elif len(circle_settings) == 3:
        v_func = venn3
        c_func = venn3_circles
    else:
        st.warning("Number of columns to venn should be 2 or 3.")
        return None, None
        
    v = v_func(
        [set(df.index[df[c_s["column"]]==True]) for c_s in circle_settings],
        set_labels=[c_s["column"] for c_s in circle_settings],
    )
    c = c_func(
        [set(df.index[df[c_s["column"]]==True]) for c_s in circle_settings],
    )
        
    # Set edge color and style
    for i, c_s in enumerate(circle_settings):
        edge_color = c_s.get("edge_color", "black")
        edge_style = c_s.get("edge_style", "solid")
        
        c[i].set_edgecolor(edge_color)
        c[i].set_linestyle(edge_style)
        v.get_label_by_id(["A", "B", "C"][i]).set_color(edge_color)
    
    # Clear all patch color
    for patch in v.patches:
        if patch:  # Some patches might be None
            patch.set_facecolor('none')

    notes = []
    for patch_setting in venn_preset["patch_settings"]:
        # Set color
        for patch_id in patch_setting["patch_ids"]:
            if v.get_patch_by_id(patch_id):
                v.get_patch_by_id(patch_id).set_color(patch_setting["color"])
        # Add notes
        notes.append(f"#### :{patch_setting['emoji']}: :{patch_setting['color']}[{patch_setting['notes']}]")

    return fig, notes

def _show_records_on_sidebar(dfs, file_name_prefix, source_str="docDB"):
    df = dfs["df"]
    df_multi_sessions_per_day = dfs["df_multi_sessions_per_day"]
    df_unique_mouse_date = dfs["df_unique_mouse_date"]

    with st.expander(f"{len(df)} records from {source_str}"):
        download_df(df, label="Download as CSV", file_name=f"{file_name_prefix}.csv")
        st.write(df)

    st.markdown(":heavy_exclamation_mark: :red[Multiple sessions per day should be resolved!]")
    with st.expander(f"{len(df_multi_sessions_per_day)} have multiple sessions per day"):
        download_df(
            df_multi_sessions_per_day,
            label="Download as CSV",
            file_name=f"{file_name_prefix}_multi_sessions_per_day.csv",
        )
        st.write(df_multi_sessions_per_day)

    with st.expander(f"{len(df_unique_mouse_date)} unique mouse-date pairs"):
        download_df(
            df_unique_mouse_date,
            label="Download as CSV",
            file_name=f"{file_name_prefix}_unique_mouse_date.csv",
        )
        st.write(df_unique_mouse_date)
        
    # if len(df_unique_mouse_date) != df_merged[query["alias"]].sum(): 
    #     st.warning('''len(df_unique_mouse_date) != df_merged[query["alias"]].sum()!''')

def add_sidebar(df_merged, dfs_docDB, df_Han_pipeline, dfs_raw_on_VAST, docDB_retrieve_time):
    # Sidebar
    with st.sidebar:
        for query in QUERY_PRESET:
            with st.expander(f"### {query['alias']}"):

                # Turn query to json with indent=4
                # with st.expander("Show docDB query"):
                query_json = json.dumps(query["filter"], indent=4)
                st.code(query_json)

                # Show records                
                _show_records_on_sidebar(dfs_docDB[query["alias"]], file_name_prefix=query["alias"], source_str="docDB")
        st.markdown('#### See [how to use above queries](https://aind-data-access-api.readthedocs.io/en/latest/UserGuide.html#document-database-docdb) in your own code.')

        st.markdown('''## 2. From Han's temporary pipeline (the "Home" page)''')
        hardwares = ["bonsai", "bpod"]
        for hardware in hardwares:
            df_this_hardware = df_Han_pipeline[
                df_Han_pipeline[f"Han_temp_pipeline ({hardware})"].notnull()
            ]
            with st.expander(
                f"### {len(df_this_hardware)} {hardware} sessions"
                + (" (old data, not growing)" if hardware == "bpod" else "")
            ):
                download_df(
                    df_this_hardware,
                    label="Download as CSV",
                    file_name=f"Han_temp_pipeline_{hardware}.csv",
                )
                st.write(df_this_hardware)
                
        st.markdown('''## 3. From VAST /scratch: existing raw data''')
        _show_records_on_sidebar(dfs_raw_on_VAST, file_name_prefix="raw_on_VAST", source_str="VAST /scratch")

@st.cache_data(ttl=3600*24)
def plot_histogram_over_time(df, venn_preset, time_period="Daily", if_sync_y_limits=True, if_separate_plots=False):
    """Generate histogram over time for the columns and patches in preset
    """
    df["Daily"] = df["session_date"]
    df["Weekly"] = df["session_date"].dt.to_period("W").dt.start_time
    df["Monthly"] = df["session_date"].dt.to_period("M").dt.start_time
    df["Quarterly"] = df["session_date"].dt.to_period("Q").dt.start_time

    # Function to count "True" values for a given column over a specific time period
    def count_true_values(df, time_period, column):
        return df.groupby(time_period)[column].apply(lambda x: (x == True).sum())

    # Preparing subplots for each circle/patch in venn
    columns = [c_s["column"] for c_s in venn_preset["circle_settings"]] + [
        str(p_s["patch_ids"]) for p_s in venn_preset.get("patch_settings", [])
        if not p_s.get("skip_timeline", False)
    ]
    colors = [c_s["edge_color"] for c_s in venn_preset["circle_settings"]] + [
        p_s["color"] for p_s in venn_preset["patch_settings"]
    ]
    if if_separate_plots:
        fig = make_subplots(
            rows=len(columns),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=columns,
        )

        # Adding traces for each column
        max_counts = 0
        for i, column in enumerate(columns):
            counts = count_true_values(df, time_period, column)
            max_counts = max(max_counts, counts.max())
            fig.add_trace(
                go.Bar(
                    x=counts.index,
                    y=counts.values,
                    name=column,
                    marker=dict(color=colors[i]),
                ),
                row=i + 1,
                col=1,
            )

        # Sync y limits
        if if_sync_y_limits:
            fig.update_yaxes(range=[0, max_counts * 1.1])

        # Updating layout
        fig.update_layout(
            height=200 * len(columns),
            showlegend=False,
            title=f"{time_period} counts",
        )
    else:  # side-by-side histograms in the same plot
        fig = go.Figure()
        for i, column in enumerate(columns):
            fig.add_trace(go.Histogram( 
                x=df[df[column]==True]["session_date"],
                xbins=dict(size="M1"), # Only monthly bins look good
                name=column,
                marker_color=colors[i],
                opacity=0.75
            ))

        # Update layout for grouped histogram
        fig.update_layout(
            height=500,
            bargap=0.05,  # Gap between bars of adjacent locations
            bargroupgap=0.1,  # Gap between bars of the same location
            barmode='group',  # Grouped style
            showlegend=True,
            legend=dict(
                orientation="h",  # Horizontal legend
                y=-0.2,  # Position below the plot
                x=0.5,  # Center the legend
                xanchor="center",  # Anchor the legend's x position
                yanchor="top"  # Anchor the legend's y position
            ),
            title="Monthly counts"
        )
        
    return fig

def app():
    # --- 1. Generate combined dataframe from docDB queries ---
    with st.sidebar:
        st.markdown('# Metadata sources:')
        st.markdown('## 1. From docDB queries')

        with st.expander("MetadataDbClient settings"):
            parallel = st.checkbox("Parallel fetching", value=False)
            pagination = st.checkbox("Pagination", value=True)
            paginate_batch_size = st.number_input("Pagination batch size", value=5000, disabled=not pagination)

        cols = st.columns([1.5, 1])
        with cols[0]:
            start_time = time.time()
            df_merged, dfs_docDB = fetch_all_queries_from_docDB(
                queries_to_merge=QUERY_PRESET,
                parallel=parallel,
                pagination=pagination,
                paginate_batch_size=paginate_batch_size,
            )
            docDB_retrieve_time = time.time() - start_time
            st.markdown(f"Finished in {docDB_retrieve_time:.3f} secs.")

        with cols[1]:
            if st.button('Re-fetch docDB queries'):
                st.cache_data.clear()
                st.rerun()

    if df_merged is None:
        st.cache_data.clear()
        return

    # --- 2. Merge in the master df in the Home page (Han's temporary pipeline) ---
    # Data from Home.init (all sessions from Janelia bpod + AIND bpod + AIND bonsai)
    df_from_Home = st.session_state.df["sessions_bonsai"]
    # Only keep AIND sessions
    df_from_Home = df_from_Home.query("institute == 'AIND'")
    df_from_Home.loc[df_from_Home.hardware == "bpod", "Han_temp_pipeline (bpod)"] = True
    df_from_Home.loc[df_from_Home.hardware == "bonsai", "Han_temp_pipeline (bonsai)"] = True

    # Only keep subject_id and session_date as index
    df_Han_pipeline = (
        df_from_Home[
            [
                "subject_id",
                "session_date",
                "Han_temp_pipeline (bpod)",
                "Han_temp_pipeline (bonsai)",
            ]
        ]
        .set_index(["subject_id", "session_date"])
        .sort_index(
            level=["session_date", "subject_id"],
            ascending=[False, False],
        )
    )

    # Merged with df_merged
    df_merged = df_merged.combine_first(df_Han_pipeline)

    # --- 3. Get raw data on VAST ---
    raw_sessions_on_VAST = load_raw_sessions_on_VAST()

    # Example entry of raw_sessions_on_VAST:
    #     Z:\svc_aind_behavior_transfer\447-3-D\751153\behavior_751153_2024-10-20_17-13-34\behavior
    #     Z:\svc_aind_behavior_transfer\2023late_DataNoMeta_Reorganized\687553_2023-11-20_09-48-24\behavior
    #     Z:\svc_aind_behavior_transfer\2023late_DataNoMeta_Reorganized\687553_2023-11-13_11-09-55\687553_2023-12-01_09-41-43\TrainingFolder
    #     Let's find the strings between two \\s that precede "behavior" or "TrainingFolder"

    # Parse "name" from full path on VAST
    re_pattern = R"\\([^\\]*)\\(?:behavior|TrainingFolder)$"
    session_names = [re.findall(re_pattern, path)[0] for path in raw_sessions_on_VAST]
    df_raw_sessions_on_VAST = pd.DataFrame(raw_sessions_on_VAST, columns=["full_path"])
    df_raw_sessions_on_VAST["name"] = session_names
    df_raw_sessions_on_VAST["raw_data_on_VAST"] = True 

    # Formatting metadata dataframe
    (
        df_raw_sessions_on_VAST, 
        df_raw_sessions_on_VAST_unique_mouse_date,
        df_raw_sessions_on_VAST_multi_sessions_per_day
    ) = _formatting_metadata_df(df_raw_sessions_on_VAST, source_prefix="VAST")

    dfs_raw_on_VAST = {
        "df": df_raw_sessions_on_VAST,
        "df_unique_mouse_date": df_raw_sessions_on_VAST_unique_mouse_date,
        "df_multi_sessions_per_day": df_raw_sessions_on_VAST_multi_sessions_per_day,
    }

    # Merging with df_merged (using the unique mouse-date dataframe)
    df_merged = df_merged.combine_first(df_raw_sessions_on_VAST_unique_mouse_date)
    df_merged.sort_index(level=["session_date", "subject_id"], ascending=[False, False], inplace=True)

    # --- Add sidebar ---
    add_sidebar(df_merged, dfs_docDB, df_Han_pipeline, dfs_raw_on_VAST, docDB_retrieve_time)

    # --- Main contents ---
    st.markdown(f"# Data inventory for dynamic foraging")
    st.markdown(f"### Merged metadata (n = {len(df_merged)}, see the sidebar for details)")
    download_df(df_merged, label="Download merged df as CSV", file_name="df_docDB_queries.csv")

    aggrid_interactive_table_basic(
        df_merged.reset_index(),
        height=400,
        configure_columns=[
            dict(
                field="session_date",
                type=["customDateTimeFormat"],
                custom_format_string="yyyy-MM-dd",
            )
        ],
    )

    # --- Venn diagram from presets ---
    with open("data_inventory_VENN_PRESET.json", "r") as f:
        VENN_PRESET = json.load(f)

    if VENN_PRESET:

        cols = st.columns([2, 1])
        cols[0].markdown("## Venn diagrams from presets")
        with cols[1].expander("Time view settings", expanded=True):
            cols_1 = st.columns([1, 1])                    
            if_separate_plots = cols_1[0].checkbox("Separate in subplots", value=True)
            if_sync_y_limits = cols_1[0].checkbox(
                "Sync Y limits", value=True, disabled=not if_separate_plots
            )
            time_period = cols_1[1].selectbox(
                "Bin size",
                ["Daily", "Weekly", "Monthly", "Quarterly"],
                index=1,
                disabled=not if_separate_plots,
            )

        for i_venn, venn_preset in enumerate(VENN_PRESET):
            # -- Venn diagrams --
            st.markdown(f"### ({i_venn+1}). {venn_preset['name']}")
            fig, notes = generate_venn(
                    df_merged,
                    venn_preset
                )
            for note in notes:
                st.markdown(note)

            cols = st.columns([1, 1])
            with cols[0]:
                st.pyplot(fig, use_container_width=True)

            # -- Show and download df for this Venn --
            circle_columns = [c_s["column"] for c_s in venn_preset["circle_settings"]]
            # Show histogram over time for the columns and patches in preset
            df_this_preset = df_merged[circle_columns]
            # Filter out rows that have at least one True in this Venn
            df_this_preset = df_this_preset[df_this_preset.any(axis=1)]

            # Create a new column to indicate sessions in patches specified by patch_ids like ["100", "101", "110", "111"]
            for patch_setting in venn_preset.get("patch_settings", []):
                idx = _filter_df_by_patch_ids(
                    df_this_preset[circle_columns],
                    patch_setting["patch_ids"]
                )
                df_this_preset.loc[idx, str(patch_setting["patch_ids"])] = True 

            # Join in other extra columns
            df_this_preset = df_this_preset.join(
                df_merged[[col for col in df_merged.columns if col not in meta_columns]], how="left"
            )

            with cols[0]:
                download_df(
                    df_this_preset,
                    label="Download as CSV for this Venn diagram",
                    file_name=f"df_{venn_preset['name']}.csv",
                )
                with st.expander(f"Show dataframe, n = {len(df_this_preset)}"):
                    st.write(df_this_preset)

            with cols[1]:
                # -- Show histogram over time --
                fig = plot_histogram_over_time(
                    df=df_this_preset.reset_index(),
                    venn_preset=venn_preset,
                    time_period=time_period,
                    if_sync_y_limits=if_sync_y_limits,
                    if_separate_plots=if_separate_plots,
                )
                plotly_events(
                    fig,
                    click_event=False,
                    hover_event=False,
                    select_event=False,
                    override_height=fig.layout.height * 1.1,
                    override_width=fig.layout.width,
                )

            st.markdown("---")

    # --- User-defined Venn diagram ---
    # Multiselect for selecting queries up to three
    # st.markdown('---')
    # st.markdown("## Venn diagram from user-selected queries")
    # selected_queries = st.multiselect(
    #     "Select queries to filter sessions",
    #     meta_columns,
    #     default=meta_columns[:3],
    #     key="selected_queries",
    # )

    # columns_to_venn = selected_queries
    # fig = generate_venn(df_merged, columns_to_venn)
    # st.columns([1, 1])[0].pyplot(fig, use_container_width=True)


if __name__ == "__main__":
    
    # Share the same master df as the Home page
    if "df" not in st.session_state or "sessions_bonsai" not in st.session_state.df.keys() or not st.session_state.bpod_loaded:
        st.spinner("Loading data from Han temp pipeline...")
        init(if_load_docDB_override=False, if_load_bpod_data_override=True)

    app()
