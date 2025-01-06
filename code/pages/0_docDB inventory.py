import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json

from matplotlib_venn import venn2, venn3
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_dynamic_filters import DynamicFilters
import pandas as pd
import numpy as np
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

QUERY_PRESET = [
    {"alias": "docDB_{raw, 'dynamic_foraging' in ANY software name}",
     "filter": {
         "$or":[
             {"session.data_streams.software.name": "dynamic-foraging-task"},
             {"session.stimulus_epochs.software.name": "dynamic-foraging-task"},            
         ],
         "name": {"$not": {"$regex": ".*processed.*"}},
     }},
    {"alias": "docDB_{raw, 'dynamic_foraging' in data_streams software name}",
     "filter": {
         "session.data_streams.software.name": "dynamic-foraging-task",
         "name": {"$not": {"$regex": ".*processed.*"}},
     }},
    {"alias": "docDB_{raw, 'dynamic_foraging' in stimulus_epochs software name}",
     "filter": {
         "session.stimulus_epochs.software.name": "dynamic-foraging-task",            
         "name": {"$not": {"$regex": ".*processed.*"}},
     }},
    {"alias": "docDB_{raw, 'fib' in 'data_description.modality'}",
     "filter": {
         "data_description.modality.abbreviation": "fib",
         "name": {"$not": {"$regex": ".*processed.*"}},
     }},
    {"alias": "docDB_{raw, 'fib' in 'rig.modalities'}",
     "filter": {
         "rig.modalities.abbreviation": "fib",
         "name": {"$not": {"$regex": ".*processed.*"}},        
     }},
    {"alias": "docDB_{raw, 'fib' in 'session.data_streams'}",
     "filter": {
         "session.data_streams.stream_modalities.abbreviation": "fib",
         "name": {"$not": {"$regex": ".*processed.*"}},
     }},
    {"alias": "docDB_{raw, ('dynamic_foraging' in ANY software name) AND ('ecephys' in data_description.modality)}",
     "filter": {
         "$or":[
             {"session.data_streams.software.name": "dynamic-foraging-task"},
             {"session.stimulus_epochs.software.name": "dynamic-foraging-task"},            
         ],
         "data_description.modality.abbreviation": "ecephys",
         "name": {"$not": {"$regex": ".*processed.*"}},        
     }},
    {"alias": "docDB_{raw, 'FIP' in name}",
     "filter": {
        "name": {
            "$regex": "^FIP.*",
            "$not": {"$regex": ".*processed.*"}
        }
    }},
    {"alias": "docDB_{processed, 'dynamic_foraging' in ANY software name}",
     "filter": {
         "$or":[
             {"session.data_streams.software.name": "dynamic-foraging-task"},
             {"session.stimulus_epochs.software.name": "dynamic-foraging-task"},            
         ],
         "name": {"$regex": ".*processed.*"},
     }},
]

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

def fetch_single_query(query):
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
        paginate=False,
    )
    print(f"Done querying {query['alias']}!")

    # --- Process data into dataframe ---
    df = pd.json_normalize(results)
    
    # Formatting dataframe
    df, df_unique_mouse_date, df_multi_sessions_per_day = _formatting_metadata_df(df)
    df_unique_mouse_date[query["alias"]] = True  # Add a column to prepare for merging

    return df, df_unique_mouse_date, df_multi_sessions_per_day

@st.cache_data(ttl=3600 * 24)
def fetch_all_queries_from_docDB(queries_to_merge):
    """ Get merged queries from selected queries """

    dfs = {}
    
    # Fetch data in parallel
    with ThreadPoolExecutor(max_workers=len(queries_to_merge)) as executor:
        future_to_query = {executor.submit(fetch_single_query, key): key for key in queries_to_merge}
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

    # Fetch data in serial 
    # for query in queries_to_merge:
    #     df, df_unique_mouse_date, df_multi_sessions_per_day = fetch_single_query(query)
    #     dfs[query["alias"]] = {
    #         "df": df,
    #         "df_unique_mouse_date": df_unique_mouse_date,
    #         "df_multi_sessions_per_day": df_multi_sessions_per_day,
    #     } 

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

def venn(df, columns_to_venn):
    """ Show venn diagram """
    if len(columns_to_venn) > 3:
        st.write("Venn diagram only supports up to 3 columns.")
        return
    
    fig, ax = plt.subplots()
    
    if len(columns_to_venn) == 2:
        venn2(
            [set(df.index[df[col]==True]) for col in columns_to_venn],
            set_labels=columns_to_venn,
        )
    else:
        venn3(
            [set(df.index[df[col]==True]) for col in columns_to_venn],
            set_labels=columns_to_venn,
        )
    return fig

def _show_records_on_sidebar(dfs, file_name_prefix, source_str="docDB"):
    df = dfs["df"]
    df_multi_sessions_per_day = dfs["df_multi_sessions_per_day"]
    df_unique_mouse_date = dfs["df_unique_mouse_date"]

    with st.expander(f"{len(df)} records from {source_str}"):
        download_df(df, label="Download as CSV", file_name=f"{file_name_prefix}.csv")
        st.write(df)

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
        st.markdown('# Metadata sources:')

        st.markdown('## 1. From docDB queries')
        st.markdown('#### See [how to use these queries](https://aind-data-access-api.readthedocs.io/en/latest/UserGuide.html#document-database-docdb) in your own code.')
        for query in QUERY_PRESET:
            with st.expander(f"### {query['alias']}"):

                # Turn query to json with indent=4
                # with st.expander("Show docDB query"):
                query_json = json.dumps(query["filter"], indent=4)
                st.code(query_json)

                # Show records                
                _show_records_on_sidebar(dfs_docDB[query["alias"]], file_name_prefix=query["alias"], source_str="docDB")

        st.markdown(f"Retrieving data from docDB (or st.cache) took {docDB_retrieve_time:.3f} secs.")

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
                
        st.markdown('''## 3. From VAST: existing raw data''')
        _show_records_on_sidebar(dfs_raw_on_VAST, file_name_prefix="raw_on_VAST", source_str="VAST /scratch")
        


def app():

    # --- 1. Generate combined dataframe from docDB queries ---
    start_time = time.time()
    df_merged, dfs_docDB = fetch_all_queries_from_docDB(queries_to_merge=QUERY_PRESET)
    docDB_retrieve_time = time.time() - start_time

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
    cols = st.columns([1, 2])
    cols[0].markdown(f"### Merged metadata (n = {len(df_merged)}, see the sidebar for details)")
    with cols[1]:
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

    # Multiselect for selecting queries up to three
    query_keys = [
        "Han_temp_pipeline (bpod)",
        "Han_temp_pipeline (bonsai)",
        "VAST_raw_data_on_VAST",
    ] + [query["alias"] for query in QUERY_PRESET]
    selected_queries = st.multiselect(
        "Select queries to filter sessions",
        query_keys,
        default=query_keys[:3],
        key="selected_queries",
    )

    columns_to_venn = selected_queries
    fig = venn(df_merged, columns_to_venn)
    st.columns([1, 1])[0].pyplot(fig, use_container_width=True)


if __name__ == "__main__":
    
    # Share the same master df as the Home page
    if "df" not in st.session_state or "sessions_bonsai" not in st.session_state.df.keys() or not st.session_state.bpod_loaded:
        init(if_load_docDB_override=False, if_load_bpod_data_override=True)

    app()
