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
import time

from util.fetch_data_docDB import load_data_from_docDB, load_client
from util.reformat import split_nwb_name

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
    {"alias": "{raw, 'dynamic_foraging' in ANY software name}",
     "filter": {
         "$or":[
             {"session.data_streams.software.name": "dynamic-foraging-task"},
             {"session.stimulus_epochs.software.name": "dynamic-foraging-task"},            
         ],
         "name": {"$not": {"$regex": ".*processed.*"}},
     }},
    {"alias": "{raw, 'dynamic_foraging' in data_streams software name}",
     "filter": {
         "session.data_streams.software.name": "dynamic-foraging-task",
         "name": {"$not": {"$regex": ".*processed.*"}},
     }},
    {"alias": "{raw, 'dynamic_foraging' in stimulus_epochs software name}",
     "filter": {
         "session.stimulus_epochs.software.name": "dynamic-foraging-task",            
         "name": {"$not": {"$regex": ".*processed.*"}},
     }},
    {"alias": "{raw, 'fib' in 'data_description.modality'}",
     "filter": {
         "data_description.modality.abbreviation": "fib",
         "name": {"$not": {"$regex": ".*processed.*"}},
     }},
    {"alias": "{raw, 'fib' in 'rig.modalities'}",
     "filter": {
         "rig.modalities.abbreviation": "fib",
         "name": {"$not": {"$regex": ".*processed.*"}},        
     }},
    {"alias": "{raw, 'fib' in 'session.data_streams'}",
     "filter": {
         "session.data_streams.stream_modalities.abbreviation": "fib",
         "name": {"$not": {"$regex": ".*processed.*"}},
     }},
    {"alias": "{raw, ('dynamic_foraging' in ANY software name) AND ('ecephys' in data_description.modality)}",
     "filter": {
         "$or":[
             {"session.data_streams.software.name": "dynamic-foraging-task"},
             {"session.stimulus_epochs.software.name": "dynamic-foraging-task"},            
         ],
         "data_description.modality.abbreviation": "ecephys",
         "name": {"$not": {"$regex": ".*processed.*"}},        
     }},
    {"alias": "{raw, 'FIP' in name}",
     "filter": {
        "name": {
            "$regex": "^FIP.*",
            "$not": {"$regex": ".*processed.*"}
        }
    }},
    {"alias": "{processed, 'dynamic_foraging' in ANY software name}",
     "filter": {
         "$or":[
             {"session.data_streams.software.name": "dynamic-foraging-task"},
             {"session.stimulus_epochs.software.name": "dynamic-foraging-task"},            
         ],
         "name": {"$regex": ".*processed.*"},
     }},
]

def download_df(df, label="Download filtered df as CSV", file_name="df.csv"):
    """ Download df as csv """
    csv = df.to_csv(index=True)
    
    # Create download buttons
    st.download_button(
        label=label,
        data=csv,
        file_name=file_name,
        mime='text/csv'
    )

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

    # --- Process data into dataframe ---
    df = pd.json_normalize(results)

    # Create index of subject_id, session_date, nwb_suffix by parsing nwb_name
    df[["subject_id", "session_date", "nwb_suffix"]] = df[f"name"].apply(
        lambda x: pd.Series(split_nwb_name(x))
    )
    df = df.set_index(["subject_id", "session_date", "nwb_suffix"]).sort_index()
    df[query["alias"]] = True  # Add a column to mark the query

    # Remove invalid subject_id
    df = df[df.index.get_level_values("subject_id").astype(int) > 300000]

    # --- Handle multiple sessions per day ---
    # Build a dataframe with unique mouse-dates.
    # If multiple sessions per day, combine them into a list of 'name'
    df_unique_mouse_date = (
        df.reset_index()
        .groupby(["subject_id", "session_date"])
        .agg({"name": list, **{col: "first" for col in df.columns if col != "name"}})
    )
    # Add a new column to indicate multiple sessions per day
    df_unique_mouse_date["multiple_sessions_per_day"] = df_unique_mouse_date["name"].apply(
        lambda x: len(x) > 1
    )

    # Also return the dataframe with multiple sessions per day
    df_multi_sessions_per_day = df_unique_mouse_date[df_unique_mouse_date["multiple_sessions_per_day"]]

    # Create a new column to mark duplicates in the original df
    df.loc[
        df.index.droplevel("nwb_suffix").isin(df_multi_sessions_per_day.index), 
        "multiple_sessions_per_day"
    ] = True

    print(f"Done querying {query['alias']}!")
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


def app():

    # Generate combined dataframe
    start_time = time.time()
    df_merged, dfs = fetch_all_queries_from_docDB(queries_to_merge=QUERY_PRESET)

    # Sidebar
    with st.sidebar:
        st.markdown('## docDB query presets ')
        st.markdown('#### See how to use these queries [in this doc.](https://aind-data-access-api.readthedocs.io/en/latest/UserGuide.html#document-database-docdb)')
        for query in QUERY_PRESET:
            with st.expander(f"{query['alias']}"):
                # Turn query to json with indent=4
                query_json = json.dumps(query["filter"], indent=4)
                st.code(query_json)

                # Show records
                df = dfs[query["alias"]]["df"]
                df_multi_sessions_per_day = dfs[query["alias"]]["df_multi_sessions_per_day"]
                df_unique_mouse_date = dfs[query["alias"]]["df_unique_mouse_date"]

                cols = st.columns([2, 1])
                cols[0].markdown(f"{len(df)} returned")
                with cols[1]:
                    download_df(df, label="Download as CSV", file_name=f"{query['alias']}.csv")

                cols = st.columns([2, 1])
                cols[0].markdown(
                    f"{df_unique_mouse_date.multiple_sessions_per_day.sum()} have multiple sessions per day")
                with cols[1]:
                    download_df(
                        df_multi_sessions_per_day,
                        label="Download as CSV",
                        file_name=f"{query['alias']}_multi_sessions_per_day.csv",
                    )

                cols = st.columns([2, 1])
                cols[0].markdown(
                    f"{len(df_unique_mouse_date)} unique mouse-date pairs"
                )
                with cols[1]:
                    download_df(
                        df_unique_mouse_date,
                        label="Download as CSV",
                        file_name=f"{query['alias']}_unique_mouse_date.csv",
                    )

                if len(df_unique_mouse_date) != df_merged[query["alias"]].sum():
                    st.warning('''len(df_unique_mouse_date) != df_merged[query["alias"]].sum()!''')
                    
        st.markdown(f"Retrieving data took {time.time() - start_time} secs")


    st.markdown(f"#### Merged dataframe (n = {len(df_merged)})")
    st.write(df_merged)
    download_df(df_merged, label="Download merged df as CSV", file_name="df_docDB_queries.csv")

    # Multiselect for selecting queries up to three
    query_keys = [query["alias"] for query in QUERY_PRESET]
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
    app()
