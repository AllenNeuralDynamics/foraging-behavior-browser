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

def query_sessions_from_docDB(query):
    return client.retrieve_docdb_records(
        filter_query=query["filter"],
        projection={
            "_id": 0,
            "name": 1,
            "rig.rig_id": 1,
            "session.experimenter_full_name": 1,
        },
        paginate=False,
    )

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

def fetch_single_query(key):
    """ Query a single query from QUERY_PRESET and process the result 
    
    Return: 
    - df: the full dataframe
    - df_multi_sessions_per_day: the dataframe with multiple sessions per day
    
    """
    query_result = query_sessions_from_docDB(key)
    df = pd.json_normalize(query_result)

    # Create index that can be joined with other dfs and main df
    df[["subject_id", "session_date", "nwb_suffix"]] = df[f"name"].apply(
        lambda x: pd.Series(split_nwb_name(x))
    )
    df[key["alias"]] = True
    df = df.set_index(["subject_id", "session_date", "nwb_suffix"]).sort_index()
    
    # Remove invalid subject_id
    df = df[df.index.get_level_values("subject_id").astype(int) > 300000]

    # Get cases where one mouse has multiple records per day
    subject_date = df.index.to_frame(index=False)[["subject_id", "session_date"]]
    df_multi_sessions_per_day = df[subject_date.duplicated(keep=False).values]

    return df, df_multi_sessions_per_day

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
                df, df_multi_sessions_per_day = future.result()
                dfs[key["alias"]] = {
                    "df": df,
                    "df_multi_sessions_per_day": df_multi_sessions_per_day,
                }
            except Exception as e:
                print(f"Error querying {key}: {e}")

    # Combine queried dfs
    df_merged = dfs[queries_to_merge[0]["alias"]]["df"]
    for df in [dfs[query["alias"]]["df"] for query in queries_to_merge[1:]]:
        df_merged = df_merged.combine_first(df)  # Combine nwb_names

    # Recover the order of QUERY_PRESET
    df_merged = df_merged.reindex(columns=[query["alias"] for query in queries_to_merge])

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
    df_merged, dfs = fetch_all_queries_from_docDB(queries_to_merge=QUERY_PRESET)
    
    # Sidebar
    with st.sidebar:
        st.markdown('## docDB query presets ')
        st.markdown('#### See how to use these queries [in this doc.](https://aind-data-access-api.readthedocs.io/en/latest/UserGuide.html#document-database-docdb)')
        for query in QUERY_PRESET:
            results_this = df_merged[query["alias"]].sum()
            with st.expander(f"n = {results_this}, {query['alias']}"):
                # Turn query to json with indent=4
                query_json = json.dumps(query["filter"], indent=4)
                st.code(query_json)

    
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
