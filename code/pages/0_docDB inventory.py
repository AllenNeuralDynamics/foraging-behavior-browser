'''Migrated from David's toy app https://codeocean.allenneuraldynamics.org/capsule/9532498/tree
'''

import logging
import re
from functools import reduce

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

QUERY_PRESET = {
    "raw, 'dynamic_foraging' in ANY software name": {
        "$or":[
            {"session.data_streams.software.name": "dynamic-foraging-task"},
            {"session.stimulus_epochs.software.name": "dynamic-foraging-task"},            
        ],
        "name": {"$not": {"$regex": ".*processed.*"}},
    },
    "raw, ('dynamic_foraging' in ANY software name) AND ('ecephys' in data_description.modality)": {
        "$or":[
            {"session.data_streams.software.name": "dynamic-foraging-task"},
            {"session.stimulus_epochs.software.name": "dynamic-foraging-task"},            
        ],
        "data_description.modality.abbreviation": "ecephys",
        "name": {"$not": {"$regex": ".*processed.*"}},        
    },
    "raw, 'dynamic_foraging' in data_streams software name": {
        "session.data_streams.software.name": "dynamic-foraging-task",
        "name": {"$not": {"$regex": ".*processed.*"}},
    },
    "raw, 'dynamic_foraging' in stimulus_epochs software name": {
        "session.stimulus_epochs.software.name": "dynamic-foraging-task",            
        "name": {"$not": {"$regex": ".*processed.*"}},
    },
    "processed, 'dynamic_foraging' in ANY software name": {
        "$or":[
            {"session.data_streams.software.name": "dynamic-foraging-task"},
            {"session.stimulus_epochs.software.name": "dynamic-foraging-task"},            
        ],
        "name": {"$regex": ".*processed.*"},
    },
    "raw, 'fib' in 'data_description.modality'": {
        "data_description.modality.abbreviation": "fib",
        "name": {"$not": {"$regex": ".*processed.*"}},
    },
    "raw, 'fib' in 'rig.modalities'": {
        "rig.modalities.abbreviation": "fib",
        "name": {"$not": {"$regex": ".*processed.*"}},        
    },
    "raw, 'fib' in 'session.data_streams'": {
        "session.data_streams.stream_modalities.abbreviation": "fib",
        "name": {"$not": {"$regex": ".*processed.*"}},
    }
}

@st.cache_data(ttl=3600 * 12)  # Cache the df_docDB up to 12 hours
def query_sessions_from_docDB(query):
    results = client.retrieve_docdb_records(
        filter_query=query,
        projection={"name": 1, "_id": 0},
    )
    
    sessions = [re.sub(r'_processed.*$', '', r["name"]) for r in results]
    return sessions


def app():

    #
    with st.expander("Show docDB queries", expanded=False):
        st.write('See how to use these queries [here](https://aind-data-access-api.readthedocs.io/en/latest/UserGuide.html#document-database-docdb)')
        for key, query in QUERY_PRESET.items():
            st.markdown(f"**{key}**")
            st.code(query)

    # Multiselect for selecting queries up to three
    query_keys = list(QUERY_PRESET.keys())
    selected_queries = st.multiselect(
        "Select queries to filter sessions",
        query_keys,
        default=query_keys[:3],
        key="selected_queries",
    )

    # Generage venn diagram of the selected queries
    query_results = {key: query_sessions_from_docDB(QUERY_PRESET[key]) for key in selected_queries}

    # -- Show dataframe that summarize the selected queries --
    st.markdown(f"#### Merged dataframe")

    dfs = []
    for query_name, query_result in query_results.items():
        df = pd.DataFrame(query_result, columns=[f"nwb_name_{query_name}"])

        # Apply the function to create new columns
        df[["subject_id", "session_date", "nwb_suffix"]] = df["nwb_name"].apply(
            lambda x: pd.Series(split_nwb_name(x))
        )
        
        df[query_name] = True
        dfs.append(df)

    # Out merge all dataframes
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on=["subject_id", "session_date", "nwb_suffix"], how="outer"), dfs
    )

    # -- Show venn --
    fig, ax = plt.subplots()
    if len(selected_queries) == 2:
        venn2(
            [query_results[key] for key in selected_queries],
            set_labels=selected_queries,
        )
    else:
        venn3(
            [query_results[key] for key in selected_queries],
            set_labels=selected_queries,
        )

    st.columns([1, 1])[0].pyplot(fig, use_container_width=True)

    # df = load_data_from_docDB()

    # st.markdown(f'### Note: the dataframe showing here has been merged in to the master table on the Home page!')

    # dynamic_filters = DynamicFilters(
    #     df=df,
    #     filters=['subject_id', 'subject_genotype'])
    # dynamic_filters.display_filters()
    # dynamic_filters.display_df()


if __name__ == "__main__":
    app()
