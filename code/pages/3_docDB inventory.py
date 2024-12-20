'''Migrated from David's toy app https://codeocean.allenneuraldynamics.org/capsule/9532498/tree
'''

import logging
import re

from matplotlib_venn import venn2, venn3
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_dynamic_filters import DynamicFilters

from util.fetch_data_docDB import load_data_from_docDB, load_client

st.markdown(
"""
<style>
    .stMultiSelect [data-baseweb=select] span{
        max-width: 1000px;
    }
</style>""",
unsafe_allow_html=True,
)

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


client = load_client()

queries = {
    "raw, 'dynamic_foraging' in ANY software name": {
        "$or":[
            {"session.data_streams.software.name": "dynamic-foraging-task"},
            {"session.stimulus_epochs.software.name": "dynamic-foraging-task"},            
        ],
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
def get_session_from_query(query):
    results = client.retrieve_docdb_records(
        filter_query=query,
        projection={"name": 1, "_id": 1},
    )
    
    sessions = [re.sub(r'_processed.*$', '', r["name"]) for r in results]
    return sessions


def app():

    # Multiselect for selecting queries up to three
    query_keys = list(queries.keys())
    selected_queries = st.multiselect(
        "Select queries to filter sessions",
        query_keys,
        default=query_keys[:3],
        key="selected_queries",
    )

    # Generage venn diagram of the selected queries
    query_results = {key: set(get_session_from_query(queries[key])) for key in selected_queries}


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
    
    
    # -- Show dataframe that summarize the selected queries --
    st.markdown(f"### Summary of selected queries")
    query_results
    


    # df = load_data_from_docDB()

    # st.markdown(f'### Note: the dataframe showing here has been merged in to the master table on the Home page!')

    # dynamic_filters = DynamicFilters(
    #     df=df, 
    #     filters=['subject_id', 'subject_genotype'])
    # dynamic_filters.display_filters()
    # dynamic_filters.display_df()
    
    
if __name__ == "__main__":
    app()
