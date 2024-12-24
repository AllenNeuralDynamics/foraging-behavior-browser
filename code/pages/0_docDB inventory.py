'''Migrated from David's toy app https://codeocean.allenneuraldynamics.org/capsule/9532498/tree
'''

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    "{raw, 'dynamic_foraging' in ANY software name}": {
        "$or":[
            {"session.data_streams.software.name": "dynamic-foraging-task"},
            {"session.stimulus_epochs.software.name": "dynamic-foraging-task"},            
        ],
        "name": {"$not": {"$regex": ".*processed.*"}},
    },
    "{raw, ('dynamic_foraging' in ANY software name) AND ('ecephys' in data_description.modality)}": {
        "$or":[
            {"session.data_streams.software.name": "dynamic-foraging-task"},
            {"session.stimulus_epochs.software.name": "dynamic-foraging-task"},            
        ],
        "data_description.modality.abbreviation": "ecephys",
        "name": {"$not": {"$regex": ".*processed.*"}},        
    },
    "{raw, 'dynamic_foraging' in data_streams software name}": {
        "session.data_streams.software.name": "dynamic-foraging-task",
        "name": {"$not": {"$regex": ".*processed.*"}},
    },
    "{raw, 'dynamic_foraging' in stimulus_epochs software name}": {
        "session.stimulus_epochs.software.name": "dynamic-foraging-task",            
        "name": {"$not": {"$regex": ".*processed.*"}},
    },
    "{processed, 'dynamic_foraging' in ANY software name}": {
        "$or":[
            {"session.data_streams.software.name": "dynamic-foraging-task"},
            {"session.stimulus_epochs.software.name": "dynamic-foraging-task"},            
        ],
        "name": {"$regex": ".*processed.*"},
    },
    "{raw, 'fib' in 'data_description.modality'}": {
        "data_description.modality.abbreviation": "fib",
        "name": {"$not": {"$regex": ".*processed.*"}},
    },
    "{raw, 'fib' in 'rig.modalities'}": {
        "rig.modalities.abbreviation": "fib",
        "name": {"$not": {"$regex": ".*processed.*"}},        
    },
    "{raw, 'fib' in 'session.data_streams'}": {
        "session.data_streams.stream_modalities.abbreviation": "fib",
        "name": {"$not": {"$regex": ".*processed.*"}},
    }
}

@st.cache_data(ttl=3600 * 12)  # Cache the df_docDB up to 12 hours
def query_sessions_from_docDB(query):
    return client.retrieve_docdb_records(
        filter_query=query,
        projection={
            "_id": 0,
            "name": 1,
            "rig.rig_id": 1,
            "session.experimenter_full_name": 1,
        },
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

@st.cache_data(ttl=3600 * 12)  # Cache the df_docDB up to 12 hours
def query_single_query(key):
    """ Query a single query from QUERY_PRESET and process the result """
    query_result = query_sessions_from_docDB(QUERY_PRESET[key])
    df = pd.json_normalize(query_result)
    
    # Create index that can be joined with other dfs and main df
    df[["subject_id", "session_date", "nwb_suffix"]] = df[f"name"].apply(
        lambda x: pd.Series(split_nwb_name(x))
    )
    df[key] = True
    df.set_index(["subject_id", "session_date", "nwb_suffix"], inplace=True)
    
    return df


def get_merged_queries(queries_to_merge):
    """ Get merged queries from selected queries """
    
    dfs = []
    progress_bar = st.progress(0, text="Querying docDB")
    
    with ThreadPoolExecutor(max_workers=len(queries_to_merge)) as executor:
        future_to_query = {executor.submit(query_single_query, key): key for key in queries_to_merge}
        for i, future in enumerate(as_completed(future_to_query), 1):
            key = future_to_query[future]
            try:
                df = future.result()
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error querying {key}: {e}")
            finally:
                progress_bar.progress(i / len(queries_to_merge), text=f"Fetching  docDB ({i}/{len(queries_to_merge)})... {key} done!")           
    
    # Combine queried dfs
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.combine_first(df)  # Combine nwb_names
    
    st.markdown(f"#### Merged dataframe")
    st.write(df_merged)
    download_df(df_merged, label="Download merged df as CSV", file_name="df_docDB_queries.csv")
    
    return df_merged

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

    # Generate combined dataframe
    import time
    start = time.time()
    df_merged = get_merged_queries(queries_to_merge=QUERY_PRESET.keys())
    st.write(f"Time taken to query and merge: {time.time() - start:.2f} seconds")
    
    columns_to_venn = selected_queries

    # -- Show venn --
    fig = venn(df_merged, columns_to_venn)
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
