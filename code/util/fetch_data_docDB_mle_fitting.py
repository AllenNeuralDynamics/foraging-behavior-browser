"""Code to fetch mle_fitting results from docDB
Han Hou
"""

import logging
import time

import pandas as pd
import streamlit as st
import numpy as np

from aind_data_access_api.document_db import MetadataDbClient

@st.cache_resource
def load_client():
    return MetadataDbClient(
        host="api.allenneuraldynamics-test.org",    # From the test docDB  
        database="behavior_analysis",
        collection="mle_fitting"
    )
    
client = load_client()

def fetch_mle_fitting_results():
    """Fetch mle fitting results from docDB
    """

    import time 
    start_time = time.time()
    
    results = client.retrieve_docdb_records(
        filter_query={
            "analysis_results.fit_settings.agent_alias": "QLearning_L1F1_CK1_softmax",
        },
        projection={"nwb_name": 1, "analysis_results.params": 1},
    )
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    
    pipeline = [
        {
            "$match": {
                "analysis_results.fit_settings.agent_alias": "QLearning_L1F1_CK1_softmax"
            }
        },
        {
            "$project": {
                "_id": 0,
                "nwb_name": 1,
                "params": "$analysis_results.params",
            }
        },
    ]
    
    records = client.aggregate_docdb_records(pipeline=pipeline)
    
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
    
    