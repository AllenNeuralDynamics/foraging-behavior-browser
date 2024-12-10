"""Code to fetch mle_fitting results from docDB
Han Hou
"""

import logging
import time

import pandas as pd
import streamlit as st
import numpy as np

from aind_data_access_api.document_db import MetadataDbClient

@st.cache_data(ttl=3600*12) # Cache the df_docDB up to 12 hours
def load_mle_results_from_docDB():
    client = load_client()
    df = fetch_mle_fitting_results(client)
    return df


@st.cache_resource
def load_client():
    return MetadataDbClient(
        host="api.allenneuraldynamics-test.org",    # From the test docDB  
        database="metadata_index",
        collection="data_assets"
    )


def fetch_mle_fitting_results(client):
    """Fetch mle fitting results from docDB
    """
    
    pass
