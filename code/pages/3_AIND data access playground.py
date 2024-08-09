'''Migrated from David's toy app https://codeocean.allenneuraldynamics.org/capsule/9532498/tree
'''

import logging
from aind_data_access_api.document_db import MetadataDbClient
from util.fetch_data_docDB import fetch_fip_data

import streamlit as st
from streamlit_dynamic_filters import DynamicFilters

@st.cache_data  
def load_data():
    df = fetch_fip_data(client)
    return df

@st.cache_resource
def load_client():
    return MetadataDbClient(
        host="api.allenneuraldynamics.org",    
        database="metadata_index",
        collection="data_assets"
    )

client = load_client()
df = load_data()

dynamic_filters = DynamicFilters(df=df, filters=['subject_id', 'subject_genotype'])
dynamic_filters.display_filters()
dynamic_filters.display_df()
