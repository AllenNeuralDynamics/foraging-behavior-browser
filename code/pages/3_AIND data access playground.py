'''Migrated from David's toy app https://codeocean.allenneuraldynamics.org/capsule/9532498/tree
'''

import logging
import streamlit as st
from streamlit_dynamic_filters import DynamicFilters

from aind_data_access_api.document_db import MetadataDbClient
from util.fetch_data_docDB import fetch_fip_data

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

@st.cache_data(ttl=3600*12) # Cache the df_docDB up to 12 hours
def load_data_from_docDB():
    client = load_client()
    df = fetch_fip_data(client)
    return df

@st.cache_resource
def load_client():
    return MetadataDbClient(
        host="api.allenneuraldynamics.org",    
        database="metadata_index",
        collection="data_assets"
    )

df = load_data_from_docDB()

dynamic_filters = DynamicFilters(
    df=df, 
    filters=['subject_id', 'subject_genotype'])
dynamic_filters.display_filters()
dynamic_filters.display_df()
