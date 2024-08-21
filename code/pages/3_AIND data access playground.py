'''Migrated from David's toy app https://codeocean.allenneuraldynamics.org/capsule/9532498/tree
'''

import logging
import streamlit as st
from streamlit_dynamic_filters import DynamicFilters

from util.fetch_data_docDB import load_data_from_docDB

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

df = load_data_from_docDB()

st.markdown(f'### Note: the dataframe showing here has been merged in to the master table on the Home page!')

dynamic_filters = DynamicFilters(
    df=df, 
    filters=['subject_id', 'subject_genotype'])
dynamic_filters.display_filters()
dynamic_filters.display_df()
