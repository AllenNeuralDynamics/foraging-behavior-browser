import streamlit as st

import importlib

from streamlit_util import filter_dataframe, aggrid_interactive_table_session, add_session_filter


def app():

    with st.sidebar:
        add_session_filter()
        






try:
    app()
except:
    st.markdown('### Something is wrong. Try going back to 🏠Home or refresh.')