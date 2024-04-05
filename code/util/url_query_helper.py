import streamlit as st

def checkbox_wrapper_for_url_query(st_prefix, label, key, default, **kwargs):
    return st_prefix.checkbox(
        label,
        value=st.session_state[key]
                if key in st.session_state else 
                st.query_params[key].lower()=='true' 
                if key in st.query_params 
                else default,
        key=key,
        **kwargs,
    )

def selectbox_wrapper_for_url_query(st_prefix, label, options, key, default, **kwargs):
    return st_prefix.selectbox(
        label,
        options=options,
        index=(
            options.index(st.session_state[key])
            if key in st.session_state
            else options.index(st.query_params[key]) 
            if key in st.query_params 
            else default
        ),
        key=key,
        **kwargs,
    )

def slider_wrapper_for_url_query(st_prefix, label, min_value, max_value, key, default, **kwargs):
    return st_prefix.slider(
        label,
        min_value,
        max_value,
        value=(
            st.session_state[key]
            if key in st.session_state
            else int(st.query_params[key]) 
            if key in st.query_params 
            else default
        ),
        key=key,
        **kwargs,
    )