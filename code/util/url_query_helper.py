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
            else type(min_value)(st.query_params[key]) 
            if key in st.query_params 
            else default
        ),
        key=key,
        **kwargs,
    )
    
    
def sync_widget_with_query(key, default):
    if key in st.query_params:
        # always get all query params as a list
        q_all = st.query_params.get_all(key)
        
        # convert type according to default
        list_default = default if isinstance(default, list) else [default]
        for d in list_default:
            _type = type(d)
            if _type: break  # The first non-None type
            
        if _type == bool:
            q_all_correct_type = [q.lower() == 'true' for q in q_all]
        else:
            q_all_correct_type = [_type(q) for q in q_all]
        
        # flatten list if only one element
        if not isinstance(default, list):
            q_all_correct_type = q_all_correct_type[0]
        
        try:
            st.session_state[key] = q_all_correct_type
        except:
            print(f'Failed to set {key} to {q_all_correct_type}')
    else:
        try:
            st.session_state[key] = default
        except:
            print(f'Failed to set {key} to {default}')