"""Code to fetch mle_fitting results from docDB
Han Hou
"""

import logging
import time

import pandas as pd
import streamlit as st
import numpy as np

from aind_data_access_api.document_db import MetadataDbClient

FORAGER_PRESET = {
    "Win-Stay-Lose-Shift": "WSLS",
    "Rescorla-Wagner": "QLearning_L1F0_epsi",
    "Bari2019": "QLearning_L1F1_CK1_softmax",
    "Hattori2019": "QLearning_L2F1_softmax",
}

@st.cache_resource
def load_client():
    return MetadataDbClient(
        host="api.allenneuraldynamics-test.org",    # From the test docDB  
        database="behavior_analysis",
        collection="mle_fitting"
    )
    
client = load_client()

@st.cache_data(ttl=3600*12) # Cache the df_docDB up to 12 hours
def fetch_mle_fitting_results(model_alias="Hattori2019"):
    """Fetch mle fitting results from docDB
    """

    # --- Fetch MLE fitting results ---
    pipeline = [
        {
            "$match": {
                "analysis_results.fit_settings.agent_alias": FORAGER_PRESET[model_alias],
            }
        },
        {
            "$project": {
                "_id": 0,
                "nwb_name": 1,
                f"{model_alias}": "$analysis_results.params",
            }
        },
    ]
    records = client.aggregate_docdb_records(pipeline=pipeline)
    df = pd.json_normalize(records)
    

    # Apply the function to create new columns
    df[["subject_id", "session_date", "nwb_suffix"]] = df["nwb_name"].apply(
        lambda x: pd.Series(split_nwb_name(x))
    )

    return df


# --- Helper functions ---

# Function to split the `nwb_name` column
def split_nwb_name(nwb_name):
    """Turn the nwb_name into subject_id, session_date, nwb_suffix in order to be merged to
    the main df.

    Parameters
    ----------
    nwb_name : str. The name of the nwb file, e.g.
        "721403_2024-08-09_08-39-12.nwb"
        "685641_2023-10-04.nwb",
        ...

    Returns
    -------
    subject_id : str. The subject ID
    session_date : str. The session date
    nwb_suffix : int. The nwb suffix
    """
    parts = nwb_name.replace(".nwb", "").split("_")
    subject_id = parts[0]
    session_date = parts[1]
    try:
        nwb_suffix = int(parts[2].replace("-", "")) if len(parts) > 2 else 0
    except:
        nwb_suffix = 0
    return subject_id, session_date, nwb_suffix




df = fetch_mle_fitting_results(model_alias="Hattori2019")
st.write(df)

