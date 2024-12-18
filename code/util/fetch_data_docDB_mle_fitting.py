"""Code to fetch mle_fitting results from docDB
Han Hou
"""

import logging
import time
from functools import reduce

import pandas as pd
import streamlit as st
import numpy as np

from aind_data_access_api.document_db import MetadataDbClient

@st.cache_resource
def load_client():
    return MetadataDbClient(
        host="api.allenneuraldynamics-test.org",    # From the test docDB  
        database="analysis",
        collection="dynamic_foraging_analysis",
    )

client = load_client()

MAPPER_PRESET_TO_ALIAS = {
    "Win-Stay-Lose-Shift": "WSLS",
    "Rescorla-Wagner": "QLearning_L1F0_epsi",
    "Bari2019": "QLearning_L1F1_CK1_softmax",
    "Hattori2019": "QLearning_L2F1_softmax",
}

@st.cache_data(ttl=3600 * 12)  # Cache the df_docDB up to 12 hours
def fetch_mle_fitting_results(
    model_presets=MAPPER_PRESET_TO_ALIAS.keys(),
    fields=[
        "params",
        "log_likelihood",
        "AIC",
        "BIC",
        "LPT",
        "LPT_AIC",
        "LPT_BIC",
        "prediction_accuracy",
        # "cross_validation.prediction_accuracy_test",
        # "cross_validation.prediction_accuracy_fit",
        # "cross_validation.prediction_accuracy_test_bias_only",        
    ],
):
    """Fetch mle fitting results from docDB

    model_presets: list of str. The model presets to fetch.
    fields: list of str. The fields to fetch.
    
    returns: pd.DataFrame. The fetched data.
    """

    # --- Fetch MLE fitting results ---
    dfs = []
    for model_preset in model_presets:
        start = time.time()
        print(f"Fetching {model_preset}...")
        model_alias = MAPPER_PRESET_TO_ALIAS[model_preset]
        
        # Using filter/projection with paginate
        filter = {"analysis_results.fit_settings.agent_alias": model_alias}
        projection = {       
                    "_id": 0,
                    "nwb_name": 1,
                    **{
                        f"{model_preset}.{field}": f"$analysis_results.{field}"
                        for field in fields
                    },
                }
        
        records = client.retrieve_docdb_records(
            filter_query=filter, 
            projection=projection,
            paginate=True,
            paginate_batch_size=5000,
            )
                
        dfs.append(pd.json_normalize(records))
        print(f"finished in {time.time() - start} sec.")

    # Merge the dfs
    df = reduce(lambda left, right: pd.merge(left, right, on="nwb_name", how="outer"), dfs)

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