"""Code to fetch data from docDB by David Feng
"""

import logging
import time

import pandas as pd
import semver
import streamlit as st
import re
import numpy as np
logger = logging.getLogger(__name__)

from aind_data_access_api.document_db import MetadataDbClient


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


def fetch_individual_procedures(r):
    version = semver.Version.parse((r.get('procedures') or {}).get('schema_version', '0.0.0'))
    if version >= "0.8.1": # post introduction of Surgery concept
        sub_procs = (r.get('procedures') or {}).get('subject_procedures') or {}        
        for sub_proc in sub_procs:        
            if sub_proc.get('procedure_type') == 'Surgery':
                yield from sub_proc['procedures']
    else: # pre Surgery
        sub_procs = (r.get('procedures') or {}).get('subject_procedures') or {}
        yield from sub_procs
        
def fetch_fiber_probes(r):
    probes = []
    for sp in fetch_individual_procedures(r):
        if sp['procedure_type'] == 'Fiber implant':
            probes += sp['probes']
    return probes
                
def fetch_injections(r):    
    injections=[]
    for sp in fetch_individual_procedures(r):        
        if sp['procedure_type'] in ('Nanoject injection', 'Nanoject (Pressure)',"Iontophoresis injection","ICM injection"):
            ims = sp['injection_materials'] or []
            injections.append({
                'injection_materials': [ im['name'] for im in ims ],
                'ap': sp['injection_coordinate_ap'],
                'ml': sp['injection_coordinate_ml'],
                'depth': sp['injection_coordinate_depth'] # somehow depth given as a list 
            })            
       
    return injections
    

def get_viruses(injections):
    virus_names = []
    NM_recorded = []
    
    if injections:
        virus_names = [inj['injection_materials'][0] for inj in injections if inj['injection_materials']]
        # TODO: fixing this once we have a better mapping of virus to NM. 
        # Not including this in the newest version of streamlit to avoid confusion 
        NM_patterns = {"DA": "DA|dLight", 
                       "NE":"NE|nLight", 
                       "Ach":"Ach", 
                       "5HT":"5HT", 
                       "GCaMP":"GCaMP"}
        
        
        for inj in injections:
            for NM, NM_names_in_virus in NM_patterns.items():
                if inj['injection_materials'] and re.search(NM_names_in_virus, inj['injection_materials'][0]):
                    if NM_names_in_virus == "GCaMP":
                        loc = inj['ap'] + ',' + inj['ml'] + ',' + inj['depth']
                        NM_recorded.append(loc)
                    else:
                        NM_recorded.append(NM)
    return virus_names, NM_recorded   
    

def strip_dict_for_id(co_asset_id_dict_list):
    result_list = []
    
    for asset_id in co_asset_id_dict_list:
        try:
            if asset_id is None or len(asset_id) == 0:
                result_list.append(None)
            elif isinstance(asset_id, list):
                result_list.append(asset_id[0]['Code Ocean'])
            elif isinstance(asset_id, dict) and 'Code Ocean' in asset_id:
                if len(asset_id['Code Ocean']) == 0:
                    result_list.append(None)
                else:
                    result_list.append(asset_id['Code Ocean'][0])
            else:
                result_list.append(None)  # Default case if none of the conditions are met
        except (KeyError, IndexError, TypeError):
            result_list.append(None)  # Catch any errors and append None
            
    return result_list                   

def fetch_fip_data(client):
    # To compare the FIP pipeline with my temporary pipeline for all behavior sessions, we should 
    # query "behavior" from the data_description.modality.abbreviation field.
    logger.warning("fetching 'behavior' records...")
    modality_results = client.retrieve_docdb_records(
        filter_query={"data_description.modality.abbreviation": "behavior"},
        paginate_batch_size=500
    )              
    logger.warning(f"found {len(modality_results)} results")

    # there are more from the past that didn't specify modality correctly. 
    # until this is fixed, need to guess by asset name 
    logger.warning("fetching FIP records by name...")
    name_results = client.retrieve_docdb_records(
        filter_query={"name": {"$regex": "^FIP.*"}},
        paginate_batch_size=500
    )
    logger.warning(f"found {len(name_results)} results")

    # in case there is overlap between these two queries, filter down to a single list with unique IDs
    unique_results_by_id = {**{ r['_id']: r for r in modality_results }, **{ r['_id']: r for r in name_results }}
    results = list(unique_results_by_id.values())
    logger.warning(f"found {len(results)} unique results")
    
    # filter out results with 'processed' in the name because I can't rely on data_description.data_level :(
    results = [ r for r in results if not 'processed' in r['name'] ]
    
    # make a dataframe
    records_df = pd.DataFrame.from_records([map_record_to_dict(d) for d in results ])
    
    # PREVIOUSLY, there are some sessions uploaded twice in two different locations.
    # let's filter out the ones in aind-ophys-data, a deprecated location
    # this is no longer a problem-- so I'm taking off the drop but keeping the dupe check on. 
    dup_df = records_df[records_df.duplicated('session_name',keep=False)]
    dup_df = dup_df[dup_df.session_loc.str.contains("aind-ophys-data")]
    if len(dup_df):
        logger.warning('duplicated entries found, please fix')
    # records_df = records_df.drop(dup_df.index.values)
    
    # let's get processed results too
    logger.warning("fetching processed results...")
    processed_results = client.retrieve_docdb_records(filter_query={
        "name": {"$regex": "^behavior_.*processed_.*"}
    }) 
    
    # converting to a dictionary
    processed_results_by_name = { r['name']: r for r in processed_results }  
        
    # adding two columns to our master dataframe - result name and result s3 location
    records_df['processed_session_name'] = records_df.session_name.apply(lambda x: find_result(x, processed_results_by_name).get('name'))
    records_df['processed_session_loc'] = records_df.session_name.apply(lambda x: find_result(x, processed_results_by_name).get('location'))
    # get external_links, strip it down to the string 
    co_data_asset_id_processed = records_df.session_name.apply(lambda x: find_result(x, processed_results_by_name).get('external_links'))
    records_df['processed_CO_dataID'] = strip_dict_for_id(co_data_asset_id_processed)
    records_df['CO_dataID'] = strip_dict_for_id(records_df['CO_dataID'])
    return records_df




def map_record_to_dict(record):
    """ function to map a metadata dictionary to a simpler dictionary with the fields we care about """
    dd = record.get('data_description', {}) or {}
    co_data_asset_id_raw = record.get('external_links')
    creation_time = dd.get('creation_time', '') or ''
    subject = record.get('subject', {}) or {}
    subject_id = subject.get('subject_id') or ''
    subject_genotype = subject.get('genotype') or ''
    session = record.get('session') or {}
    task_type = session.get('session_type') or ''

    try:
        injections = fetch_injections(record)
        virus_names, NM_recorded = get_viruses(injections)
    except:
        injections, virus_names, NM_recorded = [], [], []
        
    return {
        'session_loc': record['location'],
        'session_name': record['name'],
        'creation_time': creation_time,
        'CO_dataID' : co_data_asset_id_raw,
        'subject_id': subject_id,
        'subject_genotype': subject_genotype,
        'fiber_probes': str(fetch_fiber_probes(record)),
        'injections': str(injections),
        'task_type': task_type, 
        'virus':virus_names, 
        # 'NM_recorded':NM_recorded
        
    }

def find_result(x, lookup):
    """ lazy shortcut; we can look for a corresponding result by seeing if part of another record's prefix """
    for result_name, result in lookup.items():
        if result_name.startswith(x):
            return result
    return {}
    

