import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)

def fetch_fip_data(client):
    # search for records that have the "fib" (for fiber photometry) modality in data_description
    logger.warning("fetching 'fib' records...")
    modality_results = client.retrieve_docdb_records(filter_query={
        "data_description.modality.abbreviation": "fib"
    })              

    # there are more from the past that didn't specify modality correctly. 
    # until this is fixed, need to guess by asset name 
    logger.warning("fetching FIP records by name...")
    name_results = client.retrieve_docdb_records(filter_query={
        "name": {"$regex": "^FIP.*"}
    })
    
    # make some dataframes from these two queries
    records_by_modality_df = pd.DataFrame.from_records([ map_record_to_dict(d) for d in modality_results ])
    records_by_name_df = pd.DataFrame.from_records([ map_record_to_dict(d) for d in name_results ])

    # currently there are some sessions uploaded twice in two different locations.
    # let's filter out the ones in aind-ophys-data, a deprecated location
    dup_df = records_by_name_df[records_by_name_df.duplicated('session_name',keep=False)]
    dup_df = dup_df[dup_df.location.str.contains("aind-ophys-data")]
    records_by_name_df = records_by_name_df.drop(dup_df.index.values)

    # now we have a master data frame
    combined_df = pd.concat([records_by_modality_df, records_by_name_df], axis=0).drop_duplicates()
    
    # let's get processed results too
    logger.warning("fetching processed results...")
    processed_results = client.retrieve_docdb_records(filter_query={
        "name": {"$regex": "^behavior_.*processed_.*"}
    }) 
    
    # converting to a dictionary
    processed_results_by_name = { r['name']: r for r in processed_results }  
        
    # adding two columns to our master dataframe - result name and result s3 location
    combined_df['results'] = combined_df.session_name.apply(lambda x: find_result(x, processed_results_by_name).get('name'))
    combined_df['results_location'] = combined_df.session_name.apply(lambda x: find_result(x, processed_results_by_name).get('location'))
    
    return combined_df


def map_record_to_dict(record):
    """ function to map a metadata dictionary to a simpler dictionary with the fields we care about """
    dd = record.get('data_description', {}) or {}
    creation_time = dd.get('creation_time', '') or ''
    subject = record.get('subject', {}) or {}
    subject_id = subject.get('subject_id') or ''
    subject_genotype = subject.get('genotype') or ''

    return {
        'location': record['location'],
        'session_name': record['name'],
        'creation_time': creation_time,
        'subject_id': subject_id,
        'subject_genotype': subject_genotype,
    }


def find_result(x, lookup):
    """ lazy shortcut; we can look for a corresponding result by seeing if part of another record's prefix """
    for result_name, result in lookup.items():
        if result_name.startswith(x):
            return result
    return {}
    

