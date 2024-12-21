
""" Helper functions to reformat the data
"""
import re


# Function to split the `nwb_name` column
def split_nwb_name(nwb_name):
    """Turn the nwb_name into subject_id, session_date, nwb_suffix in order to be merged to
    the main df.

    Parameters
    ----------
    nwb_name : str. The name of the nwb file. This function can handle the following formats:
        "721403_2024-08-09_08-39-12.nwb"
        "685641_2023-10-04.nwb",
        "behavior_754280_2024-11-14_11-06-24.nwb",
        "behavior_1_2024-08-05_15-48-54",
        "
        ...

    Returns
    -------
    subject_id : str. The subject ID
    session_date : str. The session date
    nwb_suffix : int. The nwb suffix (converted from session time if available, otherwise 0)
    """

    pattern = R"(?:\w+_)?(?P<subject_id>\d+)_(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<time>\d{2}-\d{2}-\d{2}))?(?:.*)?"
    matches = re.search(pattern, nwb_name)
    
    if not matches:
        return None, None, 0

    subject_id = matches.group("subject_id")
    session_date = matches.group("date")
    session_time = matches.group("time")
    if session_time:
        nwb_suffix = int(session_time.replace("-", ""))
    else:
        nwb_suffix = 0

    return subject_id, session_date, nwb_suffix