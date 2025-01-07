
""" Helper functions to reformat the data
"""
import re
import pandas as pd

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


def formatting_metadata_df(df, source_prefix="docDB"):
    """Formatting metadata dataframe
    Given a dataframe with a column of "name" that contains nwb names
    1. parse the nwb names into subject_id, session_date, nwb_suffix.
    2. remove invalid subject_id
    3. handle multiple sessions per day
    """

    df.rename(columns={col: f"{source_prefix}_{col}" for col in df.columns}, inplace=True)
    new_name_field = f"{source_prefix}_name"

    # Create index of subject_id, session_date, nwb_suffix by parsing nwb_name
    df[["subject_id", "session_date", "nwb_suffix"]] = df[new_name_field].apply(
        lambda x: pd.Series(split_nwb_name(x))
    )
    df["session_date"] = pd.to_datetime(df["session_date"])
    df = df.set_index(["subject_id", "session_date", "nwb_suffix"]).sort_index(
        level=["session_date", "subject_id", "nwb_suffix"],
        ascending=[False, False, False],
    )

    # Remove invalid subject_id
    df = df[(df.index.get_level_values("subject_id").astype(int) > 300000) 
            & (df.index.get_level_values("subject_id").astype(int) < 999999)]

    # --- Handle multiple sessions per day ---
    # Build a dataframe with unique mouse-dates.
    # If multiple sessions per day, combine them into a list of 'name'
    df_unique_mouse_date = (
        df.reset_index()
        .groupby(["subject_id", "session_date"])
        .agg({new_name_field: list, **{col: "first" for col in df.columns if col != new_name_field}})
    ).sort_index(
        level=["session_date", "subject_id"], # Restore order 
        ascending=[False, False],
    )
    # Add a new column to indicate multiple sessions per day
    df_unique_mouse_date[f"{source_prefix}_multiple_sessions_per_day"] = df_unique_mouse_date[
        new_name_field
    ].apply(lambda x: len(x) > 1)

    # Also return the dataframe with multiple sessions per day
    df_multi_sessions_per_day = df_unique_mouse_date[
        df_unique_mouse_date[f"{source_prefix}_multiple_sessions_per_day"]
    ]

    # Create a new column to mark duplicates in the original df
    df.loc[
        df.index.droplevel("nwb_suffix").isin(df_multi_sessions_per_day.index), 
        f"{source_prefix}_multiple_sessions_per_day"
    ] = True

    return df, df_unique_mouse_date, df_multi_sessions_per_day