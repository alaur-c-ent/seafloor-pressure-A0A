
"""

Input and ouput functions for A0A seafloor pressure datasets procesisng. 

2024/2025 — alaurent
"""

import pandas as pd


def read_A0A_data(data_path):
    """
    Read raw RBR data file extracted from the RBR software from the instrument.

    Parameters : 
    data_path : str
        Path to the {serial_number}_{YYYYMMDD}_{HHMM}_data.txt file.

    Returns pandas.DataFrame
            Time-indexed dataframe containing both bottom pressures, temperature and barometer data.
    """
    df = pd.read_csv(data_path, parse_dates=True, index_col=0, skiprows=1,
        usecols=range(7),
        names=['Time',
               'BPR_temp_1',
               'BPR_pressure_1',
               'BPR_temp_2',
               'BPR_pressure_2',
               'External_temp',
               'Barometer_temp',
               'Barometer_pressure'],
        )
    return df


def read_events_log(data_path):
    """
    Read RBR events log and extract valve rotation times.

    Parameters : 
    data_path : str
        Path to the {serial_number}_{YYYYMMDD}_{HHMM}_events.txt file.

    Returns pandas.Dataframe
            Dataframe containing atmospheric, marine and error timestamps as pandas.Series.
    """
    events_df = pd.read_csv(data_path, index_col=0, parse_dates=True),
    ## "marine" ambiant-to_marine (external seafloor pressure) switch 
    times_marine = events_df.Type[events_df.Type == 'Valve movement - Marine'].index[1:]
    ## "ambiant" marine-to-ambiant (zero) switch 
    times_ambi = events_df.Type[events_df.Type == 'Valve movement - Atmospheric'].index 
    times_error  = events_df.Type[events_df.Type == 'Valve movement error'].index

    return events_df, times_marine, times_ambi, times_error



