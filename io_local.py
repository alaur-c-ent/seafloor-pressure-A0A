
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
        usecols=range(0, 8, 1),
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
    events_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    ## "marine" ambiant-to_marine (external seafloor pressure) switch 
    times_marine = events_df.Type[events_df.Type == 'Valve movement - Marine'].index[1:-1]
    ## "ambiant" marine-to-ambiant (zero) switch 
    times_ambi = events_df.Type[events_df.Type == 'Valve movement - Atmospheric'].index 
    times_error  = events_df.Type[events_df.Type == 'Valve movement error'].index

    return events_df, times_marine, times_ambi, times_error


def flag_and_extract_zeros(df, times_marine, times_ambi, times_error):
    """
    Flag A0A data as Ambient (A), Zero (Z) or False (F),
    extract calibration sequences (Z), clean data (A only) by removing bad quality data (F).

    Parameters
    ----------
    df : pandas.DataFrame
        Time-indexed dataframe containing uncorrected data.
    times_marine : array-like pandas.Series
        Timestamps of atmospheric (zero-pressure) valve switches.
    times_ambi : array-like pandas.Series
        Timestamps of marine valve switches.
    times_error : int, optional
        Timestamps of marine valve movements errors.

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe with F data removed.
    zeros_df : pandas.DataFrame
        Dataframe containing only zero-pressure sequences.
    """
    ### We tag the A0A record by adding a Type flag: 
    ### A = Ambient, Z = Zeros, F = Flagged as bad
    df.insert(len(df.iloc[0]), 'Type', 'A')
    df['Type'] = 'A'
    
    for ta, tm in zip(times_ambi, times_marine):
        ### Re-write over the flagged 'A' only for zero-measurments windows we want to keep
        Z_start = ta - pd.Timedelta(seconds=10)
        ### Extract 20 minutes long zeros
        Z_end = tm + pd.Timedelta(minutes=20, seconds=10)
        ### WARNING, it can change from one instrument to another (I don't know why)
        ### from t + 20 minutes - 2s to tm - 2s
        # Z_end = tm - pd.Timedelta(seconds=2)
        df.loc[Z_start:Z_end, 'Type'] = 'Z'

    #### FLAG valve movement errors etc
    ## Delete for the MAY33 dataset, keep values even of no calib
    for t_err in times_error:
        err_start = t_err - pd.Timedelta(minutes=25)
        df.loc[err_start:, 'Type'] = 'F'

    #### Keep only the Z FLAG 
    zeros_df = df[df.Type == 'Z']
    #### Keeep only the good quality records
    # df_clean = df[df.Type != 'F']
    df_clean = df[df.Type == 'A']

    return zeros_df, df_clean


def calibrations(zeros_df, times_ambi, window, lim_inf, lim_sup):
    """
    Compute calibration values from zero-pressure segments.

    Parameters
    -------
    zeros_df : pandas.DataFrame
        Dataframe containing only zero-pressure sequences.
    times_ambi : array-like pandas.Series
        Timestamps of marine valve switches.
    window : pandas.Timedelta
        Calibration length in seconds (20 minutes -> 1200 s).  
    lim_inf : int
        Relative time in seconds after internal valve switch 
        of the selected stable window to compute calibration value.
    lim_sup
        End time (in relative seconds) 
        of the selected stable window to compute calibration value.
             
    Returns
    -------
    calib_df : pandas.DataFrame
        Calibration time series for each pressure sensor.
    """
    results = []

    calib_n = 1
    for t in times_ambi:
        seg = zeros_df.loc[t:t+window]
        if seg.empty:
                print(f'No data for segment starting at {t}')
                continue
        elapsed_s = seg['time_seconds'] - seg['time_seconds'].iloc[0]
        
        ## Extract the stable zeros selected window
        mask = (elapsed_s >= lim_inf) & (elapsed_s <= lim_sup)
        sel = seg.loc[mask]
        if sel.empty:
            print(f"No data in {lim_inf}–{lim_sup}s window for segment starting at {t}")
        P1_zero = sel['BPR_pressure_1']
        P2_zero = sel['BPR_pressure_2']
        P_barom_zero = sel['Barometer_pressure']
        ### Correct zeros from barometer pressure
        calib1_value = (P1_zero - P_barom_zero).mean()
        calib2_value = (P2_zero - P_barom_zero).mean()
        ### Store final calibration zero-pressure value
        results.append({'id': calib_n, 
                        'Date': t.date(), 
                        'Calib_1': calib1_value, 
                        'Calib_2': calib2_value,
                        })
        calib_n += 1
  
    print(f'\nThe total number of calibration segments is: {calib_n}\n')

    ### Concatenate into a new dataframe 
    calib_df = pd.DataFrame.from_records(results, 
                                         index='id', 
                                         columns=results[0].keys()) #.set_index('id')

    ### Normalise the calibration by sutracting the inital state (first value)
    calib_df['Calib_1'] -= calib_df['Calib_1'].values[0]
    calib_df['Calib_2'] -= calib_df['Calib_2'].values[0]

    return calib_df
