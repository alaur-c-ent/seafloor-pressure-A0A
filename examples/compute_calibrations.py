#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
STEP 2: Drift estimation from calibration (zero-pressure) sequences

This example illustrates the standard usage of the STEP 2 functions.

1. Read zero-pressure dataset (output of STEP 1).
2. Compute calibration values from selected stable time windows during calibration sequences.
3. Plot zero segments (control)
    Looking for calibration "outliers" on internal pressures and/or external temperature.
4. Plot the calibration (or drift) curves.

For the drift curves modelling, see STEP 3 example (fit_calib.py)

conda activate myrsk

06/2024 — alaurent
"""

import sys, os
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt 

from src.A0A.core import read_events_log, calibrations
from src.A0A.plots import (plot_calibrations, plot_check_zeros,
                            plot_calibration_curves)


########################################
#### DEFINE PATHS ####
root_path = '/Users/alaurent/moby-data/DATA/'
mission = 'MAYOBS'


### Name the station
station_name = 'MAY_C' 
### Ruskin name of the file
rsk_reference = '{serialnumber}_{recover_date_YYMMDD}_{recover_time_HHMM}'

rsk_ref_lst = rsk_reference.split('_')
recover_date = rsk_ref_lst[1]

output_path = os.path.join(root_path, mission, recover_date,  'figures/calib/')
data_folderout = os.path.join(root_path, mission, recover_date, station_name)

########################################
#### OPTIONS & AESTHETICS ####
text_size = 'large'

channels_colors = {'BB': 'violet',
                   'BPR1': 'orange',
                   'BPR2': 'darkgreen',
                   'T_ext' : 'tab:red'}


def main():

    #### Build input file paths
    zeros_path = os.path.join(root_path, mission, recover_date, station_name, f'{station_name}_zeros.csv')

    events_path = os.path.join(root_path, mission, rsk_reference, f'{rsk_reference}_events.txt')

    os.makedirs(output_path, exist_ok=True)

    #### Read extracted zeros (from STEP1)
    zeros_df = pd.read_csv(zeros_path, parse_dates=True, index_col=0)
    #### Read log
    events_df, t_marine, t_atmo, t_error = read_events_log(events_path)

    today = date.today().strftime("%Y%m%d")   ### Warning not UTC

    ## Time vector in seconds from beginning
    zeros_df['time_seconds'] = (zeros_df.index - zeros_df.index[0]).total_seconds()

    print(f'\n{today} - Extracted zeros from {station_name}.\n')
    print(zeros_df.head())
    print()
    print(zeros_df.tail())

    ######################################
    #### RETRIEVE ZERO-PRESSURE VALUES FROM CALIBRATION SEQUENCES
   

    ### chosen borns in seconds after marine to atmospheric rotation
    if station_name == "A0A_MAY18_C":
        window = pd.Timedelta(seconds=250) # also working with datetime timedelta
        win_lims = (230, 240)
        ylims = (None, None)
        dP_ylims = (0.01, 0.45)

    elif station_name == 'A0A_MAY25_C':
        window = pd.Timedelta(seconds=20*60)
        win_lims = (600, 1000)
        ylims = (8.5, 24.)
        dP_ylims = (0.2, 0.35)   

    elif station_name == 'A0A_MAY30_R':
        window = pd.Timedelta(seconds=20*60)
        win_lims = (600, 1000)      
        ylims = (9.5, 9.7)
        dP_ylims = (0.01, 0.15)   

    elif station_name == 'A0A_MAY30_C':
        window = pd.Timedelta(seconds=20*60)
        win_lims = (600, 1000)      
        ylims = (9.1, 9.4)
        dP_ylims = (0.01, 0.15)   
        
    else:
        ### programmed duration between 2021-2025
        window = pd.Timedelta(seconds=20*60)
        win_lims = (600, 1000)
        ylims = (9.5, 9.85)
        dP_ylims = (0.01, 0.15)

    calib_df = calibrations(zeros_df, 
                            ['BPR_pressure_1', 'BPR_pressure_2', 'Barometer_pressure'], 
                            t_atmo, window, win_lims)

    print(f'\n {today} - {len(calib_df)} calibrations (zeros) values\n')
    print(calib_df.to_string())
    print()

    ### Save calibration dataframe output to csv
    # X=8 ## manually indexing of the number of deployment (coherent with previous nomenclature, but automatisation not allowed here)
    calib_df.to_csv(os.path.join(data_folderout,  f"{station_name}_Calib.csv"))

    ######################################
    #### PLOT ALL CALIBRATION SEQUENCES(ZERO SEGMENTS) IN A SINGLE WINDOW
    pressure_cols = ['BPR_pressure_1', 'BPR_pressure_2']
    # temp_cols = ['BPR_temp_1', 'BPR_temp_2', 'Barometer_temp', 'External_temp']
        
    fig = plot_calibrations(zeros_df, calibration_times=t_atmo, 
                      keys=pressure_cols, 
                      window=window, select_window=win_lims,
                      ylim=ylims, ##example from the last deployment, manually adjusted
                      cmap='viridis', figsize=(8, 4))

    fig.savefig(os.path.join(output_path, f"{station_name}_zero_segments.pdf"), dpi=300)
    plt.show()
    # plt.close(fig)

    ######################################
    #### CONTROL PLOT
    ### Barometer pressure and external temperature during calibration sequences
    #### Example with highlight ids from last deployment (see Laurent et al., 2026) 
    fig = plot_check_zeros(zeros_df, calibration_times=t_atmo, 
                           window=window, select_window=win_lims, 
                           ylim=ylims, #highlight_ids=[12, 13], #11, 14, 15
                           colors_code=None, figsize=(8, 6))

    fig.savefig(os.path.join(output_path, f"{station_name}_check_for_outliers_allvar.pdf"), dpi=300)
    plt.show()
    # plt.close(fig)

    ######################################
    #### PLOT CALIBRATION/DRIFTS CURVES
    use_cmap = False
    # use_cmap = True
    title = f'{station_name} –– Calibration curves: P_barometric - P_internal'
    fig = plot_calibration_curves(calib_df, cols=('Calib_1', 'Calib_2'), 
                                  title=title, colors_code=None, use_cmap=use_cmap,
                                  ylim=dP_ylims, text_size=text_size, figsize=(10, 5))

    if not use_cmap:
        fig.savefig(os.path.join(output_path, f"{station_name}_drift_curves.pdf"), dpi=300)
    else:
        fig.savefig(os.path.join(output_path, f"{station_name}_drift_curves_cmap.pdf"), dpi=300)
    plt.show()
    # plt.close(fig)



if __name__ == "__main__":
    main()


