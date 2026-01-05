#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
STEP 3 (Part 2) : Correct pressure datasets from long-term drift.

This example illustrates how to use the modelled long-term instrumental drift
from calibration (zero-pressure) sequences to correct the presure datasets.

1. Read (or extract) the zeros measurements stored in a .csv (ouput of STEP 1)
2. Read (or estimate the calibration curves) the calibration (zero-pressure) values stored in a .csv (ouput of STEP 2).
3. Model the calibration/drift curves
4. Save fitted drift model parameters to flat file and models into the .csv calibration file.

conda activate myrsk

07/2024 — alaurent
"""

import sys, os, glob, json
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt

# sys.path.append('/Users/alaure04/moby-data/CODES/Git_/src/A0A/')
from src.A0A.core import (read_A0A_data, read_events_log, load_drift_model)
from src.A0A.plots import plot_pressure, plot_deltaP
from src.A0A.models import exp_linear

########################################
#### DEFINE PATHS ####
root_path = '/Users/alaure04/moby-data/DATA/'
recover_date = '2025_09_30'
station_name = 'A0A_MAY33_R'
X=8 ## manually indexing of the number of deployment (coherent with previous nomenclature, but automatisation not allowed here)
rsk_reference = '208295_20250930_0551'

output_path = os.path.join(root_path, recover_date, station_name,  'figures/corrected/')

########################################
#### OPTIONS & AESTHETICS ####
text_size = 'large'

channels_colors = {'BB': 'violet',
                   'BPR1': 'orange',
                   'BPR2': 'darkgreen',
                   'T_ext' : 'tab:red'}


def main():

    #### Build input file paths
    data_path = os.path.join(root_path, recover_date, station_name, rsk_reference,f'{rsk_reference}_data_clean.txt')

    events_path = os.path.join(root_path, recover_date, station_name, rsk_reference, f'{rsk_reference}_events.txt')

    os.makedirs(output_path, exist_ok=True)

    today = date.today().strftime("%Y%m%d")   ### Warning not UTC

    #### Read pressure dataset
    ### Example made with the clean file not resampled
    A0A_clean_df = read_A0A_data(data_path)

    print(f'\n {today} - Cleaned from calibration pressure datasets\n')
    print(A0A_clean_df.head())
    print()

    #### Read models parameters
    sensor_name = 'BPR1'
    BPR1_f_path = os.path.join(root_path, recover_date, station_name, f'A0A{X}_{sensor_name}_drift_model.json')
    BPR1_fit = load_drift_model(BPR1_f_path)

    sensor_name = 'BPR2'
    BPR2_f_path = os.path.join(root_path, recover_date, station_name, f'A0A{X}_{sensor_name}_drift_model.json')
    BPR2_fit = load_drift_model(BPR2_f_path)

    #### Read log
    events_df, t_marine, t_atmo, t_error = read_events_log(events_path)

    ######################################
    #### Pre-process dataset
    ### Extract DateTimeIndex (automatically called "Time") into a real column
    A0A_clean_df['Timestamp'] = A0A_clean_df.index
    ### Add new column: convert DateTime into relative seconds from beginning
    A0A_clean_df['time_seconds'] = (A0A_clean_df['Timestamp'] - A0A_clean_df['Timestamp'].iloc[0]).dt.total_seconds()

    ######################################
    #### LONG TERM DRIFT CORRECTION

    #### GENERATE NEW MODEL WITH KNOWN PARAMETERS
    model_name = 'exp_linear'
    BPR1_model_explin = exp_linear(A0A_clean_df['time_seconds'], 
                                BPR1_fit['parameters']['a'], 
                                BPR1_fit['parameters']['tau'], 
                                BPR1_fit['parameters']['b'], 
                                BPR1_fit['parameters']['d'])
    BPR2_model_explin = exp_linear(A0A_clean_df['time_seconds'], 
                                BPR2_fit['parameters']['a'], 
                                BPR2_fit['parameters']['tau'], 
                                BPR2_fit['parameters']['b'], 
                                BPR2_fit['parameters']['d'])

    #### DRIFT CORECTION
    ### Subtract model to data in the numpy version for faster computation times
    ### Directly stored into the dataframe as a new column
    A0A_clean_df['BPR_dedrift_1'] = A0A_clean_df['BPR_pressure_1'].values - BPR1_model_explin.to_numpy()
    A0A_clean_df['BPR_dedrift_2'] = A0A_clean_df['BPR_pressure_2'].values - BPR2_model_explin.to_numpy()

    print()
    print(A0A_clean_df.head())
    print()

    ######################################
    #### STORE AND SAVE NEW DATASET WITH DEDRIFTED PRESSURE
    A0A_clean_df.to_csv(os.path.join(root_path, recover_date, station_name, rsk_reference, f'{rsk_reference}_data_corrected.txt'))

    #### Downsampling the data to minutes, hours and days
    # daily_df = A0A_clean_df.resample('1D').mean() ## not smoothed daily mean
    # hourly_df = A0A_clean_df.resample('1H').mean() ## not smoothed 1 hour averaged dataset
    A0A_1min_df = A0A_clean_df.resample('1min').mean() ## not smoothed 1 minute averaged dataset

    ### Save also the drift corrected 1 min resampled dataset
    A0A_1min_df.to_csv(os.path.join(root_path, recover_date, station_name, rsk_reference, f'{rsk_reference}_1min_corrected.txt'))

    ### With respect to YTT nomenclature
    A0A_1min_df.to_csv(os.path.join(root_path, recover_date, station_name, f'A0A{X}_OBP_1min.csv'))

    ######################################
    #### PLOT DRIFT CORRECTED TIMESERIES 
    outfile = f'Dedrifted_pressure_{station_name}_{today}.pdf'
    title = f'{station_name} - Plot dedrifted datasets'
    print(f'\n{today} -  Plot {title}.\n')
    plot_pressure(A0A_clean_df,
                keys=('BPR_dedrift_1', 'BPR_dedrift_2'),
                calibration_times=t_atmo,
                colors_code=channels_colors,
                fig_title=title,
                plot_fig=True, output_path=output_path, filenamout=outfile, savefig=True)

    ######################################
    #### PLOT DRIFT CORRECTED PRESSURE DIFFERENCE
    outfile = f'Drift_corrected_deltaP_{station_name}_{today}.pdf'
    title = u'{} - Drift corrected $\Delta$P = BPR2 - BPR1'.format(station_name)
    print(f'\n{today} -  Plot {title}.\n')
    plot_deltaP(A0A_clean_df,
                deltaP=None,
                keys=('BPR_dedrift_1', 'BPR_dedrift_2'),
                calibration_times=t_atmo,
                show_calibrations=True,
                title=title,
                text_size='large',
                plot_fig=True, output_path=output_path, filenamout=outfile, savefig=True)


if __name__ == "__main__":
    main()
