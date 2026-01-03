#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
STEP 1 

Read the A0A dataset (read {serialnumber}_{YYYYMMDD}_{HHMM}_data.txt file) 
and the events log (read {sn}_{YYYYMMDD}_{HHMM}_event.txt file). .

Diagnostic plots (full timeseries without correction) : 
- Uncorrected pressure timeseries with associated events in log (valve movements and errors). 
- Internal pressure and temperature, both pressure sensors temperatures, and seafloor bottom temperature 

FLAG zeros segments (calibration sequences) as Z, errors as F and, A, the rest.

conda activate myrsk

08/2025 — alaurent
"""

import os
import numpy as np
import pandas as pd
from datetime import date, timedelta

#sys.path.append('/Users/alaure04/moby-data/CODES/Git_/src/A0A/')
from io_local import (read_A0A_data, read_events_log, flag_and_extract_zeros)
from plots import (plotlog, plot_barometer_and_temperatures)


########################################
#### DEFINE PATHS ####
root_path = '/Users/alaure04/moby-data/DATA/'
recover_date = '2025_09_30'
station_name = 'A0A_MAY33_R'
# nbr_deploy = 8 ### 8th deployment of A0A
rsk_reference = '208295_20250930_0551'
rsk_ref_lst = rsk_reference.split('_')

output_path = os.path.join(root_path, recover_date, station_name, 'figures/raw_pressure/')

########################################
#### OPTIONS & AESTHETICS ####
text_size = 'large'

channels_colors = {'BB': 'violet',
                   'BPR1': 'orange',
                   'BPR2': 'darkgreen'}

calib_color_ = 'red'

show_figure = True
save_figure = True


def main():

    #### Build input file paths
    data_path = os.path.join(root_path, recover_date, station_name, rsk_reference,f'{rsk_reference}_data.txt')

    events_path = os.path.join(root_path, recover_date, station_name, rsk_reference, f'{rsk_reference}_events.txt')

    os.makedirs(output_path, exist_ok=True)

    #### Read data
    A0A_df = read_A0A_data(data_path)
    #### Read log
    events_df, t_marine, t_atmo, t_error = read_events_log(events_path)

    #### Cutting edges
    #### As I know that the deploy or recovery durations are about 2 hours, however depending on depth, 
    #### I prefere removing the first and last 6 hours of data
    edge_window = pd.Timedelta(hours=6)    
    start_cut = A0A_df.index.min() + edge_window
    end_cut   = A0A_df.index.max() - edge_window
    A0A_df = A0A_df.loc[start_cut:end_cut]
    
    start_date = A0A_df.index.min().date()
    end_date = A0A_df.index.max().date()

    today = date.today().strftime("%Y%m%d")   ### Warning not UTC

    print(f'\n{today} - Analysing pressure dataset from {start_date} to {end_date}.')
    print(A0A_df.head())

    #### Plot
    pressure_key = 'BPR_pressure_1'  # or 'BPR_pressure_2'
    outfile = f'log_{station_name}_BPR{pressure_key[-1]}_{start_date}_{end_date}.pdf'

    ######################################
    #### PLOT A0A UNCORRECTED DATA WITH LOG EVENTS
    print(f'\n{today} - Diagnostic plot of {station_name}.\n')
    plotlog(
        A0A_df,
        key=pressure_key,
        events=events_df,
        title=f'{station_name} — {start_date} to {end_date}',
        plot_log=show_figure, output_path=output_path, filenamout=outfile, savefig=save_figure)
    
    ######################################
    #### CHECK AT THE INTERNAL STATE OF THE INSTRUMENT 
    outfile = f'Temp_n_intern_state_{station_name}_{today}.pdf'
    title = f'{station_name} - Barometer and temperature datasets'
    print(f'\n{today} -  Plot {title}.\n')
    plot_barometer_and_temperatures(A0A_df, 
                                    calibration_times=t_atmo, 
                                    colors=channels_colors, 
                                    title=title, 
                                    text_size='large', 
                                    plot_fig=show_figure, output_path=output_path, filenamout=outfile, savefig=save_figure)

    ######################################
    ##### FLAGs WINDOWS AS AMBIANT (A), ZEROS (Z) or FALSE (to be removed)
    zeros_df, df_clean = flag_and_extract_zeros(A0A_df,  t_marine, t_atmo, t_error)

    ######################################
    #### WRITE CLEANED DATASETS
    #### Zeros and NO-ZEROS dataset outputs
    zeros_df.to_csv(os.path.join(root_path, recover_date, station_name, f"{station_name}_zeros.csv"))
    df_clean.to_csv(os.path.join(root_path, recover_date, station_name, rsk_reference, f'{rsk_reference}_data_clean.txt'))

    #### Downsampling the data to minutes, hours and days
    daily_mean_df = df_clean.resample('1D').mean() ## not smoothed daily mean
    # A0A_1min_df = df_clean.resample('1H').mean() ## not smoothed 1 hour averaged dataset
    A0A_1min_df = df_clean.resample('1min').mean() ## not smoothed 1 minute averaged dataset

    A0A_1min_df.to_csv(os.path.join(root_path, recover_date, station_name, rsk_reference, f'{rsk_reference}_1min.txt'))


if __name__ == "__main__":
    main()
