#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
# src/a0a/io.py

STEP 1

Read the A0A dataset (read {serialnumber}_{YYYYMMDD}_{HHMM}_data.txt file) 
and the events log (read {sn}_{YYYYMMDD}_{HHMM}_event.txt file). .

Diagnostic plots (full timeseries without correction) : 
- Uncorrected pressure timeseries with associated events in log (valve movements and errors). 
- Internal pressure and temperature, both pressure sensors temperatures, and seafloor bottom temperature 

FLAG zeros segments (calibration sequences) as Z, errors as F and, A, the rest.

Save clean dataset in txt file format. 
Resample dataset, save the 1 minute downsampled version only. 

Plot uncorrected cleaned seafloor pressure timeseries. 
Plot the uncorretected ∆P = BPR2 - BPR1 signal as instrumental control. 

conda activate myrsk

08/2025 — alaurent
"""

import os, sys
import pandas as pd
from datetime import date

from src.A0A.core import read_A0A_data, read_events_log, flag_and_extract_zeros
from src.A0A.plots import plotlog, plot_barometer_and_temperatures, plot_pressure, plot_deltaP

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

output_path = os.path.join(root_path, mission, recover_date,  'figures/parsing/')
data_folderout = os.path.join(root_path, mission, recover_date, station_name)

########################################
#### OPTIONS & AESTHETICS ####
text_size = 'large'

channels_colors = {'BPR_pressure_1' : 'orange',
                      'BPR_pressure_2' : 'darkgreen',
                      'Barometer_pressure' : 'violet',
                      'External_temp' : 'tab:red'}

save_figure = True


def main():

    #### Build input file paths
    data_path = os.path.join(root_path, mission, rsk_reference, f'{rsk_reference}_data.txt')

    events_path = os.path.join(root_path, mission, rsk_reference, f'{rsk_reference}_events.txt')

    ### Create new folders for figures and (cleaned) datasets if not existing already
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(data_folderout, exist_ok=True)

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

    print(f'\n{today} - Analysing pressure dataset from {start_date} to {end_date}.\n')
    print(A0A_df.head())

    ######################################
    #### PLOT A0A UNCORRECTED DATA WITH LOG EVENTS
    pressure_key = 'BPR_pressure_1'  # or 'BPR_pressure_2'
    outfile = f'log_{station_name}_BPR{pressure_key[-1]}_{start_date}_{end_date}.pdf'

    print(f'\n{today} - Diagnostic plot of {station_name}.\n')
    plotlog(
        A0A_df,
        key=pressure_key,
        events_log=events_df,
        title=f'{station_name} — {start_date} to {end_date}',
        plot_log=save_figure, output_path=output_path, filenamout=outfile, savefig=save_figure)
    
    ######################################
    #### CHECK AT THE INTERNAL STATE OF THE INSTRUMENT 
    outfile = f'Temp_n_intern_state_{station_name}_{today}.pdf'
    title = f'{station_name} - Barometer and temperature datasets'
    print(f'\n{today} -  Plot {title}.\n')
    plot_barometer_and_temperatures(A0A_df, 
                                    calibration_times=t_atmo, 
                                    colors_code=channels_colors, 
                                    title=title, 
                                    text_size='large', 
                                    plot_fig=save_figure, output_path=output_path, filenamout=outfile, savefig=save_figure)

    ######################################
    ##### FLAGs WINDOWS AS AMBIANT (A), ZEROS (Z) or FALSE (to be removed)
    ### Duration in seconds of the calibration sequence (i.e, zero-pressure measurments)
    win_calib = 1200 # in seconds
    zeros_df, A0A_clean_df = flag_and_extract_zeros(A0A_df, win_calib, t_atmo, t_error)

    #### Zeros and NO-ZEROS dataset outputs
    zeros_df.to_csv(os.path.join(data_folderout, f"{station_name}_zeros.csv"))
    A0A_clean_df.to_csv(os.path.join(data_folderout, f'{rsk_reference}_data_clean.txt'))

    #### Downsampling the data to minutes, hours and days
    # daily_df = A0A_clean_df.resample('1D').mean() ## not smoothed daily mean
    # A0A_1min_df = A0A_clean_df.resample('1H').mean() ## not smoothed 1 hour averaged dataset
    A0A_1min_df = A0A_clean_df.resample('1min').mean() ## not smoothed 1 minute averaged dataset

    A0A_1min_df.to_csv(os.path.join(data_folderout, f'{rsk_reference}_1min.txt'))

    ######################################
    #### PLOT UNCORRECTED CLEANED TIMESERIES 
    #### meaning without calibration sequences
    outfile = f'Uncorrected_pressure_{station_name}_{today}.pdf'
    title = f'{station_name} - Plot cleaned uncorrected datasets'
    print(f'\n{today} -  Plot {title}.\n')
    plot_pressure(A0A_clean_df,
                  keys=('BPR_pressure_1', 'BPR_pressure_2'),
                  calibration_times=t_atmo,
                  show_calibrations=False, 
                  colors_code=channels_colors,
                  fig_title=title,
                  plot_fig=save_figure, output_path=output_path, filenamout=outfile, savefig=save_figure)

    ######################################
    #### PLOT UNCORRECTED PRESSURE DIFFERENCE
    outfile = f'Uncorrected_deltaP_{station_name}_{today}.pdf'
    title = u'{} - Uncorrected $\Delta$P = BPR2 - BPR1'.format(station_name)
    print(f'\n{today} -  Plot {title}.\n')
    plot_deltaP(A0A_clean_df,
                keys=('BPR_pressure_1', 'BPR_pressure_2'),
                deltaP=None,
                calibration_times=t_atmo,
                show_calibrations=False,
                title=title,
                text_size='large',
                plot_fig=True, output_path=output_path, filenamout=outfile, savefig=True)


if __name__ == "__main__":
    main()

"""
### Article IAG 2025 Figure 4

#### PLOT PRESSURE DIFFERENCE (delta_P)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

diff_A0A = A0A_clean_df['BPR_pressure_2'].values - A0A_clean_df['BPR_pressure_1'].values

fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(axis='both', labelsize=text_size)
# ax.set_title('2022-2023 — East Mayotte (MAY_C)', loc='left', fontsize=text_size)
ax.set_title('2023-2024 — North-East Mayotte (MAY_R)', loc='left', fontsize=text_size)
# ax.grid(which='both', lw=0.45, color='silver', zorder=0)

ax.plot(A0A_clean_df.index, (diff_A0A - diff_A0A.mean()), c='tab:blue', rasterized=True, label=u'$\Delta$P')

# ax.legend(loc='upper left')
# ax.legend(loc='upper right')
ax.set_ylabel('Pressure difference [dBar]', fontsize=text_size)
ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
plt.tight_layout()
plt.show()
"""
