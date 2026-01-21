#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
STEP 3 (part 3) - Remove tide from de-drifted pressure datasets

1. Read de-drifted A0A pressure datasets (out of STEP 3 part 2)
2. It is advised to downsample the data to a lower temporal resolution
   (e.g. hourly) to stabilize harmonic analysis.
3. Perform free harmonic tidal analysis using UTide for each pressure
   sensor.
4. Reconstruct tidal water heights from the solved constituents.
5. Remove the reconstructed tidal signal from the pressure records.
6. Save:
   - reconstructed tidal water heights,
   - tide-corrected pressure datasets.
7. Figures:
   - bar plots of tidal constituent names and amplitudes,
   - timeseries of raw (de-drifted) pressure and detided residuals.

conda activate myrsk

07/2024 — alaurent
"""

import sys, os
import utide as ut
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt

from src.A0A.plots import plot_res_tides, plot_deltaP


def cst2dataframe(ut_output):
    """
    Convert UTide harmonic analysis output into a pandas DataFrame.

    Parameters
    ----------
    ut_output : dict
        Output dictionary returned by UTide harmonic analysis.
        Expected keys include:
        - 'name' : tidal constituent names
        - 'A'    : amplitudes
        - 'A_ci' : amplitude uncertainties (confidence intervals)
        - 'g'    : phases (degrees)

    Returns
    -------
    df : pandas.DataFrame
        DataFrame indexed by tidal constituent name, with columns:
        - 'amp'     : tidal amplitude
        - 'amp_err' : amplitude uncertainty
        - 'pha'     : tidal phase
    """
    name_list=[]; amp_list=[]; amp_err_list=[]; pha_list=[] #init list
    for name,amp,aerr,pha in zip(ut_output['name'], ut_output['A'], ut_output['A_ci'], ut_output['g']) :
        name_list.append(name)
        amp_list.append(amp)
        amp_err_list.append(aerr)
        pha_list.append(pha)
    d={'wave':name_list,'amp':amp_list,'amp_err':amp_err_list,'pha':pha_list}
    df = pd.DataFrame(data=d)
    df.set_index('wave',inplace=True)
    return df


def barplot_utide(ut_output, title='', amin=.01, text_size='medium'):
    """ 
    Plot tidal constituent amplitudes from UTide harmonic analysis.

    This figure has been initially made by L. Testut (https://orcid.org/0000-0002-3969-2919)

    Parameters
    ----------
    ut_output : dict
        Output dictionary output from the harmonic analysis.
        Expected keys include tidal constituent names, amplitudes,
        amplitude uncertainties, and phases.
    title : str
        Title of the figure.
    amin : float, optional
        Minimum tidal amplitude threshold. Constituents with amplitudes
        below this value are excluded from the plot (default: 0.01 meter).
    text_size : str, optional
        Font size used for axis labels and ticks (default: 'medium').

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    ### convert the dictionnary that result from the free analysis into a dataframe
    df = cst2dataframe(ut_output)
    ### Select only amplitude higher than the given/default threshold
    udf = df[df['amp'] >= amin]
    ### extract tide constituents (waves) code, amplitude and errors
    x, y, yerr = udf.index, udf['amp'].values, udf['amp_err'].values
    ### PLOT
    fig = plt.figure(figsize=(10, 6))
    plt.bar(x, y, yerr=yerr,
                capsize=1, color='lightblue', edgecolor='black', 
                zorder=3)

    plt.title(title)
    plt.xlabel('Tidal Constituent', fontsize=text_size)
    plt.ylabel('Amplitude [m]', fontsize=text_size)
    #plt.yticks(np.arange(amin, 10.*amin))  # Set y-axis ticks at intervals of 10
    plt.grid(axis='y', which='major', linestyle='--', color='grey', zorder=0)
    plt.tick_params(axis='both', labelsize=text_size)
    plt.tight_layout()
    fig.autofmt_xdate()
    return fig


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

output_path = os.path.join(root_path, mission, recover_date,  'figures/tides_analysis/')
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
    data_path = os.path.join(root_path, mission, recover_date, station_name, f'{rsk_reference}_data_corrected.txt')
    ### Other posible datasets :
    # data_path = os.path.join(root_path, recover_date, station_name, rsk_reference, f'{rsk_reference}_1min_corrected.txt')
    ### With respect to YTT nomenclature
    # data_path = os.path.join(root_path, recover_date, station_name, f'A0A{X}_OBP_1min.csv')

    os.makedirs(output_path, exist_ok=True)

    today = date.today().strftime("%Y%m%d")   ### Warning not UTC

    #### Read pressure dataset
    ### Example made with the dedrifted file not resampled
    ### I dont use the read_A0A_data function anymore as I want all the dataframe as it is
    A0A_dedrift_df = pd.read_csv(data_path, parse_dates=True, index_col=0)

    print(f'\n {today} - Drift corrected pressure datasets\n')
    print(A0A_dedrift_df.head())
    print()

    #### Reduce the dataset to one sample per hour
    resample_to = '1H'
    # resample_to = '10min'
    # resample_to = '1min'
    ### if dont, might crash...
    resampled_df = A0A_dedrift_df.resample(resample_to).mean() ## not smoothed 1 hour averaged dataset
    ### Drop relative time (in seconds) after beginning as it has no use anymore (and no meaning)
    resampled_df.drop(['time_seconds'], inplace=True, axis=1)
    #### Add a (new) Timestamp column from DateTimeIndex
    resampled_df['Timestamp'] = resampled_df.index

    print(f'\n {today} - 1 hour resampled pressure dataset\n')
    print(resampled_df.head())
    print()

    ########################################
    ##### FREE TIDE ANALYSIS 
    #### Harmonic analysis using UTIDE python module : 
    #### solve then reconstruct tides of a long duration periodic signal.
    """
    WARNING : UTIDE WITH THE CSV IS NEITHER TOO LONG OR MAKE MY LAPTOP PYTHON CRASH
    """
    tide_coefs = {}
    water_heights = {}

    for BPR_data in (resampled_df['BPR_dedrift_1'], resampled_df['BPR_dedrift_2']):
        sensor_code = (BPR_data.name).split('_')[0]+(BPR_data.name).split('_')[2]
        #### High-frequency tide (free harmonic analysis)
        cst_coefs_ = ut.solve(resampled_df['Timestamp'].values, BPR_data.values, lat=-12.7,  trend=False) #, constit = const)
        tide_coefs[sensor_code] = cst_coefs_
        ### All reconstructed outputs
        # u = ut.reconstruct(resampled_df['Timestamp'].values, cst_coefs_) 
        ### Keep only the reconstructed water heights
        reconstides_H = ut.reconstruct(resampled_df['Timestamp'].values, cst_coefs_).h 
        water_heights[sensor_code] = reconstides_H
        new_col = 'BPR_detided_{}'.format((BPR_data.name).split('_')[2])
        resampled_df[new_col] = BPR_data.values - reconstides_H

    print(f'{today} - Reconstructed water heights (in meters) from free harmonic analysis')
    print(water_heights)

    ########################################
    ##### SAVE WATER HEIGHTS USED TO CORRECT THE SIGNAL
    #### Not necessary to re-run tide analysis in futur data process
    waterH_outpath = os.path.join(data_folderout,f'{rsk_reference}_{resample_to}_water_heights.txt')
    waterH_df = pd.DataFrame(water_heights)
    waterH_df.to_csv(waterH_outpath, index=False)

    ########################################
    ##### SAVE NEW DATASET WITH TIDE CORRECTED PRESSURE
    resampled_df.to_csv(os.path.join(data_folderout, f'{rsk_reference}_{resample_to}_tides_corrected.txt'))

    ########################################
    ##### BAR PLOT OF THE TIDE CONSTITUENTS
    start = resampled_df.index[0].strftime("%m-%Y")
    end = resampled_df.index[-1].strftime("%m-%Y")
    ### threshold for select tide constituent to display
    amin=.01

    sensor_codes = ('BPR1',  'BPR2')
    for n_cha in sensor_codes:
        y = tide_coefs[n_cha]
        title = f'{station_name} - {n_cha} – {len(y)} solved tidal constituent > {amin*100} cm – {start}/{end}'
        fig = barplot_utide(y, title, amin=.01, text_size=text_size)
        fig.savefig(os.path.join(output_path, f"{station_name}_{n_cha}_tides_coefs.pdf"), dpi=300)
        plt.show()

    ########################################
    ##### PLOT OF THE DETIDED SIGNAL 
    figsize = (12, 6)
    tide_offset = 1 ## in meters
    legend_txt = f'{resample_to} resampled pressure'
    # keys = ('BPR_pressure_1', 'BPR_pressure_2')
    keys = ('BPR_dedrift_1', 'BPR_dedrift_2')
    # keys = ('BPR_detided_1', 'BPR_detided_2')

    fig = plot_res_tides(resampled_df, keys, legend_txt, tide_offset, 
                        colors_code=channels_colors, figsize=figsize, text_size='large')

    fig.savefig(os.path.join(output_path, f"{station_name}_tide_residual_signal_{today}.pdf"), dpi=300)

    plt.show()

    ######################################
    #### PLOT DRIFT CORRECTED PRESSURE DIFFERENCE
    outfile = f'Tides_corrected_deltaP_{station_name}_{today}.pdf'
    title = u'{} - Tides corrected $\Delta$P = BPR2 - BPR1'.format(station_name)
    print(f'\n{today} -  Plot {title}.\n')
    plot_deltaP(resampled_df,
                deltaP=None,
                keys=('BPR_detided_1', 'BPR_detided_2'),
                show_calibrations=False,
                title=title,
                text_size='large',
                plot_fig=True, output_path=output_path, filenamout=outfile, savefig=True)


if __name__ == "__main__":
    main()
