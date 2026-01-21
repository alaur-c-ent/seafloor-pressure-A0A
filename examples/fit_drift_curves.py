#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
STEP 3 (Part 1) : Fit drift models to calibration curves.

This example illustrates how to model long-term instrumental drift
from calibration (zero-pressure) sequences.

1. Read (or extract) the zeros measurements stored in a .csv (ouput of STEP 1)
2. Read (or estimate the calibration curves) the calibration (zero-pressure) values stored in a .csv (ouput of STEP 2).
3. Model the calibration/drift curves
4. Save fitted drift model parameters to flat file and models into the .csv calibration file.

conda activate myrsk

07/2024 — alaurent
"""

import sys, os, glob
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt

from src.A0A.core import read_events_log, calibrations, save_drift_model
from src.A0A.plots import plot_calibration_curves, compare_calib_models
from src.A0A.inversion import fit_drift_curve


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

    calib_path = os.path.join(root_path, mission, recover_date, station_name, f"{station_name}_Calib.csv")

    # os.makedirs(output_path, exist_ok=True)

    today = date.today().strftime("%Y%m%d")   ### Warning not UTC

    #### Read log
    events_df, t_marine, t_atmo, t_error = read_events_log(events_path)

    if station_name == 'A0A_MAY18_C':
        window = pd.Timedelta(seconds=250) # also working with datetime timedelta
        ### chosen borns in seconds after marine to atmospheric rotation
        win_lims = (230, 240)
        ylims = (None, None)
        dP_ylims = (0.01, 0.35)

    elif station_name == 'A0A_MAY25_C':
        window = pd.Timedelta(seconds=20*60)
        win_lims = (600, 1000)
        ylims = (8.5, 24.)
        dP_ylims = (0.2, 0.35)  

    elif station_name == 'A0A_MAY30_R':
        window = pd.Timedelta(seconds=20*60)
        win_lims = (600, 1000)      
        ylims = (9.5, 9.7)
        dP_ylims = (0.01, 0.10)   

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

    #### Read calibrations
    if glob.glob(calib_path):
        calib_df = pd.read_csv(calib_path, parse_dates=True, index_col=0)
        calib_df['Date'] = pd.to_datetime(calib_df['Date'])
    else:    
        #### Read extracted zeros (from STEP1)
        zeros_df = pd.read_csv(zeros_path, parse_dates=True, index_col=0)
        ## Time vector in seconds from beginning
        zeros_df['time_seconds'] = (zeros_df.index - zeros_df.index[0]).total_seconds()
            ### chosen borns in seconds after marine to atmospheric rotation


        calib_df = calibrations(zeros_df, ['BPR_pressure_1', 'BPR_pressure_2', 'Barometer_pressure'], 
                                t_atmo, window, win_lims)

    print(f'\n {today} - {len(calib_df)} calibrations (zeros) values\n')
    print(calib_df.to_string())
    print()

    ### if needed check/plot drift curves
    use_cmap = False
    title = f'{station_name} –– Calibration curves: P_barometric - P_internal'
    fig = plot_calibration_curves(calib_df, cols=('Calib_1', 'Calib_2'), 
                                title=title, colors_code=None, use_cmap=use_cmap,
                                ylim=dP_ylims, text_size=text_size, figsize=(10, 5))

    if not use_cmap:
        fig.savefig(os.path.join(output_path, f"{station_name}_drift_curves.pdf"), dpi=300)
    else:
        fig.savefig(os.path.join(output_path, f"{station_name}_drift_curves_cmap.pdf"), dpi=300)
    # plt.show()
    plt.close(fig)

    ######################################
    #### Least square inversion
    """
    The least square inversion method have been created by A.D during his intership 
    to model the drift curves and correct pressure dataset from OHA-GEODAMS's A0A instrument.
    This is a adpated replicate to do so on REVOSIMA's A0A instruments.
    """
    model_name = 'exp_linear'
    #### MODEL DRIFTS CURVES - INITIAL PARAMETERS DEFINITION
    # Tested tau values number (number of iteration)
    maxfev = 1000 
    # maxfev = 10000 

    ### Initiate nested dictionnary to store info
    models = {}
    models[model_name] = {}

    # Grid on tau values (site dependent, user choice)
    if 'R' in station_name:
        tau_grid_BPR1 = np.linspace(1e5, 1e7, maxfev) ## MAY30_A0A_R BPR1
        tau_grid_BPR2 = np.linspace(1e10, 3e10, maxfev) ## MAY30_A0A_R BPR2
    elif 'C' in station_name:
        tau_grid_BPR1 = np.linspace(1e6, 1e7, maxfev) # 2025-07-18 MAY30_A0A_C
        tau_grid_BPR2 = np.linspace(1e6, 1e7, maxfev) # same

    ### EXP-LIN INVERSION ON DRIFT CURVES
    if model_name == 'exp_linear':
        BPR1_fit = fit_drift_curve(calib_df['Calib_1'].values, 
                                tau_grid=tau_grid_BPR1, 
                                time=calib_df['Date'], 
                                model=model_name, t_event=None)
        
        models[model_name]['BPR1'] = {}
        models[model_name]['BPR1']['tau'] = BPR1_fit['parameters']['tau']

        BPR2_fit = fit_drift_curve(calib_df['Calib_2'].values, 
                                tau_grid=tau_grid_BPR2, 
                                time=calib_df['Date'], 
                                model=model_name, t_event=None)

        models[model_name]['BPR2'] = {}
        models[model_name]['BPR2']['tau'] = BPR2_fit['parameters']['tau']


    elif model_name == 'exp_lin_H':
        ### Random example
        t_event = np.datetime64("2023-10-12 20:00:00")

        BPR1_fit = fit_drift_curve(calib_df['Calib_1'].values, 
                                tau_grid=tau_grid_BPR1, 
                                time=calib_df['Date'], 
                                model='exp_lin_H', t_event=t_event)
        BPR2_fit = fit_drift_curve(calib_df['Calib_2'].values, 
                                tau_grid=tau_grid_BPR2, 
                                time=calib_df['Date'], 
                                model='exp_lin_H', t_event=t_event)

    else:
        raise ValueError(f'Unknown model {model_name}')
    
    ### Store model in the nested dictionnary 
    models[model_name]['BPR1']['col_name'] = f'{model_name}_1'
    models[model_name]['BPR2']['col_name'] = f'{model_name}_2'

    ######################################
    #### STORE AND SAVE DRIFT MODEL
    ### Add models to the calib_df
    calib_df[f'{model_name}_1'] = BPR1_fit['model']
    calib_df[f'{model_name}_2'] = BPR2_fit['model']

    print()
    print(calib_df.to_string())
    print()
    ### SAve the new dataframe with X = id nbr of deployment 
    ### WARNING : erase the previous veriosn of the file - rename if wont
    calib_df.to_csv(os.path.join(data_folderout,  f"{station_name}_Calib.csv"))


    print(f'\n{today} - Save {model_name} models with parameters into a JSON file for each sensor.\n')
    sensor_name = 'BPR1' # can also be : 'BP1', 'Paros_1' etc
    json_pathout = os.path.join(data_folderout, f'{station_name}_{sensor_name}_drift_model.json')
    save_drift_model(BPR1_fit['parameters'], sensor_name, model_name, today, json_pathout)
    sensor_name = 'BPR2' # can also be : 'BP1', 'Paros_1' etc
    json_pathout = os.path.join(data_folderout, f'{station_name}_{sensor_name}_drift_model.json')
    save_drift_model(BPR2_fit['parameters'], sensor_name, model_name, today, json_pathout)

    ######################################
    #### DISPLAY RESULT
    use_cmap = False
    if model_name == 'exp_linear':
        title = f'{station_name} –– Exponential-linear drift models'
    else : 
        title = f'{station_name} –– Drift models'
    fig = plot_calibration_curves(calib_df, cols=(f'{model_name}_1', f'{model_name}_2'), 
                                title=title, colors_code=None, use_cmap=use_cmap,
                                ylim=dP_ylims, text_size=text_size, figsize=(10, 5))

    if not use_cmap:
        fig.savefig(os.path.join(output_path, f"{station_name}_drift_{model_name}_model.pdf"), dpi=300)
    else:
        fig.savefig(os.path.join(output_path, f"{station_name}_drift_{model_name}_model_cmap.pdf"), dpi=300)
    # plt.show()
    plt.close(fig)


    ######################################
    #### PLOT DRIFTS CURVES AND MODELS 
    fig, ax = compare_calib_models(calib_df, 
                                calib_keys=('Calib_1', 'Calib_2'), 
                                models = models, 
                                colors_code=channels_colors, linestyles=None, 
                                # calib_error=[(t_error[0], t_error[-1])],
                                title='', figsize=(10, 5), text_size=text_size)

    # ax.tick_params('both', labelsize=text_size)
    ax.set_ylim(-0.16, 0.01)
    # ax.set_title('2023-2024 – North-East Mayotte (MAY_R)', loc='left', fontsize=text_size)
    ax.set_title('2023-2024 – East Mayotte (MAY_C)', loc='left', fontsize=text_size)
    fig.tight_layout()

    fig.savefig(os.path.join(output_path, f"{station_name}_calib_n_drift_models.pdf"), dpi=300)
    plt.show()
    # plt.close(fig)


if __name__ == "__main__":
    main()

"""
### Article IAG 2025 Figure 2

#### PLOT CALIBRATION/DRIFTS CURVES
import matplotlib.ticker as ticker

fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(axis='both', labelsize=text_size)
# ax.set_title('2020-2021 — East Mayotte (MAY_C)', loc='left', fontsize=text_size)
# ax.set_title('2022-2023 — East Mayotte (MAY_C)', loc='left', fontsize=text_size)
ax.set_title('2023-2024 — East Mayotte (MAY_C)', loc='left', fontsize=text_size)
# ax.set_title('2023-2024 — North-East Mayotte (MAY_R)', loc='left', fontsize=text_size)
# ax.grid(which='both', lw=0.45, color='silver', zorder=0)

# ax.axvspan(t_atmo[-1]+ pd.Timedelta(days=4), t_error[-1], color='dimgray', alpha=0.6, label='Valve movement error')
ax.plot(calib_df.Date, calib_df['Calib_1'], '-o', c=channels_colors['BPR1'], rasterized=True, label='Calibration BPR1')
ax.plot(calib_df.Date, calib_df['Calib_2'], '-o', c=channels_colors['BPR2'], rasterized=True, label='Calibration BPR2')
ax.legend(loc='upper right')
ax.set_xlabel('Dates', fontsize=text_size)
ax.set_ylabel(u'Normalised $\Delta$P presssure [dBar]', fontsize=text_size)

# ax.set_ylim(-0.10, 0.01)
# ax.set_xlim(calib_df.Date.iloc[0] - pd.Timedelta(days=4), t_error[-1])
ax.set_xlim(calib_df.Date.iloc[0] - pd.Timedelta(days=4), calib_df.Date.iloc[-1] + pd.Timedelta(days=4))

ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
plt.tight_layout()
plt.show()

"""

# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# ax.tick_params(axis='both', labelsize=text_size)
# # ax.set_xlabel('Dates', fontsize=text_size)
# ax.set_ylabel(u'$\Delta$ presssure [dBar]', fontsize=text_size)
# # ax.set_title(f'{station_name} Southeast Indian Ridge - Drift models', fontsize=text_size)
# ax.set_title('2024-2025 Southeast Indian Ridge - Drift models', loc='left', fontsize=text_size)

# ax.grid(which='both', lw=0.45, color='silver', zorder=0)
# ax.axvspan(t_atmo[-1]+timedelta(days=2), t_error[-1], color='grey', alpha=0.6, zorder=1, label='Valve movement error')

# ax.plot(model_calib_df.Date, model_calib_df['Corr_1'], '-o', c='orange', label='Calib. BPR 1', alpha=0.8)
# ax.plot(model_calib_df.Date, model_calib_df['Corr_2'], '-o', c='darkgreen', label='Calib. BPR 2', alpha=0.8)

# ax.plot(model_calib_df.Date, model_calib_df['Model_explin_1'], ':',  linewidth=1.5, c='chocolate',
#         label=r'exp-lin BPR 1 $\tau$={:.0e}'.format(explin_params['BPR 1']['tau']))
# ax.plot(model_calib_df.Date, model_calib_df['Model_explin_2'], ':',  linewidth=1.5, c='limegreen',
#         label=r'exp-lin BPR 2 $\tau$={:.0e}'.format(explin_params['BPR 2']['tau']))

# ax.plot(calib_df.Date, model_calib_df['Model_explin_H_1'], '--',  linewidth=1.5,  c='chocolate',
#          label=r'exp-lin-H BPR 1 $\tau$={:.0e}'.format(explin_H_params['BPR 1']['tau']))
# ax.plot(calib_df.Date, model_calib_df['Model_explin_H_2'], '--',  linewidth=1.5,  c='limegreen',
#          label=r'exp-lin-H BPR 2 $\tau$={:.0e}'.format(explin_H_params['BPR 2']['tau']))

# ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
# ax.legend(loc='upper right')
# plt.tight_layout()
# plt.show()
