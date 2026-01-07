#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
Process dataset from raw pressure to drift corrected pressure. 

1. Read the 1 min .csv file from A. Duvernay work folder
2. Read zeros extracted file, compute the calib curve
3. Display the calib curve + fig (raw P, detided P, temp, calib)
4. Plot the uncorrected delta P
5. Compute the slope on uncorrected delta P
6. Model the drift with least mean square inversion

conda activate myrsk

10/2025 — alaurent
"""

import sys, os
import numpy as np
import pandas as pd
from datetime import date, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import linregress

#sys.path.append('/Users/alaure04/moby-data/CODES/Git_/src/A0A/')
from src.A0A.core import read_A0A_data, read_events_log, calibrations, save_drift_model
from src.A0A.models import exp_linear, heaviside, exp_linear_H
from src.A0A.inversion import fit_drift_curve
from src.A0A.plots import (plot_pressure, 
                   plot_calibrations, 
                   plot_deltaP, 
                   plot_calibration_curves,
                   compare_calib_models)


########################################
#### DEFINE DATA PATHS ####
root_path = '/Users/alaure04/moby-data/DATA/'
mission = 'GEODAMS'
station_name = 'A0A'
rsk_reference = '233200_20250125_1007'

########################################
#### OPTIONS & AESTHETICS ####
text_size = 'large'

channels_colors = {'BB': 'violet',
                   'BPR1': 'orange',
                   'BPR2': 'darkgreen',
                   'T_ext' : 'tab:red'}

today = date.today().strftime("%Y%m%d")   ### Warning not UTC

########################################
#### Build input file paths
resample_to = '1min'
data_path = os.path.join(root_path, mission, station_name, f'{station_name}_OBP_{resample_to}.csv')

events_path =  os.path.join(root_path, mission, station_name, f'{rsk_reference}_events.txt')

zeros_path = os.path.join(root_path, mission, station_name, f'{station_name}_zeros.csv')

########################################
#### Read log
events_df, t_marine, t_atmo, t_error = read_events_log(events_path)

### Read already cleaned and resampled dataset
resampled_df = read_A0A_data(data_path)

print(f'\n {today} - 1 minute resampled pressure dataset\n')
print(resampled_df.head())
print()

### Manually cut the end 
resampled_df = resampled_df[:'2025-01-23 03:00:00']

### read extracted zeros
zeros_df = pd.read_csv(zeros_path, parse_dates=True, index_col=0)
### In this case
zeros_df = zeros_df.rename({'seconds' : 'time_seconds'}, axis=1)

print(f'\n {today} - Extracted calibrations sequences\n')
print(zeros_df.head())
print()

########################################
#### PLOT UNCORRECTED PRESSURE DATA
### showing a 4 m sudden pressure increase = seafloor motion of ~4 m
output_path = os.path.join(root_path, mission, station_name,  'figures/parsing/')
os.makedirs(output_path, exist_ok=True)
outfile = f'Uncorrected_pressure_{station_name}_{today}.pdf'
title = f'{station_name} GEODAMS - Plot cleaned uncorrected datasets'
print(f'\n{today} -  Plot {title}.\n')
plot_pressure(resampled_df,
                keys=('BPR_pressure_1', 'BPR_pressure_2'),
                calibration_times=t_atmo,
                show_calibrations=False, 
                colors_code=channels_colors,
                fig_title=title,
                plot_fig=False, output_path=output_path, filenamout=outfile, savefig=False)

######################################
#### PLOT UNCORRECTED PRESSURE DIFFERENCE
diff_A0A = resampled_df['BPR_pressure_2'].values - resampled_df['BPR_pressure_1'].values

# output_path = os.path.join(root_path, mission, station_name,  'figures/parsing/')
outfile = f'Uncorrected_deltaP_{station_name}_{today}.pdf'
title = u'{} - Uncorrected $\Delta$P = BPR2 - BPR1'.format(station_name)
print(f'\n{today} -  Plot {title}.\n')
plot_deltaP(resampled_df,
            keys=('BPR_pressure_1', 'BPR_pressure_2'),
            deltaP=diff_A0A,
            calibration_times=t_atmo,
            show_calibrations=True,
            title=title,
            text_size='large',
            plot_fig=False, output_path=output_path, filenamout=outfile, savefig=False)

######################################
#### RETRIEVE ZERO-PRESSURE VALUES FROM CALIBRATION SEQUENCES
### programmed duration until 2025
window = pd.Timedelta(minutes=20) # also working with datetime timedelta
### chosen borns in seconds after marine to atmospheric rotation
lim_inf, lim_sup = (600, 1000)

calib_df = calibrations(zeros_df, 
                        ['Zeros 1', 'Zeros 2', 'Barometer pressure'], 
                        t_atmo, window, lim_inf, lim_sup)

print(f'\n {today} - {len(calib_df)} calibrations (zeros) values\n')
print(calib_df.to_string())
print()

calib_df['Date'] = pd.to_datetime(calib_df['Date'])

### Save it (if needed)
# calib_df.to_csv(os.path.join(root_path, mission, station_name, rsk_reference,  f"A0A_Calib.csv"))

######################################
#### PLOT ALL CALIBRATION SEQUENCES(ZERO SEGMENTS) IN A SINGLE WINDOW
pressure_cols = ['Zeros 1', 'Zeros 2']
output_path = os.path.join(root_path, mission, station_name,  'figures/calib/')
os.makedirs(output_path, exist_ok=True)

fig = plot_calibrations(zeros_df, calibration_times=t_atmo, 
                    keys=pressure_cols, 
                    window=window, select_window=(lim_inf, lim_sup),
                    ylim=(9.5, np.max([zeros_df[col].max() for col in pressure_cols])+0.1),
                    cmap='viridis', figsize=(8, 4))

fig.savefig(os.path.join(output_path, f"{station_name}_zero_segments.pdf"), dpi=300)
plt.show()
# plt.close(fig)

######################################
#### PLOT CALIBRATION/DRIFTS CURVES
# output_path = os.path.join(root_path, mission, station_name,  'figures/calib/')
# use_cmap = False
use_cmap = True
title = f'{station_name} –– Calibration curves: P_barometric - P_internal'
fig = plot_calibration_curves(calib_df, cols=('Calib_1', 'Calib_2'), 
                                title=title, colors_code=None, use_cmap=use_cmap,
                                ylim=(0.01, 0.18), text_size=text_size, figsize=(10, 5))

if not use_cmap:
    fig.savefig(os.path.join(output_path, f"{station_name}_drift_curves.pdf"), dpi=300)
else:
    fig.savefig(os.path.join(output_path, f"{station_name}_drift_curves_cmap.pdf"), dpi=300)
plt.show()
# plt.close(fig)

######################################
#### MODEL THE CALIB CURVES
"""
This method have been used by A.D during his intership 
to model the drift curves from GEODAMS A0A.
This is a modified version of his work.
"""

### Initiate nested dictionnary to store info
models = {}

#### MODEL DRIFTS CURVES - INITIAL PARAMETERS DEFINITION
calib_df['Date'] = pd.to_datetime(calib_df['Date'])

maxfev = 1000 # Nombre de valeurs de tau à tester

# model_name = 'exp_linear'
model_name = 'exp_lin_H'
# model_name = 'db_explin'
# model_name = 'db_explin_H'

models[model_name] = {}

### REGRESSION MATRIX INVERSION
tau_grid = np.linspace(1e5, 1e7, maxfev) ## A0A_GEODAMS BPR1 & BPR2

### EXP-LIN INVERSION ON DRIFT CURVES
if model_name == 'exp_linear':
    BPR1_fit = fit_drift_curve(calib_df['Calib_1'].values, 
                            tau_grid=tau_grid, 
                            time=calib_df['Date'], 
                            model=model_name, t_event=None)
    
    models[model_name]['BPR1'] = {}
    models[model_name]['BPR1']['tau'] = BPR1_fit['parameters']['tau']

    BPR2_fit = fit_drift_curve(calib_df['Calib_2'].values, 
                            tau_grid=tau_grid, 
                            time=calib_df['Date'], 
                            model=model_name, t_event=None)
    
    models[model_name]['BPR2'] = {}
    models[model_name]['BPR2']['tau'] = BPR2_fit['parameters']['tau']

elif model_name == 'exp_lin_H':
    ### Random example
    t_event = np.datetime64("2024-04-26 20:00:00") ## date of the identified pressure drop 

    BPR1_fit = fit_drift_curve(calib_df['Calib_1'].values, 
                            tau_grid=tau_grid, 
                            time=calib_df['Date'], 
                            model=model_name, t_event=t_event)
    
    models[model_name]['BPR1'] = {}
    models[model_name]['BPR1']['tau'] = BPR1_fit['parameters']['tau']

    BPR2_fit = fit_drift_curve(calib_df['Calib_2'].values, 
                            tau_grid=tau_grid, 
                            time=calib_df['Date'], 
                            model=model_name, t_event=t_event)

    models[model_name]['BPR2'] = {}
    models[model_name]['BPR2']['tau'] = BPR2_fit['parameters']['tau']

else:
    raise ValueError(f'Unknown model {model_name}')

### Store it in the nested dictionnary 
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
### WARNING : erase the previous version of the file - rename if wont
# calib_df.to_csv(os.path.join(root_path, mission, station_name, f"A0A{X}_Calib.csv"))


print(f'\n{today} - Save {model_name} models with parameters into a JSON file for each sensor.\n')
sensor_name = 'BPR1' # can also be : 'BP1', 'Paros_1' etc
json_pathout = os.path.join(root_path, mission, station_name, f'A0A_{sensor_name}_drift_model.json')
save_drift_model(BPR1_fit['parameters'], sensor_name, model_name, today, json_pathout)
sensor_name = 'BPR2' # can also be : 'BP1', 'Paros_1' etc
json_pathout = os.path.join(root_path, mission, station_name, f'A0A_{sensor_name}_drift_model.json')
save_drift_model(BPR2_fit['parameters'], sensor_name, model_name, today, json_pathout)

######################################
#### DISPLAY RESULT
output_path = os.path.join(root_path, mission, station_name,  'figures/calib/')

use_cmap = False
if model_name == 'exp_linear':
    title = f'{station_name} –– Exponential-linear drift models'
elif model_name == 'exp_lin_H':
    title = f'{station_name} –– Exponential-linear-Heaviside drift models'
else : 
    title = f'{station_name} –– Drift models'
fig = plot_calibration_curves(calib_df, cols=(f'{model_name}_1', f'{model_name}_2'), 
                            title=title, colors_code=None, use_cmap=use_cmap,
                            ylim=(0.01, 0.18), text_size=text_size, figsize=(10, 5))

if not use_cmap:
    fig.savefig(os.path.join(output_path, f"{station_name}_drift_model.pdf"), dpi=300)
else:
    fig.savefig(os.path.join(output_path, f"{station_name}_drift_model_cmap.pdf"), dpi=300)
plt.show()


######################################
#### PLOT DRIFTS CURVES AND MODELS 
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

output_path = os.path.join(root_path, mission, station_name,  'figures/calib/')
fig = compare_calib_models(calib_df, 
                               calib_keys=('Calib_1', 'Calib_2'), 
                               models = models, 
                               colors_code=channels_colors, linestyles=None, 
                               calib_error=[(t_error[0], t_error[-1])],
                               title='', figsize=(10, 5), text_size='medium')


fig.savefig(os.path.join(output_path, f"{station_name}_calib_n_drift_models.pdf"), dpi=300)
plt.show()
sys.exit()

######################################
#### LONG TERM DRIFT CORRECTION

#### Pre-process dataset
### Extract DateTimeIndex (automatically called "Time") into a real column
resampled_df['Timestamp'] = resampled_df.index
### Add new column: convert DateTime into relative seconds from beginning
resampled_df['time_seconds'] = (resampled_df['Timestamp'] - resampled_df['Timestamp'].iloc[0]).dt.total_seconds()


#### GENERATE NEW MODEL WITH KNOWN PARAMETERS
if model_name == 'exp_linear':
    BPR1_model = exp_linear(resampled_df['time_seconds'], 
                                BPR1_fit['parameters']['a'], 
                                BPR1_fit['parameters']['tau'], 
                                BPR1_fit['parameters']['b'], 
                                BPR1_fit['parameters']['d'])
    BPR2_model = exp_linear(resampled_df['time_seconds'], 
                                BPR2_fit['parameters']['a'], 
                                BPR2_fit['parameters']['tau'], 
                                BPR2_fit['parameters']['b'], 
                                BPR2_fit['parameters']['d'])

elif model_name == 'exp_lin_H':
    BPR1_model = exp_linear_H(resampled_df['time_seconds'], 
                                BPR1_fit['parameters']['a'], 
                                BPR1_fit['parameters']['tau'], 
                                BPR1_fit['parameters']['b'], 
                                BPR1_fit['parameters']['c'], 
                                BPR1_fit['parameters']['d'],
                                heaviside(resampled_df['Timestamp'], t_event, x=1, y=0))
    BPR2_model = exp_linear_H(resampled_df['time_seconds'], 
                                BPR2_fit['parameters']['a'], 
                                BPR2_fit['parameters']['tau'], 
                                BPR2_fit['parameters']['b'], 
                                BPR2_fit['parameters']['c'], 
                                BPR2_fit['parameters']['d'],
                                heaviside(resampled_df['Timestamp'], t_event, x=1, y=0))
else:
    raise ValueError(f'Unknown model {model_name}')


#### DRIFT CORECTION
### Subtract model to data in the numpy version for faster computation times
### Directly stored into the dataframe as a new column
resampled_df['BPR_dedrift_1'] = resampled_df['BPR_pressure_1'].values - BPR1_model.to_numpy()
resampled_df['BPR_dedrift_2'] = resampled_df['BPR_pressure_2'].values - BPR2_model.to_numpy()

print(f'\n {today} - Drift corrected pressure datasets\n')
print(resampled_df.head())
print()

### Save the drift corrected 1 min resampled dataset
resampled_df.to_csv(os.path.join(root_path, mission, station_name, f'{station_name}_{resample_to}_corrected.txt'))

### With respect to YTT nomenclature
resampled_df.to_csv(os.path.join(root_path, mission, station_name, f'{station_name}_OBP_{resample_to}.csv'))

######################################
#### PLOT DRIFT CORRECTED PRESSURE DIFFERENCE
output_path = os.path.join(root_path, mission, station_name,  'figures/parsing/')
outfile = f'Drift_corrected_deltaP_{station_name}_{today}.pdf'
title = u'{} - Drift corrected $\Delta$P = BPR2 - BPR1'.format(station_name)
print(f'\n{today} -  Plot {title}.\n')
plot_deltaP(resampled_df,
            deltaP=None,
            keys=('BPR_dedrift_1', 'BPR_dedrift_2'),
            show_calibrations=False,
            title=title,
            text_size='large',
            plot_fig=True, output_path=output_path, filenamout=outfile, savefig=True)

sys.exit()  

"""
Following figure are for exploration the temperature anomalie
Compute min, max, baseline before event, delta_T at the time of the max, find the date of the max.
etc
"""
########################################
#### LOOKING AT THE MAX TEMPERATURE ANOMALY 

### manually select the time window :
### 4 days prior the event and over one month. 
t_eruption = '2024-04-26'
format='%Y-%m-%d'
# t_eruption = datetime.strptime(t_eruption, format)
t_start = date.fromisoformat(t_eruption) - timedelta(days=4)
t_end = t_start + timedelta(days=31)
# A0A_df_T = resampled_df['2024-04-22':'2024-05-22']
A0A_df_T = resampled_df[t_start:t_end]

# col_name = 'Temperature'
col_name = 'External_temp'

### COMPUTE AVERAGE TEMPERATURE BEFORE, MAX, AFTER EVENT
mean_T_before_ev = np.mean(A0A_df_T[col_name][:date.fromisoformat(t_eruption)]) ## 2.36
mean_T_ev = np.mean(A0A_df_T[col_name][date.fromisoformat(t_eruption):]) ## 2.57
max_T_ev = np.max(A0A_df_T[col_name]) ## 2.95
min_T_ev = np.min(A0A_df_T[col_name]) ## 2.34
delta_T_ev = max_T_ev - min_T_ev ## 0.61
delta_mean_T = mean_T_ev - mean_T_before_ev ## 0.21

print(f'\nAvg. temperatures')
print('Before {} : {:.2f}°C'.format(t_eruption, mean_T_before_ev))
print('Max temprature anomalie – {} : {:.2f}°C'.format(A0A_df_T[col_name].idxmax(), max_T_ev))
print('After {} : {:.2f}°C'.format(t_eruption, mean_T_ev))

### PLOT TEMPERATURE ANOMALIE
fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(axis='both', labelsize=text_size)
ax.plot(A0A_df_T.index, A0A_df_T[col_name], color='lightblue', label=u'$T_{ext}$')
idx_eruption = np.where(A0A_df_T.index >= t_eruption)[0][0]
ax.axhline(mean_T_before_ev, 
           xmin=0, xmax=np.interp(idx_eruption, A0A_df_T.index, A0A_df_T[col_name].values),
           linestyle=':', color='lightgrey')
ax.annotate(r'before ev. $T_{ext}$'+'={:.2f}°C'.format(mean_T_before_ev),
            xy=(A0A_df_T.index[40000], mean_T_before_ev),
            textcoords='data', 
            ha='left', va='top', zorder=10)
ax.axvline(A0A_df_T[col_name].idxmax(), #'2024-05-04' 08:45 +/- 8 days after the pressure drop
           ymin=0.08, ymax=0.95, alpha=0.7,
           color='red', zorder=1,
           label=r'$\Delta$T={:.2f}°C'.format(delta_T_ev))
ax.annotate(r'max $T_{ext}$'+'={:.2f}°C'.format(max_T_ev),
            xy=(A0A_df_T[col_name].idxmax(), A0A_df_T[col_name].max()),
            textcoords='data', color='red', fontweight='bold',
            ha='right', va='bottom', zorder=10)

ax.set_ylabel('Temperature [°C]', fontsize=text_size)
ax.legend(loc='upper right')
ax.set_xlabel('Dates', fontsize=text_size)
ax.set_xlim(t_start, t_end)
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
plt.tight_layout()


output_path = os.path.join(root_path, mission, station_name,  'figures/eruption/')
os.makedirs(output_path, exist_ok=True)
fig.savefig(os.path.join(output_path, f"{station_name}_ext_temperature_anomalie.pdf"), dpi=300)

plt.show()
# plt.close(fig)

#### SUBPLOT RAW + DETIDED PRESSURE DATA + TEMPERATURE + CALIB
data_path = os.path.join(root_path, mission, station_name, 'pressure_data_detided.csv') ## to change
A0A_df_detided = pd.read_csv(data_path, delim_whitespace=True, index_col=0, parse_dates=True)

fig, axs = plt.subplots(4, 1, figsize=(8,8), sharex=True, constrained_layout=True) #
# axs[0].set_title('2024-2025 - Southeast Indian Ridge', loc='left', fontsize=text_size)

axs[0].plot(resampled_df.index, resampled_df['BPR_pressure_1'], rasterized=True,
            color='orange', label='BPR1')
axs[0].plot(resampled_df.index, resampled_df['BPR_pressure_2'], rasterized=True,
            color='darkgreen', label='BPR2')
axs[0].set_ylabel('Pressure [dBar]', fontsize=text_size)
axs[0].legend(loc='lower right')

axs[1].plot(A0A_df_detided.index, A0A_df_detided['detided BPR 1'], rasterized=True, 
            color='orange', label='detided BP1')
axs[1].plot(A0A_df_detided.index, A0A_df_detided['detided BPR 2'], rasterized=True, 
            color='darkgreen', label='detided BP2')
axs[1].set_ylabel('Pressure [dBar]', fontsize=text_size)
axs[1].legend(loc='lower right')

axs[2].plot(resampled_df.index, resampled_df[col_name],  rasterized=True,
            color='lightblue', label=u'$T_{ext}$')
axs[2].annotate(r'max $T_{ext}$'+'={:.2f}°C'.format(resampled_df[col_name].max()),
            xy=(resampled_df[col_name].idxmax()+timedelta(days=1), resampled_df[col_name].max()-0.05),
            textcoords='data', color='red', fontweight='bold',
            ha='left', va='bottom', zorder=10)
axs[2].annotate(r'before ev. $T_{ext}$'+'={:.2f}°C'.format(mean_T_before_ev),
            # xy=(resampled_df[col_name].idxmax(), mean_T_before_ev),
            xy=(date.fromisoformat(t_eruption), mean_T_before_ev),
            textcoords='data', fontweight='bold',
            ha='left', va='top', zorder=10)
axs[2].set_ylabel('Temperature [°C]', fontsize=text_size)
axs[2].legend(loc='upper right')
# axs[2].set_xlabel('Dates', fontsize=text_size)

axs[3].plot(calib_df.Date, calib_df['Calib_1'], '-o', color='orange', label='calib. BPR1')
axs[3].plot(calib_df.Date, calib_df['Calib_2'], '-o', color='darkgreen', label='calib. BPR2')
axs[3].axvspan(t_atmo[-1]+timedelta(days=1), resampled_df.index[-1], facecolor='silver', label='Valve movement error', zorder=2)
axs[3].set_ylabel('Corrected pressure [dBar]', fontsize=text_size)
axs[3].legend(loc='upper right')


for _ax in axs:
     _ax.grid(which='both', zorder=0, lw=0.45, color='silver')
     _ax.tick_params(which='both', labelsize=text_size)
     _ax.xaxis.set_major_locator(ticker.MaxNLocator(6))

output_path = os.path.join(root_path, mission, station_name,  'figures/eruption/')
os.makedirs(output_path, exist_ok=True)
fig.savefig(os.path.join(output_path, f"{station_name}_eruption_impact_{today}.pdf"), dpi=300)

plt.show()
# plt.close(fig)


"""
Following figure are for exploration on the slope computation 
Compute linear slope over the delta P signal, or part of it (manually determined).
To reshape into function(s)... 
"""

########################################
#### COMPUTE UNCORRECTED DELTA P SLOPES 
units = {'s':1.0,
         'min':60.0,
         'h':3600.0,
         'd':86400.0}

annot_unit = 'd'

### Manually cutted df 1 month prior eruption
t_eruption = '2024-04-26'
t_start = date.fromisoformat(t_eruption) - timedelta(days=30)
A0A_df_cutted_b = resampled_df[t_start:t_eruption] ## b means before

### Create time vector
A0A_df_cutted_b['time_seconds'] = (A0A_df_cutted_b.index - A0A_df_cutted_b.index[0]).total_seconds()
t_s = A0A_df_cutted_b['time_seconds'].to_numpy().astype(float)

delta_P_b = (A0A_df_cutted_b['BPR_pressure_2'].values - A0A_df_cutted_b['BPR_pressure_1'].values)

before_ev_lr = linregress(t_s, delta_P_b)
before_ev_slope, before_ev_intercept, r2 = before_ev_lr.slope, before_ev_lr.intercept, before_ev_lr.rvalue**2
before_ev_yfit = before_ev_slope*t_s + before_ev_intercept

#### Plot delta P and linear slope (only before event)
fig, ax = plt.subplots(1, 1)
ax.grid(which='both', lw=0.4, color='silver', zorder=0)
ax.plot(resampled_df.index, (resampled_df['BPR_pressure_2'].values - resampled_df['BPR_pressure_1'].values), color='tab:blue', label=u'raw $\Delta$P')
ax.plot(A0A_df_cutted_b.index, before_ev_yfit, lw=2, color='tab:green', alpha=0.9)
ax.set_ylabel(u'$\Delta$ pressure [dBar]')
ax.set_xlabel('Time')
# ax.set_title(f"Segment slopes (~{nbr_of_days} days SG) – annot. in dBar/{annot_unit}")
ax.set_title(f'{station_name} Southeast Indian Ridge - Residual slope before event')
before_xm = A0A_df_cutted_b.index[0] + (A0A_df_cutted_b.index[-1] - A0A_df_cutted_b.index[0])/2
ax.annotate(f"{before_ev_slope*units['d']:+.3g} dBar/{annot_unit}",
            xy=(before_xm, np.interp(before_xm.value/1e9, A0A_df_cutted_b.index.view('i8')/1e9, delta_P_b)),
            xytext=(0, -12), textcoords='offset points',
            ha='center', va='top', fontsize=14, color='darkgreen',
            bbox=dict(boxstyle='round', fc='white', ec='silver', lw=0.6, alpha=0.5))
ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
plt.tight_layout()
plt.show()

#### Automatic detection of segments after eruption
t_start = date.fromisoformat(t_eruption) + timedelta(days=4+15) ### mid-may
A0A_df_cutted_a = resampled_df[t_start:] ## a means after
A0A_df_cutted_a['time_seconds'] = (A0A_df_cutted_a.index - A0A_df_cutted_a.index[0]).total_seconds()
t_s = A0A_df_cutted_a['time_seconds'].to_numpy().astype(float)

delta_P_a = (A0A_df_cutted_a['BPR_pressure_2'].values - A0A_df_cutted_a['BPR_pressure_1'].values)

after_ev_lr = linregress(t_s, delta_P_a)
after_ev_slope, after_ev_intercept, r2 = after_ev_lr.slope, after_ev_lr.intercept, after_ev_lr.rvalue**2
after_ev_yfit = after_ev_slope*t_s + after_ev_intercept

#### Plot delta P and linear slope (joining before and after)
delta_P = (resampled_df['BPR_pressure_2'].values - resampled_df['BPR_pressure_1'].values)

fig, ax = plt.subplots(1, 1)
ax.grid(which='both', lw=0.4, color='silver', zorder=0)
ax.plot(resampled_df.index, delta_P, color='tab:blue', label=u'raw $\Delta$P')
ax.plot(A0A_df_cutted_b.index, before_ev_yfit, lw=2, color='tab:green', alpha=0.9)
before_xm = A0A_df_cutted_b.index[0] + (A0A_df_cutted_b.index[-1] - A0A_df_cutted_b.index[0])/2
ax.annotate(f"{before_ev_slope*units['d']:+.3g} dBar/{annot_unit}",
            xy=(before_xm, np.interp(before_xm.value/1e9, A0A_df_cutted_b.index.view('i8')/1e9, delta_P_b)),
            xytext=(0, -12), textcoords='offset points',
            ha='center', va='top', fontsize=14, color='darkgreen',
            bbox=dict(boxstyle='round', fc='white', ec='silver', lw=0.6, alpha=0.5))
ax.plot(A0A_df_cutted_a.index, after_ev_yfit, lw=2, color='tab:green', alpha=0.9)
after_xm = A0A_df_cutted_a.index[0] + (A0A_df_cutted_a.index[-1] - A0A_df_cutted_a.index[0])/2
ax.annotate(f"{after_ev_slope*units['d']:+.3g} dBar/{annot_unit}",
            xy=(after_xm, np.interp(after_xm.value/1e9, A0A_df_cutted_a.index.view('i8')/1e9, delta_P_a)),
            xytext=(0, -12), textcoords='offset points',
            ha='center', va='top', fontsize=14, color='darkgreen',
            bbox=dict(boxstyle='round', fc='white', ec='silver', lw=0.6, alpha=0.5))
ax.set_ylabel(u'$\Delta$ pressure [dBar]')
ax.set_xlabel('Time')
# ax.set_title(f"Segment slopes (~{nbr_of_days} days SG) – annot. in dBar/{annot_unit}")
ax.set_title(f'{station_name} Southeast Indian Ridge - Residual slope')
ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
plt.tight_layout()
plt.show()

######################################
#### COMPUTE SLOPE ON POST-EVENT CORRECTED DELTA_P
t_start = date.fromisoformat(t_eruption) + timedelta(days=4+15) ### mid-may
A0A_df_cutted_a = resampled_df[t_start:] ## a means after
# A0A_df_cutted_a['time_seconds'] = (A0A_df_cutted_a.index - A0A_df_cutted_a.index[0]).total_seconds()
t_s = A0A_df_cutted_a['Time_s'].to_numpy().astype(float)

dedrifted_delta_P_a = (A0A_df_cutted_a['BPR_dedrift_2'].values - A0A_df_cutted_a['BPR_dedrift_1'].values)

dedrifted_ev_lr = linregress(t_s, dedrifted_delta_P_a)
dedrifted_ev_slope, dedrifted_ev_intercept, r2 = dedrifted_ev_lr.slope, dedrifted_ev_lr.intercept, dedrifted_ev_lr.rvalue**2
dedrifted_ev_yfit = dedrifted_ev_slope*t_s + dedrifted_ev_intercept

explin_dedrifted_delta_P_a = (A0A_df_cutted_a['BPR explin dedrifted 2'].values - A0A_df_cutted_a['BPR explin dedrifted 1'].values)
explin_dedrifted_ev_lr = linregress(t_s, explin_dedrifted_delta_P_a)
explin_dedrifted_ev_slope, explin_dedrifted_ev_intercept, r2 = explin_dedrifted_ev_lr.slope, explin_dedrifted_ev_lr.intercept, explin_dedrifted_ev_lr.rvalue**2
explin_dedrifted_ev_yfit = explin_dedrifted_ev_slope*t_s + explin_dedrifted_ev_intercept

#### Plot explin-H and explin dedrifted delta P and linear slopes (after event)
explinH_delta_P = (resampled_df['BPR_dedrift_2'].values - resampled_df['BPR_dedrift_1'].values)
explin_delta_P = (resampled_df['BPR explin dedrifted 2'].values - resampled_df['BPR explin dedrifted 1'].values)

fig, ax = plt.subplots(1, 1)
ax.grid(which='both', lw=0.4, color='silver', zorder=0)
ax.plot(resampled_df.index, explinH_delta_P, color='k', label=u'exp-lin-H dedrifted $\Delta$P')
ax.plot(resampled_df.index, explin_delta_P, color='dimgrey', label=u'exp-lin dedrifted $\Delta$P')

ax.plot(A0A_df_cutted_a.index, dedrifted_ev_yfit, lw=2, color='tab:green', alpha=0.9)

xm = A0A_df_cutted_a.index[0] + (A0A_df_cutted_a.index[-1] - A0A_df_cutted_a.index[0])/2

ax.annotate(f"{dedrifted_ev_slope*units['d']:+.3g} dBar/{annot_unit}",
            xy=(xm, np.interp(xm.value/1e9, A0A_df_cutted_a.index.view('i8')/1e9, dedrifted_delta_P_a)),
            xytext=(0, -12), textcoords='offset points',
            ha='center', va='bottom', fontsize=14, color='darkgreen',
            bbox=dict(boxstyle='round', fc='white', ec='silver', lw=0.6, alpha=0.5))

ax.plot(A0A_df_cutted_a.index, explin_dedrifted_ev_yfit, lw=2, color='tab:green', alpha=0.9)
ax.annotate(f"{explin_dedrifted_ev_slope*units['d']:+.3g} dBar/{annot_unit}",
            xy=(xm, np.interp(xm.value/1e9, A0A_df_cutted_a.index.view('i8')/1e9, explin_dedrifted_delta_P_a)),
            xytext=(0, -12), textcoords='offset points',
            ha='center', va='top', fontsize=14, color='darkgreen',
            bbox=dict(boxstyle='round', fc='white', ec='silver', lw=0.6, alpha=0.5))
ax.set_ylabel(u'$\Delta$ pressure [dBar]')
# ax.set_xlabel('Time')
# ax.set_title(f"Segment slopes (~{nbr_of_days} days SG) – annot. in dBar/{annot_unit}")
ax.set_title(f'{station_name} Southeast Indian Ridge - Residual slope')
ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
plt.tight_layout()
plt.show()


######################################
#### COMPUTE SLOPE ON POST-EVENT CORRECTED DELTA_P
t_start = date.fromisoformat(t_eruption) + timedelta(days=4+15) ### mid-may
A0A_df_cutted_a = resampled_df[t_start:] ## a means after
# A0A_df_cutted_a['time_seconds'] = (A0A_df_cutted_a.index - A0A_df_cutted_a.index[0]).total_seconds()
t_s = A0A_df_cutted_a['Time_s'].to_numpy().astype(float)

dedrifted_delta_P_a = (A0A_df_cutted_a['BPR_dedrift_2'].values - A0A_df_cutted_a['BPR_dedrift_1'].values)

dedrifted_ev_lr = linregress(t_s, dedrifted_delta_P_a)
dedrifted_ev_slope, dedrifted_ev_intercept, r2 = dedrifted_ev_lr.slope, dedrifted_ev_lr.intercept, dedrifted_ev_lr.rvalue**2
dedrifted_ev_yfit = dedrifted_ev_slope*t_s + dedrifted_ev_intercept

explin_dedrifted_delta_P_a = (A0A_df_cutted_a['BPR explin dedrifted 2'].values - A0A_df_cutted_a['BPR explin dedrifted 1'].values)
explin_dedrifted_ev_lr = linregress(t_s, explin_dedrifted_delta_P_a)
explin_dedrifted_ev_slope, explin_dedrifted_ev_intercept, r2 = explin_dedrifted_ev_lr.slope, explin_dedrifted_ev_lr.intercept, explin_dedrifted_ev_lr.rvalue**2
explin_dedrifted_ev_yfit = explin_dedrifted_ev_slope*t_s + explin_dedrifted_ev_intercept

#### Plot explin-H and explin dedrifted delta P and linear slopes (after event)
explinH_delta_P = (resampled_df['BPR_dedrift_2'].values - resampled_df['BPR_dedrift_1'].values)
explin_delta_P = (resampled_df['BPR explin dedrifted 2'].values - resampled_df['BPR explin dedrifted 1'].values)

fig, ax = plt.subplots(1, 1)
ax.grid(which='both', lw=0.4, color='silver', zorder=0)
ax.plot(resampled_df.index, explinH_delta_P, color='k', label=u'exp-lin-H dedrifted $\Delta$P')
ax.plot(resampled_df.index, explin_delta_P, color='dimgrey', label=u'exp-lin dedrifted $\Delta$P')

ax.plot(A0A_df_cutted_a.index, dedrifted_ev_yfit, lw=2, color='tab:green', alpha=0.9)

xm = A0A_df_cutted_a.index[0] + (A0A_df_cutted_a.index[-1] - A0A_df_cutted_a.index[0])/2

ax.annotate(f"{dedrifted_ev_slope*units['d']:+.3g} dBar/{annot_unit}",
            xy=(xm, np.interp(xm.value/1e9, A0A_df_cutted_a.index.view('i8')/1e9, dedrifted_delta_P_a)),
            xytext=(0, -12), textcoords='offset points',
            ha='center', va='bottom', fontsize=14, color='darkgreen',
            bbox=dict(boxstyle='round', fc='white', ec='silver', lw=0.6, alpha=0.5))

ax.plot(A0A_df_cutted_a.index, explin_dedrifted_ev_yfit, lw=2, color='tab:green', alpha=0.9)
ax.annotate(f"{explin_dedrifted_ev_slope*units['d']:+.3g} dBar/{annot_unit}",
            xy=(xm, np.interp(xm.value/1e9, A0A_df_cutted_a.index.view('i8')/1e9, explin_dedrifted_delta_P_a)),
            xytext=(0, -12), textcoords='offset points',
            ha='center', va='top', fontsize=14, color='darkgreen',
            bbox=dict(boxstyle='round', fc='white', ec='silver', lw=0.6, alpha=0.5))
ax.set_ylabel(u'$\Delta$ pressure [dBar]')
# ax.set_xlabel('Time')
# ax.set_title(f"Segment slopes (~{nbr_of_days} days SG) – annot. in dBar/{annot_unit}")
ax.set_title(f'{station_name} Southeast Indian Ridge - Residual slope')
ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
plt.tight_layout()
plt.show()
