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

# sys.path.append('/Users/alaure04/moby-data/CODES/Git_/src/A0A/')
from src.A0A.core import (read_events_log, calibrations, save_drift_model)
from src.A0A.plots import plot_calibration_curves
from src.A0A.inversion import fit_drift_curve

########################################
#### DEFINE PATHS ####
root_path = '/Users/alaure04/moby-data/DATA/'
recover_date = '2025_09_30'
station_name = 'A0A_MAY33_R'
X=8 ## manually indexing of the number of deployment (coherent with previous nomenclature, but automatisation not allowed here)
rsk_reference = '208295_20250930_0551'
rsk_ref_lst = rsk_reference.split('_')

output_path = os.path.join(root_path, recover_date, station_name,  'figures/calib/')

########################################
#### OPTIONS & AESTHETICS ####
text_size = 'large'

channels_colors = {'BB': 'violet',
                   'BPR1': 'orange',
                   'BPR2': 'darkgreen',
                   'T_ext' : 'tab:red'}


def main():

  #### Build input file paths
  zeros_path = os.path.join(root_path, recover_date, station_name, f'{station_name}_zeros.csv')
  
  events_path = os.path.join(root_path, recover_date, station_name, rsk_reference, f'{rsk_reference}_events.txt')
  
  calib_path = os.path.join(root_path,recover_date, station_name, rsk_reference,  f"A0A{X}_Calib.csv")
  
  os.makedirs(output_path, exist_ok=True)
  
  today = date.today().strftime("%Y%m%d")   ### Warning not UTC
  
  #### Read log
  events_df, t_marine, t_atmo, t_error = read_events_log(events_path)
  
  #### Read calibrations
  if glob.glob(calib_path):
      calib_df = pd.read_csv(calib_path, parse_dates=True, index_col=0)
      calib_df['Date'] = pd.to_datetime(calib_df['Date'])
  else:    
      #### Read extracted zeros (from STEP1)
      zeros_df = pd.read_csv(zeros_path, parse_dates=True, index_col=0)
      ## Time vector in seconds from beginning
      zeros_df['time_seconds'] = (zeros_df.index - zeros_df.index[0]).total_seconds()
      window = pd.Timedelta(minutes=20) # also working with datetime timedelta
      ### chosen borns in seconds after marine to atmospheric rotation
      lim_inf, lim_sup = (600, 1000)
  
      calib_df = calibrations(zeros_df, t_atmo, window, lim_inf, lim_sup)
  
  print(f'\n {today} - {len(calib_df)} calibrations (zeros) values\n')
  print(calib_df.to_string())
  print()
  
  ### if needed check/plot drift curves
  use_cmap = False
  title = f'{station_name} –– Calibration curves: P_barometric - P_internal'
  fig = plot_calibration_curves(calib_df, cols=('Calib_1', 'Calib_2'), 
                                title=title, colors_code=None, use_cmap=use_cmap,
                                ylim=(0.01, 0.15), text_size=text_size, figsize=(10, 5))
  
  if not use_cmap:
      fig.savefig(os.path.join(output_path, f"{station_name}_drift_curves.pdf"), dpi=300)
  else:
      fig.savefig(os.path.join(output_path, f"{station_name}_drift_curves_cmap.pdf"), dpi=300)
  plt.show()
  # plt.close(fig)
  
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
  
  # Grid on tau values (site dependent, user choice)
  if 'R' in station_name:
      tau_grid_BPR1 = np.linspace(1e5, 1e7, maxfev) ## MAY30_A0A_R BPR1
      tau_grid_BPR2 = np.linspace(1e10, 3e10, maxfev) ## MAY30_A0A_R BPR2
  elif 'C' in station_name:
      tau_grid_BPR1 = np.linspace(1e6, 1e7, maxfev) # 2025-07-18 MAY30_A0A_C
      tau_grid_BPR2 = np.linspace(1e6, 1e7, maxfev) # same
  
  ### EXP-LIN INVERSION ON DRIFT CURVES
  ## To improve : add the model_name to function to choose model type
  BPR1_fit = fit_drift_curve(calib_df, col='Calib_1', tau_grid=tau_grid_BPR1)
  BPR2_fit = fit_drift_curve(calib_df, col='Calib_2', tau_grid=tau_grid_BPR2)
  
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
  calib_df.to_csv(os.path.join(root_path, recover_date, station_name, f"A0A{X}_Calib.csv"))
  
  
  print(f'\n{today} - Save {model_name} models with parameters into a JSON file for each sensor.\n')
  sensor_name = 'BPR1' # can also be : 'BP1', 'Paros_1' etc
  json_pathout = os.path.join(root_path, recover_date, station_name, f'A0A{X}_{sensor_name}_drift_model.json')
  save_drift_model(BPR1_fit['params'], sensor_name, model_name, today, json_pathout)
  sensor_name = 'BPR2' # can also be : 'BP1', 'Paros_1' etc
  json_pathout = os.path.join(root_path, recover_date, station_name, f'A0A{X}_{sensor_name}_drift_model.json')
  save_drift_model(BPR2_fit['params'], sensor_name, model_name, today, json_pathout)
  
  ######################################
  #### DISPLAY RESULT
  ### if needed on checking/ploting only the modelled curves
  use_cmap = False
  if model_name == 'exp_linear':
      title = f'{station_name} –– Exponential-linear drift models'
  else : 
      title = f'{station_name} –– Drift models'
  fig = plot_calibration_curves(calib_df, cols=(f'{model_name}_1', f'{model_name}_2'), 
                                title=title, colors_code=None, use_cmap=use_cmap,
                                ylim=(0.01, 0.15), text_size=text_size, figsize=(10, 5))
  
  if not use_cmap:
      fig.savefig(os.path.join(output_path, f"{station_name}_drift_model.pdf"), dpi=300)
  else:
      fig.savefig(os.path.join(output_path, f"{station_name}_drift_model_cmap.pdf"), dpi=300)
  plt.show()
  # plt.close(fig)


if __name__ == "__main__":
    main()
