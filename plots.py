#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
All plots

2024/2025 - alaurent 
"""


import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.dates as mdates
from datetime import date, datetime, timedelta

def plotlog(df, key, events_log, title='', plot_log=None, output_path=None, filenamout=None, savefig=None):
    """
    Plot raw A0A pressure timeseries with events extracted from log.

    Parameters
    ----------
    df : pandas.DataFrame
         Time-indexed dataframe containing both bottom pressures, temperature and barometer data.
    key : str
        Name of the column to plot.
    events_log : pandas.Dataframe
        Time-indexed dataframe containing timestamps of the valve movements. 
    title : str
        Title of the figure. Default is empty title (meaning no title)
    plot_log : boolean
        Will display or not the figure. Should be True if wanna save it. 
    output_path : str
    filenamout : str
    savefig : boolean
    
    Returns if displayed
    -------
    matplotlib.figure.Figure
        The generated figure.           
    """
    if plot_log:
        if '1' in key:
            color='orange'
        elif '2' in key:
            color='darkgreen'
        plt.figure() #figsize=(10, 6))
        plt.grid(which='both', lw=0.45, color='dimgrey', zorder=0)
        plt.plot(df.index, df[key], label=key, color=color)
        plt.xlabel('Dates')
        plt.ylabel('Pressure [dBar]')
        plt.title(title, loc='left')
        plt.legend(loc='upper center', framealpha=1.)
        plt.twinx()
        plt.plot(events_log.Type.astype(str), c = 'r', lw = 0, marker = '.')
        plt.tight_layout()

        if savefig:
            if output_path is not None and filenamout is not None:
                plt.savefig(os.path.join(output_path, filenamout), dpi=300)
            else:
                print('Ouput path and/or name of the figure are None.')
        return plt.show()


def plot_barometer_and_temperatures(df, calibration_times, channels_colors, title='', text_size='large', plot_fig=None, output_path=None, filenamout=None, savefig=None):
    """
    Plot barometric pressure and temperature time series with calibration events.

    Parameters
    ----------
    df : pandas.DataFrame
        A0A dataframe containing pressure and temperature data.
    calibration_times : array-like of datetime
        Times of atmospheric (zero-pressure) calibration sequences.
    channels_colors : dict
        Dictionary defining uniq color codes for channels.
    text_size : str, optional
        Font size for ticks, ticks labels and title.
    title : str
        Title of the figure. Default is empty title (meaning no title)
    plot_fig : boolean, optional
        Will display or not the figure. Should be True if wanna save it. 
    output_path : str, optional
    filenamout : str, optional
    savefig : boolean, optional
    
    Returns if displayed
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    if plot_fig:
        _, axs = plt.subplots(2, 1, sharex=True) #figsize=(12, 6)
       
        axs[0].set_title(title, fontsize=text_size)
        ## Barometer pressure
        axs[0].plot(df.index, df['Barometer_pressure'], 
                    linestyle='-', c=channels_colors['BB'], lw=1.5,
                    label='P_barometric')
        axs[0].set_ylabel('Confined presssure [dBar]', fontsize=text_size)
        axs[0].set_ylim(9.3, 9.4)

        ## TemperatureS
        for t in calibration_times:
            axs[1].axvline(t, color='r', lw=0.8, zorder=1) #, alpha=0.8)
        axs[1].plot(df.index, df['Barometer_temp'], 
                linestyle='-', c=channels_colors['BB'], lw=0.8, #alpha=0.6,
                    label='T_barom')
        axs[1].plot(df.index, df['BPR_temp_1'], 
                linestyle='dashed', c='orange', lw=0.8, #alpha=0.6,
                    label='T_BPR1')
        axs[1].plot(df.index, df['BPR_temp_2'], 
                linestyle='dashed', c='darkgreen', lw=0.8, #alpha=0.6,
                    label='T_BPR2')
        axs[1].plot(df.index, df['External_temp'], 
                linestyle='-', c='tab:red', lw=0.8, #alpha=0.6,
                    label='T_ext')
        axs[1].set_ylabel('Degrees [°C]', fontsize=text_size)
        axs[1].set_ylim(2., 5.)
        axs[-1].set_xlabel('Dates', fontsize=text_size)

        for _ax in axs:
            _ax.grid(which='both', lw=0.45, color='dimgrey', zorder=0)
            for i, t in enumerate(calibration_times):
                _ax.axvline(t, color='r', lw=0.8, zorder=1, alpha=0.8, label='Calib.' if i == 0 else '')
                axs[0].annotate(f'{i+1}', 
                            xy=(t, np.max(df['Barometer pressure']+0.01)), 
                            textcoords='data', 
                            ha='center', va='bottom',
                            zorder=10)
            _ax.tick_params(axis='both', labelsize=text_size)
            _ax.legend(loc='lower center', ncol=5, fontsize=text_size, labelspacing=0.2)
        plt.tight_layout()
        if savefig:
            if output_path is not None and filenamout is not None:
                plt.savefig(os.path.join(output_path, filenamout), dpi=300)
            else:
                print('Ouput path and/or name of the figure are None.')
        return plt.show()
