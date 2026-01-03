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


def plot_barometer_and_temperatures(df, calibration_times, colors, title='', text_size='large', plot_fig=None, output_path=None, filenamout=None, savefig=None):
    """
    Plot barometric pressure and temperature time series with calibration events.

    Parameters
    ----------
    df : pandas.DataFrame
        A0A dataframe containing pressure and temperature data.
    calibration_times : array-like of datetime
        Times of atmospheric (zero-pressure) calibration sequences.
    colors : dict
        Dictionary defining uniform color codes for channels.
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
        _, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
       
        axs[0].set_title(title, fontsize=text_size)
        ## Barometer pressure
        axs[0].plot(df.index, df['Barometer_pressure'], 
                    linestyle='-', c=colors['BB'], lw=1.5,
                    label='P_barometric', rasterized=True)
        axs[0].set_ylabel('Confined presssure [dBar]', fontsize=text_size)
        #axs[0].set_ylim(9.2, 9.7)

        ## TemperatureS
        for t in calibration_times:
            axs[1].axvline(t, color='r', lw=0.8, zorder=1) #, alpha=0.8)
        axs[1].plot(df.index, df['Barometer_temp'], 
                linestyle='-', c=colors['BB'], lw=0.8, #alpha=0.6,
                    label='T_barom', rasterized=True)
        axs[1].plot(df.index, df['BPR_temp_1'], 
                linestyle='dashed', c=colors['BP1'], lw=0.8, #alpha=0.6,
                    label='T_BPR1', rasterized=True)
        axs[1].plot(df.index, df['BPR_temp_2'], 
                linestyle='dashed', c=colors['BP2'], lw=0.8, #alpha=0.6,
                    label='T_BPR2', rasterized=True)
        axs[1].plot(df.index, df['External_temp'], 
                linestyle='-', c='tab:red', lw=0.8, #alpha=0.6,
                    label='T_ext', rasterized=True)
        axs[1].set_ylabel('Degrees [°C]', fontsize=text_size)
        #axs[1].set_ylim(2., 5.)
        axs[-1].set_xlabel('Dates', fontsize=text_size)

        for _ax in axs:
            _ax.grid(which='both', lw=0.45, color='dimgrey', zorder=0)
            for i, t in enumerate(calibration_times):
                _ax.axvline(t, color='r', lw=0.8, zorder=1, alpha=0.8, label='Calib.' if i == 0 else '')
                axs[0].annotate(f'{i+1}', 
                            xy=(t, np.max(df['Barometer_pressure']-0.01)), 
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


def plot_calibrations(zeros_df, calibration_times, keys, window, select_window=(600, 1000),
                      cmap='viridis', figsize=(8, 4)):
    """
    Subplot calibration sequences of each pressure sensors.

    This function superposes all zero segments in function of the relative time since valve switch,
    with color encoding based on the calibration sequence index.

    Parameters
    ----------
    zeros_df : pandas.DataFrame
        DataFrame containing zero segments.
    calibration_times : array-like
        Start times of zero-pressure calibration sequences.
    keys : list
        List of column names to plot (e.g. BPR pressure 1, 2).
    window : timedelta
        Duration of each zero segment.
    select_window : tuple
        Start and end relative times (in seconds) of the selected stable window.
    cmap : str
        Matplotlib colormap name.
    """

    n_calib = len(calibration_times)
    cmap_obj = plt.get_cmap(cmap)
    norm = colors.Normalize(vmin=1, vmax=n_calib)

    fig, axs = plt.subplots(
        1, len(keys), figsize=figsize, sharey=True
    )

    if len(keys) == 1:
        axs = [axs]

    for ax, col in zip(axs, keys):
        ax.grid(which='both', lw=0.4, color='lightgrey', zorder=0)
        ax.set_xlabel('Elapsed time [s]')
        ax.set_ylabel('Internal pressure [dBar]')
        ax.set_title(col)

        calib_id = 1
        for t in calibration_times:
            seg = zeros_df.loc[t:t+window]
            if seg.empty:
                calib_id += 1
                continue

            elapsed = seg['time_seconds'] - seg['time_seconds'].iloc[0]
            color = cmap_obj(norm(calib_id))

            ax.plot(elapsed, seg[col], color=color, lw=0.8)
            calib_id += 1

        # Highlight selected time window
        ax.axvspan(select_window[0], select_window[1], 
                   color='silver', alpha=0.4, zorder=1,)

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    cbar = fig.colorbar(sm, ax=axs, pad=0.02)
    cbar.set_label('Calibration index')

    fig.tight_layout()
    return fig




def plot_check_zeros(zeros_df, calibration_times, window, select_window=(600, 1000), 
                      highlight_ids=None, colors=None, figsize=(10, 8)):
    """
    This function display all zero segments for multiple variables 
    in function of the relative time from valve switch, 
    optionnal highlighting of selected calibration sequences.

    Parameters
    ----------
    zeros_df : pandas.DataFrame
        DataFrame containing zero segments.
    calibration_times : array-like
        Start times of zero-pressure calibration sequences.
    window : timedelta
        Duration of each zero segment.
    select_window : tuple
        Start and and relative times (in seconds)of the selected stable window.
    highlight_ids : list, optional
        List of calibration segment indices to highlight.
    colors : dict, optional
        Dictionary defining color scheme for channels.
    """

    if highlight_ids is None:
        highlight_ids = []

    if colors is None:
        colors = {'BPR_pressure_1' : 'orange',
                  'BPR_pressure_2' : 'darkgreen',
                  'Barometer_pressure' : 'violet',
                  'External_temp' : 'tab:red'}

    variables = [
        ('BPR_pressure_1', 'Internal pressure [dBar]'),
        ('BPR_pressure_2', 'Internal pressure [dBar]'),
        ('Barometer_pressure', 'Internal pressure [dBar]'),
        ('External_temp', 'Temperature [°C]')]

    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axs = axs.flatten()

    for ax, (var, ylabel) in zip(axs, variables):
        ax.set_ylabel(ylabel)
        ax.grid(which='both', lw=0.4, color='lightgrey', zorder=0)

        calib_id = 1
        for t in calibration_times:
            seg = zeros_df.loc[t:t+window]
            if seg.empty:
                calib_id += 1
                continue

            elapsed = seg['time_seconds'] - seg['time_seconds'].iloc[0]

            # Default appearance
            color = 'lightgrey'
            lw = 0.8
            zorder = 2

            # Channel-based color (if provided)
            for key, c in colors.items():
                if key in var:
                    color = c

            # Highlight selected segments
            if calib_id in highlight_ids:
                color = 'purple'
                lw = 1.5
                zorder = 5

            ax.plot(elapsed, seg[var], color=color, lw=lw, zorder=zorder)
            calib_id += 1

        # Highlight selected time window
        ax.axvspan(select_window[0], select_window[1], 
                   color='silver', alpha=0.4, zorder=1,)

        ax.set_title(var, fontsize='medium')

    for ax in axs[2:]:
        ax.set_xlabel('Elapsed time [s]')

    fig.tight_layout()
    return fig
