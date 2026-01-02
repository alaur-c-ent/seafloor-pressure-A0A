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

def plotlog(df, key, events_log, title, output_path=None, filenamout=None, plot_log=None, savefig=None):
    """
    Plot raw A0A pressure timeseries with events extracted from log.

    Parameters : 
    df : pandas.DataFrame
         Time-indexed dataframe containing both bottom pressures, temperature and barometer data.
    key : str
        Name of the column to plot.
    events_log : pandas.Dataframe
        Time-indexed dataframe containing timestamps of the valve movements. 
    title : str
    output_path : str
    filenamout : str
    plot_log : boolean
    savefig : boolean
           
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


