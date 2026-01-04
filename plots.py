#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
All plots

2024/2025 - alaurent 
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

def get_color_from_name(name, colors_code, default='black'):
    for key, value in colors_code.items():
        if key[-1].isdigit() and key[-1] in name:
            return value
    return default


def plotlog(df, key, events_log, colors_code=None, title='', plot_log=None, output_path=None, filenamout=None, savefig=None):
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
    colors_code : dict, optionnal
        Dictionary defining uniform color scheme for channels.
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
        if colors_code is None:
            colors_code = {'BPR_pressure_1' : 'orange',
                      'BPR_pressure_2' : 'darkgreen',
                      'Barometer_pressure' : 'violet',
                      'External_temp' : 'tab:red'}
        plt.figure(figsize=(10, 6))
        plt.grid(which='both', lw=0.45, color='dimgrey', zorder=0)
        plt.plot(df.index, df[key], label=key, color=get_color_from_name(key, colors_code))
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


def plot_barometer_and_temperatures(df, calibration_times, colors_code=None, title='', text_size='large', plot_fig=None, output_path=None, filenamout=None, savefig=None):
    """
    Plot barometric pressure and temperature time series with calibration events.

    Parameters
    ----------
    df : pandas.DataFrame
        A0A dataframe containing pressure and temperature data.
    calibration_times : array-like of datetime
        Times of atmospheric (zero-pressure) calibration sequences.
    colors_code : dict, optionnal
        Dictionary defining uniform color scheme for channels.
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
        if colors_code is None:
            colors_code = {'BPR_pressure_1' : 'orange',
                      'BPR_pressure_2' : 'darkgreen',
                      'Barometer_pressure' : 'violet',
                      'External_temp' : 'tab:red'}
            
        _, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
       
        axs[0].set_title(title, fontsize=text_size)
        ## Barometer pressure
        axs[0].plot(df.index, df['Barometer_pressure'], 
                    linestyle='-', c=colors_code['BB'] or colors_code['Barometer_pressure'], lw=1.,
                    label='P_barometric', rasterized=True)
        axs[0].set_ylabel('Confined presssure [dBar]', fontsize=text_size)
        # axs[0].set_ylim(9.3, 9.4)

        ## TemperatureS
        for t in calibration_times:
            axs[1].axvline(t, color='r', lw=0.8, zorder=1) #, alpha=0.8)
        axs[1].plot(df.index, df['Barometer_temp'], 
                linestyle='-', c=colors_code['BB'] or colors_code['Barometer_pressure'], lw=0.8, #alpha=0.6,
                    label='T_barom', rasterized=True)
        axs[1].plot(df.index, df['BPR_temp_1'], 
                linestyle='dashed', c=get_color_from_name('BPR_temp_1', colors_code), lw=0.8, #alpha=0.6,
                    label='T_BPR1', rasterized=True)
        axs[1].plot(df.index, df['BPR_temp_2'], 
                linestyle='dashed', c=get_color_from_name('BPR_temp_2', colors_code), lw=0.8, #alpha=0.6,
                    label='T_BPR2', rasterized=True)
        axs[1].plot(df.index, df['External_temp'], 
                linestyle='-', c=colors_code['External_temp'] or 'tab:red', lw=0.8, #alpha=0.6,
                    label='T_ext', rasterized=True)
        axs[1].set_ylabel('Degrees [°C]', fontsize=text_size)
        # axs[1].set_ylim(2., 5.)
        axs[-1].set_xlabel('Dates', fontsize=text_size)

        for _ax in axs:
            _ax.grid(which='both', lw=0.45, color='dimgrey', zorder=0)
            for i, t in enumerate(calibration_times):
                _ax.axvline(t, color='r', lw=0.8, zorder=1, alpha=0.8, label='Calib.' if i == 0 else '')
                axs[0].annotate(f'{i+1}', 
                            xy=(t, np.max(df['Barometer_pressure']+0.01)), 
                            textcoords='data', 
                            ha='center', va='bottom',
                            zorder=10)
            _ax.tick_params(axis='both', labelsize=text_size)
            _ax.legend(loc='upper center', ncol=5, labelspacing=0.2, fontsize=text_size)
        plt.tight_layout()
        if savefig:
            if output_path is not None and filenamout is not None:
                plt.savefig(os.path.join(output_path, filenamout), dpi=300)
            else:
                print('Ouput path and/or name of the figure are None.')
        return plt.show()
    

def plot_pressure(clean_df, calibration_times, colors_code=None, fig_title='', text_size='medium', 
                  plot_fig=None, output_path=None, filenamout=None, savefig=None):
    """
    Plot seafloor pressure timeseries.

    This function displays the full pressure time series recorded by the two
    internal pressure sensors (BPR1 and BPR2), after data cleaning (STEP 1),
    and overlays the timing of zero-pressure calibration sequences.

    No correction is applied to the data in this plot.

    Parameters
    ----------
    clean_df : pandas.DataFrame
        Time-indexed cleaned pressure dataset (end of STEP 1).
        Must contain columns:
        - 'BPR_pressure_1'
        - 'BPR_pressure_2'
    calibration_times : array-like
        Datetime values corresponding to the start times of calibration
        (zero-pressure) sequences.
    colors_code : dict, optional
        Dictionary defining the color scheme for channels.
        If None, a default color scheme is used.
    fig_title : str, optional
        Global title of the figure.
    text_size : str, optional
        Font size for labels and titles.
    plot_fig : bool, optional
        If True, the figure is created and displayed.
    output_path : str, optional
        Directory where the figure is saved (if savefig is True).
    filenamout : str, optional
        Output filename for the figure.
    savefig : bool, optional
        If True, save the figure to disk.

    Returns (if displayed)
    -------
        The function displays the figure and optionally saves it to disk.
    """
    if plot_fig:
        if colors_code is None:
            colors_code = {'BPR_pressure_1' : 'orange',
                      'BPR_pressure_2' : 'darkgreen',
                      'Barometer_pressure' : 'violet',
                      'External_temp' : 'tab:red'}
            
        fig, axs = plt.subplots(2, 1, sharex=True, constrained_layout=True)

        axs[0].set_title(f'Paros 1 (BPR1)', loc='left') #, fontsize=text_size)
        axs[0].plot(clean_df.index, (clean_df['BPR_pressure_1'].values), # this way make it less longer to plot if large dataset
                c=get_color_from_name('BPR_pressure_1', colors_code), 
                label = 'Uncorrected pressure', 
                rasterized=True, lw=0.8, zorder=2)

        axs[1].set_title(f'Paros 2 (BPR2)', loc='left') #, fontsize=text_size)
        axs[1].plot(clean_df.index, (clean_df['BPR_pressure_2'].values), 
                c=get_color_from_name('BPR_pressure_2', colors_code), 
                label = 'Uncorrected pressure', 
                rasterized=True, lw=0.8, zorder=2)

        axs[-1].set_xlim(clean_df.index[0]-timedelta(days=2), 
                         clean_df.index[-1]+timedelta(days=4))
        axs[-1].xaxis.set_major_locator(mdates.YearLocator())
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) # fmt “Jan 2025”
        axs[-1].xaxis.set_minor_locator(mdates.MonthLocator())
        axs[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        # axs[-1].set_xlabel('Dates')#, labelsize=text_size)

        for _ax in axs:
            for i, t in enumerate(calibration_times):
                    _ax.axvline(t, color='tomato', lw=0.8, zorder=1, alpha=0.8, 
                                label='Calib.' if i == 0 else '')
            _ax.set_ylabel('Seafloor pressure [dBar]') #, labelsize=text_size)
            _ax.grid(which='both', lw=0.45, color='dimgrey', zorder=0)
            _ax.tick_params(axis='both') #, labelsize=text_size)
            _ax.legend(loc='upper right', labelspacing=0.05)

        fig.suptitle(fig_title)

        if savefig:
            if output_path is not None and filenamout is not None:
                plt.savefig(os.path.join(output_path, filenamout), dpi=300)
            else:
                print('Ouput path and/or name of the figure are None.')
        return plt.show()
        

def plot_calibrations(zeros_df, calibration_times, keys, window, select_window=(600, 1000),
                      ylim=(9., 10.), cmap='viridis', text_size='large', figsize=(8, 4)):
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
        Default values are 600 to 1000 seconds after internal valve switch.
    ylim : tuple,
        Internal pressure (barometric) limitation for the y axis. 
    cmap : str
        Matplotlib colormap name.
    """

    n_calib = len(calibration_times)
    cmap_obj = plt.get_cmap(cmap)
    norm = colors.Normalize(vmin=1, vmax=n_calib)

    fig, axs = plt.subplots(1, len(keys), figsize=figsize, sharey=True)

    if len(keys) == 1:
        axs = [axs]

    for ax, col in zip(axs, keys):
        ax.tick_params(axis='both', labelsize=text_size)
        ax.grid(which='both', lw=0.4, color='lightgrey', zorder=0)
        ax.set_xlabel('Elapsed time [s]', fontsize=text_size)
        ax.set_title(col, fontsize=text_size)

        calib_id = 1
        for t in calibration_times:
            seg = zeros_df.loc[t:t+window]
            if seg.empty:
                calib_id += 1
                continue

            elapsed = seg['time_seconds'] - seg['time_seconds'].iloc[0]
            color = cmap_obj(norm(calib_id))

            ax.plot(elapsed, seg[col], color=color, lw=1.)
            calib_id += 1

        # Highlight selected time window
        ax.axvspan(select_window[0], select_window[1], 
                   color='silver', alpha=0.4, zorder=1,)

    axs[0].set_ylim(*ylim)
    axs[0].set_ylabel('Internal pressure [dBar]', fontsize=text_size)
    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    cbar = fig.colorbar(sm, ax=axs[1], pad=0.02)
    cbar.set_label('Calibration index', fontsize='medium')
    cbar.ax.tick_params(labelsize='medium') 
    cbar.ax.invert_yaxis()
    ## only integer ticks
    cbar.locator = ticker.MaxNLocator(integer=True, prune=None)  
    cbar.update_ticks()

    fig.tight_layout()
    return fig


def plot_check_zeros(zeros_df, calibration_times, window, select_window=(600, 1000), 
                    ylim=(9.3, 9.8), highlight_ids=None, colors_code=None, figsize=(10, 8)):
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
    ylim : tuple,
        Internal pressure (barometric) limitation for the y axis. 
    highlight_ids : list, optional
        List of calibration segment indices to highlight.
    colors_code : dict, optional
        Dictionary defining color scheme for channels.
    """

    if highlight_ids is None:
        highlight_ids = []

    if colors_code is None:
        colors_code = {'BPR_pressure_1' : 'orange',
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
            for key, c in colors_code.items():
                if key in var:
                    color = c

            if var in ['BPR_pressure_1', 'BPR_pressure_2']:
                ax.set_ylim(*ylim)

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


def plot_calibration_curves(calib_df, cols=('Calib_1', 'Calib_2'), title='', colors_code=None, use_cmap=False,
                            ylim=(0, 20), text_size='large', figsize=(10, 5)):
    """
    Plot calibration (drift) curves derived from zero-pressure values.

    Parameters
    ----------
    calib_df : pandas.DataFrame
        DataFrame containing calibration values. Must have 'Date', 'Calib_1', 'Calib_2' named columns.
    cols : tuple
        Columns names to plot (default Calib_1, Calib_2).
    title : str, 
        Title of the figure. Default is empty title (meaning no title)
    colors_code : dict, optional
        Dictionary defining uniform color scheme for sensors.
    use_cmap : bool, optional
        If True, color points using a colormap as a function of calibration sequences indexes.
    ylim : tuple, optional
        Y-axis limits (e.g., (0, 20)). Max drifting is about 20 cm in worst case scenario.
    """

    if colors_code is None:
        colors_code = {'Calib_1' : 'orange',
                  'Calib_2' : 'darkgreen',
                  'Barometer_pressure' : 'violet',
                  'External_temp' : 'tab:red'}
        
    fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    ax.tick_params(axis='both', labelsize=text_size)
    ax.set_title(title, fontsize=text_size)

    ax.grid(which='both', lw=0.45, color='lightgrey', zorder=0)

    if use_cmap:
        ### Colors options 
        N=len(calib_df)                 
        cmap = plt.cm.viridis  
        # cmap = plt.get_cmap(colormap)
        norm = colors.Normalize(vmin=1, vmax=N)

    for i, col in enumerate(cols):
        label = col.replace("_", " ")
        
        if use_cmap:
            ax.plot(calib_df.Date, calib_df[col], 
                color=get_color_from_name(col, colors_code), 
                zorder=1, alpha=0.8, linestyle='-',
                label=label)
            sc = ax.scatter(calib_df.Date, calib_df[col], 
                        linestyle='-', marker='o', s=50,
                        c=calib_df.index, 
                        cmap=cmap, norm=norm, 
                        zorder=4)
        else:
            ax.plot(calib_df.Date, calib_df[col], 
                    '-o', c=get_color_from_name(col, colors_code), 
                    label=label)
    if ylim:
        # ax.set_ylim(*ylim)
        ax.set_ylim(-ylim[1], ylim[0]) ## max drifting is about 20 cm.
        
    # ax.set_xlabel('Dates', fontsize=text_size)
    ax.set_ylabel(u'Normalised internal presssure [dBar]', fontsize=text_size)
    ax.legend(loc='lower left')

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) # fmt “Jan 2025”
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))


    if use_cmap:
        # sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        # cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('Calibration index', fontsize='medium')
        cbar.ax.tick_params(labelsize='medium') 
        cbar.ax.invert_yaxis()
        ## only integer ticks
        cbar.locator = ticker.MaxNLocator(integer=True, prune=None)  
        cbar.update_ticks()

    return fig

