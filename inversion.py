#!/usr/bin/env python
# -*- coding: utf8 -*-


"""
Least square method inversion to find the best parameters including the exponential decay.

Initial version constructed by A. Duvernay (https://orcid.org/0009-0008-5667-6559) 

April-June 2025, LIENSs (UMR 7266, La Rochelle), La Rochelle Universite
"""

import numpy as np
from numpy.linalg import inv

def invert_Jmatrix(time_s, tau_grid, calib, maxfev):
    """
    Invert exponential + linear drift model using a grid search on tau
    and linear least squares for remaining parameters (a, b, d).

    Parameters
    ----------
    tau_grid : 
        Grid for the exponential decay parameter
    time_s : 
        Time vector (in seconds)
    calib : 
        calibration dataset (observations)
    
    Returns
    -------
    params : ndarray
        Model parameters [d, b, a]
    best_tau : float
        Optimal relaxation time.
    """
    # ## Initate storage of residual variance
    res_var = np.zeros(maxfev)
    ### Build the regression matrix (M dimension = 3 columns)
    M = 3
    J = np.ones([len(time_s), M])
    # E = np.exp(-time_s / tau)
    # partial derivative :
    J[:, 1] = time_s

    for i, tau_ in enumerate(tau_grid):
        J[:, 2] = np.exp(-time_s/tau_) #+ b*time_s + d  # Exponenial decay
        invN = inv(J.T @ J)                             # Invert direct norm matrix
        params = invN @ J.T @ np.array(calib)           # Model parameters = [d, b, a]
        V = calib - J @ params                          # Residuals
        var = V.T @ V / (len(time_s) - M)               # Residuals variance 
        res_var[i] = var                                # Store error of the corresponding tau value 

    ### Best tau value that lowers the error
    best_tau = tau_grid[np.argmin(res_var)]
    ### Invert again to find a, b, d parameters in adequation to the best tau value
    J[:, 2] = np.exp(-time_s/best_tau)
    invN = inv(J.T @ J)  
    params = invN @ J.T @ np.array(calib)
    V = calib - J @ params
    var = V.T @ V / (len(time_s) - M)  

    return params, best_tau


def fit_drift_curve(calib_df, col, tau_grid, time_col='Date', maxfev=1000):
    """
    Fit an exponential + linear drift model to a calibration curve.

    Parameters
    ----------
    calib_df : pandas.DataFrame
        Calibration DataFrame (output of STEP 2).
    col : str
        Name of the column to fit (e.g. 'Calib_1', 'BPR_pressure_1').
    tau_grid : array-like
        Grid of tau values (in seconds) explored for the exponential decay.
    time_col : str, optional
        Column containing dates vector.
    maxfev : int, optionnal
        Number of iteration on tau parameter. Default is 1000.
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'params' : model parameters {a, tau, b, d}
        - 'model'  : modeled calibration values
        - 'time_s' : time vector in seconds relative to start time
    """

    #### Time vector in seconds
    ### Add new column: convert DateTime into relative seconds from beginning
    calib_df['time_seconds'] = (calib_df['Date'] - calib_df['Date'].iloc[0]).dt.total_seconds()


    #### Least Square Inversion (Gauss–Newton + tau grid)
    params, best_tau = invert_Jmatrix(time_s=calib_df['time_seconds'].values, 
                                      obs=calib_df[col].values, 
                                      tau_grid=tau_grid, maxfev=maxfev)

    #### Retrieve parameters
    ### Can be merged with next step using (*kwarg)
    d, b, a = params

    #### Modelling
    model = exp_linear(calib_df['time_seconds'].values, a, best_tau, b, d)

    return {"params": {
                        "a": a,
                        "tau": best_tau,
                        "b": b,
                        "d": d},
            "model": model,
            "time_s": calib_df['time_seconds'].values}

