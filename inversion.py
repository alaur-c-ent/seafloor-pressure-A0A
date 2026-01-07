#!/usr/bin/env python
# -*- coding: utf8 -*-


"""
Least square method inversion to find the best parameters including the exponential decay.

Initial version constructed by A. Duvernay (https://orcid.org/0009-0008-5667-6559) 

April-June 2025, LIENSs (UMR 7266, La Rochelle), La Rochelle Universite
"""

import numpy as np
from numpy.linalg import inv

def invert_Jmatrix(time, tau_grid, calibrations, model='exp_linear', t_event=None):
    """
    Invert regression matrix to estimate drift model paramters 
    using least squares method with exploration of the exponential decay
    time constant (tau).

    Supported models :
    - 'exp_linear' : exponential + linear trend (default)
    - 'exp_lin_H' : exponential + linear trend + Heaviside step
    
    Parameters
    ----------
    time : array-like
        Time vector (in DateTime or in relative seconds). 
        If model is 'exp_lin_H', DateTime is required.
    tau_grid : array-like
        Grid of tau values (in seconds) explored for the exponential decay.
    calibrations : pandas.DataFrame
        Calibrations dates and values to fit. Dates must be DateTime object.
    model : str, optional
        Drift model type ('exp_linear' or 'exp_lin_H').
    t_event : datetime-like, optional
        Time of the step used for the Heaviside function (required if model
        is 'exp_lin_H').

    Returns
    -------
    params : ndarray
        Model parameters [d, b, a]
    best_tau : float
        Optimal relaxation time.
    H : ndarray or None
        Heaviside vector if used, otherwise None.
    """

    #### Model configuration
    if model == 'exp_linear':
        M = 3 # M the matrix dimension is model dependent
        use_heaviside = False
    elif model == 'exp_lin_H':
        if t_event is None:
            raise ValueError('Date time of the event must be provided for model exp_lin_H')
        M = 4 # M the matrix dimension is model dependent
        use_heaviside = True
    else:
        raise ValueError(f'Unknown model {model}')

    #### Convert Dates/Timestamp etc into relative time vector (in seconds) since beginning 
    ### if needed
    if np.issubdtype(time.dtype, np.datetime64):
        # time_s = np.asarray([t.total_seconds() for t in (time - time[0])])
        time_s = np.asanyarray((time - time.iloc[0]).dt.total_seconds())
    elif np.issubdtype(time.dtype, np.floating):
        time_s = np.asarray(time_s)

    ### Build the regression matrix (M dimension is model dependent)
    J = np.ones([len(time_s), M])
    # E = np.exp(-time_s / tau)
    ### Partial derivative :
    J[:, 1] = time_s

    ### Add Heaviside term
    H = None
    if use_heaviside:
        H = heaviside(np.asarray(time), t0=t_event, x=1, y=0)
        J[:, 2] = H
        exp_col = 3
    else:
        exp_col = 2

    ### Initate storage of residual variance (tau exploration)
    res_var = np.zeros(len(tau_grid))

    for i, tau in enumerate(tau_grid):
        J[:, exp_col] = np.exp(-time_s / tau) #+ b*time_s + d  # Exponenial decay

        invN = inv(J.T @ J)                             # Invert direct norm matrix
        params = invN @ J.T @ np.asarray(calibrations)  # Models parameters = [d, b, a]
        V = calibrations - J @ params                   # Residuals           
        res_var[i] = V.T @ V / (len(time_s) - M)        # Store residuals variance of the corresponding tau value 

    ### Best tau value that lowers the error
    best_tau = tau_grid[np.argmin(res_var)]

    ### Final inversion
    ### Invert again to find a, b, d parameters in adequation to the best tau value
    J[:, exp_col] = np.exp(-time_s / best_tau)
    invN = inv(J.T @ J)  
    params = invN @ J.T @ np.asarray(calibrations)

    print(params)

    ### Check on final residual variance (not used)
    # V = calib - J @ params
    # var = V.T @ V / (len(time_s) - M)  

    return time_s, params, best_tau, H


def fit_drift_curve(val, tau_grid, time, model='exp_linear', t_event=None):
    """
    Fit an exponential + linear drift model to a calibration curve.

    Parameters
    ----------
    val : array-like
        Values to fit (from e.g. 'Calib_1' or 'BPR_pressure_1').
    tau_grid : array-like
        Grid of tau values (in seconds) explored for the exponential decay.
    time : array-like
        Time vector. 
    model : str, optional
        Drift model type ('exp_linear' or 'exp_lin_H').
    t_event : datetime-like, optional
        Time of the step used for the Heaviside function (required if model
        is 'exp_lin_H').
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'params' : model parameters {a, tau, b, d}
        - 'model_type' : drift model type 
        - 'model'  : modeled calibration values
        - 'time_s' : time vector in seconds relative to start time
    """

    # #### Time vector in seconds
    # if 'time_seconds' in calib_df.columns:
    #     pass
    # else:
    #     ### Add new column: convert DateTime into relative seconds from beginning
    #     calib_df['time_seconds'] = (calib_df[time_col] - calib_df[time_col].iloc[0]).dt.total_seconds()

    #### Least Square Inversion (Gauss–Newton + tau grid)
    if model == 'exp_linear':
        time_s, params, best_tau, H = invert_Jmatrix(time=time,
                                        tau_grid=tau_grid,
                                        calibrations=val, 
                                        model=model, 
                                        t_event=None)
        #### Retrieve parameters
        ### Can be merged with next step using (*kwarg)
        d, b, a = params
        c = None

        #### Modelling
        model_values = exp_linear(time_s, a, best_tau, b, d)

    elif model == 'exp_lin_H':
        time_s, params, best_tau, H = invert_Jmatrix(time=time,
                                        tau_grid=tau_grid,
                                        calibrations=val, 
                                        model=model, 
                                        t_event=t_event)
        #### Retrieve parameters
        ### Can be merged with next step using (*kwarg)
        d, b, c, a = params

        #### Modelling
        model_values = exp_linear_H(time_s, a, best_tau, b, c, d, H)
        
    else:
        raise ValueError(f'Unknown model {model}')
    
    return {"params": {
                        'a' : a,
                        'b' : b,
                        'c' : c,
                        'd' : d,
                        'tau' : best_tau,
                        },
            "model_type" : model,
            "model": model_values,
            "time_s": time_s}


