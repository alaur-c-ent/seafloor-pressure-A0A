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
    Invert the regression matrix to find the best a, b, d parameters
    and to explore the exponential decay slope: tau
    Initially made by A. Duvernay. 

    This is an adapted Gauss–Newton method 
    tau_grid : search grid for the exponential decay parameter
    time_s : time vector (in seconds)
    calib : calibration dataset (observations)
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
        X = invN @ J.T @ np.array(calib)                # Models coefficients X = [d, b, a]
        V = calib - J @ X                               # Residuals
        var = V.T @ V / (len(time_s) - M)               # Residuals variance 
        res_var[i] = var                                # Store error of the corresponding tau value 

    ### Best tau value that lowers the error
    best_tau = tau_grid[np.argmin(res_var)]
    ### Invert again to find a, b, d parameters in adequation to the best tau value
    J[:, 2] = np.exp(-time_s/best_tau)
    invN = inv(J.T @ J)  
    X = invN @ J.T @ np.array(calib)
    V = calib - J @ X
    var = V.T @ V / (len(time_s) - M)  

    return X, best_tau


