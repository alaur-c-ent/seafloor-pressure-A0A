#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
List of potential drift models to invert. 

2024/2025 — alaurent
"""

import numpy as np

def exp_linear(t, a, tau, b, d):
    """
    Combined exponential decay + linear trend: 
    p(t) = a*exp(-t/tau) + b*t + d
    
    Model from Wilcock et al., (2021), 
    that combine an exponential decay (effect of the relax of the system after deployment) 
    and a linear trend (long term drift), 
    without any dependency to temperature.
    """
    return a * np.exp(-t / tau) + b * t + d


def residuals(params, t, p):
    return exp_linear(t, *params) - p


def double_exp_linear(t, a1, tau1, a2, tau2, b, c):
    """
    Double exponential decay + linear trend: 
    p(t) = a1*exp(-t/tau1) + a2*exp(-t/tau2) + b*t + c
    
    I want to test this model under the assumption of two relaxations:
    A rapid relaxation related to the pressurisation of the oil within the pipe right after the deployment,
    and, a more slower long term drift of exp+lin form as below, explaining the Quartz drift and slow oil diffusion.
    """
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + b * t + c


def residuals(params, t, p):
    return double_exp_linear(t, *params) - p


def heaviside(t, t0, x, y):
     """
     Heaviside function composed of 1 and 0. 
     time vector should be type class numpy.dtype[datetime64]
     in order to be compare with conditionnal (np.where())
     """
     Harray = np.where(t >= t0, x, y)   
     return Harray


def exp_linear_H(t, a, tau, b, c, d, H):
    """
    Combined exponential decay + linear trend + Heaviside function: 
    p(t) = a*exp(-t/tau) + b*t + c*H(t) + d
    
    Adding a Heaviside function to account for the sharp pressure drop 
    that seems to have affected calibration values.
    """
    return a * np.exp(-t / tau) + b * t + c * H + d

