#!/usr/bin/env python
# -*- coding: utf8 -*-

import pandas as pd

"""
Utile fonctions to manipulate A0A pressure dataset. 
More as an exploration of fonctons library than real treatment.

Example :
- Correct the barometer by computing an estimate of the pressure following the Ideal Gas Law

2024/2025 — alaurent
"""

def compute_barometer_diagnostics(df, reference_time):
    """
    Compute similar barometric quantities using the Ideal Gas Law.

    Parameters
    ----------
    df : pandas.DataFrame
      Time-indexed 

    reference_time : pandas.DateTimeIndex or pandas.Timestamp
      Date choosen to compute nRV constant for Ideal Gas Law

    Return pandas.DataFrame
    --------
      Another time-indexed dataframe with additionnal columns, 
      such as pressure in pascal and temperature in Kelvin
    """
    diag = pd.DataFrame(index=df.index)

    diag['P_Pa'] = df['Barometer_pressure'] * 1e4
    diag['T_K'] = df['Barometer temperature'] + 273.15

    P0 = diag.loc[reference_time, 'P_Pa']
    T0 = diag.loc[reference_time, 'T_K']

    nRV = P0 / T0
    diag['P_ideal_dBar'] = (nRV * diag['T_K']) / 1e4

    return diag
