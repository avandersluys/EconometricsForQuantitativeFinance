#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepdata_sp500.py

Purpose:
    Prepare the data on SP500

Version:
    1       First start, outline for students to start with

Date:
    2025/9/22

Author:
    ???
"""
###########################################################
### Imports
import numpy as np
import pandas as pd
import glob

###########################################################
### df= PrepSPY(dtArg)
def PrepSPY(dtArg):
    """
    Purpose:
        Read the data on the symbols of this group

    Inputs:
        dtArg   dictionary, settings

    Return value:
        df      dataframe, prices
    """
    sGlob= f'data/Price*_all_*_i0.xlsx'
    asF= np.sort(glob.glob(sGlob))

    # Read the data, best to limit to only the symbols of interest
    # ... fill in code

    return df

###########################################################
### main
def main():
    # Magic numbers
    dtArg= {
        'symbols': 'SPX5.L SPY5.P SPY5.MIL',        # Change list of symbols to the symbols of your group
        'group': 'g0'
    }

    # Initialisation
    # Initialise(dtArg)

    # Estimation
    df= PrepSPY(dtArg)

    # Output
    sOut= f'output/sp_{dtArg["group"]}.csv.gz'
    df.to_csv(sOut)

    print (f'See {df.shape} observations in {sOut}')
    print ('Beginning of dataset:')
    print (df.head())

###########################################################
### start main
if __name__ == "__main__":
    main()
