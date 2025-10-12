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
    sGlob = f'/Users/siddharth/Documents/GitHub/EconometricsForQuantitativeFinance/Case2/data/Price*_all_*_i0.xlsx'
    asF = np.sort(glob.glob(sGlob))

    # Read the data, best to limit to only the symbols of interest
    # Parse your symbols
    symbols_list = dtArg['symbols'].split()

    # Read and combine all monthly files
    dfs = []
    for file_path in asF:
        df_month = pd.read_excel(file_path, index_col=3)
        # Filter for our symbols only
        available_symbols = [sym for sym in symbols_list if sym in df_month.columns]
        if available_symbols:
            df_filtered = df_month[available_symbols]
            dfs.append(df_filtered)

    # Combine all monthly data
    df = pd.concat(dfs, axis=0)
    df = df.sort_index()

    return df

###########################################################
### main
def main():
    # Magic numbers
    dtArg= {
        'symbols': 'SPX5.L SPY5z.CHIX SPY5.P',        # Change list of symbols to the symbols of your group
        'group': '9'
    }

    # Initialisation
    # Initialise(dtArg)

    # Estimation
    df= PrepSPY(dtArg)

    # Output
    sOut= f'/Users/siddharth/Documents/GitHub/EconometricsForQuantitativeFinance/Case2/sp_{dtArg["group"]}.csv.gz'
    df.to_csv(sOut)

    print (f'See {df.shape} observations in {sOut}')
    print ('Beginning of dataset:')
    print (df.head())
    print(f"Date range: {df.index.min()} to {df.index.max()}")
###########################################################
### start main
if __name__ == "__main__":
    main()
