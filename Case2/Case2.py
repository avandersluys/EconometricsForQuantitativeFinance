#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case2.py

Purpose:
    Main analysis file for Case 2 - VAR Assignment Group 9
    Uses data prepared by prepdata_sp500.py

Version:
    1       Group 9 implementation

Date:
    2025/10/12

Author:
    Group 9
"""
###########################################################
### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

###########################################################

### Load Data
github_url = "https://raw.githubusercontent.com/avandersluys/EconometricsForQuantitativeFinance/7f88e429a23a03593352b07a64f9d882017f5246/Case2/sp_9.csv.gz"
df = pd.read_csv(github_url, index_col=0, parse_dates=True)

# Exploratory Data Analysis
print("")
print("Exploratory Data Analysis")
print("")
print(df.head(5))
print(df.tail(5))
print(f"Shape: {df.shape[0]:,} observations × {df.shape[1]} symbols")
print(f"Symbols: {list(df.columns)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Time span: {(df.index.max() - df.index.min()).days} days")

# Missing Values
print("Missing values:")
for col in df.columns:
    missing = df[col].isnull().sum()
    pct = (missing/len(df))*100
    print(f"  {col}: {missing:,} ({pct:.1f}%)")

