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
import warnings
warnings.filterwarnings('ignore')

###########################################################

### Load Data
github_url = "https://raw.githubusercontent.com/avandersluys/EconometricsForQuantitativeFinance/7f88e429a23a03593352b07a64f9d882017f5246/Case2/sp_9.csv.gz"
df = pd.read_csv(github_url, index_col=0, parse_dates=True)

# Exploratory Data Analysis
print("Exploratory Data Analysis")
print("")
print(f"Shape: {df.shape[0]:,} observations × {df.shape[1]} symbols")
print(f"Symbols: {list(df.columns)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Time span: {(df.index.max() - df.index.min()).days} days")
print("")

# Missing Values Analysis  
print("Missing values:")
for col in df.columns:
    missing = df[col].isnull().sum()
    pct = (missing/len(df))*100
    print(f"  {col}: {missing:,} ({pct:.1f}%)")

# Missing data patterns for SPY5.P
spy5p_series = df['SPY5.P'].copy()
missing_mask = spy5p_series.isnull()
missing_periods = df.index[missing_mask]

if len(missing_periods) > 0:
    print(f"\nMissing data periods:")
    missing_df = pd.DataFrame({'datetime': missing_periods})
    missing_df['date'] = missing_df['datetime'].dt.date
    missing_by_date = missing_df.groupby('date').size()
    for date, count in missing_by_date.items():
        print(f"  {date}: {count} missing observations")

# Market structure validation
returns_spy5p = np.log(spy5p_series / spy5p_series.shift(1)).dropna()
print(f"\nTrading hours coverage:")
hourly_coverage = returns_spy5p.groupby(returns_spy5p.index.hour).size().sort_index()
for hour, count in hourly_coverage.items():
    if hour == 17:
        # last bucket is only a half hour
        print(f"  17:00-17:30 - {count:,} observations")
    else:
        end_hour = (hour + 1) % 24
        print(f"  {hour:02d}:00-{end_hour:02d}:00 - {count:,} observations")


# ARMA Data Preparation
arma_data = returns_spy5p.dropna()
print(f"\nARMA input data: {len(arma_data):,} observations")

# Time gap validation
time_gaps = arma_data.index.to_series().diff()[1:]
normal_gap = pd.Timedelta(minutes=1)
large_gaps = time_gaps[time_gaps > normal_gap * 10]

print(f"Normal 1-minute intervals: {(time_gaps == normal_gap).sum():,}")
print(f"Large gaps (>10min): {len(large_gaps)}")
if len(large_gaps) > 0:
    print("Large gaps occur on:")
    for gap_time in large_gaps.index[:5]:
        print(f"  {gap_time.date()}")

# Price and Returns Visualization
print("")
print("Price and Returns Visualization")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

spy5p_prices = df['SPY5.P'].dropna()
ax1.plot(spy5p_prices.index, spy5p_prices.values, linewidth=0.8, color='blue', alpha=0.8)
ax1.set_title('SPY5.P Price Levels', fontsize=12)
ax1.set_ylabel('Price (EUR)')
ax1.grid(True, alpha=0.3)

ax2.plot(arma_data.index, arma_data.values, linewidth=0.5, color='red', alpha=0.7)
ax2.set_title('SPY5.P Log Returns', fontsize=12)
ax2.set_ylabel('Log Returns')
ax2.set_xlabel('Date')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.axhline(y=arma_data.mean(), color='green', linestyle='--', alpha=0.7, 
           label=f'Mean = {arma_data.mean():.2e}')
ax2.legend()

plt.tight_layout()
plt.show()

print(f"Price range: {spy5p_prices.min():.2f} to {spy5p_prices.max():.2f} EUR")
print(f"Returns range: {arma_data.min():.6f} to {arma_data.max():.6f}")
print(f"Returns mean: {arma_data.mean():.2e}")

# Stationarity Test
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(arma_data, autolag='AIC')
print(f"\nADF Test Statistic: {adf_result[0]:.6f}")
print(f"p-value: {adf_result[1]:.6f}")

# Basic Return Properties
print(f"\nMean return: {arma_data.mean():.8f}")
print(f"Std deviation: {arma_data.std():.6f}")
print(f"Skewness: {arma_data.skew():.4f}")
print(f"Kurtosis: {arma_data.kurtosis():.4f}")

# ACF/PACF Analysis
from statsmodels.tsa.stattools import acf, pacf

lags = 10  
acf_values = acf(arma_data, nlags=lags, fft=True)
pacf_values = pacf(arma_data, nlags=lags, method='ols')

print("\nFirst 10 ACF values:")
for i in range(11):
    print(f"  Lag {i}: {acf_values[i]:.6f}")

print("\nFirst 10 PACF values:")  
for i in range(11):
    print(f"  Lag {i}: {pacf_values[i]:.6f}")

n = len(arma_data)
bound = 1.96 / np.sqrt(n)
print(f"\n95% significance bound: ±{bound:.6f}")

# Visual ACF/PACF Analysis
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

plot_acf(arma_data, lags=10, ax=ax1, alpha=0.05, 
         title='SPY5.P Returns - ACF (Lags 1-10)')
ax1.set_xlim(1, 10)
ax1.set_ylim(-0.15, 0.15)
ax1.grid(True, alpha=0.3)

plot_pacf(arma_data, lags=10, ax=ax2, alpha=0.05,
          title='SPY5.P Returns - PACF (Lags 1-10)')
ax2.set_xlim(1, 10)
ax2.set_ylim(-0.15, 0.15)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Report note: Traditional ARMA patterns show exponential decay, but data shows sporadic specific-lag patterns

# Sample Split
print("")
print("Sample Split")
split_date = '2025-01-02'
in_sample = arma_data[arma_data.index < split_date]
out_sample = arma_data[arma_data.index >= split_date]

print(f"In-sample: {len(in_sample):,} obs ({in_sample.index[0]} to {in_sample.index[-1]})")
print(f"Out-sample: {len(out_sample):,} obs ({out_sample.index[0]} to {out_sample.index[-1]})")

# ARMA Model Estimation
print("")
print("ARMA Model Estimation")
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import itertools

# Suppress warnings
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
warnings.filterwarnings('ignore', category=ValueWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='No supported index is available')

# Assignment models: p,q ∈ {0,1,2}
# itertools.product creates all combinations: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
assignment_combinations = list(itertools.product([0, 1, 2], [0, 1, 2]))
assignment_combinations.remove((0, 0))  # Remove white noise

# Higher-order models based on ACF/PACF patterns (lags 4,5)
# Report note: Only selected combinations estimated for computational efficiency
higher_order_combinations = [(0, 5), (1, 5), (4, 4), (5, 5)]

all_combinations = assignment_combinations + higher_order_combinations
results_dict = {}

print(f"Estimating {len(all_combinations)} ARMA models on {len(in_sample):,} observations")
print("Assignment models: p,q ∈ {0,1,2} + selected higher-order models")
print("")

# Estimate models
for p, q in all_combinations:
    model_name = f"ARMA({p},{q})"
    
    try:
        model = ARIMA(in_sample, order=(p, 0, q))
        fitted_model = model.fit()
        
        # Ljung-Box diagnostic test
        ljung_box = acorr_ljungbox(fitted_model.resid, lags=10, return_df=True)
        ljung_box_pvalue = ljung_box['lb_pvalue'].iloc[-1]
        ljung_box_stat = ljung_box['lb_stat'].iloc[-1]
        
        # Out-of-sample evaluation
        if len(out_sample) > 0:
            try:
                forecast = fitted_model.forecast(steps=len(out_sample))
                out_sample_residuals = out_sample.values - forecast
                out_sample_mse = np.mean(out_sample_residuals**2)
                
                # Out-of-sample Ljung-Box
                if len(out_sample_residuals) > 10:
                    ljung_box_out = acorr_ljungbox(out_sample_residuals, lags=10, return_df=True)
                    ljung_box_out_pvalue = ljung_box_out['lb_pvalue'].iloc[-1]
                    ljung_box_out_stat = ljung_box_out['lb_stat'].iloc[-1]
                else:
                    ljung_box_out_pvalue = np.nan
                    ljung_box_out_stat = np.nan
            except:
                out_sample_mse = np.nan
                ljung_box_out_pvalue = np.nan
                ljung_box_out_stat = np.nan
        else:
            out_sample_mse = np.nan
            ljung_box_out_pvalue = np.nan
            ljung_box_out_stat = np.nan
        
        results_dict[model_name] = {
            'model': fitted_model,
            'converged': fitted_model.mle_retvals['converged'],
            # In-sample statistics
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'loglik_in': fitted_model.llf,
            'mse_in': fitted_model.mse,
            'ljung_box_stat_in': ljung_box_stat,
            'ljung_box_pvalue_in': ljung_box_pvalue,
            # Out-of-sample statistics  
            'mse_out': out_sample_mse,
            'ljung_box_stat_out': ljung_box_out_stat,
            'ljung_box_pvalue_out': ljung_box_out_pvalue,
            # Parameters
            'params': fitted_model.params,
            'std_errors': fitted_model.bse,
        }
        
        conv_status = "Conv" if fitted_model.mle_retvals['converged'] else "Fail"
        print(f"{model_name:12} | AIC: {fitted_model.aic:8.2f} | BIC: {fitted_model.bic:8.2f} | LB: {ljung_box_pvalue:.3f} | {conv_status}")
        
    except Exception:
        print(f"{model_name:12} | ERROR")

# Assignment Required Table Format
print("")
print("=" * 80)
print("ASSIGNMENT TABLE: Required Models (p,q ∈ {0,1,2})")
print("=" * 80)

# Assignment models only for main table
assignment_models = ['ARMA(0,1)', 'ARMA(0,2)', 'ARMA(1,0)', 'ARMA(1,1)', 
                    'ARMA(1,2)', 'ARMA(2,0)', 'ARMA(2,1)', 'ARMA(2,2)']
assignment_converged = {k: v for k, v in results_dict.items() 
                      if k in assignment_models and v.get('converged', False)}

if assignment_converged:
    model_names = sorted(assignment_converged.keys())
    
    # Parameter estimates section
    print("\nPARAMETER ESTIMATES:")
    print("Parameter    " + "".join([f"{name:>12}" for name in model_names]))
    print("-" * (13 + 12 * len(model_names)))
    
    # Find all unique parameters
    all_params = set()
    for result in assignment_converged.values():
        all_params.update(result['params'].index)
    
    # Sort parameters logically
    param_order = ['const'] + [f'ar.L{i}' for i in range(1, 6)] + [f'ma.L{i}' for i in range(1, 6)]
    ordered_params = [p for p in param_order if p in all_params]
    
    for param in ordered_params:
        param_row = f"{param:<12}"
        for model_name in model_names:
            if param in assignment_converged[model_name]['params']:
                est = assignment_converged[model_name]['params'][param]
                param_row += f"{est:12.6f}"
            else:
                param_row += f"{'--':>12}"
        print(param_row)
    
       # Standard errors section - FIXED INDENTATION
    print(f"\nSTANDARD ERRORS:")
    print("Parameter    " + "".join([f"{name:>12}" for name in model_names]))
    print("-" * (13 + 12 * len(model_names)))
    
    for param in ordered_params:
        param_row = f"{param}_SE"[:12]
        param_row = f"{param_row:<12}"
        for model_name in model_names:
            if param in assignment_converged[model_name]['std_errors']:
                se = assignment_converged[model_name]['std_errors'][param]
                # Use scientific notation for tiny values
                if abs(se) < 1e-5:
                    param_row += f"{se:12.2e}"  # Scientific notation
                else:
                    param_row += f"{se:12.6f}"
            else:
                param_row += f"{'--':>12}"
        print(param_row)

    # Statistics section  
    print("")
    print("=" * 80)
    print("MODEL STATISTICS")
    print("=" * 80)

    print(f"\nIN-SAMPLE STATISTICS (n = {len(in_sample):,}):")
    print("Statistic    " + "".join([f"{name:>12}" for name in model_names]))
    print("-" * (13 + 12 * len(model_names)))

    stats_in = ['aic', 'bic', 'loglik_in', 'mse_in', 'ljung_box_stat_in', 'ljung_box_pvalue_in']
    stat_labels = ['AIC', 'BIC', 'Log-Lik', 'MSE', 'LB-Stat', 'LB-pval']

    for stat, label in zip(stats_in, stat_labels):
        stat_row = f"{label:<12}"
        for model_name in model_names:
            value = assignment_converged[model_name][stat]
            if stat in ['aic', 'bic', 'loglik_in']:
                stat_row += f"{value:12.2f}"
            elif stat in ['mse_in']:  # Special handling for MSE
                stat_row += f"{value:12.2e}"  # Scientific notation
            else:
                stat_row += f"{value:12.6f}"
        print(stat_row)

    print(f"\nOUT-OF-SAMPLE STATISTICS (n = {len(out_sample):,}):")
    print("Statistic    " + "".join([f"{name:>12}" for name in model_names]))
    print("-" * (13 + 12 * len(model_names)))

    stats_out = ['mse_out', 'ljung_box_stat_out', 'ljung_box_pvalue_out']
    stat_labels_out = ['MSE', 'LB-Stat', 'LB-pval']

    for stat, label in zip(stats_out, stat_labels_out):
        stat_row = f"{label:<12}"
        for model_name in model_names:
            value = assignment_converged[model_name][stat]
            if np.isnan(value):
                stat_row += f"{'N/A':>12}"
            elif stat in ['mse_out']:  # Special handling for MSE  
                stat_row += f"{value:12.2e}"  # Scientific notation
            else:
                stat_row += f"{value:12.6f}"
        print(stat_row)



# Extended Analysis: All Models Comparison
print("")
print("=" * 80)
print("EXTENDED ANALYSIS: All Models Including Higher-Order")
print("=" * 80)

all_converged = {k: v for k, v in results_dict.items() if v.get('converged', False)}

if all_converged:
    # Top models by AIC
    aic_ranking = sorted(all_converged.items(), key=lambda x: x[1]['aic'])
    print("Top 5 models by AIC:")
    for i, (model_name, results) in enumerate(aic_ranking[:5]):
        print(f"  {i+1}. {model_name:12} AIC: {results['aic']:8.2f} | LB: {results['ljung_box_pvalue_in']:.3f}")

# Model Selection Summary
print("")
print("Model Selection Summary")
if assignment_converged and all_converged:
    best_assignment = min(assignment_converged.items(), key=lambda x: x[1]['aic'])
    best_overall = min(all_converged.items(), key=lambda x: x[1]['aic'])
    
    aic_improvement = best_assignment[1]['aic'] - best_overall[1]['aic']
    
    print(f"Best assignment model: {best_assignment[0]} (AIC: {best_assignment[1]['aic']:.2f})")
    print(f"Best overall model: {best_overall[0]} (AIC: {best_overall[1]['aic']:.2f})")
    print(f"AIC improvement: {aic_improvement:.0f} points")

# Microstructure Model Exploration
print("")
print("Microstructure Model Exploration")

# Direct lag regression
from sklearn.linear_model import LinearRegression

# Create lagged variables
lag_data = pd.DataFrame({
    'return': arma_data,
    'lag1': arma_data.shift(1),
    'lag4': arma_data.shift(4), 
    'lag5': arma_data.shift(5)
}).dropna()

# Split sample
split_idx = int(0.8 * len(lag_data))
train = lag_data[:split_idx]
test = lag_data[split_idx:]

# Fit sparse model
X_train = train[['lag1', 'lag4', 'lag5']]
y_train = train['return']
X_test = test[['lag1', 'lag4', 'lag5']]
y_test = test['return']

# Simple OLS regression
reg = LinearRegression().fit(X_train, y_train)

# Compare with best ARMA
pred_sparse = reg.predict(X_test)
mse_sparse = np.mean((y_test - pred_sparse)**2)

print(f"Sparse model (lags 1,4,5):")
print(f"  Coefficients: {reg.coef_}")
print(f"  Out-sample MSE: {mse_sparse:.2e}")
print(f"  R-squared: {reg.score(X_test, y_test):.6f}")

if all_converged:
    best_arma_mse = min([v['mse_out'] for v in all_converged.values() if not np.isnan(v['mse_out'])])
    print(f"\nBest ARMA out-sample MSE: {best_arma_mse:.2e}")
    print(f"MSE ratio (sparse/best ARMA): {mse_sparse / best_arma_mse:.3f}")


###########################################################
### 3.4
print(" === Question 3.3 ===")

###########################################################

### 3.4
print(" === Question 3.4 ===")


# GARCH Model Estimation