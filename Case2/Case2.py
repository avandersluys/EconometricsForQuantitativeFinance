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
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import itertools
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

# Missing Data Handling for ARMA (Assignment Option 4)
# Option 4: Drop r=max(p,q) periods after each missing observation
# This maintains regular time intervals required for ARMA likelihood calculation

max_lag = 2  # For assignment models p,q ∈ {0,1,2}, max(p,q) = 2

# Find missing data positions
missing_mask = spy5p_series.isnull()
missing_positions = missing_mask[missing_mask].index

# Create exclusion mask: drop max_lag periods after each missing observation
exclude_mask = missing_mask.copy()
for missing_time in missing_positions:
    for i in range(1, max_lag + 1):
        next_time = missing_time + pd.Timedelta(minutes=i)
        if next_time in spy5p_series.index:
            exclude_mask[next_time] = True

# Apply exclusion and compute returns
spy5p_clean = spy5p_series[~exclude_mask]
returns_clean = np.log(spy5p_clean / spy5p_clean.shift(1)).dropna()

# ARMA Data Preparation  
arma_data = returns_clean

print(f"Missing data treatment: Option 4 (drop max(p,q) periods after missing)")
print(f"ARMA input data: {len(arma_data):,} observations")

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
adf_result = adfuller(arma_data, autolag='AIC')
print(f"\nADF Test Statistic: {adf_result[0]:.6f}")
print(f"p-value: {adf_result[1]:.6f}")

# Basic Return Properties
print(f"\nMean return: {arma_data.mean():.8f}")
print(f"Std deviation: {arma_data.std():.6f}")
print(f"Skewness: {arma_data.skew():.4f}")
print(f"Kurtosis: {arma_data.kurtosis():.4f}")

# ACF/PACF Analysis

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


############################################################## 3.3a (simplified)
print("\n=== QUESTION 3.3a — VAR(5) on returns (simplified) ===")

from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox

# 1) Two parallel listings of the same ETF
symbols_var = ['SPY5.P', 'SPY5z.CHIX']
prices = df[symbols_var]

# 2) Minute log-returns (I(0))
R = np.log(prices).diff().dropna()

# 3) Split (uses split_date defined above)
R_in  = R[R.index < split_date]
R_out = R[R.index >= split_date]

print(f"Series: {symbols_var}")
print(f"In-sample:  {R_in.index[0]} → {R_in.index[-1]}  (n={len(R_in):,})")
print(f"Out-sample: {R_out.index[0]} → {R_out.index[-1]} (n={len(R_out):,})")

# 4) Fit VAR(5) with constant
p = 5
res = VAR(R_in).fit(p, trend='c')

print("\nVAR(5) — key fit stats")
print(f"  LLF: {res.llf:.2f}")
print(f"  AIC: {res.aic:.6f}   BIC: {res.bic:.6f}   HQIC: {res.hqic:.6f}")

# 5) Compact table: coefficients + standard errors (+ t, p)
#    (one tidy table; easy to copy to the report)
params = res.params.copy()      # rows = equations, cols = ['const','L1.SPY5.P',...]
bse    = res.stderr.copy()
tvals  = res.tvalues.copy()
pvals  = res.pvalues.copy()

tidy = []
for eq in params.index:             # equation (dependent var)
    for term in params.columns:     # parameter name
        tidy.append({
            'equation': eq,
            'parameter': term,
            'beta': params.loc[eq, term],
            'se':   bse.loc[eq, term],
            't':    tvals.loc[eq, term],
            'p':    pvals.loc[eq, term]
        })
coef_table = (pd.DataFrame(tidy)
              .loc[:, ['equation','parameter','beta','se','t','p']]
              .sort_values(['equation','parameter']))

# Nice rounding for printing
def fmt(s, nd=6): return s.map(lambda x: f"{x:.{nd}f}")
show = coef_table.copy()
show['beta'] = fmt(show['beta'], 6)
show['se']   = fmt(show['se'],   6)
show['t']    = fmt(show['t'],    2)
show['p']    = show['p'].map(lambda x: f"{x:.3g}")

print("\nParameter estimates (β) with standard errors (se)")
print(show.to_string(index=False))

# 6) In-sample residual diagnostics
resid = res.resid
mse_in = (resid**2).mean()

print("\nIn-sample MSE by equation")
for s in resid.columns:
    print(f"  {s}: {mse_in[s]:.4e}")

print("\nResidual Ljung–Box Q (lag 10)")
for s in resid.columns:
    lb = acorr_ljungbox(resid[s], lags=[10], return_df=True)
    print(f"  {s}: Q={lb['lb_stat'].iloc[-1]:.2f}, p={lb['lb_pvalue'].iloc[-1]:.3f}")

print("\nResidual contemporaneous correlation")
print(resid.corr())

# 7) Optional: quick OOS MSE with static multi-step forecast (no refit/rolling)
if len(R_out) > 0:
    try:
        steps = len(R_out)
        fcst = res.forecast(y=R_in.values[-res.k_ar:], steps=steps)
        fcst_df = pd.DataFrame(fcst, index=R_out.index, columns=R_out.columns)
        mse_out = ((R_out - fcst_df)**2).mean()
        print("\nOut-of-sample MSE (static multi-step forecast)")
        for s in mse_out.index:
            print(f"  {s}: {mse_out[s]:.4e}")
    except Exception as e:
        print("OOS MSE not computed:", e)

# 8) One-line lag order hint (AIC/BIC/HQIC) to discuss whether p=5 is reasonable
try:
    order_sel = VAR(R_in).select_order(maxlags=10)
    print("\nLag order suggestion (IC minima)")
    print(order_sel.summary())
except Exception:
    pass


print("\n=== QUESTION 3.3b — Significance & Cross-effects ===")

# --- Collect coefficients, SEs, t, p for a tidy table
params = var_res.params.copy()        # DataFrame: rows = equations, cols = coeffs (const, L1.x, ... or x.L1)
bse    = var_res.stderr.copy()
tvals  = var_res.tvalues.copy()
pvals  = var_res.pvalues.copy()

# Build long-format table with nice labels and significance stars
def stars(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

# --- Robust coefficient label parser (supports both label styles)
import re
def parse_coef_label(coef: str, eq_name: str):
    """
    Returns (lag:int, src:str, kind:str) for a VAR coefficient label.
    Supports both 'Lk.SYMBOL' and 'SYMBOL.Lk' styles. 'const' -> (0, '-', 'intercept')
    """
    if coef == 'const':
        return 0, '-', 'intercept'

    # Style A: 'L3.SPY5.P'
    m = re.match(r'^L(\d+)\.(.+)$', coef)
    if m:
        lag = int(m.group(1))
        src = m.group(2)
        kind = 'own-lag' if src == eq_name else 'cross-lag'
        return lag, src, kind

    # Style B: 'SPY5.P.L3'
    m = re.match(r'^(.+)\.L(\d+)$', coef)
    if m:
        src = m.group(1)
        lag = int(m.group(2))
        kind = 'own-lag' if src == eq_name else 'cross-lag'
        return lag, src, kind

    # Fallback: unknown formatting
    return np.nan, coef, 'other'

long_rows = []
for eq in params.index:                 # equation/dependent variable names
    for coef in params.columns:         # 'const', 'L1.SPY5.P' OR 'SPY5.P.L1', etc.
        b = params.loc[eq, coef]
        se = bse.loc[eq, coef]
        t = tvals.loc[eq, coef]
        p = pvals.loc[eq, coef]
        lag, src, kind = parse_coef_label(coef, eq)
        long_rows.append({
            'equation': eq,
            'term': coef,
            'source_var': src,
            'lag': lag,
            'type': kind,
            'beta': b,
            'stderr': se,
            't': t,
            'p': p,
            'sig': stars(p)
        })

coef_df = (pd.DataFrame(long_rows)
           .sort_values(['equation','type','lag','source_var'], na_position='last'))

# Pretty print per equation
for dep in coef_df['equation'].unique():
    sub = coef_df[coef_df['equation']==dep].copy()
    print(f"\n--- Coefficient table for equation: {dep} ---")
    # columns to show
    show = sub[['term','type','source_var','lag','beta','stderr','t','p','sig']].copy()
    # nice rounding
    show['beta']   = show['beta'].map(lambda x: f"{x: .6f}")
    show['stderr'] = show['stderr'].map(lambda x: f"{x: .6f}")
    show['t']      = show['t'].map(lambda x: f"{x: .2f}")
    show['p']      = show['p'].map(lambda x: f"{x: .3g}")
    # add significance stars right next to beta
    show['beta±']  = show['beta'] + show['sig']
    show = show.drop(columns=['beta','sig'])
    show = show.rename(columns={'term':'parameter','source_var':'from','lag':'L'})
    # order and print
    show = show[['parameter','type','from','L','beta±','stderr','t','p']]
    print(show.to_string(index=False))

# --- Summarise cross-effects: how many and which lags are significant?
alpha = 0.05
cross_sig = coef_df[(coef_df['type']=='cross-lag') & (coef_df['p']<alpha)].copy()
if cross_sig.empty:
    print("\nCross-effects: None significant at 5% level.")
else:
    print("\nCross-effects (p<0.05):")
    # aggregate by direction
    summary = (cross_sig
               .assign(direction = cross_sig['source_var'] + " → " + cross_sig['equation'])
               .groupby(['direction','lag'])
               .apply(lambda g: f"β≈{g['beta'].mean():.4g} (avg)")
               .reset_index(name='coef_info'))
    for direction in summary['direction'].unique():
        chunk = summary[summary['direction']==direction]
        # guard against any NaN lags
        lags  = ", ".join([f"L{int(L)} {info}" for L, info in zip(chunk['lag'], chunk['coef_info']) if pd.notnull(L)])
        print(f"  {direction}: {lags}")

# --- Who drives whom? Granger causality tests (per slides)
#     H0: past of X does NOT help predict Y (reject => X Granger-causes Y)
#     Use same lag length p as in the fitted VAR
p_gr = var_res.k_ar
series = list(R_in.columns)

def granger(direction_y, direction_x):
    # Does X -> Y ?
    res = var_res.test_causality(caused=direction_y, causing=[direction_x], kind='f')
    return {'caused': direction_y, 'causing': direction_x,
            'stat': float(res.statistic), 'pval': float(res.pvalue), 'df': tuple(res.df)}

print("\nGranger causality (F-tests, lag length = %d):" % p_gr)
gc_rows = []
for y in series:
    for x in series:
        if x == y: 
            continue
        try:
            gc_rows.append(granger(y, x))
        except Exception:
            gc_rows.append({'caused': y, 'causing': x, 'stat': np.nan, 'pval': np.nan, 'df': None})

gc_df = pd.DataFrame(gc_rows)
for _, r in gc_df.iterrows():
    verdict = "YES (reject H0)" if (r['pval'] < 0.05) else "no"
    print(f"  {r['causing']} ⇒ {r['caused']}: F={r['stat']:.2f}, p={r['pval']:.3g}  → Granger-cause: {verdict}")

# Quick directional summary
drivers = (gc_df
           .assign(drives=lambda d: np.where(d['pval']<0.05, 1, 0))
           .groupby('causing')['drives'].sum())
if len(drivers)>0:
    top_driver = drivers.sort_values(ascending=False).index[0]
    print(f"\nDirectionality summary: strongest Granger ‘driver’ (count of significant targets) → {top_driver}")


print("\n=== QUESTION 3.3c — Residual diagnostics & zero-correlation check ===")

# Grab residuals from your fitted VAR (already computed above as `resid`)
resid = var_res.resid.copy()
eq_names = list(resid.columns)
nT = len(resid)

# ---------- 1) Normality-ish visuals (hist + QQ) ----------
import scipy.stats as st

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for j, s in enumerate(eq_names):
    axh = axes[j, 0]
    axq = axes[j, 1]
    rj = resid[s].values

    # histogram + normal fit line
    axh.hist(rj, bins=60, density=True, alpha=0.6)
    mu, sd = np.mean(rj), np.std(rj)
    xs = np.linspace(mu-4*sd, mu+4*sd, 201)
    axh.plot(xs, st.norm.pdf(xs, mu, sd), lw=1.2)
    axh.set_title(f"{s} residuals: hist + N({mu:.2e},{sd:.2e})")
    axh.grid(True, alpha=0.3)

    # QQ-plot vs normal
    osm, osr = st.probplot(rj, dist="norm")[:2]  # returns (theoretical, ordered), (slope, intercept, r)
    st.probplot(rj, dist="norm", plot=axq)
    axq.set_title(f"{s} residuals: QQ-normal")
    axq.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------- 2) Per-equation whiteness already checked via Ljung–Box above ----------
# (kept for context in the report)

# ---------- 3) Contemporaneous correlation: test ρ=0 ----------
r = float(resid.corr().iloc[0,1])
t_stat = r * np.sqrt((nT - 2) / (1 - r**2))
p_val = 2 * (1 - st.t.cdf(abs(t_stat), df=nT-2))

# 95% CI using Fisher z-transform
z = np.arctanh(np.clip(r, -0.999999, 0.999999))
se_z = 1 / np.sqrt(nT - 3)
z_lo, z_hi = z - 1.96*se_z, z + 1.96*se_z
ci_lo, ci_hi = np.tanh([z_lo, z_hi])

print("\nZero-correlation (lag 0) test for residuals:")
print(f"  Corr({eq_names[0]}, {eq_names[1]}) = {r:.6f}")
print(f"  t = {t_stat:.2f},  p = {p_val:.3g}  (H0: rho=0)")
print(f"  95% CI for rho: [{ci_lo:.6f}, {ci_hi:.6f}]")

# ---------- 4) Scatter of residuals (visual quick check on Σ_u off-diagonal) ----------
plt.figure(figsize=(6,5))
plt.scatter(resid[eq_names[0]], resid[eq_names[1]], s=3, alpha=0.2)
m, b = np.polyfit(resid[eq_names[0]], resid[eq_names[1]], 1)
xs = np.linspace(resid[eq_names[0]].min(), resid[eq_names[0]].max(), 100)
plt.plot(xs, m*xs + b, lw=1)
plt.title(f"Residual scatter: {eq_names[0]} vs {eq_names[1]} (r={r:.3f})")
plt.xlabel(eq_names[0]); plt.ylabel(eq_names[1])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---------- 5) Cross-correlation function (CCF) with 95% bands ----------
max_lag = 10
x = resid[eq_names[0]].values
y = resid[eq_names[1]].values

def ccf(x, y, L):
    x = x - x.mean(); y = y - y.mean()
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    out = []
    for k in range(-L, L+1):
        if k >= 0:
            num = np.sum(x[k:] * y[:len(y)-k])
        else:
            num = np.sum(x[:len(x)+k] * y[-k:])
        out.append(num / denom)
    return np.array(out)

cc = ccf(x, y, max_lag)
lags = np.arange(-max_lag, max_lag+1)
# Approximate 95% band for zero CCF under independence: ±1.96/sqrt(T)
band = 1.96 / np.sqrt(nT)

plt.figure(figsize=(10,4))
plt.stem(lags, cc, use_line_collection=True)
plt.axhline(band, ls='--', lw=1); plt.axhline(-band, ls='--', lw=1)
plt.axhline(0, color='k', lw=0.8)
plt.title(f"Residual cross-correlation (±95% ~ {band:.3f})")
plt.xlabel("Lag (y leads >0)"); plt.ylabel("CCF")
plt.tight_layout()
plt.show()

# ---------- 6) Quick multivariate summary for report ----------
print("\nResiduals summary for report:")
print(f"  • Per-equation Ljung–Box(10): already above (whiteness).")
print(f"  • Corr matrix off-diagonal: {r:.3f} (p={p_val:.3g}); 95% CI [{ci_lo:.3f}, {ci_hi:.3f}].")
print(f"  • CCF lags within ±{band:.3f}? Inspect the stem plot; spikes outside the band indicate remaining cross-dependence.")




"""
###########################################################
### 3.3
print(" === Question 3.3 ===")

###########################################################

### 3.4
print(" === Question 3.4 ===")


# Extract SPY5.P series (already defined but ensure it's clean)
spy5p_series = df['SPY5.P'].dropna()

# Daily Data Extraction for GARCH Analysis
daily_prices = spy5p_series.groupby(spy5p_series.index.date).last()
daily_prices.index = pd.to_datetime(daily_prices.index)

# Daily percentage log returns
daily_returns = np.log(daily_prices / daily_prices.shift(1)).dropna() * 100
print(f"Original minute observations: {len(spy5p_series):,}")
print(f"Daily closing prices: {len(daily_prices):,}")
print(f"Daily returns: {len(daily_returns):,}")
print(f"Date range: {daily_returns.index[0].date()} to {daily_returns.index[-1].date()}")
print("")

# Basic Daily Return Characteristics
print("Daily Return Characteristics (Percentage Points)")
print(f"Mean: {daily_returns.mean():.4f}%")
print(f"Std Dev: {daily_returns.std():.4f}%")
print(f"Min: {daily_returns.min():.4f}% (worst daily loss)")  
print(f"Max: {daily_returns.max():.4f}% (best daily gain)")
print(f"Skewness: {daily_returns.skew():.4f}")
print(f"Excess Kurtosis: {daily_returns.kurtosis():.4f}")
print("")

# Simple volatility bands plot
plt.figure(figsize=(12, 6))

# Plot returns
plt.plot(daily_returns.index, daily_returns.values, linewidth=0.8, color='blue', alpha=0.7)

# Add ±2σ bands
std_dev = daily_returns.std()
plt.axhline(y=2*std_dev, color='red', linestyle='--', alpha=0.7, label='+2σ')
plt.axhline(y=-2*std_dev, color='red', linestyle='--', alpha=0.7, label='-2σ')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

plt.title('Daily Percentage Log Returns with ±2σ Bands')
plt.xlabel('Date')
plt.ylabel('Returns (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# GARCH imports
from scipy import stats
from statsmodels.stats.diagnostic import het_arch

"""