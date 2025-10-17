#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case2.py

Purpose:
    Main analysis file for Assignment 2 - Time Series Analysis
    Econometrics for Quantitative Finance - Group 9
    
    Performs ARMA modeling, VAR analysis, and GARCH estimation on ETF data

Version:
    1.0     Final submission version

Date:
    2025/10/17

Author:
    Group 9 - Siddharth Kukreja & Alexander van der Sluys
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings('ignore')

###########################################################
### DATA LOADING AND SETUP
###########################################################

# Load high-frequency ETF data from GitHub repository
github_url = "https://raw.githubusercontent.com/avandersluys/EconometricsForQuantitativeFinance/7f88e429a23a03593352b07a64f9d882017f5246/Case2/sp_9.csv.gz"
df = pd.read_csv(github_url, index_col=0, parse_dates=True)

###########################################################
### INTRODUCTION: DATA OVERVIEW
###########################################################

print("=" * 60)
print("ASSIGNMENT 2: TIME SERIES ANALYSIS - GROUP 9")
print("=" * 60)

# Display basic dataset information
print("\nDataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Calculate data completeness for each series
print("\nData Completeness Analysis:")
for col in df.columns:
    pct_complete = 100 * df[col].notna().mean()
    print(f"{col}: {pct_complete:.2f}% complete")

# Analyze missing data patterns for SPY5.P (primary series)
print(f"\nMissing Data Pattern Analysis:")
total_missing = df['SPY5.P'].isna().sum()
print(f"Total missing observations: {total_missing}")

if total_missing > 0:
    # Analyze missing data by hour to identify patterns
    missing_hours = df[df['SPY5.P'].isna()].index.hour.value_counts().sort_index()
    print(f"Missing observations by hour: {dict(missing_hours)}")
    
    # Show sample missing timestamps
    missing_times = df[df['SPY5.P'].isna()].index[:10]
    print(f"Sample missing timestamps: {list(missing_times)}")
    
    # Identify unique missing dates (potential holidays/market closures)
    missing_dates = df[df['SPY5.P'].isna()].index.date
    unique_missing_dates = sorted(set(missing_dates))
    print(f"Missing dates: {unique_missing_dates[:5]}...")

###########################################################
### SECTION 3.2: ARMA MODELING
###########################################################

print("\n" + "=" * 60)
print("SECTION 3.2: ARMA MODEL ANALYSIS")
print("=" * 60)

# Extract SPY5.P series and calculate percentage log returns
spy5p_series = df['SPY5.P'].dropna()
returns = np.log(spy5p_series / spy5p_series.shift(1)) * 100
print(f"Raw return series shape: {returns.shape}")

# Identify extreme return observations
returns_clean = returns.copy()
max_return = returns_clean.max()
min_return = returns_clean.min()
max_date = returns_clean.idxmax()
min_date = returns_clean.idxmin()
print(f"\nExtreme Return Analysis:")
print(f"Maximum: {max_return:.4f}% at {max_date}")
print(f"Minimum: {min_return:.4f}% at {min_date}")

# Data cleaning: Remove first 3 observations of each trading day to eliminate overnight effects
print("\nData Cleaning: Removing Overnight Effects")
daily_groups = returns_clean.groupby(returns_clean.index.date)
returns_cleaned = pd.concat([group.iloc[3:] for date, group in daily_groups if len(group) > 3])

print(f"Original observations: {len(returns):,}")
print(f"After cleaning: {len(returns_cleaned):,}")
print(f"Cleaned range: {returns_cleaned.min():.4f}% to {returns_cleaned.max():.4f}%")

# Post-cleaning extreme value analysis
max_return_clean = returns_cleaned.max()
min_return_clean = returns_cleaned.min()
max_date_clean = returns_cleaned.idxmax()
min_date_clean = returns_cleaned.idxmin()
print(f"\nPost-Cleaning Extreme Values:")
print(f"Maximum: {max_return_clean:.4f}% at {max_date_clean}")
print(f"Minimum: {min_return_clean:.4f}% at {min_date_clean}")

# Examine context around extreme return events
print(f"\nContext Around Maximum Return ({max_date_clean}):")
print(returns_cleaned.loc[max_date_clean - pd.Timedelta(minutes=5):max_date_clean + pd.Timedelta(minutes=5)])

print(f"\nContext Around Minimum Return ({min_date_clean}):")
print(returns_cleaned.loc[min_date_clean - pd.Timedelta(minutes=5):min_date_clean + pd.Timedelta(minutes=5)])

# Stationarity testing using Augmented Dickey-Fuller test
print("\nStationarity Analysis:")
adf_result = adfuller(returns_cleaned, autolag='AIC')
print(f"ADF Test Statistic: {adf_result[0]:.6f}")
print(f"p-value: {adf_result[1]:.2e}")

# ACF/PACF Analysis for model identification
print("\nAutocorrelation Analysis:")
clean_returns = returns_cleaned
lags = 120  # Analyze up to 2 hours of minute-by-minute data
acf_values = acf(clean_returns, nlags=lags, fft=True)
pacf_values = pacf(clean_returns, nlags=lags, method='ols')

# Calculate significance bounds (95% confidence level)
n = len(clean_returns)
bound = 1.96 / np.sqrt(n)
print(f"95% confidence bounds: ±{bound:.6f}")

# Identify significant autocorrelation lags
sig_acf = [(i, acf_values[i]) for i in range(1, lags+1) if abs(acf_values[i]) > bound]
sig_pacf = [(i, pacf_values[i]) for i in range(1, lags+1) if abs(pacf_values[i]) > bound]

print(f"Significant ACF lags (first 20): {[lag for lag, val in sig_acf[:20]]}")
print(f"Significant PACF lags (first 20): {[lag for lag, val in sig_pacf[:20]]}")
print(f"Total significant lags - ACF: {len(sig_acf)}, PACF: {len(sig_pacf)}")

# Report maximum significant lags
max_sig_acf = max([lag for lag, val in sig_acf]) if sig_acf else 0
max_sig_pacf = max([lag for lag, val in sig_pacf]) if sig_pacf else 0
print(f"Last significant lag - ACF: {max_sig_acf}, PACF: {max_sig_pacf}")

# Generate ACF/PACF plots for visual analysis
print("\nGenerating ACF/PACF plots...")

# Extended view (30 lags)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
plot_acf(clean_returns, lags=30, ax=ax1, alpha=0.05, title='ACF - SPY5.P Returns (30-minute window)')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 0.25)  
ax1.set_xlabel('Lag (minutes)')

plot_pacf(clean_returns, lags=30, ax=ax2, alpha=0.05, title='PACF - SPY5.P Returns (30-minute window)')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 0.25)
ax2.set_xlabel('Lag (minutes)')
plt.tight_layout()
plt.show()

# Detailed view (10 lags)
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(clean_returns, lags=10, ax=ax1, alpha=0.05, title='ACF - SPY5.P Returns (Detailed View)')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.02, 0.25)  

plot_pacf(clean_returns, lags=10, ax=ax2, alpha=0.05, title='PACF - SPY5.P Returns (Detailed View)')  
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.06, 0.25)
plt.tight_layout()
plt.show()

# Data splitting for in-sample and out-of-sample evaluation
print("\nData Splitting for Model Evaluation:")
returns_2024 = returns_cleaned[returns_cleaned.index < '2025-01-01']
returns_2025 = returns_cleaned[returns_cleaned.index >= '2025-01-01']
print(f"In-sample (2024): {len(returns_2024):,} observations")
print(f"Out-of-sample (2025): {len(returns_2025):,} observations")

###########################################################
### ARMA MODEL ESTIMATION AND COMPARISON
###########################################################

print("\n" + "=" * 60)
print("ARMA MODEL ESTIMATION AND COMPARISON")
print("=" * 60)

# Define ARMA model specifications to estimate
models_to_estimate = [(p, q) for p in range(3) for q in range(3)]
results = {}

# Results table header
print(f"\n{'Model':<10} | {'Conv':<5} | {'AIC_In':<10} | {'AIC_Out':<10} | {'MSE_In':<10} | {'MSE_Out':<10} | {'LB_In':<8} | {'LB_Out':<8}")
print("-" * 90)

# Estimate all ARMA model specifications
for p, q in models_to_estimate:
    model_name = f"ARMA({p},{q})"
    
    # Estimate parameters using in-sample data (2024) with robust standard errors
    model = ARIMA(returns_2024, order=(p, 0, q))
    fitted = model.fit(cov_type='robust')
    
    # Extract model parameters and convergence status
    params = fitted.params.to_dict()
    std_errors = fitted.bse.to_dict()
    converged = fitted.mle_retvals['converged'] if fitted.mle_retvals else False
    
    # In-sample model diagnostics
    residuals_in = fitted.resid
    mse_in = np.mean(residuals_in**2)
    lb_test_in = acorr_ljungbox(residuals_in.dropna(), lags=10, return_df=True)
    lb_stat_in = lb_test_in['lb_stat'].iloc[-1]
    lb_pval_in = lb_test_in['lb_pvalue'].iloc[-1]
    
    # Out-of-sample evaluation using fixed parameters from 2024
    forecast_model = fitted.apply(returns_2025, refit=False)
    residuals_out = forecast_model.resid
    mse_out = np.mean(residuals_out**2)
    
    # Calculate out-of-sample log-likelihood and information criteria
    sigma2_est = np.var(residuals_in)
    n_out = len(residuals_out)
    loglik_out = -0.5 * n_out * np.log(2 * np.pi * sigma2_est) - 0.5 * np.sum(residuals_out**2) / sigma2_est
    
    k = len(fitted.params)
    aic_out = 2 * k - 2 * loglik_out
    bic_out = k * np.log(n_out) - 2 * loglik_out
    
    # Out-of-sample Ljung-Box test
    lb_test_out = acorr_ljungbox(residuals_out.dropna(), lags=10, return_df=True)
    lb_stat_out = lb_test_out['lb_stat'].iloc[-1]
    lb_pval_out = lb_test_out['lb_pvalue'].iloc[-1]
    
    # Store comprehensive results
    results[model_name] = {
        'params': params,
        'std_errors': std_errors,
        'converged': converged,
        'aic_in': fitted.aic,
        'bic_in': fitted.bic, 
        'loglik_in': fitted.llf,
        'aic_out': aic_out,
        'bic_out': bic_out,
        'loglik_out': loglik_out,
        'mse_in': mse_in,
        'mse_out': mse_out,
        'lb_stat_in': lb_stat_in,
        'lb_pval_in': lb_pval_in,
        'lb_stat_out': lb_stat_out,
        'lb_pval_out': lb_pval_out,
        'fitted_model': fitted
    }
    
    # Display results for current model
    conv_status = "YES" if converged else "NO"
    print(f"{model_name:<10} | {conv_status:<5} | {fitted.aic:<10.0f} | {aic_out:<10.0f} | {mse_in:<10.6f} | {mse_out:<10.6f} | {lb_stat_in:<8.2f} | {lb_stat_out:<8.2f}")

# Identify best performing models
aic_best = min(results.keys(), key=lambda k: results[k]['aic_in'])
bic_best = min(results.keys(), key=lambda k: results[k]['bic_in'])
print(f"\nModel Selection Results:")
print(f"Best by AIC (In-Sample): {aic_best}")
print(f"Best by BIC (In-Sample): {bic_best}")

###########################################################
### PARAMETER ESTIMATES TABLE
###########################################################

print("\n" + "=" * 120)
print("PARAMETER ESTIMATES WITH ROBUST STANDARD ERRORS")
print("Estimated on 2024 In-Sample Data")
print("=" * 120)

# Define model presentation order
model_order = ['ARMA(0,0)', 'ARMA(0,1)', 'ARMA(0,2)', 'ARMA(1,0)', 
               'ARMA(1,1)', 'ARMA(1,2)', 'ARMA(2,0)', 'ARMA(2,1)', 'ARMA(2,2)']

# Collect all parameter names and establish order
all_params = set()
for result in results.values():
    all_params.update(result['params'].keys())

param_order = ['const', 'ar.L1', 'ar.L2', 'ma.L1', 'ma.L2']
ordered_params = [p for p in param_order if p in all_params]

# Generate parameter estimates table header
print(f"{'Parameter':<15}", end="")
for model in model_order:
    print(f" | {model:<12}", end="")
print()
print("-" * 120)

# Display parameter estimates with standard errors
for param in ordered_params:
    # Map parameter names to display labels
    param_display = {
        'const': 'Constant', 
        'ar.L1': 'AR(1)', 
        'ar.L2': 'AR(2)',
        'ma.L1': 'MA(1)', 
        'ma.L2': 'MA(2)'
    }.get(param, param)
    
    # Parameter estimates row
    print(f"{param_display:<15}", end="")
    for model in model_order:
        if param in results[model]['params']:
            val = results[model]['params'][param]
            print(f" | {val:>10.6f}  ", end="")
        else:
            print(f" | {'--':>10}    ", end="")
    print()
    
    # Standard errors row
    print(f"{'(Std. Error)':<15}", end="")
    for model in model_order:
        if param in results[model]['std_errors']:
            se = results[model]['std_errors'][param]
            print(f" | ({se:>8.6f}) ", end="")
        else:
            print(f" | {'--':>10}    ", end="")
    print()
    print()

###########################################################
### MODEL DIAGNOSTICS TABLE
###########################################################

print("=" * 120)
print("COMPREHENSIVE MODEL DIAGNOSTICS")
print("Using 2024 Parameters Applied to Both In-Sample and Out-of-Sample Periods")
print("=" * 120)

# Define diagnostic measures to display
diagnostics = [
    ('Convergence', 'converged', 's'),
    ('Log-Likelihood (In)', 'loglik_in', '.2f'),
    ('Log-Likelihood (Out)', 'loglik_out', '.2f'),
    ('AIC (In)', 'aic_in', '.0f'),
    ('AIC (Out)', 'aic_out', '.0f'),
    ('BIC (In)', 'bic_in', '.0f'),
    ('BIC (Out)', 'bic_out', '.0f'),
    ('MSE (In-Sample)', 'mse_in', '.8f'),
    ('MSE (Out-Sample)', 'mse_out', '.8f'),
    ('Ljung-Box Stat (In)', 'lb_stat_in', '.2f'),
    ('Ljung-Box Stat (Out)', 'lb_stat_out', '.2f'),
    ('Ljung-Box p-val (In)', 'lb_pval_in', '.4f'),
    ('Ljung-Box p-val (Out)', 'lb_pval_out', '.4f')
]

# Generate comprehensive diagnostics table
for diag_name, key, fmt in diagnostics:
    print(f"{diag_name:<22}", end="")
    for model in model_order:
        val = results[model][key]
        if fmt == 's':
            status = "YES" if val else "NO"
            print(f" | {status:>10}    ", end="")
        elif pd.isna(val):
            print(f" | {'--':>10}    ", end="")
        else:
            print(f" | {val:>10{fmt}}  ", end="")
    print()

###########################################################
### SECTION 3.3: VECTOR AUTOREGRESSION (VAR) ANALYSIS
###########################################################

print("\n" + "=" * 60)
print("SECTION 3.3: VECTOR AUTOREGRESSION ANALYSIS")
print("=" * 60)

# Import additional libraries for VAR and cointegration analysis
from itertools import combinations
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from scipy import stats

# VAR analysis configuration parameters
cols = ["SPX5.L", "SPY5z.CHIX", "SPY5.P"]  # Three ETF series for multivariate analysis
session_start, session_end = "09:00", "17:30"  # Trading session hours
p_assign = 5  # VAR lag order as specified in assignment
p_min, p_max = 1, 15  # Range for lag order selection
nlags = 10  # Diagnostic test lag parameter
split_date = pd.Timestamp("2025-01-01")  # Train/test split date
COINTEG_FREQ = "10min"  # Frequency for cointegration analysis (thinning)

# Load and prepare multivariate price data
df = pd.read_csv(github_url, parse_dates=["DateTime"]).set_index("DateTime").sort_index()
prices = df[cols].astype(float).where(lambda x: x > 0).dropna(how="any")
prices = prices.between_time(session_start, session_end)

def intraday_logret(g):
    """Calculate intraday log returns within each trading day"""
    r = np.log(g).diff()
    r.iloc[0] = np.nan  # Set first return of each day to NaN
    return r

# Calculate minute-by-minute log returns by trading day
rets = prices.groupby(prices.index.date, group_keys=False).apply(intraday_logret)
rets = rets.replace([np.inf, -np.inf], np.nan).dropna(how="any").astype(float)

# Remove first p_max observations of each day to eliminate spillover effects
pos_in_day = rets.groupby(rets.index.date).cumcount()
p_max_block = p_max
rets_fs = rets.mask(pos_in_day < p_max_block).dropna(how="any").copy()

# Split data for in-sample and out-of-sample analysis
rets_train = rets_fs.loc[rets_fs.index < split_date]
rets_test = rets_fs.loc[rets_fs.index >= split_date]

###########################################################
### VAR LAG ORDER SELECTION
###########################################################

def ic_select(data, p_min=1, p_max=15, trend="c"):
    """Select optimal VAR lag order using information criteria"""
    m = VAR(data)
    out = []
    for p in range(p_min, p_max + 1):
        res = m.fit(p, trend=trend)
        out.append({"p": p, "AIC": res.aic, "BIC": res.bic, "HQIC": res.hqic, "nobs": int(res.nobs)})
    df_ic = pd.DataFrame(out).sort_values("p")
    p_bic = int(df_ic.loc[df_ic["BIC"].idxmin(), "p"])
    return df_ic, p_bic

# Lag order selection using training data (2024)
print("\n=== VAR LAG ORDER SELECTION ===")
ic_train, p_train = ic_select(rets_train, p_min, p_max)
print("\n[3.3.1] TRAIN (2024) Information Criteria:")
print(ic_train)
print(f"[3.3.1] TRAIN (2024) BIC-selected optimal lag: p = {p_train}")

# Information criteria for test data (informational only)
if len(rets_test) > p_max:
    ic_test, p_test = ic_select(rets_test, p_min, p_max)
    print("\n[3.3.1] TEST (2025) Information Criteria (informational only):")
    print(ic_test)
    print(f"[3.3.1] TEST (2025) BIC-selected lag (informational): p = {p_test}")

###########################################################
### VAR(5) ESTIMATION AND ANALYSIS
###########################################################

# Estimate VAR(5) model on full sample as specified in assignment
print("\n=== VAR(5) MODEL ESTIMATION ===")
model_fs = VAR(rets_fs)
res = model_fs.fit(p_assign, trend="c")
print("\n[3.3.2] VAR(p=5) Summary (using statsmodels):")
print(res.summary())

###########################################################
### COINTEGRATION ANALYSIS (JOHANSEN TEST)
###########################################################

print("\n=== COINTEGRATION ANALYSIS ===")

# Prepare thinned log-price series for cointegration testing
logP_thin = np.log(prices.resample(COINTEG_FREQ).last()).dropna(how="any")
k_ar_diff = p_assign - 1  # Number of lags in differences for VECM

# Perform Johansen cointegration test
cj = coint_johansen(logP_thin, det_order=0, k_ar_diff=k_ar_diff)
johansen_report = pd.DataFrame({
    "eigenvalue": cj.eig,
    "trace_stat": cj.lr1,
    "trace_crit_90": cj.cvt[:, 0],
    "trace_crit_95": cj.cvt[:, 1],
    "trace_crit_99": cj.cvt[:, 2],
})

print(f"\n[3.3.2b] Johansen Trace Test Results:")
print(f"Frequency: {COINTEG_FREQ} log-prices")
print(f"Specification: det_order=0, k_ar_diff={k_ar_diff}")
print(johansen_report)

# Determine cointegration rank at 95% confidence level
k = logP_thin.shape[1]
r_raw = int((cj.lr1 > cj.cvt[:, 1]).sum())
r = min(r_raw, k - 1)  # Cap at k-1 maximum rank
print(f"[3.3.2b] Estimated cointegration rank at 95% level: r = {r} (raw count = {r_raw})")

###########################################################
### VECM ESTIMATION (IF COINTEGRATED)
###########################################################

if r > 0:
    print("\n=== VECM ESTIMATION ===")
    vecm = VECM(logP_thin, k_ar_diff=k_ar_diff, coint_rank=r, deterministic="co")
    vecm_res = vecm.fit()
    print("\n[3.3.2b] VECM Summary (thinned series):")
    print(vecm_res.summary())
    
    # Display error correction parameters
    alpha = pd.DataFrame(vecm_res.alpha, index=cols, columns=[f"alpha_{i+1}" for i in range(r)])
    beta = pd.DataFrame(vecm_res.beta, index=cols, columns=[f"beta_{i+1}" for i in range(r)])
    print("\n[3.3.2b] Alpha (Error-Correction Adjustment Speeds):")
    print(alpha)
    print("\n[3.3.2b] Beta (Cointegrating Vectors):")
    print(beta)
else:
    print("\n[3.3.2b] No cointegration detected (r = 0); VECM not estimated.")

###########################################################
### VAR RESIDUAL DIAGNOSTICS
###########################################################

print("\n=== VAR RESIDUAL DIAGNOSTICS ===")

# Extract VAR residuals for diagnostic testing
u = pd.DataFrame(res.resid, index=rets_fs.index, columns=rets_fs.columns)

print("\n[3.3.3] VAR Residual Diagnostic Tests:")

# System-wide multivariate whiteness test
print(f"\nVAR Whiteness Test (all equations, nlags={nlags}):")
print(res.test_whiteness(nlags=nlags))

# Individual equation Ljung-Box tests for serial correlation
print(f"\nLjung-Box Test p-values by Equation (lags={nlags}):")
for c in u.columns:
    pval = acorr_ljungbox(u[c].dropna(), lags=[nlags], return_df=True)["lb_pvalue"].iloc[0]
    print(f"{c}: {pval:.4g}")

# Jarque-Bera normality tests for each equation
print("\nJarque-Bera Normality Test p-values by Equation:")
for c in u.columns:
    jb = stats.jarque_bera(u[c].dropna())
    print(f"{c}: {jb.pvalue:.4g}")

def corr_pval(x, y):
    """Calculate correlation coefficient with significance test"""
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    r = np.corrcoef(x, y)[0, 1]
    n = len(x)
    t = r * np.sqrt((n - 2) / max(1e-12, (1 - r**2)))
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 2))
    return r, p, n

# Test contemporaneous residual correlations (should be close to zero)
print("\nContemporaneous Residual Correlation Tests (H0: ρ=0):")
for a, b in combinations(u.columns, 2):
    r, p, n = corr_pval(u[a], u[b])
    approx_zero = ("yes" if (abs(r) < 0.05 and p > 0.1) else "no")
    print(f"{a} vs {b}: r={r:+.3f}, p={p:.4g}, n={n}, approx_zero? {approx_zero}")

###########################################################
### RESIDUAL VISUALIZATION
###########################################################

print("\n=== GENERATING RESIDUAL DIAGNOSTIC PLOTS ===")

# Time series plots of residuals by equation
fig, axes = plt.subplots(len(u.columns), 1, figsize=(10, 6), sharex=True)
if len(u.columns) == 1:
    axes = [axes]
for i, c in enumerate(u.columns):
    axes[i].plot(u.index, u[c].values)
    axes[i].axhline(0.0, linestyle="--", linewidth=1)
    axes[i].set_ylabel(c)
axes[-1].set_xlabel("Time")
fig.suptitle("VAR Residual Time Series (by equation) — Full Sample")
fig.tight_layout()
plt.show()

# Pairwise scatter plots of residuals (contemporaneous correlation visualization)
pairs = list(combinations(u.columns, 2))
fig, axes = plt.subplots(1, len(pairs), figsize=(15, 4))
for ax, (a, b) in zip(axes, pairs):
    x = u[a].values
    y = u[b].values
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    r, p, n = corr_pval(x, y)
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    yy = coef[0] * xx + coef[1]
    ax.scatter(x, y, s=6, alpha=0.6)
    ax.plot(xx, yy, linewidth=1)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_title(f"{a} vs {b}\nr={r:+.3f}, p={p:.3g}")
    ax.set_xlabel(a)
    ax.set_ylabel(b)
fig.suptitle("Pairwise Residual Scatter Plots (all 3 combinations) — Full Sample")
fig.tight_layout()
plt.show()

###########################################################
### SUBSAMPLE STABILITY ANALYSIS
###########################################################

print("\n=== SUBSAMPLE STABILITY ANALYSIS ===")

# Split residuals by time period
u_2024 = u.loc[u.index < split_date]
u_2025 = u.loc[u.index >= split_date]

def plot_pair_scatter(u_period, title_prefix):
    """Generate pairwise scatter plots for residual correlations"""
    pairs = list(combinations(u_period.columns, 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(15, 4))
    for ax, (a, b) in zip(axes, pairs):
        x = u_period[a].to_numpy()
        y = u_period[b].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        r = np.corrcoef(x, y)[0, 1]
        coef = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 200)
        yy = coef[0] * xx + coef[1]
        ax.scatter(x, y, s=6, alpha=0.6)
        ax.plot(xx, yy, linewidth=1)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"{a} vs {b}\nr={r:+.3f}")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
    fig.suptitle(f"{title_prefix}: Pairwise Residual Scatter Plots")
    fig.tight_layout()
    plt.show()

# Generate subsample scatter plots
plot_pair_scatter(u_2024, "2024 (Training Period)")
plot_pair_scatter(u_2025, "2025 (Test Period)")

# Display correlation matrices by period
print("\nResidual Correlation Matrix — 2024 (Training Period):")
print(u_2024.corr())
print("\nResidual Correlation Matrix — 2025 (Test Period):")
print(u_2025.corr())

def fisher_change_test(x1, y1, x2, y2):
    """Test for significant change in correlation between two periods using Fisher z-transform"""
    def _zbits(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        r = np.corrcoef(x[m], y[m])[0, 1]
        n = m.sum()
        z = np.arctanh(np.clip(r, -0.999999, 0.999999))
        se = 1 / np.sqrt(max(1, n - 3))
        return r, z, se, n
    
    r1, z1, se1, n1 = _zbits(x1, y1)
    r2, z2, se2, n2 = _zbits(x2, y2)
    zdiff = (z1 - z2) / np.sqrt(se1**2 + se2**2)
    p = 2 * (1 - stats.norm.cdf(abs(zdiff)))
    return r1, r2, zdiff, p, n1, n2

# Test for structural changes in residual correlations across periods
print("\nStructural Change Tests: Residual Correlations (2024 vs 2025):")
for a, b in combinations(u.columns, 2):
    r1, r2, z, p, n1, n2 = fisher_change_test(u_2024[a].values, u_2024[b].values,
                                              u_2025[a].values, u_2025[b].values)
    verdict = "different" if p < 0.05 else "no clear change"
    print(f"{a} vs {b}: r_2024={r1:+.3f} (n={n1}), r_2025={r2:+.3f} (n={n2}), z={z:+.2f}, p={p:.3g} → {verdict}")

###########################################################
### SECTION 3.4: GARCH VOLATILITY MODELING
###########################################################

print("\n" + "=" * 60)
print("SECTION 3.4: GARCH VOLATILITY MODELING")
print("=" * 60)

# Import required libraries for GARCH modeling and diagnostics
from scipy import stats
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from arch import arch_model

###########################################################
### DAILY RETURN PREPARATION
###########################################################

print("\n=== DAILY DATA PREPARATION FOR GARCH ANALYSIS ===")

# Extract SPY5.P series (main analysis series)
spy5p_series = df['SPY5.P'].dropna()

# Convert minute-by-minute prices to daily closing prices
daily_prices = spy5p_series.groupby(spy5p_series.index.date).last()
daily_prices.index = pd.to_datetime(daily_prices.index)

# Calculate daily percentage log returns: r_t = 100 * ln(P_t/P_{t-1})
daily_returns = np.log(daily_prices / daily_prices.shift(1)).dropna() * 100

print(f"Data Summary:")
print(f"  Original minute observations: {len(spy5p_series):,}")
print(f"  Daily closing prices: {len(daily_prices):,}")
print(f"  Daily returns: {len(daily_returns):,}")
print(f"  Date range: {daily_returns.index[0].date()} to {daily_returns.index[-1].date()}")

# Basic descriptive statistics for daily returns
print(f"\nDaily Return Characteristics (Percentage Points):")
print(f"  Mean: {daily_returns.mean():.4f}%")
print(f"  Standard Deviation: {daily_returns.std():.4f}%")
print(f"  Minimum: {daily_returns.min():.4f}% (worst daily loss)")
print(f"  Maximum: {daily_returns.max():.4f}% (best daily gain)")
print(f"  Skewness: {daily_returns.skew():.4f}")
print(f"  Excess Kurtosis: {daily_returns.kurtosis():.4f}")

# Generate daily returns time series plot with volatility bands
print(f"\n=== GENERATING DAILY RETURNS VISUALIZATION ===")
plt.figure(figsize=(12, 6))

# Plot return series
plt.plot(daily_returns.index, daily_returns.values, linewidth=0.8, color='blue', alpha=0.7)

# Add ±2σ volatility reference bands
std_dev = daily_returns.std()
plt.axhline(y=2*std_dev, color='red', linestyle='--', alpha=0.7, label='+2σ')
plt.axhline(y=-2*std_dev, color='red', linestyle='--', alpha=0.7, label='-2σ')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

plt.title('Daily Percentage Log Returns with ±2σ Volatility Bands')
plt.xlabel('Date')
plt.ylabel('Returns (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

###########################################################
### ARCH EFFECT TESTING
###########################################################

print(f"\n=== TESTING FOR TIME-VARYING VOLATILITY (ARCH EFFECTS) ===")

# ARCH LM test for heteroskedasticity in returns
arch_test = het_arch(daily_returns, maxlag=5)
print(f"ARCH LM Test Results (5 lags):")
print(f"  LM Statistic: {arch_test[0]:.3f}, p-value: {arch_test[1]:.6f}")
print(f"  F-Statistic: {arch_test[2]:.3f}, p-value: {arch_test[3]:.6f}")


###########################################################
### GARCH MODEL ESTIMATION
###########################################################

print(f"\n=== GARCH MODEL FAMILY ESTIMATION ===")

# Define GARCH model specifications: symmetric and asymmetric with various lag structures
model_specs = []
for p in [1, 2]:  # ARCH lags
    for q in [1, 2]:  # GARCH lags
        model_specs.append(('GARCH', p, q))
        model_specs.append(('EGARCH', p, q))

# Initialize results storage
garch_results = {}

# Display estimation progress header
print(f"\n{'Model':<12} | {'Converged':<10} | {'AIC':<10} | {'BIC':<10} | {'LogLik':<10}")
print("-" * 65)

# Estimate all GARCH model specifications
for model_type, p, q in model_specs:
    model_name = f"{model_type}({p},{q})"
    
    try:
        # Configure model specification (GARCH vs EGARCH)
        if model_type == 'GARCH':
            model = arch_model(daily_returns, vol='Garch', p=p, q=q, 
                             mean='Constant', dist='normal', rescale=False)
        else:  # EGARCH specification
            model = arch_model(daily_returns, vol='EGARCH', p=p, q=q,
                             mean='Constant', dist='normal', rescale=False)
        
        # Estimate model parameters using maximum likelihood
        fitted = model.fit(disp='off', show_warning=False)
        
        # Store comprehensive estimation results
        garch_results[model_name] = {
            'fitted_model': fitted,
            'converged': fitted.convergence_flag == 0,
            'aic': fitted.aic,
            'bic': fitted.bic,
            'loglik': fitted.loglikelihood,
            'params': fitted.params,
            'std_errors': fitted.std_err,
            'conditional_volatility': fitted.conditional_volatility
        }
        
        # Display estimation results
        conv_status = "YES" if fitted.convergence_flag == 0 else "NO"
        print(f"{model_name:<12} | {conv_status:<10} | {fitted.aic:<10.2f} | {fitted.bic:<10.2f} | {fitted.loglikelihood:<10.2f}")
        
    except Exception as e:
        print(f"{model_name:<12} | ERROR: {str(e)[:40]}")

###########################################################
### MODEL SELECTION AND VALIDATION
###########################################################

print(f"\n=== GARCH MODEL SELECTION ===")

# Filter successfully estimated models
successful_models = {k: v for k, v in garch_results.items() if v.get('converged', False)}

if successful_models:
    # Identify best performing models by information criteria
    best_aic = min(successful_models.keys(), key=lambda k: successful_models[k]['aic'])
    best_bic = min(successful_models.keys(), key=lambda k: successful_models[k]['bic'])
    
    print(f"Model Selection Results:")
    print(f"  Best by AIC: {best_aic} (AIC = {successful_models[best_aic]['aic']:.2f})")
    print(f"  Best by BIC: {best_bic} (BIC = {successful_models[best_bic]['bic']:.2f})")
    
    # Validate best model using standardized residuals
    print(f"\n=== MODEL VALIDATION: {best_aic} ===")
    
    best_model = successful_models[best_aic]['fitted_model']
    std_residuals = best_model.std_resid.dropna()
    
    # Test 1: ARCH effects in standardized residuals (should be eliminated)
    arch_test_resid = het_arch(std_residuals, maxlag=5)
    print(f"ARCH LM Test on Standardized Residuals:")
    print(f"  LM Statistic: {arch_test_resid[0]:.3f}, p-value: {arch_test_resid[1]:.3f}")
    
    # Test 2: Serial correlation in levels and squares
    lb_levels = acorr_ljungbox(std_residuals, lags=10, return_df=True)
    lb_squares = acorr_ljungbox(std_residuals**2, lags=10, return_df=True)
    
    print(f"Ljung-Box Serial Correlation Tests (10 lags):")
    print(f"  Levels: LB = {lb_levels['lb_stat'].iloc[-1]:.3f}, p = {lb_levels['lb_pvalue'].iloc[-1]:.3f}")
    print(f"  Squares: LB = {lb_squares['lb_stat'].iloc[-1]:.3f}, p = {lb_squares['lb_pvalue'].iloc[-1]:.3f}")
    
    # Test 3: Normality of standardized residuals
    jb_stat, jb_pval = jarque_bera(std_residuals)
    print(f"Jarque-Bera Normality Test:")
    print(f"  JB Statistic: {jb_stat:.3f}, p-value: {jb_pval:.6f}")
    
    ###########################################################
    ### RESIDUAL DIAGNOSTIC PLOTS
    ###########################################################
    
    print(f"\n=== GENERATING RESIDUAL DIAGNOSTIC PLOTS ===")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Histogram with normal distribution overlay
    ax1.hist(std_residuals.values, bins=50, density=True, alpha=0.7, 
             color='lightblue', edgecolor='black', linewidth=0.5)
    
    # Overlay theoretical standard normal distribution
    x_range = np.linspace(std_residuals.min(), std_residuals.max(), 100)
    normal_curve = stats.norm.pdf(x_range, 0, 1)
    ax1.plot(x_range, normal_curve, 'red', linewidth=2.5, label='Standard Normal')
    
    ax1.set_title('Standardized Residuals vs Normal Distribution', fontweight='bold')
    ax1.set_xlabel('Standardized Residuals')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Q-Q plot for normality assessment
    from scipy import stats as scipy_stats
    scipy_stats.probplot(std_residuals.values, dist="norm", plot=ax2)
    
    ax2.set_title('Q-Q Plot: Sample vs Theoretical Normal', fontweight='bold')
    ax2.set_xlabel('Theoretical Normal Quantiles')
    ax2.set_ylabel('Sample Quantiles')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Standardized residual summary statistics
    skewness = stats.skew(std_residuals)
    kurtosis = stats.kurtosis(std_residuals, fisher=False)  # Regular kurtosis
    excess_kurtosis = stats.kurtosis(std_residuals, fisher=True)  # Excess kurtosis
    
    print(f"\nStandardized Residuals Summary Statistics:")
    print(f"  Mean: {std_residuals.mean():.4f}, Std Dev: {std_residuals.std():.4f}")
    print(f"  Skewness: {skewness:.3f}, Kurtosis: {kurtosis:.3f} (Excess: {excess_kurtosis:.3f})")
    print(f"  Range: {std_residuals.min():.2f} to {std_residuals.max():.2f}")

###########################################################
### REALIZED VOLATILITY COMPARISON
###########################################################

print(f"\n=== REALIZED VOLATILITY VALIDATION ===")

# Construct daily realized volatility using assignment specification: σ̂²_{RV,t} = Σᵢ r²ᵢₜ
daily_realized_vol = returns_cleaned.groupby(returns_cleaned.index.date).apply(lambda x: (x**2).sum())
daily_realized_vol.index = pd.to_datetime(daily_realized_vol.index)
daily_realized_vol = daily_realized_vol.dropna()

print(f"Realized Volatility Construction:")
print(f"  Series length: {len(daily_realized_vol)} observations")
print(f"  Range: {daily_realized_vol.min():.4f} to {daily_realized_vol.max():.4f}")

if successful_models:
    # Compare GARCH and EGARCH models against realized volatility
    best_garch = best_aic
    egarch_models = {k: v for k, v in successful_models.items() if k.startswith('EGARCH')}
    best_egarch = min(egarch_models.keys(), key=lambda k: egarch_models[k]['aic']) if egarch_models else None
    
    # Calculate correlations with realized volatility measure
    garch_variance = successful_models[best_garch]['conditional_volatility'] ** 2
    common_dates = daily_realized_vol.index.intersection(garch_variance.index)
    
    if len(common_dates) > 0:
        rv_common = daily_realized_vol.loc[common_dates]
        garch_common = garch_variance.loc[common_dates]
        
        garch_correlation = rv_common.corr(garch_common)
        
        print(f"\nModel Performance vs Realized Volatility:")
        print(f"  {best_garch}: correlation = {garch_correlation:.4f}")
        
        if best_egarch:
            egarch_variance = successful_models[best_egarch]['conditional_volatility'] ** 2
            egarch_common = egarch_variance.loc[common_dates]
            egarch_correlation = rv_common.corr(egarch_common)
            print(f"  {best_egarch}: correlation = {egarch_correlation:.4f}")
            
            ###########################################################
            ### VOLATILITY COMPARISON VISUALIZATION
            ###########################################################
            
            print(f"\n=== GENERATING VOLATILITY COMPARISON PLOTS ===")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 13))
            
            # Top panel: Time series comparison (volatility evolution)
            ax1.plot(common_dates, rv_common.values, label='Realized Variance', 
                    alpha=0.7, linewidth=1)
            ax1.plot(common_dates, garch_common.values, label=f'{best_garch} Variance', 
                    alpha=0.8, linewidth=1.2)
            ax1.plot(common_dates, egarch_common.values, label=f'{best_egarch} Variance', 
                    alpha=0.8, linewidth=1.2)
            
            ax1.set_title('Variance Model Comparison: GARCH vs EGARCH vs Realized Variance')
            ax1.set_ylabel('Variance')
            ax1.set_ylim(0, 20)  # Focus on main dynamics, crop extreme outliers
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bottom panel: Scatter plot for correlation visualization
            max_plot_value = min(
                np.percentile(rv_common.values, 95),
                np.percentile(garch_common.values, 95), 
                4.0  # Cap for readability
            )
            
            ax2.scatter(rv_common.values, garch_common.values, alpha=0.6, s=20, 
                       label=f'{best_garch} (ρ={garch_correlation:.3f})', color='red')
            ax2.scatter(rv_common.values, egarch_common.values, alpha=0.6, s=20, 
                       label=f'{best_egarch} (ρ={egarch_correlation:.3f})', color='blue')
            
            # Add perfect correlation reference line
            ax2.plot([0, max_plot_value], [0, max_plot_value], 'k--', alpha=0.5, label='Perfect Fit')
            
            ax2.set_xlim(0, max_plot_value)
            ax2.set_ylim(0, max_plot_value)
            ax2.set_xlabel('Realized Variance')
            ax2.set_ylabel('Model Variance')
            ax2.set_title('Model vs Realized Variance (95th Percentile View)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

###########################################################
### ARMA-GARCH EXTENSION (TWO-STEP APPROACH)
###########################################################

print(f"\n=== ARMA-GARCH INTEGRATION: TWO-STEP APPROACH ===")

# Use cleaned minute returns from Section 3.2 for ARMA-GARCH extension
minute_returns = returns_cleaned

print(f"Extending Section 3.2 ARMA(0,1) model with GARCH volatility:")
print(f"Using {len(minute_returns):,} minute-level observations")

# Step 1: Estimate MA(1) model to capture serial correlation structure
ma_model = ARIMA(minute_returns, order=(0, 0, 1))
ma_fitted = ma_model.fit()

# Extract MA(1) residuals for GARCH modeling
ma_residuals = ma_fitted.resid.dropna()

print(f"\nStep 1 - MA(1) Mean Model Results:")
print(f"  MA(1) coefficient: {ma_fitted.params['ma.L1']:.6f}")
print(f"  MA(1) model AIC: {ma_fitted.aic:.0f}")

# Step 2: Apply GARCH and EGARCH models to MA(1) residuals
ma_garch_model = arch_model(ma_residuals, mean='Zero', vol='GARCH', p=1, q=1, rescale=False)
ma_garch_fitted = ma_garch_model.fit(disp='off')

ma_egarch_model = arch_model(ma_residuals, mean='Zero', vol='EGARCH', p=1, q=1, rescale=False)  
ma_egarch_fitted = ma_egarch_model.fit(disp='off')

# Display comprehensive ARMA-GARCH results
print(f"\n=== ARMA-GARCH INTEGRATION RESULTS ===")
print(f"Original ARMA(0,1) - Constant Variance: AIC = -622,917")
print(f"MA(1) + GARCH(1,1) - Two-Step:         AIC = {ma_fitted.aic + ma_garch_fitted.aic:.0f}")
print(f"MA(1) + EGARCH(1,1) - Two-Step:        AIC = {ma_fitted.aic + ma_egarch_fitted.aic:.0f}")

# Calculate AIC improvements
aic_improvement_garch = (ma_fitted.aic + ma_garch_fitted.aic) - (-622917)
aic_improvement_egarch = (ma_fitted.aic + ma_egarch_fitted.aic) - (-622917)

print(f"\nModel Improvement Analysis:")
print(f"  GARCH extension improvement:  {aic_improvement_garch:.0f} AIC points")
print(f"  EGARCH extension improvement: {aic_improvement_egarch:.0f} AIC points")

