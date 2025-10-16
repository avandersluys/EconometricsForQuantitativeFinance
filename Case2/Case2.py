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
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import itertools

warnings.filterwarnings('ignore')

# Load data
github_url = "https://raw.githubusercontent.com/avandersluys/EconometricsForQuantitativeFinance/7f88e429a23a03593352b07a64f9d882017f5246/Case2/sp_9.csv.gz"
df = pd.read_csv(github_url, index_col=0, parse_dates=True)

# Basic info
print("Basic Data Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Calculate and print percentage of non-missing data for each series
for col in df.columns:
    pct_complete = 100 * df[col].notna().mean()
    print(f"{col}: {pct_complete:.2f}% complete")

# Check missing data pattern before dropna()
print(f"Missing data check:")
print(f"Total missing: {df['SPY5.P'].isna().sum()}")
if df['SPY5.P'].isna().sum() > 0:
    missing_hours = df[df['SPY5.P'].isna()].index.hour.value_counts().sort_index()
    print(f"Missing by hour: {dict(missing_hours)}")
    
    # Show first few missing timestamps
    missing_times = df[df['SPY5.P'].isna()].index[:10]
    print(f"First missing times: {list(missing_times)}")

# Check if missing days are holidays
missing_dates = df[df['SPY5.P'].isna()].index.date
unique_missing_dates = sorted(set(missing_dates))
print(f"Missing dates: {unique_missing_dates[:5]}...")  # Show first few


# Extract and clean SPY5.P
spy5p_series = df['SPY5.P'].dropna()
returns = np.log(spy5p_series / spy5p_series.shift(1))*100
print(f"Shape: {returns.shape}")
returns_clean = returns.copy()

#Look at outliers
max_return = returns_clean.max()
min_return = returns_clean.min()
max_date = returns_clean.idxmax()
min_date = returns_clean.idxmin()
print(f"Maximum: {max_return:.4f}% at {max_date}")
print(f"Minimum: {min_return:.4f}% at {min_date}")

# Remove first 3 observations of each trading day 
daily_groups = returns_clean.groupby(returns_clean.index.date)
returns_cleaned = pd.concat([group.iloc[3:] for date, group in daily_groups if len(group) > 3])
"""
print(f"Original returns: {len(returns)} observations")
print(f"After cleaning: {len(returns_cleaned)} observations")
print(f"Range: {returns_cleaned.min():.4f}% to {returns_cleaned.max():.4f}%")

#Look at outliers for this cleaned data 
max_return = returns_cleaned.max()
min_return = returns_cleaned.min()
max_date = returns_cleaned.idxmax()
min_date = returns_cleaned.idxmin()
print(f"Maximum: {max_return:.4f}% at {max_date}")
print(f"Minimum: {min_return:.4f}% at {min_date}")

# Simple look around extreme values
print(f"\nAround max return:")
print(returns_cleaned.loc[max_date - pd.Timedelta(minutes=5):max_date + pd.Timedelta(minutes=5)])

print(f"\nAround min return:")
print(returns_cleaned.loc[min_date - pd.Timedelta(minutes=5):min_date + pd.Timedelta(minutes=5)])


# ADF Test
adf_result = adfuller(returns_cleaned, autolag='AIC')
print(f"\\nADF Test Statistic: {adf_result[0]:.6f}")
print(f"p-value: {adf_result[1]:.2e}")

# ACF/PACF Analysis 
clean_returns = returns_cleaned
lags = 120  # 2 hours of minute data
acf_values = acf(clean_returns, nlags=lags, fft=True)
pacf_values = pacf(clean_returns, nlags=lags, method='ols')

# Significance bounds
n = len(clean_returns)
bound = 1.96 / np.sqrt(n)
print(f"\n95% confidence bounds: ±{bound:.6f}")

# Find all significant lags
sig_acf = [(i, acf_values[i]) for i in range(1, lags+1) if abs(acf_values[i]) > bound]
sig_pacf = [(i, pacf_values[i]) for i in range(1, lags+1) if abs(pacf_values[i]) > bound]

print(f"\nSignificant ACF lags (first 20): {[lag for lag, val in sig_acf[:20]]}")
print(f"Significant PACF lags (first 20): {[lag for lag, val in sig_pacf[:20]]}")
print(f"Total significant - ACF: {len(sig_acf)}, PACF: {len(sig_pacf)}")

# Show pattern 
if len(sig_acf) > 0:
    max_sig_acf = max([lag for lag, val in sig_acf])
    max_sig_pacf = max([lag for lag, val in sig_pacf]) if len(sig_pacf) > 0 else 0
    print(f"Last significant lag - ACF: {max_sig_acf}, PACF: {max_sig_pacf}")

# Create plots showing more detail
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# ACF plot - show first 30 lags 
plot_acf(clean_returns, lags=30, ax=ax1, alpha=0.05, title='ACF - SPY5.P Returns (30-minute window)')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 0.25)  
ax1.set_xlabel('Lag (minutes)')

# PACF plot - show first 30 lags  
plot_pacf(clean_returns, lags=30, ax=ax2, alpha=0.05, title='PACF - SPY5.P Returns (30-minute window)')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 0.25)
ax2.set_xlabel('Lag (minutes)')

plt.tight_layout()
plt.show()

# Also create a zoomed version for first 10 lags
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(clean_returns, lags=10, ax=ax1, alpha=0.05, title='ACF - SPY5.P Returns (Detailed View)')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.02, 0.25)  

plot_pacf(clean_returns, lags=10, ax=ax2, alpha=0.05, title='PACF - SPY5.P Returns (Detailed View)')  
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.06, 0.25)

plt.tight_layout()
plt.show()


# Split data 
returns_2024 = returns_cleaned[returns_cleaned.index < '2025-01-01']
returns_2025 = returns_cleaned[returns_cleaned.index >= '2025-01-01']
print(f"In-sample (2024): {len(returns_2024)} obs")
print(f"Out-sample (2025): {len(returns_2025)} obs")

# ARMA Model Estimation
print(f"\n=== ARMA MODEL ESTIMATION ===")

models_to_estimate = [(p,q) for p in range(3) for q in range(3)]
results = {}

print(f"\n{'Model':<10} | {'Conv':<5} | {'AIC_In':<10} | {'AIC_Out':<10} | {'MSE_In':<10} | {'MSE_Out':<10} | {'LB_In':<8} | {'LB_Out':<8}")
print("-" * 90)

for p, q in models_to_estimate:
    model_name = f"ARMA({p},{q})"
    
    # ESTIMATE PARAMETERS ON 2024 DATA ONLY
    model = ARIMA(returns_2024, order=(p, 0, q))
    fitted = model.fit(cov_type='robust')
    
    params = fitted.params.to_dict()
    std_errors = fitted.bse.to_dict()
    converged = fitted.mle_retvals['converged'] if fitted.mle_retvals else False
    
    # IN-SAMPLE STATISTICS (2024 period)
    residuals_in = fitted.resid
    mse_in = np.mean(residuals_in**2)
    lb_test_in = acorr_ljungbox(residuals_in.dropna(), lags=10, return_df=True)
    lb_stat_in = lb_test_in['lb_stat'].iloc[-1]
    lb_pval_in = lb_test_in['lb_pvalue'].iloc[-1]
    
    # OUT-OF-SAMPLE STATISTICS (2025 period using fixed 2024 parameters)
    forecast_model = fitted.apply(returns_2025, refit=False)
    residuals_out = forecast_model.resid
    
    # Out-of-sample MSE
    mse_out = np.mean(residuals_out**2)
    
    # Out-of-sample log-likelihood using 2024 variance estimate
    sigma2_est = np.var(residuals_in)
    n_out = len(residuals_out)
    loglik_out = -0.5 * n_out * np.log(2 * np.pi * sigma2_est) - 0.5 * np.sum(residuals_out**2) / sigma2_est
    
    # Out-of-sample AIC/BIC
    k = len(fitted.params)
    aic_out = 2 * k - 2 * loglik_out
    bic_out = k * np.log(n_out) - 2 * loglik_out
    
    # Out-of-sample Ljung-Box
    lb_test_out = acorr_ljungbox(residuals_out.dropna(), lags=10, return_df=True)
    lb_stat_out = lb_test_out['lb_stat'].iloc[-1]
    lb_pval_out = lb_test_out['lb_pvalue'].iloc[-1]
    
    # Store results
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
    
    conv_status = "YES" if converged else "NO"
    print(f"{model_name:<10} | {conv_status:<5} | {fitted.aic:<10.0f} | {aic_out:<10.0f} | {mse_in:<10.6f} | {mse_out:<10.6f} | {lb_stat_in:<8.2f} | {lb_stat_out:<8.2f}")

# Best models (based on in-sample criteria)
aic_best = min(results.keys(), key=lambda k: results[k]['aic_in'])
bic_best = min(results.keys(), key=lambda k: results[k]['bic_in'])
print(f"\nBest by AIC (In-Sample): {aic_best}")
print(f"Best by BIC (In-Sample): {bic_best}")

# Parameter estimates table
print("PARAMETER ESTIMATES (with Standard Errors) - Estimated on 2024 Data")
print("="*120)

model_order = ['ARMA(0,0)', 'ARMA(0,1)', 'ARMA(0,2)', 'ARMA(1,0)', 
               'ARMA(1,1)', 'ARMA(1,2)', 'ARMA(2,0)', 'ARMA(2,1)', 'ARMA(2,2)']

# Get parameters
all_params = set()
for result in results.values():
    all_params.update(result['params'].keys())

param_order = ['const', 'ar.L1', 'ar.L2', 'ma.L1', 'ma.L2']
ordered_params = [p for p in param_order if p in all_params]

# Header
print(f"{'Parameter':<15}", end="")
for model in model_order:
    print(f" | {model:<12}", end="")
print()
print("-" * 120)

# Parameters with standard errors
for param in ordered_params:
    param_display = {
        'const': 'Constant', 
        'ar.L1': 'AR(1)', 
        'ar.L2': 'AR(2)',
        'ma.L1': 'MA(1)', 
        'ma.L2': 'MA(2)'
    }.get(param, param)
    
    print(f"{param_display:<15}", end="")
    for model in model_order:
        if param in results[model]['params']:
            val = results[model]['params'][param]
            print(f" | {val:>10.6f}  ", end="")
        else:
            print(f" | {'--':>10}    ", end="")
    print()
    
    print(f"{'(Std. Error)':<15}", end="")
    for model in model_order:
        if param in results[model]['std_errors']:
            se = results[model]['std_errors'][param]
            print(f" | ({se:>8.6f}) ", end="")
        else:
            print(f" | {'--':>10}    ", end="")
    print()
    print()

# Diagnostics table
print("="*120)
print("MODEL DIAGNOSTICS - USING 2024 PARAMETERS ON BOTH PERIODS")
print("="*120)

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

print("="*120)

############################################################## 3.3a (simplified)
split_date = pd.Timestamp('2025-01-01')
print("\n=== QUESTION 3.3a — VAR(5) on returns (simplified) ===")

from statsmodels.tsa.api import VAR

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
params = res.params.copy()        # DataFrame: rows = equations, cols = coeffs (const, L1.x, ... or x.L1)
bse    = res.stderr.copy()
tvals  = res.tvalues.copy()
pvals  = res.pvalues.copy()

# Build long-format table with nice labels and significance stars
def stars(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

# --- Robust coefficient label parser (supports both label styles)
import re
def parse_coef_label(coef: str, eq_name: str):
    
    #Returns (lag:int, src:str, kind:str) for a VAR coefficient label.
    #Supports both 'Lk.SYMBOL' and 'SYMBOL.Lk' styles. 'const' -> (0, '-', 'intercept')
    
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
p_gr = res.k_ar
series = list(R_in.columns)

def granger(direction_y, direction_x):
    # Does X -> Y ?
    res = res.test_causality(caused=direction_y, causing=[direction_x], kind='f')
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
resid = res.resid.copy()
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
plt.stem(lags, cc)
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


"""

# Import required libraries for GARCH modeling and diagnostics
from scipy import stats
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np

print(f"\n" + "="*60)
print("SECTION 3.4: GARCH MODELING")
print("="*60)

# Convert minute-by-minute data to daily returns
print("\n=== DAILY RETURN CONVERSION ===")

# Extract daily closing prices (last observation each trading day)
daily_prices = spy5p_series.groupby(spy5p_series.index.date).last()
daily_prices.index = pd.to_datetime(daily_prices.index)

# Calculate daily percentage log returns: r_t = 100 * ln(P_t/P_{t-1})
daily_returns = np.log(daily_prices / daily_prices.shift(1)) * 100
daily_returns = daily_returns.dropna()

print(f"Daily prices: {len(daily_prices)} observations")
print(f"Daily returns: {len(daily_returns)} observations")
print(f"Date range: {daily_returns.index.min().date()} to {daily_returns.index.max().date()}")
print(f"Return range: {daily_returns.min():.2f}% to {daily_returns.max():.2f}%")
print(f"Mean return: {daily_returns.mean():.3f}%, Std dev: {daily_returns.std():.3f}%")

# Identify extreme return events and their dates
max_return = daily_returns.max()
min_return = daily_returns.min()
max_date = daily_returns.idxmax()
min_date = daily_returns.idxmin()
print(f"\nExtreme returns:")
print(f"  Maximum: {max_return:.2f}% on {max_date.date()}")
print(f"  Minimum: {min_return:.2f}% on {min_date.date()}")

# Test for ARCH effects (time-varying volatility)
print(f"\n=== TESTING FOR TIME-VARYING VOLATILITY ===")
arch_test = het_arch(daily_returns, maxlag=5)
print(f"ARCH LM Test (5 lags):")
print(f"  LM Statistic: {arch_test[0]:.3f}, p-value: {arch_test[1]:.6f}")
print(f"  F-Statistic: {arch_test[2]:.3f}, p-value: {arch_test[3]:.6f}")

# Estimate GARCH family models
print(f"\n=== GARCH MODEL ESTIMATION ===")

# Define model combinations: GARCH and EGARCH with p,q ∈ {1,2}
model_specs = []
for p in [1, 2]:
    for q in [1, 2]:
        model_specs.append(('GARCH', p, q))
        model_specs.append(('EGARCH', p, q))

# Store estimation results
garch_results = {}
print(f"\n{'Model':<12} | {'Converged':<10} | {'AIC':<10} | {'BIC':<10} | {'LogLik':<10}")
print("-" * 65)

# Estimate each model specification
for model_type, p, q in model_specs:
    model_name = f"{model_type}({p},{q})"
    
    try:
        # Configure GARCH or EGARCH model
        if model_type == 'GARCH':
            model = arch_model(daily_returns, vol='Garch', p=p, q=q, 
                             mean='Constant', dist='normal', rescale=False)
        else:  # EGARCH
            model = arch_model(daily_returns, vol='EGARCH', p=p, q=q,
                             mean='Constant', dist='normal', rescale=False)
        
        # Estimate model parameters
        fitted = model.fit(disp='off', show_warning=False)
        
        # Extract key results
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
        
        # Display results
        conv_status = "YES" if fitted.convergence_flag == 0 else "NO"
        print(f"{model_name:<12} | {conv_status:<10} | {fitted.aic:<10.2f} | {fitted.bic:<10.2f} | {fitted.loglikelihood:<10.2f}")
        
    except Exception as e:
        print(f"{model_name:<12} | ERROR: {str(e)[:40]}")

# Identify best performing models
successful_models = {k: v for k, v in garch_results.items() if v.get('converged', False)}

if successful_models:
    best_aic = min(successful_models.keys(), key=lambda k: successful_models[k]['aic'])
    best_bic = min(successful_models.keys(), key=lambda k: successful_models[k]['bic'])
    print(f"\nBest models:")
    print(f"  By AIC: {best_aic} (AIC = {successful_models[best_aic]['aic']:.2f})")
    print(f"  By BIC: {best_bic} (BIC = {successful_models[best_bic]['bic']:.2f})")

# Validate the best model using standardized residuals
print(f"\n=== MODEL VALIDATION: {best_aic} ===")

if successful_models:
    best_model = successful_models[best_aic]['fitted_model']
    std_residuals = best_model.std_resid.dropna()
    
    # Test 1: ARCH effects in standardized residuals (should be none)
    arch_test_resid = het_arch(std_residuals, maxlag=5)
    print(f"ARCH LM Test on Standardized Residuals:")
    print(f"  LM Statistic: {arch_test_resid[0]:.3f}, p-value: {arch_test_resid[1]:.3f}")
    
    # Test 2: Serial correlation tests
    lb_levels = acorr_ljungbox(std_residuals, lags=10, return_df=True)
    lb_squares = acorr_ljungbox(std_residuals**2, lags=10, return_df=True)
    
    print(f"Ljung-Box Tests (10 lags):")
    print(f"  Levels: LB = {lb_levels['lb_stat'].iloc[-1]:.3f}, p = {lb_levels['lb_pvalue'].iloc[-1]:.3f}")
    print(f"  Squares: LB = {lb_squares['lb_stat'].iloc[-1]:.3f}, p = {lb_squares['lb_pvalue'].iloc[-1]:.3f}")
    
    # Test 3: Normality test
    jb_stat, jb_pval = jarque_bera(std_residuals)
    print(f"Jarque-Bera Normality Test:")
    print(f"  JB Statistic: {jb_stat:.3f}, p-value: {jb_pval:.6f}")

    # Generate diagnostic plots for standardized residuals
    print(f"\n=== RESIDUAL DIAGNOSTICS PLOTS ===")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Histogram vs normal distribution
    ax1.hist(std_residuals.values, bins=50, density=True, alpha=0.7, 
             color='lightblue', edgecolor='black', linewidth=0.5)
    
    # Overlay standard normal curve for comparison
    x_range = np.linspace(std_residuals.min(), std_residuals.max(), 100)
    normal_curve = stats.norm.pdf(x_range, 0, 1)
    ax1.plot(x_range, normal_curve, 'red', linewidth=2.5, label='Standard Normal')
    
    ax1.set_title('Standardized Residuals vs Normal Distribution', fontweight='bold')
    ax1.set_xlabel('Standardized Residuals')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Quantile-Quantile plot
    n_obs = len(std_residuals)
    sorted_residuals = np.sort(std_residuals.values)
    
    # Right panel: Simple Q-Q plot
    from scipy import stats as scipy_stats
    
    # Use scipy's probplot function for standard Q-Q plot
    scipy_stats.probplot(std_residuals.values, dist="norm", plot=ax2)
    
    ax2.set_title('Q-Q Plot: Sample vs Theoretical Normal', fontweight='bold')
    ax2.set_xlabel('Theoretical Normal Quantiles')
    ax2.set_ylabel('Sample Quantiles')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics for standardized residuals
    skewness = stats.skew(std_residuals)
    kurtosis = stats.kurtosis(std_residuals, fisher=False)  # Regular kurtosis
    excess_kurtosis = stats.kurtosis(std_residuals, fisher=True)  # Excess kurtosis
    
    print(f"\nStandardized Residuals Summary:")
    print(f"  Mean: {std_residuals.mean():.4f}, Std Dev: {std_residuals.std():.4f}")
    print(f"  Skewness: {skewness:.3f}, Kurtosis: {kurtosis:.3f} (Excess: {excess_kurtosis:.3f})")
    print(f"  Range: {std_residuals.min():.2f} to {std_residuals.max():.2f}")

# Calculate realized volatility for model comparison
print(f"\n=== REALIZED VOLATILITY COMPARISON ===")

# Construct realized volatility from minute-by-minute returns
daily_realized_vol = returns_clean.groupby(returns_clean.index.date).apply(lambda x: (x**2).sum())
daily_realized_vol.index = pd.to_datetime(daily_realized_vol.index)
daily_realized_vol = daily_realized_vol.dropna()

print(f"Realized volatility series: {len(daily_realized_vol)} observations")
print(f"RV range: {daily_realized_vol.min():.4f} to {daily_realized_vol.max():.4f}")

if successful_models:
    # Get best GARCH and EGARCH models for comparison
    best_garch = best_aic
    egarch_models = {k: v for k, v in successful_models.items() if k.startswith('EGARCH')}
    best_egarch = min(egarch_models.keys(), key=lambda k: egarch_models[k]['aic']) if egarch_models else None
    
    # Calculate correlations with realized volatility
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
            
            # Create volatility comparison plot
            print(f"\n=== VOLATILITY COMPARISON PLOT ===")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 13))
            
            # Top panel: Time series comparison (cropped at 20 for visibility)
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
            
            # Bottom panel: Scatter plot (zoomed to 95th percentile)
            max_plot_value = min(
                np.percentile(rv_common.values, 95),
                np.percentile(garch_common.values, 95), 
                4.0  # Cap for readability
            )
            
            ax2.scatter(rv_common.values, garch_common.values, alpha=0.6, s=20, 
                       label=f'{best_garch} (ρ={garch_correlation:.3f})', color='red')
            ax2.scatter(rv_common.values, egarch_common.values, alpha=0.6, s=20, 
                       label=f'{best_egarch} (ρ={egarch_correlation:.3f})', color='blue')
            
            # Add perfect fit reference line
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

# Use the same minute-by-minute cleaned returns from Section 3.2
minute_returns = returns_cleaned  # Cleaned minute returns

# Step 1: Estimate ARIMA(0,0,1) = MA(1) to get residuals
ma_model = ARIMA(minute_returns, order=(0, 0, 1))
ma_fitted = ma_model.fit()

# Extract residuals (these are the "demeaned" returns for GARCH)
ma_residuals = ma_fitted.resid.dropna()

print(f"MA(1) coefficient: {ma_fitted.params['ma.L1']:.6f}")
print(f"MA(1) AIC: {ma_fitted.aic:.0f}")

# Step 2: Apply GARCH to the MA residuals
ma_garch_model = arch_model(ma_residuals, mean='Zero', vol='GARCH', p=1, q=1, rescale=False)
ma_garch_fitted = ma_garch_model.fit(disp='off')

ma_egarch_model = arch_model(ma_residuals, mean='Zero', vol='EGARCH', p=1, q=1, rescale=False)  
ma_egarch_fitted = ma_egarch_model.fit(disp='off')

print(f"\n=== MA(1) + GARCH RESULTS ===")
print(f"Original ARMA(0,1): AIC = -622,917")
print(f"MA(1)-GARCH(1,1):   AIC = {ma_fitted.aic + ma_garch_fitted.aic:.0f}")
print(f"MA(1)-EGARCH(1,1):  AIC = {ma_fitted.aic + ma_egarch_fitted.aic:.0f}")


