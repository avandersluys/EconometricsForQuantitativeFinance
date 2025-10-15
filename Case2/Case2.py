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

# Extract and clean SPY5.P
spy5p_series = df['SPY5.P'].dropna()
returns = np.log(spy5p_series / spy5p_series.shift(1))*100
returns_clean = returns.copy()

# Remove first 3 observations of each trading day 
daily_groups = returns_clean.groupby(returns_clean.index.date)
returns_cleaned = pd.concat([group.iloc[3:] for date, group in daily_groups if len(group) > 3])

print(f"Original returns: {len(returns)} observations")
print(f"After cleaning: {len(returns_cleaned)} observations")
print(f"Range: {returns_cleaned.min():.4f}% to {returns_cleaned.max():.4f}%")

# Split data
returns_2024 = returns_cleaned[returns_cleaned.index < '2025-01-01']
returns_2025 = returns_cleaned[returns_cleaned.index >= '2025-01-01']
print(f"In-sample (2024): {len(returns_2024)} obs")
print(f"Out-sample (2025): {len(returns_2025)} obs")


# ADF Test
adf_result = adfuller(returns_2024.dropna(), autolag='AIC')
print(f"\\nADF Test Statistic: {adf_result[0]:.6f}")
print(f"p-value: {adf_result[1]:.2e}")

# ACF/PACF Analysis
clean_returns = returns_2024.dropna()
lags = 60
acf_values = acf(clean_returns, nlags=lags, fft=True)
pacf_values = pacf(clean_returns, nlags=lags, method='ols')

# Significance bounds
n = len(clean_returns)
bound = 1.96 / np.sqrt(n)
print(f"\\n95% confidence bounds: ±{bound:.6f}")

print("\\nFirst 10 ACF values:")
for i in range(11):
    sig = "***" if abs(acf_values[i]) > bound else ""
    print(f"  Lag {i:2d}: {acf_values[i]:8.6f} {sig}")

print("\\nFirst 10 PACF values:")  
for i in range(11):
    sig = "***" if abs(pacf_values[i]) > bound else ""
    print(f"  Lag {i:2d}: {pacf_values[i]:8.6f} {sig}")

# Create plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(clean_returns, lags=20, ax=ax1, alpha=0.05, title='SPY5.P Returns - ACF')
ax1.grid(True, alpha=0.3)

plot_pacf(clean_returns, lags=20, ax=ax2, alpha=0.05, title='SPY5.P Returns - PACF')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ARMA Model Estimation
print(f"\\n=== ARMA MODEL ESTIMATION ===")

# All combinations p,q ∈ {0,1,2}
models_to_estimate = [(p,q) for p in range(3) for q in range(3)]
results = {}

print(f"\\n{'Model':<10} | {'Converged':<10} | {'AIC':<10} | {'BIC':<10} | {'LogLik':<10} | {'LB_stat':<10} | {'LB_pval':<10}")
print("-" * 80)

for p, q in models_to_estimate:
    model_name = f"ARMA({p},{q})"
    model = ARIMA(clean_returns, order=(p, 0, q))
    fitted = model.fit()
    
    params = fitted.params.to_dict()
    std_errors = fitted.bse.to_dict()
    residuals = fitted.resid
    aic = fitted.aic
    bic = fitted.bic
    loglik = fitted.llf
    converged = fitted.mle_retvals['converged'] if fitted.mle_retvals else False

    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
    lb_stat = lb_test['lb_stat'].iloc[-1]
    lb_pval = lb_test['lb_pvalue'].iloc[-1]
    
    # Out-of-sample MSE
    if len(returns_2025) > 0 and fitted is not None:
        forecast = fitted.forecast(steps=len(returns_2025))
        mse_out = np.mean((returns_2025.dropna().values - forecast.values[:len(returns_2025.dropna())])**2)
    else:
        mse_out = np.nan
    
    # Store results
    results[model_name] = {
        'params': params,
        'std_errors': std_errors,
        'converged': converged,
        'aic': aic,
        'bic': bic,
        'loglik': loglik,
        'mse_in': np.mean(residuals**2),
        'mse_out': mse_out,
        'lb_stat': lb_stat,
        'lb_pval': lb_pval
    }
    
    conv_status = "YES" if converged else "NO"
    print(f"{model_name:<10} | {conv_status:<10} | {aic:<10.2f} | {bic:<10.2f} | {loglik:<10.2f} | {lb_stat:<10.2f} | {lb_pval:<10.3f}")


# Best models
aic_best = min(results.keys(), key=lambda k: results[k]['aic'])
bic_best = min(results.keys(), key=lambda k: results[k]['bic'])

print(f"\\nBest by AIC: {aic_best}")
print(f"Best by BIC: {bic_best}")


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
from arch import arch_model  

print(f"\n" + "="*60)
print("SECTION 3.4: GARCH MODELING")
print("="*60)

# Convert minute returns to daily returns
print("\n=== DAILY RETURN CONVERSION ===")

# Get daily closing prices (last price each day)
daily_prices = spy5p_series.groupby(spy5p_series.index.date).last()
daily_prices.index = pd.to_datetime(daily_prices.index)

# Calculate daily percentage log returns
daily_returns = np.log(daily_prices / daily_prices.shift(1)) * 100
daily_returns = daily_returns.dropna()

print(f"Daily prices: {len(daily_prices)} observations")
print(f"Daily returns: {len(daily_returns)} observations")
print(f"Date range: {daily_returns.index.min().date()} to {daily_returns.index.max().date()}")
print(f"Daily return range: {daily_returns.min():.4f}% to {daily_returns.max():.4f}%")
print(f"Daily return std: {daily_returns.std():.4f}%")

# Basic daily return statistics
print("\nDaily Return Statistics:")
print(daily_returns.describe())

# Test for ARCH effects
print(f"\n=== ARCH EFFECTS TEST ===")
arch_test = het_arch(daily_returns.dropna(), maxlag=5)
print(f"ARCH Test (5 lags):")
print(f"  LM Statistic: {arch_test[0]:.6f}")
print(f"  p-value: {arch_test[1]:.6f}")
print(f"  F-Statistic: {arch_test[2]:.6f}")
print(f"  F p-value: {arch_test[3]:.6f}")

# GARCH Model Estimation
print(f"\n=== GARCH MODEL ESTIMATION ===")

# Model combinations: p,q ∈ {1,2} for GARCH and EGARCH (8 models total)
garch_models = []
for p in [1, 2]:
    for q in [1, 2]:
        garch_models.append(('GARCH', p, q))
        garch_models.append(('EGARCH', p, q))

garch_results = {}

print(f"\n{'Model':<12} | {'Converged':<10} | {'AIC':<10} | {'BIC':<10} | {'LogLik':<10}")
print("-" * 65)

for model_type, p, q in garch_models:
    model_name = f"{model_type}({p},{q})"
    
    try:
        if model_type == 'GARCH':
            # Standard GARCH model
            model = arch_model(daily_returns.dropna(), vol='Garch', p=p, q=q, 
                             mean='Constant', dist='normal', rescale=False)
        else:  # EGARCH
            # EGARCH model  
            model = arch_model(daily_returns.dropna(), vol='EGARCH', p=p, q=q,
                             mean='Constant', dist='normal', rescale=False)
        
        # Fit model
        fitted = model.fit(disp='off', show_warning=False)
        
        # Extract results
        converged = fitted.convergence_flag == 0
        aic = fitted.aic
        bic = fitted.bic
        loglik = fitted.loglikelihood
        
        # Store results
        garch_results[model_name] = {
            'fitted_model': fitted,
            'converged': converged,
            'aic': aic,
            'bic': bic,
            'loglik': loglik,
            'params': fitted.params,
            'std_errors': fitted.std_err,
            'conditional_volatility': fitted.conditional_volatility
        }
        
        conv_status = "YES" if converged else "NO"
        print(f"{model_name:<12} | {conv_status:<10} | {aic:<10.2f} | {bic:<10.2f} | {loglik:<10.2f}")
        
    except Exception as e:
        print(f"{model_name:<12} | ERROR: {str(e)[:40]}")
        garch_results[model_name] = {'error': str(e)}

# Find best models
successful_garch = {k: v for k, v in garch_results.items() if 'error' not in v}

if successful_garch:
    aic_best_garch = min(successful_garch.keys(), key=lambda k: successful_garch[k]['aic'])
    bic_best_garch = min(successful_garch.keys(), key=lambda k: successful_garch[k]['bic'])
    
    print(f"\nBest GARCH by AIC: {aic_best_garch}")
    print(f"Best GARCH by BIC: {bic_best_garch}")

# GARCH Results Table
print(f"\n=== GARCH PARAMETER ESTIMATES ===")

if successful_garch:
    models = list(successful_garch.keys())
    
    # Get all parameter names
    all_params = set()
    for res in successful_garch.values():
        all_params.update(res['params'].index)
    
    print(f"\n{'Parameter':<15}", end="")
    for model in models:
        print(f" | {model:<12}", end="")
    print()
    print("-" * (15 + 15 * len(models)))
    
    # Parameter estimates
    for param in sorted(all_params):
        # Parameter values
        row = f"{param:<15}"
        for model in models:
            if param in successful_garch[model]['params']:
                val = successful_garch[model]['params'][param]
                row += f" | {val:>10.6f}  "
            else:
                row += f" | {'--':>10}    "
        print(row)
        
        # Standard errors  
        row = f"{'(SE)':<15}"
        for model in models:
            if param in successful_garch[model]['std_errors']:
                se = successful_garch[model]['std_errors'][param]
                row += f" | ({se:>8.6f}) "
            else:
                row += f" | {'--':>10}    "
        print(row)

# Model diagnostics
print(f"\n=== GARCH MODEL DIAGNOSTICS ===")

if successful_garch:
    print(f"\n{'Diagnostic':<15}", end="")
    for model in models:
        print(f" | {model:<12}", end="")
    print()
    print("-" * (15 + 15 * len(models)))
    
    diagnostics = [
        ('Converged', 'converged', 's'),
        ('AIC', 'aic', '.2f'),
        ('BIC', 'bic', '.2f'),
        ('Log-Likelihood', 'loglik', '.2f')
    ]
    
    for diag_name, key, fmt in diagnostics:
        row = f"{diag_name:<15}"
        for model in models:
            val = successful_garch[model][key]
            if fmt == 's':
                status = "YES" if val else "NO" 
                row += f" | {status:>10}    "
            else:
                row += f" | {val:>10{fmt}}  "
        print(row)

# Calculate Realized Volatility from intraday returns
print(f"\n=== REALIZED VOLATILITY CALCULATION ===")

# Use cleaned minute returns for RV calculation
intraday_returns = returns_cleaned.copy()

# Calculate daily realized volatility (sum of squared intraday returns)
daily_rv = intraday_returns.groupby(intraday_returns.index.date).apply(lambda x: (x**2).sum())
daily_rv.index = pd.to_datetime(daily_rv.index)
daily_rv = daily_rv.dropna()

print(f"Realized Volatility series: {len(daily_rv)} observations")
print(f"RV range: {daily_rv.min():.6f} to {daily_rv.max():.6f}")
print(f"RV mean: {daily_rv.mean():.6f}")

# Align dates for comparison (common dates only)
if successful_garch:
    # Get best GARCH and EGARCH models
    best_garch_name = aic_best_garch
    best_egarch_name = None
    
    # Find best EGARCH
    egarch_models = {k: v for k, v in successful_garch.items() if k.startswith('EGARCH')}
    if egarch_models:
        best_egarch_name = min(egarch_models.keys(), key=lambda k: egarch_models[k]['aic'])
    
    print(f"\nBest GARCH model: {best_garch_name}")
    if best_egarch_name:
        print(f"Best EGARCH model: {best_egarch_name}")
    
    # Extract conditional volatilities (convert to variance by squaring)
    garch_vol = successful_garch[best_garch_name]['conditional_volatility']
    garch_var = garch_vol ** 2
    
    if best_egarch_name:
        egarch_vol = successful_garch[best_egarch_name]['conditional_volatility'] 
        egarch_var = egarch_vol ** 2
    
    # Align all series to common dates
    common_dates = daily_rv.index.intersection(garch_var.index)
    
    if len(common_dates) > 0:
        rv_aligned = daily_rv.loc[common_dates]
        garch_aligned = garch_var.loc[common_dates] 
        
        print(f"\nCommon observations for comparison: {len(common_dates)}")
        print(f"Date range: {common_dates.min().date()} to {common_dates.max().date()}")
        
        # Summary statistics comparison
        print(f"\nVariance Comparison:")
        print(f"  Realized Variance:  Mean={rv_aligned.mean():.6f}, Std={rv_aligned.std():.6f}")
        print(f"  GARCH Variance:     Mean={garch_aligned.mean():.6f}, Std={garch_aligned.std():.6f}")
        
        if best_egarch_name:
            egarch_aligned = egarch_var.loc[common_dates]
            print(f"  EGARCH Variance:    Mean={egarch_aligned.mean():.6f}, Std={egarch_aligned.std():.6f}")

print(f"\n" + "="*60)
print("GARCH ANALYSIS COMPLETE")
print("="*60)

# 1. PLOT GARCH vs EGARCH vs RV
print(f"\n=== VOLATILITY COMPARISON PLOTS ===")

if successful_garch and len(common_dates) > 0:
    # Get data for plotting
    rv_plot = daily_rv.loc[common_dates]
    garch_plot = successful_garch[best_garch_name]['conditional_volatility'].loc[common_dates] ** 2
    egarch_plot = successful_garch[best_egarch_name]['conditional_volatility'].loc[common_dates] ** 2
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top panel: Time series
    ax1.plot(common_dates, rv_plot.values, label='Realized Variance', alpha=0.7, linewidth=1)
    ax1.plot(common_dates, garch_plot.values, label=f'{best_garch_name} Variance', alpha=0.8, linewidth=1.2)
    ax1.plot(common_dates, egarch_plot.values, label=f'{best_egarch_name} Variance', alpha=0.8, linewidth=1.2)
    ax1.set_title('Variance Comparison: GARCH vs EGARCH vs Realized Variance')
    ax1.set_ylabel('Variance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Scatter plot RV vs GARCH
    ax2.scatter(rv_plot.values, garch_plot.values, alpha=0.6, s=20, label=f'{best_garch_name}', color='red')
    ax2.scatter(rv_plot.values, egarch_plot.values, alpha=0.6, s=20, label=f'{best_egarch_name}', color='blue')
    
    # 45-degree line (perfect fit)
    min_val = min(rv_plot.min(), garch_plot.min(), egarch_plot.min())
    max_val = max(rv_plot.max(), garch_plot.max(), egarch_plot.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Fit')
    
    ax2.set_xlabel('Realized Variance')
    ax2.set_ylabel('Model Variance')
    ax2.set_title('Model vs Realized Variance Scatter')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate correlations
    corr_garch_rv = rv_plot.corr(garch_plot)
    corr_egarch_rv = rv_plot.corr(egarch_plot)
    
    print(f"Correlations with Realized Variance:")
    print(f"  {best_garch_name}: {corr_garch_rv:.4f}")
    print(f"  {best_egarch_name}: {corr_egarch_rv:.4f}")

