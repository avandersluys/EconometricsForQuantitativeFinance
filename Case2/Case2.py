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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import stats
from scipy.stats import norm
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

path = "/Users/alexandervds/Documents/GitHub/EconometricsForQuantitativeFinance/Case2/sp_9.csv"
cols = ["SPX5.L","SPY5z.CHIX","SPY5.P"]
session_start, session_end = "09:00", "17:30"
p_assign = 5
p_min, p_max = 1, 15
nlags = 10
split_date = pd.Timestamp("2025-01-01")
COINTEG_FREQ = "10min"

df = pd.read_csv(path, parse_dates=["DateTime"]).set_index("DateTime").sort_index()
prices = df[cols].astype(float).where(lambda x: x > 0).dropna(how="any")
prices = prices.between_time(session_start, session_end)

def intraday_logret(g):
    r = np.log(g).diff()
    r.iloc[0] = np.nan
    return r

rets = prices.groupby(prices.index.date, group_keys=False).apply(intraday_logret)
rets = rets.replace([np.inf, -np.inf], np.nan).dropna(how="any").astype(float)

pos_in_day = rets.groupby(rets.index.date).cumcount()
p_max_block = p_max
rets_fs = rets.mask(pos_in_day < p_max_block).dropna(how="any").copy()

rets_train = rets_fs.loc[rets_fs.index < split_date]
rets_test  = rets_fs.loc[rets_fs.index >= split_date]

def ic_select(data, p_min=1, p_max=15, trend="c"):
    m = VAR(data)
    out = []
    for p in range(p_min, p_max + 1):
        res = m.fit(p, trend=trend)
        out.append({"p": p, "AIC": res.aic, "BIC": res.bic, "HQIC": res.hqic, "nobs": int(res.nobs)})
    df_ic = pd.DataFrame(out).sort_values("p")
    p_bic = int(df_ic.loc[df_ic["BIC"].idxmin(), "p"])
    return df_ic, p_bic

ic_train, p_train = ic_select(rets_train, p_min, p_max)
print("\n[3.3.1] TRAIN (2024) Information criteria:")
print(ic_train)
print(f"[3.3.1] TRAIN (2024) BIC-selected p: {p_train}")

if len(rets_test) > p_max:
    ic_test, p_test = ic_select(rets_test, p_min, p_max)
    print("\n[3.3.1] TEST (2025) Information criteria (info only):")
    print(ic_test)
    print(f"[3.3.1] TEST (2025) BIC-selected p (info only): {p_test}")

model_fs = VAR(rets_fs)
res = model_fs.fit(p_assign, trend="c")
print("\n[3.3.2] VAR(p=5) summary (statsmodels):")
print(res.summary())

logP_thin = np.log(prices.resample(COINTEG_FREQ).last()).dropna(how="any")
k_ar_diff = p_assign - 1

cj = coint_johansen(logP_thin, det_order=0, k_ar_diff=k_ar_diff)
johansen_report = pd.DataFrame({
    "eigenvalue": cj.eig,
    "trace_stat": cj.lr1,
    "trace_crit_90": cj.cvt[:,0],
    "trace_crit_95": cj.cvt[:,1],
    "trace_crit_99": cj.cvt[:,2],
})
print(f"\n[3.3.2b] Johansen trace test on {COINTEG_FREQ} log-prices (det_order=0, k_ar_diff={k_ar_diff}):")
print(johansen_report)

k = logP_thin.shape[1]
r_raw = int((cj.lr1 > cj.cvt[:,1]).sum())
r = min(r_raw, k - 1)
print(f"[3.3.2b] Estimated cointegration rank at 95% (capped at k-1): r = {r} (raw count = {r_raw})")

if r > 0:
    vecm = VECM(logP_thin, k_ar_diff=k_ar_diff, coint_rank=r, deterministic="co")
    vecm_res = vecm.fit()
    print("\n[3.3.2b] VECM summary (thinned series):")
    print(vecm_res.summary())
    alpha = pd.DataFrame(vecm_res.alpha, index=cols, columns=[f"alpha_{i+1}" for i in range(r)])
    beta  = pd.DataFrame(vecm_res.beta,  index=cols, columns=[f"beta_{i+1}"  for i in range(r)])
    print("\n[3.3.2b] Alpha (error-correction speeds):")
    print(alpha)
    print("\n[3.3.2b] Beta (cointegrating vectors):")
    print(beta)
else:
    print("\n[3.3.2b] Johansen rank r = 0 at 95%; VECM not estimated.")

u = pd.DataFrame(res.resid, index=rets_fs.index, columns=rets_fs.columns)

print("\n[3.3.3] Residual diagnostics:")
print(f"\nVAR whiteness test (all residuals, nlags={nlags}):")
print(res.test_whiteness(nlags=nlags))

print(f"\nLjung–Box p-values per equation (lags={nlags}):")
for c in u.columns:
    pval = acorr_ljungbox(u[c].dropna(), lags=[nlags], return_df=True)["lb_pvalue"].iloc[0]
    print(f"{c}: {pval:.4g}")

print("\nJarque–Bera normality p-values per equation:")
for c in u.columns:
    jb = stats.jarque_bera(u[c].dropna())
    print(f"{c}: {jb.pvalue:.4g}")

def corr_pval(x, y):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    r = np.corrcoef(x, y)[0,1]
    n = len(x)
    t = r * np.sqrt((n - 2) / max(1e-12, (1 - r**2)))
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 2))
    return r, p, n

print("\nContemporaneous residual correlations (H0: rho=0):")
for a, b in combinations(u.columns, 2):
    r, p, n = corr_pval(u[a], u[b])
    approx_zero = ("yes" if (abs(r) < 0.05 and p > 0.1) else "no")
    print(f"{a} vs {b}: r={r:+.3f}, p={p:.4g}, n={n}, approx_zero? {approx_zero}")

fig, axes = plt.subplots(len(u.columns), 1, figsize=(10, 6), sharex=True)
if len(u.columns) == 1:
    axes = [axes]
for i, c in enumerate(u.columns):
    axes[i].plot(u.index, u[c].values)
    axes[i].axhline(0.0, linestyle="--", linewidth=1)
    axes[i].set_ylabel(c)
axes[-1].set_xlabel("Time")
fig.suptitle("VAR Residuals (per series) — full sample")
fig.tight_layout()
plt.show()

pairs = list(combinations(u.columns, 2))
fig, axes = plt.subplots(1, len(pairs), figsize=(15, 4))
for ax, (a, b) in zip(axes, pairs):
    x = u[a].values; y = u[b].values
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    r, p, n = corr_pval(x, y)
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    yy = coef[0]*xx + coef[1]
    ax.scatter(x, y, s=6, alpha=0.6)
    ax.plot(xx, yy, linewidth=1)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_title(f"{a} vs {b}\nr={r:+.3f}, p={p:.3g}")
    ax.set_xlabel(a); ax.set_ylabel(b)
fig.suptitle("Pairwise residual scatter plots (all 3 combinations) — full sample")
fig.tight_layout()
plt.show()

u_2024 = u.loc[u.index < split_date]
u_2025 = u.loc[u.index >= split_date]

def plot_pair_scatter(u_period, title_prefix):
    pairs = list(combinations(u_period.columns, 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(15, 4))
    for ax, (a, b) in zip(axes, pairs):
        x = u_period[a].to_numpy(); y = u_period[b].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]; y = y[m]
        r = np.corrcoef(x, y)[0,1]
        coef = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 200)
        yy = coef[0]*xx + coef[1]
        ax.scatter(x, y, s=6, alpha=0.6)
        ax.plot(xx, yy, linewidth=1)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"{a} vs {b}\nr={r:+.3f}")
        ax.set_xlabel(a); ax.set_ylabel(b)
    fig.suptitle(f"{title_prefix}: pairwise residual scatters")
    fig.tight_layout()
    plt.show()

plot_pair_scatter(u_2024, "2024 (train)")
plot_pair_scatter(u_2025, "2025 (test)")

print("\nResidual correlation matrix — 2024 (train):")
print(u_2024.corr())
print("\nResidual correlation matrix — 2025 (test):")
print(u_2025.corr())

def fisher_change_test(x1, y1, x2, y2):
    def _zbits(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        r = np.corrcoef(x[m], y[m])[0,1]
        n = m.sum()
        z = np.arctanh(np.clip(r, -0.999999, 0.999999))
        se = 1/np.sqrt(max(1, n-3))
        return r, z, se, n
    r1, z1, se1, n1 = _zbits(x1, y1)
    r2, z2, se2, n2 = _zbits(x2, y2)
    zdiff = (z1 - z2) / np.sqrt(se1**2 + se2**2)
    p = 2 * (1 - stats.norm.cdf(abs(zdiff)))
    return r1, r2, zdiff, p, n1, n2

print("\nChange in contemporaneous residual correlation (2024 vs 2025):")
for a, b in combinations(u.columns, 2):
    r1, r2, z, p, n1, n2 = fisher_change_test(u_2024[a].values, u_2024[b].values,
                                              u_2025[a].values, u_2025[b].values)
    verdict = "different" if p < 0.05 else "no clear change"
    print(f"{a} vs {b}: r_2024={r1:+.3f} (n={n1}), r_2025={r2:+.3f} (n={n2}), z={z:+.2f}, p={p:.3g} → {verdict}")


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


