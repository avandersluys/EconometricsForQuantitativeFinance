# Econometrics Case 1

# Libraries
import pandas as pd
import numpy as np 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from scipy.stats import chi2  # for hausman test
from linearmodels.panel import RandomEffects
from linearmodels.panel import PanelOLS

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"linearmodels\.panel\..*"
)

# Base URL for raw GitHub repo
BASE_URL = "https://raw.githubusercontent.com/avandersluys/EconometricsForQuantitativeFinance/main"

# Loading datasets directly from GitHub
df_ff = pd.read_csv(f"{BASE_URL}/capmff_2010-2025_ff.csv")
for c in ["Mkt-RF", "SMB", "HML", "RF"]:
    df_ff[c] = pd.to_numeric(df_ff[c], errors="coerce") / 100.0

df_sector = pd.read_csv(f"{BASE_URL}/capmff_2010-2025_sector.csv")
df_prices = pd.read_csv(f"{BASE_URL}/capmff_2010-2025_prices.csv")

# Reformatting the dataframe to multiindex panel style
prices_long = (
    df_prices
    .assign(Date=pd.to_datetime(df_prices['Date']))
    .melt(id_vars='Date', var_name='Ticker', value_name='Price')
    .dropna(subset=['Price'])
    .sort_values(['Ticker','Date'])
    .reset_index(drop=True)
)

# MultiIndex
prices_panel = prices_long.set_index(['Ticker','Date']).sort_index()

print(prices_panel.shape)

industries_of_interest = ['Basic Materials', 'Communication Services', 'Consumer Cyclical', 'Consumer Defensive']
tickersOfInterest = df_sector.loc[df_sector['sector'].isin(industries_of_interest), 'Ticker'].tolist()

prices_sel = prices_panel[prices_panel.index.get_level_values('Ticker').isin(tickersOfInterest)]

df_returns = prices_sel.pct_change(fill_method=None).dropna()

df_ff['Date'] = pd.to_datetime(df_ff['Date'])
rf = df_ff.set_index('Date')['RF']          

returns_df = (
    prices_sel
    .groupby(level='Ticker')           
    .pct_change(fill_method=None)
    .dropna()
    .rename(columns={'Price': 'Return'})
)

# Compute excess returns
df_ff['Date'] = pd.to_datetime(df_ff['Date'])
rf = df_ff.set_index('Date')['RF']

returns_df = returns_df.copy()
returns_df['RF'] = returns_df.index.get_level_values('Date').map(rf)
returns_df = returns_df.dropna(subset=['RF'])
returns_df['ExcessReturn'] = returns_df['Return'] - returns_df['RF']

ff_factors = df_ff[['Date', 'Mkt-RF', 'SMB', 'HML']].copy()
ff_factors['Date'] = pd.to_datetime(ff_factors['Date'])

returns_df = (returns_df.reset_index()
              .merge(ff_factors, on='Date', how='left')
              .set_index(['Ticker','Date'])
              .dropna(subset=['Mkt-RF','SMB','HML']))  # keep only rows with factors

panel_data_with_sector = returns_df.reset_index().merge(
    df_sector[['Ticker', 'sector']],
    on='Ticker',
    how='left'
)

panel_data_filtered = panel_data_with_sector.set_index(['Ticker', 'Date'])

# Filter to sectors of interest
panel_data_filtered = panel_data_filtered[
    panel_data_filtered['sector'].isin(industries_of_interest)
]

print("\nFinal Panel Data Info:")
print(f"Shape: {panel_data_filtered.shape}")
print(f"Unique Tickers: {panel_data_filtered.index.get_level_values('Ticker').nunique()}")
print(f"Unique Dates: {panel_data_filtered.index.get_level_values('Date').nunique()}")
print(f"Sectors: {sorted(panel_data_filtered['sector'].unique())}")
print(panel_data_filtered.head(10))

# 5. Model Estimation 


# Model 1: Common Beta 
print("")
print("Estimating Common Beta Model:")
# Entity fixed effects for α_i and common β (no extra intercept; absorbed by EntityEffects)
mod1_common = PanelOLS.from_formula(
    'ExcessReturn ~ EntityEffects + `Mkt-RF` + SMB + HML',
    data=panel_data_filtered
)
# Clustered (by entity) SEs
res1_common = mod1_common.fit(cov_type='clustered', cluster_entity=True)
print(res1_common)

# Hausman Test
print("\nHausman Test: Fixed Effects vs Random Effects:")

# Estimate RE model
print("Estimating Random Effects Model:")
mod1_re = RandomEffects.from_formula(
    'ExcessReturn ~ 1 + `Mkt-RF` + SMB + HML',
    data=panel_data_filtered
)
# Clustered (by entity) SEs
res1_re = mod1_re.fit(cov_type='clustered', cluster_entity=True)

# Estimate Fixed Effects model for comparison (already have res1_common)
print("Comparing Fixed Effects vs Random Effects:")

# Manual Hausman Test Implementation
def hausman_test(fe_results, re_results):
    """
    Hausman test FE vs RE.
    H0: RE is consistent (no correlation between effects and regressors).
    Uses V = Var(FE) - Var(RE) and np.linalg.pinv for stability.
    """
    keep = ['Mkt-RF', 'SMB', 'HML']  # same regressors in both models

    b_fe = fe_results.params[keep].values
    b_re = re_results.params[keep].values

    V_fe = fe_results.cov.loc[keep, keep].values
    V_re = re_results.cov.loc[keep, keep].values

    diff = b_fe - b_re
    V = V_fe - V_re  # FE minus RE (as in slides)

    stat = float(diff.T @ np.linalg.pinv(V) @ diff)
    df = diff.size
    p = 1 - chi2.cdf(stat, df)
    return stat, p, df

# Perform Hausman test
hausman_stat, hausman_p, hausman_df = hausman_test(res1_common, res1_re)

print(f"\n--- Hausman Test Results ---")
print(f"Test Statistic: {hausman_stat:.4f}")
print(f"Degrees of Freedom: {hausman_df}")
print(f"P-value: {hausman_p:.4f}")

print("Model 2: Sector-Specific Betas")

mod2_sector = PanelOLS.from_formula(
    'ExcessReturn ~ EntityEffects + `Mkt-RF`:C(sector) + SMB:C(sector) + HML:C(sector)',
    data=panel_data_filtered,
    drop_absorbed=True 
)
# Clustered (by entity) SEs
res2_sector = mod2_sector.fit(cov_type='clustered', cluster_entity=True)
print(res2_sector)

# Difference of Betas Across Sectors
print("Wald test for difference of Betas")

# Build and test linear contrasts: for each factor, test equality across sectors vs a base sector
def wald_equal_across_sectors(results, factor_name, sectors, base_sector):
    """
    H0: beta_factor[sector] - beta_factor[base_sector] = 0 for all sectors != base
    Uses Moore-Penrose pseudo-inverse for stability.
    """
    param_names = results.params.index.to_list()
    term_prefix = f"{factor_name}:C(sector)"
    keys = {s: f"{term_prefix}[{s}]" for s in sectors if f"{term_prefix}[{s}]" in param_names}
    if base_sector not in keys or len(keys) <= 1:
        return np.nan, np.nan, 0

    p = len(param_names)
    rows = []
    for s, k in keys.items():
        if s == base_sector:
            continue
        r = np.zeros(p)
        r[param_names.index(k)] = 1.0
        r[param_names.index(keys[base_sector])] = -1.0
        rows.append(r)

    if not rows:
        return np.nan, np.nan, 0

    R = np.vstack(rows)
    beta_hat = results.params.values
    cov_beta = results.cov.values
    diff = R @ beta_hat
    var_diff = R @ cov_beta @ R.T
    stat = float(diff.T @ np.linalg.pinv(var_diff) @ diff)
    df = R.shape[0]
    p_value = 1 - chi2.cdf(stat, df)
    return stat, p_value, df

sectors = ['Basic Materials', 'Communication Services', 'Consumer Cyclical', 'Consumer Defensive']
base_sector = 'Basic Materials'

# Per-factor tests
stat_mkt, pval_mkt, df_mkt = wald_equal_across_sectors(res2_sector, 'Mkt-RF', sectors, base_sector)
if df_mkt > 0:
    print(f"Mkt-RF sector betas equality (vs {base_sector}): Wald stat = {stat_mkt:.4f}, df = {df_mkt}, p-value = {pval_mkt:.4f}")

stat_smb, pval_smb, df_smb = wald_equal_across_sectors(res2_sector, 'SMB', sectors, base_sector)
if df_smb > 0:
    print(f"SMB sector betas equality (vs {base_sector}): Wald stat = {stat_smb:.4f}, df = {df_smb}, p-value = {pval_smb:.4f}")

stat_hml, pval_hml, df_hml = wald_equal_across_sectors(res2_sector, 'HML', sectors, base_sector)
if df_hml > 0:
    print(f"HML sector betas equality (vs {base_sector}): Wald stat = {stat_hml:.4f}, df = {df_hml}, p-value = {pval_hml:.4f}")

# Overall joint test for all sector-specific betas together (all factors, all sectors vs base)
def wald_overall(results, factors, sectors, base_sector):
    param_names = results.params.index.to_list()
    p = len(param_names)
    rows = []
    for f in factors:
        term_prefix = f"{f}:C(sector)"
        keys = {s: f"{term_prefix}[{s}]" for s in sectors if f"{term_prefix}[{s}]" in param_names}
        if base_sector not in keys or len(keys) <= 1:
            continue
        for s, k in keys.items():
            if s == base_sector:
                continue
            r = np.zeros(p)
            r[param_names.index(k)] = 1.0
            r[param_names.index(keys[base_sector])] = -1.0
            rows.append(r)
    if not rows:
        return np.nan, np.nan, 0
    R = np.vstack(rows)
    beta_hat = results.params.values
    cov_beta = results.cov.values
    diff = R @ beta_hat
    var_diff = R @ cov_beta @ R.T
    stat = float(diff.T @ np.linalg.pinv(var_diff) @ diff)
    df = R.shape[0]
    p_value = 1 - chi2.cdf(stat, df)
    return stat, p_value, df

stat_all, pval_all, df_all = wald_overall(res2_sector, ['Mkt-RF','SMB','HML'], sectors, base_sector)
if df_all > 0:
    print(f"Overall sector betas equality (all factors, vs {base_sector}): Wald stat = {stat_all:.4f}, df = {df_all}, p-value = {pval_all:.4f}")

    # Interpretation
    alpha = 0.05
    print("\n--- Overall Test Conclusion ---")
    if pval_all < alpha:
        print(f"Reject H0: Betas differ across sectors (p-value = {pval_all:.4f} < {alpha})")
        print("Model 2 (sector-specific betas) preferred.")
    else:
        print(f"Fail to reject H0: No strong evidence of beta differences (p-value = {pval_all:.4f} >= {alpha})")
        print("Model 1 (common beta) might suffice.")

print("Sector beta coefficients with 95% CIs:")
params = res2_sector.params
conf_int = res2_sector.conf_int()
sectors = ['Basic Materials', 'Communication Services', 'Consumer Cyclical', 'Consumer Defensive']

for factor in ['Mkt-RF', 'SMB', 'HML']:
    print(f"\n{factor} Sector Betas:")
    for sector in sectors:
        key = f"{factor}:C(sector)[{sector}]"
        if key in params:
            est = params[key]
            lower, upper = conf_int.loc[key]
            print(f"{sector}: Est={est:.5f}, 95% CI=({lower:.5f}, {upper:.5f})")
