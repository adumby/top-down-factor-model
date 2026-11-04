"""Per-asset macro factor regressions.

For each holding i (excess of cash):   r_i,t - rf_t = a_i + b_i' f_t + e_i,t
estimated by OLS with Newey-West (HAC) standard errors. Cash has zero betas by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

from config import CASH


@dataclass
class FactorModel:
    keys: list[str]
    betas: pd.DataFrame      # tickers x factors (NaN where not estimable)
    tstats: pd.DataFrame
    se: pd.DataFrame
    alpha: pd.Series         # monthly %, in excess of rf
    r2: pd.Series
    resid_vol: pd.Series     # monthly %
    nobs: pd.Series
    start: pd.Series         # first month used per asset
    resid: pd.DataFrame      # residuals, month x tickers
    factor_cov: pd.DataFrame
    factor_mean: pd.Series

    @property
    def betas0(self) -> pd.DataFrame:
        """Betas with unestimable rows treated as zero (flagged separately in diagnostics)."""
        return self.betas.fillna(0.0)


def fit_models(returns: pd.DataFrame, rf: pd.Series, factors: pd.DataFrame, keys: list[str],
               hac_lags: int = 3, min_obs: int = 36) -> FactorModel:
    keys = list(keys)
    X_all = factors[keys]
    tick = list(returns.columns)
    betas = pd.DataFrame(np.nan, index=tick, columns=keys)
    tstats, se = betas.copy(), betas.copy()
    alpha = pd.Series(np.nan, index=tick)
    r2 = alpha.copy(); rvol = alpha.copy(); nobs = pd.Series(0, index=tick, dtype=int)
    start = pd.Series([pd.NaT] * len(tick), index=tick, dtype=object)
    resid = pd.DataFrame(np.nan, index=returns.index, columns=tick)

    for t in tick:
        if t == CASH:
            betas.loc[t] = 0.0; tstats.loc[t] = 0.0; se.loc[t] = 0.0
            alpha[t] = 0.0; r2[t] = 0.0; rvol[t] = 0.0; nobs[t] = int(rf.notna().sum())
            resid[t] = 0.0
            continue
        y = (returns[t] - rf).rename("y")
        df = pd.concat([y, X_all], axis=1).dropna()
        if len(df) < max(min_obs, len(keys) + 10):
            nobs[t] = len(df)
            continue
        X = sm.add_constant(df[keys])
        fit = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
        betas.loc[t] = fit.params[keys]
        tstats.loc[t] = fit.tvalues[keys]
        se.loc[t] = fit.bse[keys]
        alpha[t] = fit.params["const"]
        r2[t] = fit.rsquared
        rvol[t] = float(np.std(fit.resid, ddof=len(keys) + 1))
        nobs[t] = len(df)
        start[t] = df.index[0]
        resid.loc[df.index, t] = fit.resid

    Xc = X_all.dropna()
    return FactorModel(keys, betas, tstats, se, alpha, r2, rvol, nobs, start, resid,
                       Xc.cov(), Xc.mean())


def factor_diagnostics(factors: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean, stdev, ADF p-value (stationarity) and VIF (collinearity) per factor."""
    X = factors[keys].dropna()
    rows = {}
    Xc = sm.add_constant(X)
    for i, k in enumerate(keys):
        vif = variance_inflation_factor(Xc.values, i + 1) if len(keys) > 1 else np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = adfuller(X[k], autolag="AIC")[1]
        except Exception:
            p = np.nan
        rows[k] = {"mean": X[k].mean(), "stdev": X[k].std(), "ADF p-value": p, "VIF": vif, "months": len(X)}
    return pd.DataFrame(rows).T


def realized_exposures(excess: pd.Series, factors: pd.DataFrame, keys: list[str], hac_lags: int = 3):
    """Direct regression of a (portfolio) excess-return series on the factors."""
    df = pd.concat([excess.rename("y"), factors[keys]], axis=1).dropna()
    if len(df) < len(keys) + 12:
        return None
    fit = sm.OLS(df["y"], sm.add_constant(df[keys])).fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
    return fit
