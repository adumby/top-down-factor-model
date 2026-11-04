"""Portfolio maths: weights, backtest, attribution, stress tests, Monte Carlo."""
from __future__ import annotations

import numpy as np
import pandas as pd

from model import FactorModel


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
def to_fractions(w_pct: pd.Series) -> pd.Series:
    """Percent weights -> fractions summing to 1 (negative values clipped to 0)."""
    w = w_pct.clip(lower=0.0)
    s = w.sum()
    return w / s if s > 0 else w


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
def portfolio_returns(returns: pd.DataFrame, w: pd.Series, min_cover: float = 0.8):
    """Monthly-rebalanced portfolio return (%). If some holdings have no data yet, the remaining
    weights are re-scaled; months with < `min_cover` of the target weight available are dropped.
    Returns (portfolio return, effective weights by month)."""
    w = w[w > 0]
    R = returns[w.index]
    mask = R.notna()
    raw = mask.mul(w, axis=1)
    cover = raw.sum(axis=1) / w.sum()
    W = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0)
    rp = (R.fillna(0.0) * W.fillna(0.0)).sum(axis=1).where(cover >= min_cover)
    W = W.where(cover >= min_cover)
    return rp.rename("portfolio"), W


def performance_stats(r: pd.Series, rf: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 12:
        return {}
    growth = (1 + r / 100).cumprod()
    n = len(r)
    ann_ret = growth.iloc[-1] ** (12 / n) - 1
    vol = r.std() * np.sqrt(12) / 100
    ex = (r - rf.reindex(r.index)).mean() * 12 / 100
    dd = (growth / growth.cummax() - 1).min()
    return {"Ann. return": ann_ret, "Ann. volatility": vol,
            "Sharpe (vs T-bills)": ex / vol if vol > 0 else np.nan, "Max drawdown": dd,
            "Months": n}


def attribution(model: FactorModel, W: pd.DataFrame, factors: pd.DataFrame,
                rp: pd.Series, rf: pd.Series) -> pd.DataFrame:
    """Monthly contribution (% pts) of each factor to the portfolio's return, using full-sample
    betas, plus 'rf' and 'Alpha / unexplained' (risk premia + idiosyncratic)."""
    B = model.betas0.reindex(W.columns).fillna(0.0)
    f = factors[model.keys]
    port_beta_t = W.fillna(0.0) @ B                   # month x factor (effective exposure)
    contrib = port_beta_t * f
    idx = rp.dropna().index.intersection(contrib.dropna().index)
    contrib = contrib.loc[idx]
    out = contrib.copy()
    out["Risk-free"] = rf.reindex(idx)
    out["Alpha / unexplained"] = rp.loc[idx] - contrib.sum(axis=1) - out["Risk-free"]
    return out


def rolling_ols(y: pd.Series, X: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(df) < window:
        return pd.DataFrame(columns=X.columns)
    Y = df["y"].to_numpy()
    Xm = np.column_stack([np.ones(len(df)), df[list(X.columns)].to_numpy()])
    out = np.full((len(df), X.shape[1]), np.nan)
    for t in range(window, len(df) + 1):
        out[t - 1] = np.linalg.lstsq(Xm[t - window:t], Y[t - window:t], rcond=None)[0][1:]
    return pd.DataFrame(out, index=df.index, columns=X.columns).dropna(how="all")


# --------------------------------------------------------------------------- #
# Stress test (instantaneous factor shock)
# --------------------------------------------------------------------------- #
def stress(betas: pd.DataFrame, w: pd.Series, shock: pd.Series) -> dict:
    """Impact (% of portfolio) = sum_i w_i * b_i . shock.   `betas` must be tickers x factors."""
    b = betas.reindex(w.index).fillna(0.0)
    m = b.mul(shock.reindex(b.columns).fillna(0.0), axis=1).mul(w, axis=0) * 1.0
    return {"total": float(m.values.sum()), "by_asset": m.sum(axis=1), "by_factor": m.sum(axis=0), "matrix": m}


def portfolio_betas(betas: pd.DataFrame, w: pd.Series) -> pd.Series:
    return betas.reindex(w.index).fillna(0.0).mul(w, axis=0).sum()


def risk_decomposition(betas: pd.DataFrame, w: pd.Series, factor_cov: pd.DataFrame,
                       resid: pd.DataFrame) -> pd.Series:
    """Share of ex-ante monthly portfolio variance from each factor and from idiosyncratic risk."""
    s = portfolio_betas(betas, w)
    Sf = factor_cov.loc[s.index, s.index]
    comp = s * (Sf @ s)                                   # contributions sum to s' Sf s
    wr = w.reindex(resid.columns).fillna(0.0)
    Se = nearest_psd(resid.cov().fillna(0.0).values)
    idio = float(wr.values @ Se @ wr.values)
    tot = comp.sum() + idio
    out = pd.concat([comp, pd.Series({"Idiosyncratic": idio})]) / tot
    return out


# --------------------------------------------------------------------------- #
# Monte Carlo forecast
# --------------------------------------------------------------------------- #
def nearest_psd(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    a = (a + a.T) / 2
    vals, vecs = np.linalg.eigh(a)
    return (vecs * np.clip(vals, eps, None)) @ vecs.T


def simulate(model: FactorModel, weights: dict[str, pd.Series], rf_now: float, horizon: int,
             n: int, mu_total: pd.Series, vol_mult: float, alpha_keep: float, seed: int = 0) -> dict:
    """Simulate monthly factor changes ~ N(mu_total/horizon, vol_mult^2 * hist cov) and idiosyncratic
    shocks ~ N(0, hist residual cov). Same random draws are used for every weight set, so
    differences between allocations are not Monte Carlo noise."""
    rng = np.random.default_rng(seed)
    keys = model.keys
    tick = list(model.betas.index)
    B = model.betas0.loc[tick, keys].to_numpy()                    # A x K
    alpha = model.alpha.loc[tick].fillna(0.0).to_numpy() * alpha_keep
    Sf = nearest_psd(model.factor_cov.loc[keys, keys].to_numpy()) * vol_mult ** 2
    Se = nearest_psd(model.resid.cov().fillna(0.0).loc[tick, tick].to_numpy())
    mu = (mu_total.reindex(keys).fillna(0.0) / horizon).to_numpy()

    f = rng.multivariate_normal(mu, Sf, size=(n, horizon))        # n x H x K
    e = rng.multivariate_normal(np.zeros(len(tick)), Se, size=(n, horizon))   # n x H x A
    is_cash = np.array([t == "CASH" for t in tick])
    r = rf_now + alpha + f @ B.T + e                              # asset total returns (%)
    r[..., is_cash] = rf_now

    res = {}
    for name, w in weights.items():
        wv = w.reindex(tick).fillna(0.0).to_numpy()
        rp = r @ wv                                               # n x H (%)
        growth = np.cumprod(1 + rp / 100, axis=1)
        res[name] = growth
    return res


def sim_stats(growth: np.ndarray, horizon: int) -> dict:
    term = (growth[:, -1] - 1) * 100
    q = lambda p: float(np.percentile(term, p))
    k = max(1, int(0.05 * len(term)))
    worst = np.sort(term)[:k]
    return {"Mean": float(term.mean()), "Median": q(50), "Std dev": float(term.std()),
            "5th pct": q(5), "95th pct": q(95), "CVaR 5% (avg worst 5%)": float(worst.mean()),
            "P(loss)": float((term < 0).mean() * 100)}
