"""Oil -> inflation pass-through, with a 'net oil price increase' (Hamilton) term.

Why: a rise in oil back to a level it has already been at recently tends to hit inflation less than a
rise to a fresh high. The model lets the data say how much less.

Monthly variables (all in %, log-based):
  o_t   = 100 * ln(Brent, monthly average)
  d_t   = o_t - o_{t-1}                              every oil move, up or down
  N_t   = max(0, o_t - max(o_{t-1}, ..., o_{t-12}))  only moves to a NEW 12-month high
  pi_t  = 100 * ln(P_t / P_{t-1})                    monthly inflation (headline or core)

Core:      pi^c_t = c + rho * avg(pi^c_{t-1..t-3}) + b1*sum d_{t-1..t-6} + b2*sum d_{t-7..t-12}
                      + g1*sum N_{t..t-6} + g2*sum N_{t-7..t-12} + e
Headline:  pi^h_t = c + k * avg(pi^c_{t-1..t-3}) + a0*d_t + a1*d_{t-1} + a2*d_{t-2} + a3*sum d_{t-3..t-12}
                      + h0*N_t + h1*N_{t-1} + h2*N_{t-2} + e

A move back to an old level only gets the 'd' terms. A move to a new high gets 'd' + 'N' terms.
Estimated by OLS with Newey-West standard errors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def oil_terms(brent: pd.Series) -> pd.DataFrame:
    o = 100.0 * np.log(brent.astype(float))
    prev_max = o.shift(1).rolling(12, min_periods=12).max()
    return pd.DataFrame({"o": o, "d": o.diff(), "N": (o - prev_max).clip(lower=0.0)})


def mom(level: pd.Series) -> pd.Series:
    return 100.0 * np.log(level.astype(float)).diff()


def _lagsum(s: pd.Series, a: int, b: int) -> pd.Series:
    return sum(s.shift(l) for l in range(a, b + 1))


def core_design(core_mm: pd.Series, d: pd.Series, N: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "trend": core_mm.shift(1).rolling(3).mean(),
        "oil chg, months 1-6": _lagsum(d, 1, 6),
        "oil chg, months 7-12": _lagsum(d, 7, 12),
        "new high, months 0-6": _lagsum(N, 0, 6),
        "new high, months 7-12": _lagsum(N, 7, 12),
    })


def head_design(core_mm: pd.Series, d: pd.Series, N: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "trend": core_mm.shift(1).rolling(3).mean(),
        "oil chg, month 0": d, "oil chg, month 1": d.shift(1), "oil chg, month 2": d.shift(2),
        "oil chg, months 3-12": _lagsum(d, 3, 12),
        "new high, month 0": N, "new high, month 1": N.shift(1), "new high, month 2": N.shift(2),
    })


@dataclass
class PassThrough:
    family: str                 # "PCE" or "CPI"
    core: sm.regression.linear_model.RegressionResultsWrapper
    head: sm.regression.linear_model.RegressionResultsWrapper
    core_sd: float
    head_sd: float
    head_mm: pd.Series          # actual monthly inflation history
    core_mm: pd.Series
    brent: pd.Series            # monthly average Brent history
    start: pd.Period
    end: pd.Period


def fit(brent: pd.Series, head_level: pd.Series, core_level: pd.Series, family: str,
        hac_lags: int = 6, start: str | None = None) -> PassThrough:
    ot = oil_terms(brent)
    hm, cm = mom(head_level), mom(core_level)
    Xc, Xh = core_design(cm, ot["d"], ot["N"]), head_design(cm, ot["d"], ot["N"])
    dc = pd.concat([cm.rename("y"), Xc], axis=1).dropna()
    dh = pd.concat([hm.rename("y"), Xh], axis=1).dropna()
    if start:
        dc, dh = dc.loc[pd.Period(start, "M"):], dh.loc[pd.Period(start, "M"):]
    kw = dict(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    fc = sm.OLS(dc["y"], sm.add_constant(dc.drop(columns="y"))).fit(**kw)
    fh = sm.OLS(dh["y"], sm.add_constant(dh.drop(columns="y"))).fit(**kw)
    return PassThrough(family, fc, fh, float(fc.resid.std()), float(fh.resid.std()),
                       hm.dropna(), cm.dropna(), brent.dropna(), dh.index[0], dh.index[-1])


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def project(pt: PassThrough, path: list[float] | np.ndarray, n_sims: int = 0, seed: int = 0,
            brent_hist: pd.Series | None = None) -> dict:
    """Run a future monthly Brent path (starting the month after the last Brent observation) through
    the model. Inflation is simulated from the month after the last actual inflation print, using actual
    Brent where known. Returns monthly (m/m) and year-over-year paths for headline and core.
    n_sims = 0 -> deterministic (no shocks); > 0 -> adds residual shocks and returns percentiles."""
    bh = (brent_hist if brent_hist is not None else pt.brent).dropna()
    fut_idx = pd.period_range(bh.index[-1] + 1, periods=len(path), freq="M")
    brent = pd.concat([bh, pd.Series(np.asarray(path, float), index=fut_idx)])
    ot = oil_terms(brent)
    last_inf = min(pt.head_mm.index[-1], pt.core_mm.index[-1])
    sim_idx = pd.period_range(last_inf + 1, fut_idx[-1], freq="M")
    ns = max(1, n_sims)
    rng = np.random.default_rng(seed)

    pc, ph = pt.core.params, pt.head.params
    common = pt.core_mm.loc[:last_inf].index.intersection(pt.head_mm.loc[:last_inf].index)
    core_hist = pt.core_mm.loc[common].to_numpy()
    head_hist = pt.head_mm.loc[common].to_numpy()
    C = np.tile(core_hist, (ns, 1))
    H = np.tile(head_hist, (ns, 1))
    d, N = ot["d"], ot["N"]
    zero = pd.Series(np.nan, index=d.index)
    Xc = core_design(zero, d, N).drop(columns="trend")   # oil terms only; trend comes from simulation
    Xh = head_design(zero, d, N).drop(columns="trend")
    bc, bh_ = pc.drop(["const", "trend"]).reindex(Xc.columns).to_numpy(), ph.drop(["const", "trend"]).reindex(Xh.columns).to_numpy()
    for t in sim_idx:
        trend = C[:, -3:].mean(axis=1)
        c_t = pc["const"] + pc["trend"] * trend + float(Xc.loc[t].to_numpy() @ bc)
        h_t = ph["const"] + ph["trend"] * trend + float(Xh.loc[t].to_numpy() @ bh_)
        if n_sims > 0:
            c_t = c_t + rng.normal(0, pt.core_sd, ns)
            h_t = h_t + rng.normal(0, pt.head_sd, ns)
        C = np.column_stack([C, c_t])
        H = np.column_stack([H, h_t])

    # year-over-year from the last 12 monthly log changes
    def yoy(M):
        cs = np.cumsum(M, axis=1)
        out = np.full_like(M, np.nan)
        out[:, 12:] = cs[:, 12:] - cs[:, :-12]
        out[:, 11] = cs[:, 11]
        return 100.0 * (np.exp(out / 100.0) - 1.0)

    n_new = len(sim_idx)
    idx_all = pd.period_range(sim_idx[0] - (C.shape[1] - n_new), sim_idx[-1], freq="M")
    res = {"index": idx_all, "sim_index": sim_idx, "path_index": fut_idx, "brent": brent,
           "last_actual": last_inf}
    for name, M in (("head", H), ("core", C)):
        Y = yoy(M)
        res[f"{name}_mm"] = pd.Series(M.mean(axis=0), index=idx_all)
        res[f"{name}_yoy"] = pd.Series(np.nanmean(Y, axis=0), index=idx_all)
        if n_sims > 0:
            for q in (10, 50, 90):
                res[f"{name}_yoy_p{q}"] = pd.Series(np.nanpercentile(Y, q, axis=0), index=idx_all)
    return res


def impulse_comparison(pt: PassThrough, shock_pct: float = 10.0, horizons=(3, 6, 12)) -> pd.DataFrame:
    """Effect on y/y inflation (pp) of a permanent `shock_pct` oil rise, in two situations:
      * 'New 12-month high': oil was flat before, so the rise breaks a new high.
      * 'Back to an earlier level': oil was 20% higher within the past year, so the rise only returns
        part-way to a level it has already seen (no new high).
    Each is measured against the same history with no rise, so only the shock is counted."""
    base = 100.0
    hist_len = 36
    idx = pd.period_range("2000-01", periods=hist_len, freq="M")
    flat = pd.Series(base, index=idx)
    earlier_high = flat.copy()
    earlier_high.iloc[-12:-6] = base * 1.2       # 6 to 12 months ago, oil was 20% higher

    # steady-state inflation history so the trend term is neutral
    p = pt.core.params
    css = p["const"] / (1 - p["trend"]) if abs(1 - p["trend"]) > 1e-6 else pt.core_mm.mean()
    hh = PassThrough(pt.family, pt.core, pt.head, pt.core_sd, pt.head_sd,
                     pd.Series(css, index=idx), pd.Series(css, index=idx), flat, idx[0], idx[-1])
    H = max(horizons)
    rows = {}
    for label, hist in (("New 12-month high", flat), ("Back to an earlier level", earlier_high)):
        shocked = project(hh, [base * (1 + shock_pct / 100.0)] * H, brent_hist=hist)
        calm = project(hh, [base] * H, brent_hist=hist)
        dh = shocked["head_yoy"] - calm["head_yoy"]
        dc = shocked["core_yoy"] - calm["core_yoy"]
        fut = shocked["path_index"]
        row = {f"Headline after {h}m": float(dh.loc[fut[h - 1]]) for h in horizons}
        row[f"Core after {H}m"] = float(dc.loc[fut[H - 1]])
        rows[label] = row
    return pd.DataFrame(rows).T


def view_paths(last_brent: float, months: int = 12, ramp: int = 12) -> dict[str, np.ndarray]:
    """Monthly Brent paths for the three 12-month views in config.VIEWS (held flat after `ramp`)."""
    slide = np.concatenate([np.linspace(last_brent, 75.0, ramp + 1)[1:], np.full(max(0, months - ramp), 75.0)])
    return {
        "Oil re-spikes to $110 and holds": np.full(months, 110.0),
        "Oil eases to $90 and holds": np.full(months, 90.0),
        "Brent slides steadily to $75": slide[:months],
    }


def custom_path(last_brent: float, target: float, ramp: int, months: int) -> np.ndarray:
    ramp = max(1, min(ramp, months))
    return np.concatenate([np.linspace(last_brent, target, ramp + 1)[1:], np.full(months - ramp, target)])


def new_high_pvalues(pt: PassThrough) -> dict[str, float]:
    """Wald test p-value that all 'new high' coefficients are zero (i.e. no extra kick from new highs)."""
    out = {}
    for name, res in (("headline", pt.head), ("core", pt.core)):
        names = list(res.params.index)
        R = np.array([[1.0 if n == m else 0.0 for n in names] for m in names if m.startswith("new high")])
        out[name] = float(res.wald_test(R, scalar=True).pvalue)
    return out


def cpi_view_value(pt_cpi: PassThrough, path, months: int = 12, hist_mean_mm: float | None = None) -> float:
    """CPI factor value for a view: projected inflation over the path months, minus the historical
    average pace (the factor model's regressions measure CPI relative to its average)."""
    r = project(pt_cpi, path)
    fut = r["path_index"][:months]
    total = float(r["head_mm"].loc[fut].sum())
    avg = hist_mean_mm if hist_mean_mm is not None else float(pt_cpi.head_mm.mean())
    return total - avg * len(fut)
