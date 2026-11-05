"""Fill the factors a scenario doesn't specify with their historical co-movement.

The oil model sets 5 factors (oil, CPI, breakeven, 2y, real yield). The other factors (credit, VIX, dollar,
growth, copper, term premium, mortgage spread) are set to their expected value GIVEN those 5 moves:

    E[f_u | f_s] = mean_u + Cov(u, s) Cov(s, s)^-1 (f_s - mean_s)

estimated on overlapping 12-month sums of monthly factor changes (the scenarios are 12-month moves).
Windows that overlap the 2008-09 crisis or the 2020 Covid crash are left out by default, so that "yields
fall" doesn't automatically import crash behaviour. Results are returned as deviations from each factor's
average 12-month change, which is how the factor regressions read them (the intercept holds the average).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CRISES = [("2008-09", "2009-06"), ("2020-02", "2020-06")]
# Factors whose scenario values are already deviations from their average pace
ALREADY_DEVIATION = {"CPI"}


def twelve_month_changes(factors: pd.DataFrame, keys: list[str], start=None,
                         exclude=CRISES) -> pd.DataFrame:
    F = factors[keys]
    if start is not None:
        F = F.loc[start:]
    S = F.rolling(12, min_periods=12).sum().dropna()
    if exclude:
        bad = pd.Series(False, index=S.index)
        for a, b in exclude:
            a, b = pd.Period(a, "M"), pd.Period(b, "M")
            # window ending t covers t-11..t: drop it if it overlaps [a, b]
            bad |= (S.index >= a) & (S.index - 11 <= b)
        S = S.loc[~bad]
    return S


def fill(view: dict[str, float], S: pd.DataFrame, ridge: float = 1e-3) -> tuple[dict[str, float], list[str]]:
    """Return (full view, list of filled keys). Specified values are kept exactly as given."""
    keys = list(S.columns)
    spec = [k for k in keys if k in view]
    free = [k for k in keys if k not in view]
    if not spec or not free:
        return dict(view), []
    mu = S.mean()
    C = S.cov()
    x = np.array([view[k] if k in ALREADY_DEVIATION else view[k] - mu[k] for k in spec])
    Css = C.loc[spec, spec].to_numpy()
    Css = Css + ridge * np.diag(np.diag(Css))       # small ridge: oil, CPI and rates are correlated
    beta = C.loc[free, spec].to_numpy() @ np.linalg.solve(Css, x)
    out = dict(view)
    for k, v in zip(free, beta):
        out[k] = round(float(v), 3)
    return out, free


def explained_share(S: pd.DataFrame, spec: list[str]) -> pd.Series:
    """R-squared of each free factor on the specified ones: how much the fill actually pins down."""
    C = S.cov()
    free = [k for k in S.columns if k not in spec]
    Css = C.loc[spec, spec].to_numpy()
    out = {}
    for k in free:
        b = np.linalg.solve(Css, C.loc[spec, k].to_numpy())
        out[k] = float(C.loc[k, spec].to_numpy() @ b / C.loc[k, k])
    return pd.Series(out)
