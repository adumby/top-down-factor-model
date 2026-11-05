"""Data loading: live (FRED + Yahoo Finance) or a synthetic demo dataset.

Everything is converted to a common monthly PeriodIndex:
  * factors: monthly changes (pp for yields/spreads, % for prices) -> see config.FACTORS
  * returns: monthly total returns in percent (adjusted-close based), incl. a CASH column
  * rf:      monthly risk-free return in percent (3m T-bill / 12, lagged one month)
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import (ASSETS, CASH, FACTOR_KEYS, FACTORS, FRED_IDS, INFLATION_SERIES, OIL_LEVEL_FRED,
                    RF_FRED_ID, TICKERS)

CACHE_DIR = Path(__file__).parent / "data_cache"
START = "2003-12-01"


@dataclass
class Dataset:
    factors: pd.DataFrame
    returns: pd.DataFrame
    rf: pd.Series
    source: str
    notes: list[str] = field(default_factory=list)
    proxy_used: dict[str, str] = field(default_factory=dict)   # ticker -> proxy ticker
    true_betas: pd.DataFrame | None = None                     # demo only (for testing)
    levels: pd.DataFrame | None = None     # monthly BRENT (avg $) + price indexes (PCE_H, PCE_C, CPI_H, CPI_C)


# --------------------------------------------------------------------------- #
# Helpers (pure functions, unit-tested with synthetic daily data)
# --------------------------------------------------------------------------- #
def to_monthly_last(s: pd.Series) -> pd.Series:
    """Daily/irregular series -> last observation of each calendar month (PeriodIndex)."""
    s = s.dropna().copy()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.groupby(s.index.to_period("M")).last()


def build_factors(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """Factors whose FRED series are missing from `raw` are skipped (column left out)."""
    out = {}
    for f in FACTORS:
        ids = f.fred.split("-")          # "A-B" = spread between two series, each at month-end
        if any(sid not in raw or raw[sid].dropna().empty for sid in ids):
            continue
        parts = [to_monthly_last(raw[sid]) for sid in ids]
        m = parts[0] if len(parts) == 1 else parts[0].sub(parts[1])
        if f.transform == "diff":
            out[f.key] = m.diff()
        elif f.transform == "logdiff":
            out[f.key] = np.log(m.where(m > 0)).diff() * 100.0
        else:
            raise ValueError(f.transform)
    return pd.DataFrame(out)[[k for k in FACTOR_KEYS if k in out]]


def build_levels(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """Brent (monthly average) and the price-index levels used by the inflation module."""
    out = {}
    if OIL_LEVEL_FRED in raw:
        b = raw[OIL_LEVEL_FRED].dropna().copy()
        b.index = pd.to_datetime(b.index)
        out["BRENT"] = b.groupby(b.index.to_period("M")).mean()
    for key, (sid, _) in INFLATION_SERIES.items():
        if sid in raw:
            out[key] = to_monthly_last(raw[sid])
    return pd.DataFrame(out)


def build_rf(tbill_annual_pct: pd.Series) -> pd.Series:
    """Monthly rf in %, using the PRIOR month's 3m yield (known at the start of the month)."""
    m = to_monthly_last(tbill_annual_pct)
    return (m / 12.0).shift(1).rename("RF")


def monthly_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily adjusted closes -> monthly % returns."""
    m = prices.apply(to_monthly_last)
    return m.pct_change(fill_method=None) * 100.0


def splice_proxies(rets: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Use the proxy fund's returns before the holding's own history starts."""
    out = pd.DataFrame(index=rets.index)
    used: dict[str, str] = {}
    for a in ASSETS:
        if a.ticker == CASH:
            continue
        merged = rets.get(a.ticker, pd.Series(index=rets.index, dtype=float))
        chain = []
        for p in a.proxy:
            if p in rets.columns:
                filled = merged.combine_first(rets[p])
                if filled.notna().sum() > merged.notna().sum():
                    chain.append(p)
                merged = filled
        if chain:
            used[a.ticker] = " then ".join(chain)
        out[a.ticker] = merged
    return out, used


def drop_incomplete_month(df, today: pd.Timestamp | None = None):
    today = today or pd.Timestamp.today()
    cur = today.to_period("M")
    return df.loc[df.index < cur] if len(df) and df.index[-1] >= cur else df


# --------------------------------------------------------------------------- #
# Live fetchers
# --------------------------------------------------------------------------- #
def fetch_fred(series_id: str, start: str = START) -> pd.Series:
    """FRED series. Uses the JSON API if FRED_API_KEY is set, else the public CSV endpoint."""
    key = os.environ.get("FRED_API_KEY")
    if key:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params={"series_id": series_id, "api_key": key, "file_type": "json",
                                 "observation_start": start}, timeout=30)
        r.raise_for_status()
        obs = r.json()["observations"]
        s = pd.Series({pd.Timestamp(o["date"]): pd.to_numeric(o["value"], errors="coerce") for o in obs})
        return s.rename(series_id)
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                     params={"id": series_id, "cosd": start}, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.Series(pd.to_numeric(df[series_id], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df.iloc[:, 0]), name=series_id)
    return s


def fetch_prices(tickers: list[str], start: str = START) -> pd.DataFrame:
    import yfinance as yf
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close.dropna(how="all")


class MissingDataError(RuntimeError):
    """Raised when any required series is missing or stale. The app refuses to run on partial data."""


def check_complete(factors: pd.DataFrame, returns: pd.DataFrame, max_lag_months: int = 3,
                   today: pd.Timestamp | None = None) -> list[str]:
    """Return a list of problems (empty = OK): missing factors/holdings, or series that stopped updating."""
    cur = (today or pd.Timestamp.today()).to_period("M")
    problems = []
    for f in FACTORS:
        if f.key not in factors.columns or factors[f.key].dropna().empty:
            problems.append(f"Factor '{f.label}' (FRED {f.fred}): no data")
        else:
            last = factors[f.key].dropna().index[-1]
            if (cur - last).n > max_lag_months:
                problems.append(f"Factor '{f.label}' (FRED {f.fred}): last data {last.strftime('%b %Y')}, stale")
    for t in TICKERS:
        if t == CASH:
            continue
        if t not in returns.columns or returns[t].dropna().empty:
            problems.append(f"Holding {t}: no price data")
        else:
            last = returns[t].dropna().index[-1]
            if (cur - last).n > max_lag_months:
                problems.append(f"Holding {t}: last price {last.strftime('%b %Y')}, stale")
    return problems


def check_levels(levels: pd.DataFrame, max_lag_months: int = 3, today: pd.Timestamp | None = None) -> list[str]:
    cur = (today or pd.Timestamp.today()).to_period("M")
    problems = []
    need = {"BRENT": f"Brent (FRED {OIL_LEVEL_FRED})"} | {k: f"{v[1]} (FRED {v[0]})" for k, v in INFLATION_SERIES.items()}
    for k, label in need.items():
        if k not in levels.columns or levels[k].dropna().empty:
            problems.append(f"{label}: no data")
        else:
            last = levels[k].dropna().index[-1]
            if (cur - last).n > max_lag_months:
                problems.append(f"{label}: last data {last.strftime('%b %Y')}, stale")
    return problems


def load_live() -> Dataset:
    """Download everything fresh. Raises MissingDataError if anything is missing: no partial runs,
    no fallback to old cached data."""
    raw, failed = {}, []
    for sid in FRED_IDS:
        try:
            raw[sid] = fetch_fred(sid)
        except Exception as exc:
            failed.append(f"FRED {sid} ({type(exc).__name__})")
    want = [t for t in TICKERS if t != CASH] + [p for a in ASSETS for p in a.proxy]
    try:
        prices = fetch_prices(sorted(set(want)))
    except Exception as exc:
        raise MissingDataError(f"Yahoo Finance download failed ({type(exc).__name__}: {exc})") from exc
    if failed:
        raise MissingDataError("Download failed for: " + "; ".join(failed))

    factors = build_factors(raw)
    rf = build_rf(raw[RF_FRED_ID])
    rets_all = monthly_returns_from_prices(prices)
    rets, used = splice_proxies(rets_all)

    idx = factors.index.union(rets.index).union(rf.index)
    factors, rets, rf = (x.reindex(idx) for x in (factors, rets, rf))
    factors, rets, rf = (drop_incomplete_month(x) for x in (factors, rets, rf))
    rf = rf.ffill()
    rets[CASH] = rf

    levels = drop_incomplete_month(build_levels(raw))

    problems = check_complete(factors, rets) + check_levels(levels)
    if problems:
        raise MissingDataError("Missing or stale data:\n- " + "\n- ".join(problems))
    return Dataset(factors, rets[TICKERS], rf, "live", [], used, levels=levels)


# --------------------------------------------------------------------------- #
# Synthetic demo data (known betas, so the model can be validated)
# --------------------------------------------------------------------------- #
_DEMO_BETAS = {
    #        REAL   BE    FED   CRED   GROW   USD    OIL    VIX
    "VGT":  [-4.0,  0.0, -1.0, -3.0,  0.4, -0.30,  0.00, -0.35],
    "XLY":  [-2.5,  0.0, -1.5, -3.0,  0.5, -0.30, -0.03, -0.30],
    "XLC":  [-2.5,  0.0, -1.0, -2.5,  0.3, -0.20,  0.00, -0.25],
    "SCHH": [-6.0,  1.0, -1.0, -2.5,  0.3, -0.20,  0.00, -0.20],
    "XLE":  [ 0.5,  3.0,  0.5, -2.0,  0.5, -0.40,  0.30, -0.15],
    "XLB":  [-1.5,  1.5,  0.0, -3.0,  0.8, -0.70,  0.08, -0.30],
    "XLU":  [-4.5,  1.5, -1.5, -0.5, -0.1,  0.00,  0.00, -0.10],
    "XLI":  [-1.5,  0.5, -0.5, -3.0,  0.9, -0.40,  0.03, -0.30],
    "XLF":  [ 1.0,  0.5,  2.5, -4.0,  0.6, -0.10,  0.00, -0.30],
    "XLV":  [-1.0,  0.0, -0.5, -1.0,  0.1,  0.00,  0.00, -0.15],
    "XLP":  [-2.5,  0.5, -1.0, -0.5,  0.0, -0.10,  0.00, -0.08],
    "IGSB": [-0.5,  0.2, -1.8, -1.2,  0.0,  0.00,  0.00, -0.02],
    "IBID": [-1.8,  0.9, -0.5, -0.3,  0.0,  0.00,  0.00,  0.00],
    "AGZ":  [-2.5, -2.0, -1.0, -0.3,  0.0,  0.00,  0.00,  0.02],
    "VGIT": [-4.2, -4.2, -0.8,  0.5,  0.0,  0.00,  0.00,  0.05],
    "IAU":  [-2.5,  1.5, -0.8,  0.0,  0.0, -1.00,  0.05,  0.05],
}
_DEMO_ALPHA = {t: (0.70 if a.group == "Equity" else 0.50 if a.group == "Commodity" else 0.15)
               for a in ASSETS for t in [a.ticker] if t != CASH}
_DEMO_RESID = {"VGT": 3.0, "XLY": 2.8, "XLC": 2.8, "SCHH": 3.5, "XLE": 4.5, "XLB": 3.0, "XLU": 3.2,
               "XLI": 2.5, "XLF": 3.0, "XLV": 2.5, "XLP": 2.3, "IGSB": 0.4, "IBID": 0.5,
               "AGZ": 0.6, "VGIT": 0.6, "IAU": 3.2}


def make_demo(seed: int = 7, start: str = "2007-01", end: str = "2026-10") -> Dataset:
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, end, freq="M")
    n = len(idx)
    keys = FACTOR_KEYS
    sd = pd.Series({"REAL_YIELD": 0.20, "BREAKEVEN": 0.15, "FED_EXPECT": 0.22, "CREDIT": 0.20,
                    "GROWTH": 0.7, "USD": 1.4, "OIL": 9.0, "VIX": 5.0, "TERM_PREMIUM": 0.15,
                    "COPPER": 6.0, "MORTGAGE": 0.12, "CPI": 0.30}).reindex(keys).fillna(1.0)
    corr = pd.DataFrame(np.eye(len(keys)), index=keys, columns=keys)

    def setc(a, b, v):
        corr.loc[a, b] = corr.loc[b, a] = v
    setc("REAL_YIELD", "FED_EXPECT", 0.55); setc("REAL_YIELD", "BREAKEVEN", 0.20)
    setc("CREDIT", "VIX", 0.70); setc("CREDIT", "GROWTH", -0.40); setc("OIL", "BREAKEVEN", 0.40)
    setc("OIL", "GROWTH", 0.20); setc("USD", "OIL", -0.35); setc("USD", "VIX", 0.30)
    setc("VIX", "GROWTH", -0.40)
    for a, b, v in [("COPPER", "GROWTH", 0.35), ("COPPER", "OIL", 0.35), ("CPI", "OIL", 0.45),
                    ("CPI", "BREAKEVEN", 0.30), ("TERM_PREMIUM", "REAL_YIELD", 0.50),
                    ("MORTGAGE", "CREDIT", 0.30)]:
        if a in keys and b in keys:
            setc(a, b, v)
    cov = corr.values * np.outer(sd.values, sd.values)
    f = pd.DataFrame(rng.multivariate_normal(np.zeros(len(keys)), cov, size=n), index=idx, columns=keys)

    rf = pd.Series(np.clip(0.25 * np.exp(-((np.arange(n) - 20) / 60.0) ** 2) + 0.05, 0.02, 0.45)
                   + np.where(np.arange(n) > 190, 0.2, 0.0), index=idx, name="RF")
    base_keys = ["REAL_YIELD", "BREAKEVEN", "FED_EXPECT", "CREDIT", "GROWTH", "USD", "OIL", "VIX"]
    betas = pd.DataFrame(_DEMO_BETAS, index=base_keys).T.reindex(columns=keys).fillna(0.0)
    extra = {"VGIT": {"TERM_PREMIUM": -4.0}, "AGZ": {"TERM_PREMIUM": -2.5}, "VGT": {"TERM_PREMIUM": -2.0},
             "XLB": {"COPPER": 0.25}, "XLI": {"COPPER": 0.12}, "SCHH": {"MORTGAGE": -4.0},
             "XLY": {"MORTGAGE": -2.0, "CPI": -1.5}, "XLP": {"CPI": -1.0}, "IBID": {"CPI": 0.4},
             "IAU": {"CPI": 1.0}}
    for t, d in extra.items():
        for k, v in d.items():
            if k in betas.columns:
                betas.loc[t, k] = v
    mkt = rng.standard_normal(n)
    rets = {}
    for a in ASSETS:
        t = a.ticker
        if t == CASH:
            rets[t] = rf
            continue
        z = rng.standard_normal(n)
        e = _DEMO_RESID[t] * ((0.6 * mkt + 0.8 * z) if a.group == "Equity" else z)
        rets[t] = rf + _DEMO_ALPHA[t] + f.values @ betas.loc[t].values + e
    rets = pd.DataFrame(rets)[TICKERS]
    f.iloc[0] = np.nan
    return Dataset(f, rets, rf, "demo",
                   ["Synthetic data generated from known betas. For testing the app only: not real markets."],
                   {}, betas, levels=make_demo_levels(seed, start, end))


# Known pass-through used to generate demo inflation (so the inflation module can be validated)
DEMO_PT = {
    "core": {"const": 0.08, "trend": 0.55, "oil chg, months 1-6": 0.002, "oil chg, months 7-12": 0.001,
             "new high, months 0-6": 0.004, "new high, months 7-12": 0.002},
    "head": {"const": 0.02, "trend": 0.90, "oil chg, month 0": 0.030, "oil chg, month 1": 0.012,
             "oil chg, month 2": 0.004, "oil chg, months 3-12": 0.001,
             "new high, month 0": 0.020, "new high, month 1": 0.010, "new high, month 2": 0.004},
    "core_sd": 0.07, "head_sd": 0.10,
}


def make_demo_levels(seed: int = 7, start: str = "2007-01", end: str = "2026-10",
                     pre: int = 36) -> pd.DataFrame:
    """Brent: mean-reverting log price with occasional spikes. Inflation: generated from DEMO_PT."""
    from inflation import core_design, head_design, oil_terms
    rng = np.random.default_rng(seed + 101)
    idx = pd.period_range(pd.Period(start, "M") - pre, end, freq="M")
    n = len(idx)
    lo = np.empty(n); lo[0] = np.log(70)
    for t in range(1, n):
        spike = rng.choice([0.0, 0.18, -0.18], p=[0.94, 0.03, 0.03])
        lo[t] = lo[t - 1] + 0.04 * (np.log(80) - lo[t - 1]) + rng.normal(0, 0.07) + spike
    brent = pd.Series(np.exp(lo), index=idx)
    brent *= 95.0 / brent.iloc[-1]
    ot = oil_terms(brent)
    d, N = ot["d"].fillna(0), ot["N"].fillna(0)
    cm = pd.Series(0.18, index=idx); hm = pd.Series(0.20, index=idx)
    pc, ph = DEMO_PT["core"], DEMO_PT["head"]
    for i in range(15, n):
        t = idx[i]
        xc = core_design(cm, d, N).loc[t]
        cm.iloc[i] = pc["const"] + sum(pc[k] * xc[k] for k in pc if k != "const") + rng.normal(0, DEMO_PT["core_sd"])
        xh = head_design(cm, d, N).loc[t]
        hm.iloc[i] = ph["const"] + sum(ph[k] * xh[k] for k in ph if k != "const") + rng.normal(0, DEMO_PT["head_sd"])
    lvl = lambda m: 100.0 * np.exp(m.cumsum() / 100.0)
    cpi_h = hm * 1.15 + rng.normal(0, 0.04, n)
    cpi_c = cm * 1.12 + rng.normal(0, 0.03, n)
    out = pd.DataFrame({"BRENT": brent, "PCE_H": lvl(hm), "PCE_C": lvl(cm),
                        "CPI_H": lvl(cpi_h), "CPI_C": lvl(cpi_c)})
    return out.loc[pd.Period(start, "M") - 24:]


def load(source: str) -> Dataset:
    return make_demo() if source == "demo" else load_live()
