"""Data loading: live (FRED + Yahoo Finance) or a synthetic demo dataset.

Everything is converted to a common monthly PeriodIndex:
  * factors: monthly changes (pp for yields/spreads, % for prices) -> see config.FACTORS
  * returns: monthly total returns in percent (adjusted-close based), incl. a CASH column
  * rf:      monthly risk-free return in percent (3m T-bill / 12, lagged one month)
"""
from __future__ import annotations

import io
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import (ASSETS, CASH, FACTOR_KEYS, FACTORS, FRED_IDS, RF_FRED_ID, TICKERS)

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
    out = {}
    for f in FACTORS:
        # "A-B" means the spread between two FRED series, each taken at month-end
        parts = [to_monthly_last(raw[sid]) for sid in f.fred.split("-")]
        m = parts[0] if len(parts) == 1 else parts[0].sub(parts[1])
        if f.transform == "diff":
            out[f.key] = m.diff()
        elif f.transform == "logdiff":
            out[f.key] = np.log(m.where(m > 0)).diff() * 100.0
        else:
            raise ValueError(f.transform)
    return pd.DataFrame(out)[FACTOR_KEYS]


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


def load_live() -> Dataset:
    """Download everything, build monthly dataset, cache to disk. Falls back to the disk cache."""
    cache = CACHE_DIR / "live_dataset.pkl"
    try:
        raw = {sid: fetch_fred(sid) for sid in FRED_IDS}
        want = [t for t in TICKERS if t != CASH] + [p for a in ASSETS for p in a.proxy]
        prices = fetch_prices(sorted(set(want)))
        missing = [t for t in want if t not in prices.columns or prices[t].dropna().empty]

        factors = build_factors(raw)
        rf = build_rf(raw[RF_FRED_ID])
        rets_all = monthly_returns_from_prices(prices)
        rets, used = splice_proxies(rets_all)
        rets[CASH] = rf

        idx = factors.index.union(rets.index).union(rf.index)
        factors, rets, rf = (x.reindex(idx) for x in (factors, rets, rf))
        factors, rets, rf = (drop_incomplete_month(x) for x in (factors, rets, rf))
        rf = rf.ffill()
        rets[CASH] = rf

        notes = []
        if missing:
            notes.append(f"No Yahoo data returned for: {', '.join(missing)}. These holdings are excluded "
                         "from regressions until data is available.")
        ds = Dataset(factors, rets[TICKERS], rf, "live", notes, used)
        CACHE_DIR.mkdir(exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump(ds, fh)
        return ds
    except Exception as exc:  # network down, API change, etc.
        if cache.exists():
            with open(cache, "rb") as fh:
                ds = pickle.load(fh)
            ds.notes = [f"Live download failed ({type(exc).__name__}); showing last cached data."] + ds.notes
            return ds
        raise


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
                   {}, betas)


def load(source: str) -> Dataset:
    return make_demo() if source == "demo" else load_live()
