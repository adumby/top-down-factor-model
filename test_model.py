"""Run:  python -m pytest -q   (from the project folder)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ASSETS, FACTOR_KEYS, FACTORS, PRESETS, TICKERS  # noqa: E402
from data import (build_factors, build_rf, make_demo, monthly_returns_from_prices,  # noqa: E402
                  splice_proxies, to_monthly_last)
from model import fit_models  # noqa: E402
from portfolio import (portfolio_betas, portfolio_returns, sim_stats, simulate, stress,  # noqa: E402
                       to_fractions)


def test_default_weights_sum_to_100():
    assert abs(sum(a.weight for a in ASSETS) - 100.0) < 0.01


def test_month_end_and_transforms():
    days = pd.bdate_range("2020-01-01", "2020-04-30")
    raw = {}
    for f in FACTORS:
        raw[f.fred] = pd.Series(np.linspace(1, 2, len(days)), index=days)
    raw["TB3MS"] = pd.Series([1.2, 2.4, 3.6, 4.8], index=pd.date_range("2020-01-01", periods=4, freq="MS"))
    fac = build_factors(raw)
    s = raw["DGS2"]
    me = s.groupby(s.index.to_period("M")).last()
    assert np.isclose(fac["FED_EXPECT"].iloc[1], me.iloc[1] - me.iloc[0])
    assert np.isclose(fac["OIL"].iloc[1], 100 * np.log(me.iloc[1] / me.iloc[0]))
    rf = build_rf(raw["TB3MS"])
    assert np.isnan(rf.iloc[0]) and np.isclose(rf.iloc[1], 0.1)   # lagged one month, /12


def test_nonpositive_price_gives_nan_not_crash():
    days = pd.date_range("2020-01-31", periods=3, freq="ME")
    raw = {f.fred: pd.Series([1.0, 2.0, 3.0], index=days) for f in FACTORS}
    raw["DCOILWTICO"] = pd.Series([50.0, -5.0, 40.0], index=days)
    fac = build_factors(raw)
    assert fac["OIL"].isna().all()


def test_monthly_returns_and_proxy_splice():
    days = pd.bdate_range("2019-01-01", "2019-06-30")
    px = pd.DataFrame({"VNQ": np.linspace(100, 110, len(days))}, index=days)
    px["SCHH"] = px["VNQ"].where(days >= "2019-04-01")
    r = monthly_returns_from_prices(px)
    out, used = splice_proxies(r)
    assert used.get("SCHH") == "VNQ"
    assert out["SCHH"].iloc[1:].notna().all()
    assert np.isclose(out["SCHH"].iloc[1], r["VNQ"].iloc[1])


def test_regression_recovers_true_betas():
    ds = make_demo(seed=3)
    m = fit_models(ds.returns, ds.rf, ds.factors, FACTOR_KEYS)
    z = ((m.betas.drop("CASH") - ds.true_betas) / m.se.drop("CASH")).values.ravel()
    assert abs(z.mean()) < 0.3
    assert (np.abs(z) < 3.5).all()
    assert (m.betas.loc["CASH"] == 0).all()


def test_stress_is_linear_and_matches_manual():
    ds = make_demo()
    m = fit_models(ds.returns, ds.rf, ds.factors, FACTOR_KEYS)
    w = to_fractions(pd.Series({a.ticker: a.weight for a in ASSETS}))
    shock = pd.Series(PRESETS["Hawkish Fed surprise"]).reindex(FACTOR_KEYS).fillna(0)
    r = stress(m.betas0, w, shock)
    manual = float(portfolio_betas(m.betas0, w) @ shock)
    assert np.isclose(r["total"], manual)
    assert np.isclose(r["by_asset"].sum(), r["total"]) and np.isclose(r["by_factor"].sum(), r["total"])
    assert np.isclose(stress(m.betas0, w, 2 * shock)["total"], 2 * manual)


def test_all_cash_portfolio():
    ds = make_demo()
    m = fit_models(ds.returns, ds.rf, ds.factors, FACTOR_KEYS)
    w = pd.Series(0.0, index=TICKERS); w["CASH"] = 1.0
    assert stress(m.betas0, w, pd.Series(1.0, index=FACTOR_KEYS))["total"] == 0
    g = simulate(m, {"c": w}, 0.3, 12, 500, pd.Series(dtype=float), 1.0, 1.0)["c"]
    assert np.allclose(g[:, -1], 1.003 ** 12)


def test_portfolio_rescales_missing_holdings():
    idx = pd.period_range("2020-01", periods=3, freq="M")
    R = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [np.nan, 3.0, 3.0]}, index=idx)
    rp, W = portfolio_returns(R, pd.Series({"A": 0.9, "B": 0.1}))
    assert np.isclose(rp.iloc[0], 1.0) and np.isclose(rp.iloc[1], 1.2)


def test_forecast_drift_moves_mean_in_right_direction():
    ds = make_demo()
    m = fit_models(ds.returns, ds.rf, ds.factors, FACTOR_KEYS)
    w = to_fractions(pd.Series({a.ticker: a.weight for a in ASSETS}))
    base = sim_stats(simulate(m, {"w": w}, 0.3, 12, 4000, pd.Series(dtype=float), 1.0, 0.0)["w"], 12)
    view = pd.Series({"REAL_YIELD": 1.0})          # real yields +1pp over the year
    hit = sim_stats(simulate(m, {"w": w}, 0.3, 12, 4000, view, 1.0, 0.0)["w"], 12)
    expected = portfolio_betas(m.betas0, w)["REAL_YIELD"] * 1.0
    assert expected < 0
    assert hit["Mean"] - base["Mean"] == pytest.approx(expected, abs=0.6)
