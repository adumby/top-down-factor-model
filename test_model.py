"""Run:  python -m pytest -q   (from the project folder)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ASSETS, FACTOR_KEYS, FACTORS, FRED_IDS, PRESETS, TICKERS  # noqa: E402
from data import (build_factors, build_rf, make_demo, monthly_returns_from_prices,  # noqa: E402
                  splice_proxies, to_monthly_last)
from model import fit_models  # noqa: E402
from portfolio import (portfolio_betas, portfolio_returns, sim_stats, simulate, stress,  # noqa: E402
                       to_fractions)


def test_default_weights_sum_to_100():
    assert abs(sum(a.weight for a in ASSETS) - 100.0) < 0.01


def test_month_end_and_transforms():
    days = pd.bdate_range("2020-01-01", "2020-04-30")
    raw = {sid: pd.Series(np.linspace(1, 2, len(days)), index=days) for sid in FRED_IDS}
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
    raw = {sid: pd.Series([1.0, 2.0, 3.0], index=days) for sid in FRED_IDS}
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


def test_spread_factor_is_difference_of_two_series():
    days = pd.date_range("2020-01-31", periods=3, freq="ME")
    raw = {sid: pd.Series([1.0, 2.0, 3.0], index=days) for sid in FRED_IDS}
    raw["MORTGAGE30US"] = pd.Series([6.0, 6.5, 6.6], index=days)
    raw["DGS10"] = pd.Series([4.0, 4.2, 4.5], index=days)
    fac = build_factors(raw)
    assert np.allclose(fac["MORTGAGE"].iloc[1:].values, [0.3, -0.2])


def test_no_factor_is_an_exact_combination_of_others():
    ds = make_demo()
    X = ds.factors[FACTOR_KEYS].dropna().values
    assert np.linalg.matrix_rank(X) == len(FACTOR_KEYS)


def test_check_complete_flags_missing_and_stale():
    from data import check_complete
    ds = make_demo(end="2026-10")
    today = pd.Timestamp("2026-11-05")
    assert check_complete(ds.factors, ds.returns, today=today) == []
    f = ds.factors.drop(columns=["COPPER"])
    r = ds.returns.copy(); r.loc[pd.Period("2026-05", "M"):, "XLE"] = np.nan
    probs = check_complete(f, r, today=today)
    assert any("Copper" in p for p in probs) and any("XLE" in p and "stale" in p for p in probs)


def test_load_live_refuses_partial_data(monkeypatch):
    import data as D
    def fake_fred(sid, start=D.START):
        if sid == "THREEFYTP10":
            raise ConnectionError("down")
        idx = pd.bdate_range("2006-01-01", "2026-10-31")
        return pd.Series(np.linspace(1, 3, len(idx)), index=idx, name=sid)
    def fake_prices(tickers, start=D.START):
        idx = pd.bdate_range("2006-01-01", "2026-10-31")
        return pd.DataFrame({t: np.linspace(50, 100, len(idx)) for t in tickers}, index=idx)
    monkeypatch.setattr(D, "fetch_fred", fake_fred)
    monkeypatch.setattr(D, "fetch_prices", fake_prices)
    with pytest.raises(D.MissingDataError, match="THREEFYTP10"):
        D.load_live()
    monkeypatch.setattr(D, "fetch_fred", lambda sid, start=D.START: fake_fred("X") if sid != "x" else None)
    ds = D.load_live()
    assert list(ds.factors.columns) == FACTOR_KEYS


# ---------------------------- inflation module ---------------------------- #
def _demo_pt(fam="PCE"):
    import inflation as I
    from data import make_demo_levels
    L = make_demo_levels()
    return I, L, I.fit(L["BRENT"], L[f"{fam}_H"], L[f"{fam}_C"], fam)


def test_net_oil_price_increase_only_counts_new_highs():
    import inflation as I
    idx = pd.period_range("2020-01", periods=16, freq="M")
    b = pd.Series([100] * 12 + [120, 100, 110, 125], index=idx, dtype=float)
    N = I.oil_terms(b)["N"]
    assert N.iloc[12] > 0                      # 120 breaks the 12m high of 100
    assert N.iloc[14] == 0                     # 110 is below the recent 120 -> no new high
    assert np.isclose(N.iloc[15], 100 * np.log(125 / 120))


def test_passthrough_recovers_demo_parameters():
    from data import DEMO_PT
    I, L, pt = _demo_pt()
    for k, v in DEMO_PT["head"].items():
        if k.startswith("oil chg, month 0") or k.startswith("new high, month 0"):
            assert abs(pt.head.params[k] - v) < 0.006, k


def test_second_wave_smaller_and_views_ordered():
    I, L, pt = _demo_pt()
    imp = I.impulse_comparison(pt)
    assert imp.loc["New 12-month high", "Headline after 12m"] > imp.loc["Back to an earlier level", "Headline after 12m"] > 0
    _, _, ptc = _demo_pt("CPI")
    v = {n: I.cpi_view_value(ptc, p) for n, p in I.view_paths(float(L["BRENT"].iloc[-1])).items()}
    assert v["Oil re-spikes to $110 and holds"] > v["Oil eases to $90 and holds"] > v["Brent slides steadily to $75"]


def test_projection_shapes_and_bands():
    I, L, pt = _demo_pt()
    r = I.project(pt, [95.0] * 12, n_sims=200, seed=0)
    assert len(r["path_index"]) == 12 and r["path_index"][0] == L.index[-1] + 1
    a12 = r["path_index"][-1]
    assert r["head_yoy_p10"].loc[a12] <= r["head_yoy"].loc[a12] <= r["head_yoy_p90"].loc[a12]


def test_check_levels_flags_missing_inflation_series():
    from data import check_levels, make_demo_levels
    L = make_demo_levels()
    assert check_levels(L, today=pd.Timestamp("2026-11-05")) == []
    assert any("Core PCE" in p for p in check_levels(L.drop(columns=["PCE_C"]), today=pd.Timestamp("2026-11-05")))


def test_yield_links_recover_demo_slopes_and_order_views():
    from data import DEMO_LINKS
    I, L, pt = _demo_pt()
    lk = I.fit_links(L, pt)
    for k, true in DEMO_LINKS.items():
        assert abs(lk.fits[k].params["core inflation chg (pp)"] - true["core"]) < 0.15, k
    lb = float(L["BRENT"].iloc[-1])
    v = {n: I.link_view_values(lk, I.scenario_drivers(pt, p, lb)) for n, p in I.view_paths(lb).items()}
    assert v["Oil re-spikes to $110 and holds"]["FED_EXPECT"] > v["Oil eases to $90 and holds"]["FED_EXPECT"] \
        > v["Brent slides steadily to $75"]["FED_EXPECT"]


# ---------------------------- oil-model views ---------------------------- #
def test_oil_model_views_load_and_are_consistent():
    import oil_views as OV
    om = OV.load()
    v_core = OV.views(om, "core", 0.2)
    v_hawk = OV.views(om, "hawkish", 0.2)
    assert set(v_core) == set(om.labels.values())
    s = OV.summary(om, "core")
    o = om.origin_rate("GS2")
    for lbl in v_core:
        assert np.isclose(v_core[lbl]["FED_EXPECT"], s.loc[lbl, "2y"] - o, atol=1e-3)
    # breakevens and CPI don't depend on the Fed rule; the 2y does
    k = "Brent slides steadily to $75"
    assert v_core[k]["BREAKEVEN"] == v_hawk[k]["BREAKEVEN"] and v_core[k]["CPI"] == v_hawk[k]["CPI"]
    assert v_hawk[k]["FED_EXPECT"] < v_core[k]["FED_EXPECT"]
    # more oil -> more CPI and higher breakevens
    assert v_core["Oil re-spikes to $110 and holds"]["CPI"] > v_core["Oil eases to $90 and holds"]["CPI"] > v_core[k]["CPI"]


def test_oil_views_measured_from_current_levels():
    import oil_views as OV
    om = OV.load()
    cur = {"GS2": 4.74, "FII10": 2.78, "T10YIEM": 2.35, "BRENT": 95.0}
    v = OV.views(om, "core", 0.2, cur)["Oil re-spikes to $110 and holds"]
    assert np.isclose(v["OIL"], 100 * np.log(110 / 95), atol=0.01)
    end = OV.scenario_end(om, "hold_110")
    assert np.isclose(v["FED_EXPECT"], end["GS2"] - 4.74, atol=1e-3)


def test_missing_oil_export_raises(tmp_path):
    import oil_views as OV
    with pytest.raises(OV.OilModelMissing):
        OV.load(tmp_path)
