"""Top Down Fund: macro factor model.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Force-reload the project's own modules on every run, so newly uploaded config.py / data.py
# are always picked up (Streamlit Cloud can otherwise keep the old versions in memory).
import importlib
import config as _config
import charts as C
import inflation as INF
import data as D
import model as _model
import portfolio as _portfolio
for _m in (_config, C, INF, D, _model, _portfolio):
    importlib.reload(_m)

from config import (ASSETS, ASSET_BY_TICKER, CASH, EPISODES, FACTOR_BY_KEY, FACTOR_KEYS, FACTORS,
                    PRESETS, TICKERS, VIEW_NOTES, VIEWS)
from model import factor_diagnostics, fit_models
from portfolio import (attribution, performance_stats, portfolio_betas, portfolio_returns,
                       risk_decomposition, rolling_ols, sim_stats, simulate, stress, to_fractions)

st.set_page_config(page_title="Top Down Macro Factor Model", page_icon="📈", layout="wide")

NAME = {a.ticker: a.name for a in ASSETS}
LABEL = {f.key: f.label for f in FACTORS}
SHORT = {f.key: f.short or f.label for f in FACTORS}
UNIT = {f.key: f.unit for f in FACTORS}


def lab(t: str) -> str:
    return f"{NAME.get(t, t)} ({t})" if t != CASH else "Cash"


def pct(x, d=1):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}f}%"


# --------------------------------------------------------------------------- #
# Sidebar: data + model settings
# --------------------------------------------------------------------------- #
st.sidebar.title("Model settings")
src_label = st.sidebar.radio("Data source", ["Live (FRED + Yahoo Finance)", "Demo (synthetic)"],
                             help="Live downloads real data (needs internet). Demo uses made-up data "
                                  "with known answers, for trying the app offline.")
SRC = "live" if src_label.startswith("Live") else "demo"


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading FRED and Yahoo Finance data...")
def get_data(src: str, signature: tuple) -> D.Dataset:
    # `signature` changes whenever factors/holdings change in config.py, which forces a fresh download
    return D.load(src)


if st.sidebar.button("Refresh data", help="Clear the cache and re-download"):
    get_data.clear()

try:
    ds = get_data(SRC, (tuple(FACTOR_KEYS), tuple(TICKERS), tuple(D.FRED_IDS)))
except D.MissingDataError as exc:
    st.error("**The model won't run on incomplete data.**\n\n" + str(exc) +
             "\n\nThis is usually a temporary FRED or Yahoo outage: click **Refresh data** in the sidebar "
             "in a few minutes. If one series keeps failing, check its ID on fred.stlouisfed.org.")
    st.stop()
except Exception as exc:
    st.error(f"Could not download live data: {type(exc).__name__}: {exc}\n\n"
             "Check your internet connection, or switch the data source to Demo in the sidebar.")
    st.stop()

AVAILABLE = FACTOR_KEYS
sel_keys = st.sidebar.multiselect(
    "Factors in the model", AVAILABLE, default=AVAILABLE, format_func=lambda k: LABEL[k],
    help="Fewer factors = more stable betas. Real yield, breakevens and the 2y yield move together, "
         "so check the VIF column in Diagnostics.")
if not sel_keys:
    st.warning("Pick at least one factor.")
    st.stop()

first_month = ds.factors.dropna().index.min()
last_month = ds.factors.dropna().index.max()
years = list(range(first_month.year, last_month.year - 4))
est_start_year = st.sidebar.selectbox("Estimate betas from", years, index=0,
                                      help="Start year of the regression window. Later start = betas "
                                           "reflect the recent regime but are noisier.")
hac = st.sidebar.slider("Newey-West lags", 0, 12, 3, help="Corrects standard errors for autocorrelation.")
roll_win = st.sidebar.slider("Rolling window (months)", 24, 120, 60, step=12)

est_start = pd.Period(f"{est_start_year}-01", "M")


@st.cache_data(show_spinner="Estimating factor betas...")
def get_model(src: str, keys: tuple, start: str, lags: int, _ds: D.Dataset):
    sl = slice(pd.Period(start, "M"), None)
    return fit_models(_ds.returns.loc[sl], _ds.rf.loc[sl], _ds.factors.loc[sl], list(keys), hac_lags=lags)


model = get_model(SRC, tuple(sel_keys), str(est_start), hac, ds)

SIGNATURE = (tuple(FACTOR_KEYS), tuple(TICKERS), tuple(D.FRED_IDS))


@st.cache_data(show_spinner="Estimating oil-to-inflation pass-through...")
def get_inflation(src: str, signature: tuple, _ds: D.Dataset):
    L = _ds.levels
    if L is None:
        return None
    return {fam: INF.fit(L["BRENT"], L[f"{fam}_H"], L[f"{fam}_C"], fam) for fam in ("PCE", "CPI")}


INFL = get_inflation(SRC, SIGNATURE, ds)
LAST_BRENT = float(ds.levels["BRENT"].dropna().iloc[-1]) if ds.levels is not None else None


def calibrate_views():
    """Replace the judgment-based CPI value in each 12m view with the pass-through model's projection."""
    out = {k: dict(v) for k, v in VIEWS.items()}
    if INFL is None or "CPI" not in ds.factors.columns:
        return out
    avg = float(INFL["CPI"].head_mm.loc[est_start:].mean())   # same series and window as the CPI factor
    for name, path in INF.view_paths(LAST_BRENT).items():
        if name in out:
            out[name]["CPI"] = round(INF.cpi_view_value(INFL["CPI"], path, hist_mean_mm=avg), 2)
    return out


VIEWS_LIVE = calibrate_views()
CPI_NOTE = (" CPI move comes from the pass-through model (Inflation tab)." if INFL is not None else "")

for n in ds.notes:
    st.sidebar.warning(n)
st.sidebar.caption(f"Data: {first_month.strftime('%b %Y')} to {last_month.strftime('%b %Y')} (monthly). "
                   f"Source: {'FRED, Yahoo Finance' if SRC == 'live' else 'synthetic demo'}.")

# --------------------------------------------------------------------------- #
# Allocation editor
# --------------------------------------------------------------------------- #
st.title("Top Down Fund: macro factor model")

BASE_PCT = pd.Series({a.ticker: a.weight for a in ASSETS})

if "w_version" not in st.session_state:
    st.session_state.w_version = 0
    st.session_state.w_start = BASE_PCT.copy()

with st.expander("Allocation (edit weights here)", expanded=True):
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        start_df = pd.DataFrame({"Holding": [NAME[t] for t in TICKERS], "Ticker": TICKERS,
                                 "Group": [ASSET_BY_TICKER[t].group for t in TICKERS],
                                 "Baseline %": BASE_PCT.values,
                                 "Weight %": st.session_state.w_start.reindex(TICKERS).fillna(0).values})
        edited = st.data_editor(
            start_df, key=f"weights_{st.session_state.w_version}", hide_index=True, num_rows="fixed",
            disabled=["Holding", "Ticker", "Group", "Baseline %"], height=640, width="stretch",
            column_config={"Weight %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0,
                                                                     step=0.01, format="%.2f"),
                           "Baseline %": st.column_config.NumberColumn(format="%.2f")})
    with c2:
        w_pct = pd.Series(edited["Weight %"].to_numpy(dtype=float), index=TICKERS).fillna(0.0)
        total = w_pct.sum()
        st.metric("Total", f"{total:.2f}%", delta=f"{total - 100:+.2f} vs 100%" if abs(total - 100) > 0.005 else None,
                  delta_color="off")
        fix = st.radio("If the total isn't 100%", ["Scale all weights to 100%", "Put the difference in cash"],
                       horizontal=False)
        if fix.startswith("Scale"):
            w_cur = to_fractions(w_pct)
        else:
            w_cur = w_pct / 100.0
            w_cur[CASH] = w_cur[CASH] + (1.0 - w_cur.sum())
            if w_cur[CASH] < 0:
                st.warning(f"Cash is {w_cur[CASH]:.1%}: that's leverage (borrowing at the T-bill rate).")
        w_base = to_fractions(BASE_PCT)

        grp = pd.DataFrame({"Current": w_cur, "Baseline": w_base}).groupby(
            lambda t: ASSET_BY_TICKER[t].group).sum() * 100
        st.dataframe(grp.style.format("{:.1f}%"), width="stretch")

        b1, b2 = st.columns(2)
        if b1.button("Reset to baseline", width="stretch"):
            st.session_state.w_start = BASE_PCT.copy()
            st.session_state.w_version += 1
            st.rerun()
        out = pd.DataFrame({"Ticker": TICKERS, "Holding": [NAME[t] for t in TICKERS],
                            "Weight %": (w_cur * 100).round(4).values})
        b2.download_button("Save weights (CSV)", out.to_csv(index=False), "weights.csv", "text/csv",
                           width="stretch")
        up = st.file_uploader("Load weights (CSV with Ticker and Weight % columns)", type="csv")
        if up is not None and st.session_state.get("last_upload") != up.file_id:
            try:
                u = pd.read_csv(up)
                wcol = [c for c in u.columns if "weight" in c.lower()][0]
                loaded = pd.Series(u[wcol].to_numpy(dtype=float), index=u["Ticker"].str.upper())
                unknown = [t for t in loaded.index if t not in TICKERS]
                st.session_state.w_start = loaded.reindex(TICKERS).fillna(0.0)
                st.session_state.w_version += 1
                st.session_state.last_upload = up.file_id
                if unknown:
                    st.session_state.upload_msg = (f"Ignored tickers not in the model: {', '.join(unknown)}. "
                                                   "Add them to config.py first.")
                st.rerun()
            except Exception as exc:
                st.error(f"Couldn't read that file: {exc}")
        if msg := st.session_state.pop("upload_msg", None):
            st.warning(msg)
        st.caption("Blue = your current weights, orange = baseline (the original sheet) everywhere below.")

WEIGHTS = {"Current": w_cur, "Baseline": w_base}
COLORS = {"Current": C.CURRENT, "Baseline": C.BASELINE}

# shared calculations
B = model.betas0
f_sd = model.factor_cov.values.diagonal() ** 0.5
f_sd = pd.Series(f_sd, index=model.keys)
hist = {k: portfolio_returns(ds.returns, w) for k, w in WEIGHTS.items()}       # (rp, W)

tabs = st.tabs(["Overview", "History", "Exposures", "Stress test", "Forecast", "Inflation", "Diagnostics"])

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tabs[0]:
    stats = {k: performance_stats(hist[k][0], ds.rf) for k in WEIGHTS}
    exante = {}
    for k, w in WEIGHTS.items():
        s = portfolio_betas(B, w)
        fv = float(s @ model.factor_cov.loc[s.index, s.index] @ s)
        wr = w.reindex(model.resid.columns).fillna(0)
        iv = float(wr @ model.resid.cov().fillna(0) @ wr)
        exante[k] = np.sqrt(12 * (fv + iv))

    cols = st.columns(5)
    sc, sb = stats["Current"], stats["Baseline"]
    same = bool(np.allclose(w_cur.values, w_base.reindex(w_cur.index).values, atol=1e-9))

    def dl(text):
        return None if same else text
    if sc:
        cols[0].metric("Ann. return (history)", pct(sc["Ann. return"] * 100),
                       dl(f"{(sc['Ann. return'] - sb['Ann. return']) * 100:+.2f} pp vs baseline"), delta_color="normal")
        cols[1].metric("Ann. volatility (history)", pct(sc["Ann. volatility"] * 100),
                       dl(f"{(sc['Ann. volatility'] - sb['Ann. volatility']) * 100:+.2f} pp"), delta_color="inverse")
        cols[2].metric("Sharpe", f"{sc['Sharpe (vs T-bills)']:.2f}",
                       dl(f"{sc['Sharpe (vs T-bills)'] - sb['Sharpe (vs T-bills)']:+.2f}"))
        cols[3].metric("Max drawdown", pct(sc["Max drawdown"] * 100),
                       dl(f"{(sc['Max drawdown'] - sb['Max drawdown']) * 100:+.2f} pp"))
    cols[4].metric("Model volatility (ex-ante)", pct(exante["Current"]),
                   dl(f"{exante['Current'] - exante['Baseline']:+.2f} pp"), delta_color="inverse",
                   help="Annualised volatility implied by today's weights x factor betas x factor covariance + "
                        "idiosyncratic risk.")
    if sc:
        rp0 = hist["Current"][0].dropna()
        st.caption(f"History uses today's weights held constant (rebalanced monthly), {rp0.index[0].strftime('%b %Y')} to "
                   f"{rp0.index[-1].strftime('%b %Y')}. Holdings without data in early months are re-scaled out.")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for k, w in WEIGHTS.items():
            s = portfolio_betas(B, w) * f_sd
            C.bars(fig, [SHORT[x] for x in s.index], s.values, k, COLORS[k], horizontal=True)
        fig.update_yaxes(autorange="reversed")
        C.finish(fig, "Portfolio return from a typical (1σ) monthly move in each factor", 420,
                 xtitle="% portfolio return")
        st.plotly_chart(fig, width="stretch", theme=None)
        st.caption("Bars = portfolio beta × one standard deviation of the factor's monthly change, so factors "
                   "with different units are comparable. Negative = the portfolio loses when the factor rises.")
    with c2:
        fig = go.Figure()
        for k, w in WEIGHTS.items():
            rd = risk_decomposition(B, w, model.factor_cov, model.resid) * 100
            rd.index = [SHORT.get(x, x) for x in rd.index]
            C.bars(fig, list(rd.index), rd.values, k, COLORS[k], horizontal=True, fmt=".1f")
        fig.update_yaxes(autorange="reversed")
        C.finish(fig, "Where the portfolio's risk comes from (% of variance)", 420, xtitle="% of variance")
        st.plotly_chart(fig, width="stretch", theme=None)
        st.caption("Shares can be negative when a factor hedges others (e.g. rates vs. credit).")

    pb = pd.DataFrame({k: portfolio_betas(B, w) for k, w in WEIGHTS.items()})
    pb["Unit of factor move"] = [UNIT[k] for k in pb.index]
    pb.index = [LABEL[k] for k in pb.index]
    st.markdown("**Portfolio betas**: % portfolio return per 1-unit monthly factor move")
    st.dataframe(pb.style.format({"Current": "{:+.3f}", "Baseline": "{:+.3f}"}), width="stretch")

# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
with tabs[1]:
    rp_c = hist["Current"][0].dropna()
    months = list(rp_c.index)
    if len(months) < 13:
        st.info("Not enough history for this allocation.")
    else:
        r0, r1 = st.select_slider("Period", options=months, value=(months[0], months[-1]),
                                  format_func=lambda p: p.strftime("%b %Y"))
        sl = slice(r0, r1)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for k in WEIGHTS:
                r = hist[k][0].loc[sl].dropna()
                g = 100 * (1 + r / 100).cumprod()
                C.line(fig, g.index.to_timestamp(), g.values, k, COLORS[k])
            C.finish(fig, "Growth of $100", 380, hover="x unified")
            st.plotly_chart(fig, width="stretch", theme=None)
        with c2:
            fig = go.Figure()
            for k in WEIGHTS:
                r = hist[k][0].loc[sl].dropna()
                g = (1 + r / 100).cumprod()
                dd = (g / g.cummax() - 1) * 100
                C.line(fig, dd.index.to_timestamp(), dd.values, k, COLORS[k], fmt=".1f")
            C.finish(fig, "Drawdown (%)", 380, hover="x unified")
            st.plotly_chart(fig, width="stretch", theme=None)

        # attribution
        att = attribution(model, hist["Current"][1], ds.factors, hist["Current"][0], ds.rf).loc[sl]
        st.subheader("What drove returns (current weights)")
        c1, c2 = st.columns([2, 3])
        with c1:
            tot = att.sum()
            tot.index = [SHORT.get(x, x) for x in tot.index]
            tot = tot.sort_values()
            fig = go.Figure()
            fig.add_trace(go.Bar(y=tot.index, x=tot.values, orientation="h",
                                 marker=dict(color=[C.CURRENT if v >= 0 else C.SERIES[7] for v in tot.values],
                                             cornerradius=4, line=dict(color=C.SURFACE, width=2)),
                                 hovertemplate="%{y}: %{x:.1f} pp<extra></extra>"))
            C.finish(fig, "Total contribution over the period (sum of monthly % pts)", 420, legend=False,
                     xtitle="% pts")
            st.plotly_chart(fig, width="stretch", theme=None)
            st.caption("Uses full-sample betas. 'Alpha / unexplained' = average excess return the factors "
                       "don't explain + idiosyncratic noise.")
        with c2:
            fk = st.selectbox("Cumulative contribution of", model.keys, format_func=lambda k: LABEL[k])
            fig = go.Figure()
            cc = att[fk].cumsum()
            C.line(fig, cc.index.to_timestamp(), cc.values, LABEL[fk], C.CURRENT)
            fl = ds.factors[fk].loc[sl].cumsum()
            C.finish(fig, f"Cumulative contribution of {SHORT[fk]} to the portfolio (% pts)", 210,
                     hover="x unified", legend=False)
            st.plotly_chart(fig, width="stretch", theme=None)
            fig = go.Figure()
            C.line(fig, fl.index.to_timestamp(), fl.values, LABEL[fk], C.INK2)
            C.finish(fig, f"{LABEL[fk]}: cumulative change ({UNIT[fk]})", 210, hover="x unified", legend=False)
            st.plotly_chart(fig, width="stretch", theme=None)

        # rolling portfolio beta
        st.subheader(f"Rolling {roll_win}-month portfolio betas")
        rk = st.multiselect("Factors to plot", model.keys, default=model.keys[:3], format_func=lambda k: LABEL[k],
                            max_selections=4, key="roll_port")
        if rk:
            ex = hist["Current"][0] - ds.rf
            rb = rolling_ols(ex, ds.factors[model.keys], roll_win)
            fig = go.Figure()
            for i, k in enumerate(rk):
                if k in rb:
                    C.line(fig, rb.index.to_timestamp(), rb[k].values, SHORT[k], C.SERIES[i], fmt=".2f")
            C.finish(fig, "Rolling beta of the current portfolio (estimated on its own history)", 360,
                     hover="x unified", ytitle="% per unit move")
            st.plotly_chart(fig, width="stretch", theme=None)
            st.caption("If a line drifts a lot, that sensitivity is regime-dependent: lean on recent "
                       "estimates (later 'Estimate betas from' year) when stress testing.")

    # episodes
    st.subheader("Historical episodes")
    rows = []
    for name, (a, b) in EPISODES.items():
        p0, p1 = pd.Period(a, "M"), pd.Period(b, "M")
        row = {"Episode": name}
        for k in WEIGHTS:
            r = hist[k][0].loc[p0:p1]
            row[f"{k}: actual"] = ((1 + r / 100).prod() - 1) * 100 if r.notna().all() and len(r) else np.nan
        fm = ds.factors[model.keys].loc[p0:p1]
        if len(fm) and fm.notna().all().all():
            row["Current: factor-implied"] = float((fm.values @ portfolio_betas(B, w_cur).values).sum()
                                                   + ds.rf.loc[p0:p1].sum())
        else:
            row["Current: factor-implied"] = np.nan
        rows.append(row)
    ep = pd.DataFrame(rows).set_index("Episode")
    st.dataframe(ep.style.format("{:+.1f}%", na_rep="no data"), width="stretch")
    st.caption("Actual = what today's weights would have returned (rebalanced monthly). Factor-implied = "
               "what the betas say from that period's factor moves alone (no alpha). A big gap means "
               "something outside the model drove that episode.")

# --------------------------------------------------------------------------- #
# Exposures
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.markdown("Each holding's beta: **% monthly return per 1-unit factor move** (pp for yields/spreads, "
                "% for prices, points for VIX). Color = t-stat (blue positive, red negative; pale = not "
                "significant).")
    est = model.betas.drop(index=CASH)
    tt = model.tstats.drop(index=CASH).clip(-4, 4)
    fig = go.Figure(go.Heatmap(
        z=tt.values, x=[SHORT[k] for k in model.keys], y=[lab(t) for t in est.index],
        text=est.map(lambda v: "" if pd.isna(v) else f"{v:+.2f}").values, texttemplate="%{text}",
        textfont=dict(size=12, color=C.INK), colorscale=C.DIVERGING, zmin=-4, zmax=4, xgap=2, ygap=2,
        colorbar=dict(title="t-stat", thickness=12),
        customdata=model.tstats.drop(index=CASH).values,
        hovertemplate="%{y}<br>%{x}<br>beta %{text}<br>t-stat %{customdata:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top", tickangle=-30)
    C.finish(fig, "", 640, legend=False)
    st.plotly_chart(fig, width="stretch", theme=None)

    fit_tbl = pd.DataFrame({
        "R²": model.r2, "Alpha (ann., vs T-bills)": model.alpha * 12,
        "Residual vol (ann.)": model.resid_vol * np.sqrt(12), "Months": model.nobs,
        "From": model.start.map(lambda p: p.strftime("%b %Y") if isinstance(p, pd.Period) else "n/a"),
        "Backfilled with": pd.Series(ds.proxy_used).reindex(model.betas.index).fillna(""),
    }).drop(index=CASH)
    fit_tbl.index = [lab(t) for t in fit_tbl.index]
    st.dataframe(fit_tbl.style.format({"R²": "{:.2f}", "Alpha (ann., vs T-bills)": "{:+.1f}%",
                                       "Residual vol (ann.)": "{:.1f}%"}, na_rep="n/a"), width="stretch")
    missing = model.betas.index[model.betas.isna().any(axis=1)].tolist()
    if missing:
        st.warning(f"Not enough data to estimate: {', '.join(missing)}. Treated as zero factor exposure.")

    st.subheader("Rolling betas for one holding")
    c1, c2 = st.columns([1, 2])
    t_sel = c1.selectbox("Holding", [t for t in TICKERS if t != CASH], format_func=lab)
    rk2 = c2.multiselect("Factors", model.keys, default=model.keys[:3], format_func=lambda k: LABEL[k],
                         max_selections=4, key="roll_asset")
    if rk2:
        rb = rolling_ols(ds.returns[t_sel] - ds.rf, ds.factors[model.keys], roll_win)
        fig = go.Figure()
        for i, k in enumerate(rk2):
            if k in rb:
                C.line(fig, rb.index.to_timestamp(), rb[k].values, SHORT[k], C.SERIES[i])
        C.finish(fig, f"{lab(t_sel)}: rolling {roll_win}-month betas", 360, hover="x unified",
                 ytitle="% per unit move")
        st.plotly_chart(fig, width="stretch", theme=None)

# --------------------------------------------------------------------------- #
# Stress test
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.markdown("Set a **one-month move** in each factor and see the instant hit to each holding and the "
                "portfolio. Linear: uses the estimated betas, ignores alpha and the T-bill return.")

    def _apply_preset():
        p = st.session_state.preset
        vals = {}
        if p.startswith("Episode: "):
            a, b = EPISODES[p.removeprefix("Episode: ")]
            vals = ds.factors.loc[pd.Period(a, "M"):pd.Period(b, "M")].sum().to_dict()
        elif p.startswith("12m view: "):
            vals = VIEWS_LIVE[p.removeprefix("12m view: ")]
        else:
            vals = PRESETS.get(p, {})
        for f in FACTORS:
            st.session_state[f"shock_{f.key}"] = float(round(vals.get(f.key, 0.0) / f.step) * f.step)

    opts = list(PRESETS) + [f"12m view: {v}" for v in VIEWS] + [f"Episode: {e}" for e in EPISODES]
    DEFAULT_PRESET = "Hawkish Fed surprise"
    st.session_state.setdefault("preset", DEFAULT_PRESET)
    st.selectbox("Scenario", opts, key="preset", on_change=_apply_preset,
                 help="Episodes use the actual cumulative factor moves from that period, applied as one shock.")
    if st.session_state.preset.startswith("12m view: "):
        st.info(VIEW_NOTES[st.session_state.preset.removeprefix("12m view: ")] + CPI_NOTE +
                " Applied here as the full 12-month move at once (total impact, no alpha or T-bill return).")
    for f in FACTORS:
        st.session_state.setdefault(f"shock_{f.key}", float(PRESETS[DEFAULT_PRESET].get(f.key, 0.0)))

    cols = st.columns(4)
    for i, f in enumerate(FACTORS):
        v = st.session_state[f"shock_{f.key}"]
        cols[i % 4].slider(f"{f.label} ({f.unit})", min_value=float(min(f.lo, v)), max_value=float(max(f.hi, v)),
                           step=f.step, key=f"shock_{f.key}", help=f.desc,
                           disabled=f.key not in model.keys)
    shock = pd.Series({k: st.session_state[f"shock_{k}"] for k in model.keys})
    unused = [LABEL[f.key] for f in FACTORS if f.key not in model.keys and st.session_state[f"shock_{f.key}"] != 0]
    if unused:
        st.caption(f"Ignored (factor not in model): {', '.join(unused)}")

    res = {k: stress(B, w, shock) for k, w in WEIGHTS.items()}
    m1, m2, m3 = st.columns(3)
    m1.metric("Portfolio impact: current", f"{res['Current']['total']:+.2f}%")
    m2.metric("Portfolio impact: baseline", f"{res['Baseline']['total']:+.2f}%")
    m3.metric("Difference (current - baseline)", f"{res['Current']['total'] - res['Baseline']['total']:+.2f} pp")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for k in WEIGHTS:
            ba = res[k]["by_asset"].drop(CASH)
            C.bars(fig, [lab(t) for t in ba.index], ba.values, k, COLORS[k], horizontal=True)
        fig.update_yaxes(autorange="reversed")
        C.finish(fig, "Contribution to portfolio impact, by holding (% pts)", 560, xtitle="% pts")
        st.plotly_chart(fig, width="stretch", theme=None)
    with c2:
        fig = go.Figure()
        for k in WEIGHTS:
            bf = res[k]["by_factor"]
            C.bars(fig, [SHORT[x] for x in bf.index], bf.values, k, COLORS[k], horizontal=True)
        fig.update_yaxes(autorange="reversed")
        C.finish(fig, "Contribution by factor (% pts)", 560, xtitle="% pts")
        st.plotly_chart(fig, width="stretch", theme=None)

    own = (B.drop(index=CASH) * shock).sum(axis=1)
    band = (np.sqrt(((model.se.drop(index=CASH).fillna(0) * shock) ** 2).sum(axis=1)) * 1.96)
    tbl = pd.DataFrame({"Holding's own return": own, "± (95% beta uncertainty)": band,
                        "Weight (current)": w_cur.drop(CASH) * 100,
                        "Contribution (current)": res["Current"]["by_asset"].drop(CASH),
                        "Contribution (baseline)": res["Baseline"]["by_asset"].drop(CASH)})
    tbl.index = [lab(t) for t in tbl.index]
    st.dataframe(tbl.style.format({"Holding's own return": "{:+.2f}%", "± (95% beta uncertainty)": "±{:.2f}%",
                                   "Weight (current)": "{:.2f}%", "Contribution (current)": "{:+.3f} pp",
                                   "Contribution (baseline)": "{:+.3f} pp"}), width="stretch")
    st.caption("The ± band treats beta errors as independent, so it is a rough guide. Correlations between "
               "factors are not enforced: if you only move one rate factor, remember the others usually move too.")

# --------------------------------------------------------------------------- #
# Forecast
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.markdown("Monte Carlo: simulate factor paths from their historical volatility and correlations, "
                "add your macro view as a drift, and push them through the betas. Both allocations see the "
                "same random paths.")
    c1, c2, c3, c4 = st.columns(4)
    st.session_state.setdefault("horizon", 12)
    horizon = c1.slider("Horizon (months)", 1, 36, key="horizon")
    nsim = c2.select_slider("Simulations", [1000, 2000, 5000, 10000], value=5000)
    volm = c3.slider("Volatility multiplier", 0.5, 3.0, 1.0, 0.1,
                     help=">1 = more turbulent than history (e.g. 1.5 for a stressed regime).")
    akeep = c4.slider("Keep historical alpha", 0.0, 1.0, 0.5, 0.1,
                      help="How much of each holding's past unexplained return to assume continues. 0 = none "
                           "(conservative), 1 = all of it. Past alpha is mostly noise, so 0.5 or less is sensible.")

    def _apply_view():
        v = st.session_state.view_preset
        vals = VIEWS_LIVE.get(v, PRESETS.get(v, {}))
        if v in VIEWS:
            st.session_state["horizon"] = 12
        for f in FACTORS:
            st.session_state[f"view_{f.key}"] = float(vals.get(f.key, 0.0))

    with st.expander("Your macro view: expected total change in each factor over the horizon", expanded=True):
        st.selectbox("Start from a scenario", ["Custom (all zero)"] + list(VIEWS) +
                     [p for p in PRESETS if p != "Custom (all zero)"], key="view_preset", on_change=_apply_view,
                     help="Treats a stress scenario as the expected total move over the horizon.")
        if st.session_state.view_preset in VIEW_NOTES:
            st.info(VIEW_NOTES[st.session_state.view_preset] + CPI_NOTE)
        vc = st.columns(4)
        for i, f in enumerate(FACTORS):
            st.session_state.setdefault(f"view_{f.key}", 0.0)
            vc[i % 4].number_input(f"{f.label} ({f.unit})", step=f.step, key=f"view_{f.key}",
                                   disabled=f.key not in model.keys)
    mu_total = pd.Series({k: st.session_state[f"view_{k}"] for k in model.keys})
    rf_now = float(ds.rf.dropna().iloc[-1])

    @st.cache_data(show_spinner="Simulating...")
    def run_sim(_model, key, weights_tuple, rf_now, horizon, nsim, mu_items, volm, akeep):
        wts = {n: pd.Series(dict(w)) for n, w in weights_tuple}
        return simulate(_model, wts, rf_now, horizon, nsim, pd.Series(dict(mu_items)), volm, akeep, seed=42)

    wt = tuple((n, tuple(w.round(8).items())) for n, w in WEIGHTS.items())
    mkey = (SRC, tuple(model.keys), str(est_start), hac)
    sims = run_sim(model, mkey, wt, rf_now, horizon, nsim, tuple(mu_total.items()), volm, akeep)

    st_tbl = pd.DataFrame({k: sim_stats(sims[k], horizon) for k in WEIGHTS})
    c1, c2 = st.columns([3, 2])
    with c1:
        g = (sims["Current"] - 1) * 100
        x = np.arange(0, horizon + 1)
        qs = {p: np.concatenate([[0], np.percentile(g, p, axis=0)]) for p in (5, 25, 50, 75, 95)}
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([qs[95], qs[5][::-1]]),
                                 fill="toself", fillcolor="rgba(42,120,214,0.12)", line=dict(width=0),
                                 name="Current 5-95%", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([qs[75], qs[25][::-1]]),
                                 fill="toself", fillcolor="rgba(42,120,214,0.25)", line=dict(width=0),
                                 name="Current 25-75%", hoverinfo="skip"))
        C.line(fig, x, qs[50], "Current median", C.CURRENT, fmt=".1f")
        gb = (sims["Baseline"] - 1) * 100
        C.line(fig, x, np.concatenate([[0], np.median(gb, axis=0)]), "Baseline median", C.BASELINE,
               dash="dash", fmt=".1f")
        C.finish(fig, "Cumulative return paths (%)", 420, xtitle="Months ahead", hover="x unified")
        st.plotly_chart(fig, width="stretch", theme=None)
    with c2:
        fig = go.Figure()
        terms = {k: (sims[k][:, -1] - 1) * 100 for k in WEIGHTS}
        edges = np.histogram_bin_edges(np.concatenate(list(terms.values())), bins=50)
        mids = (edges[:-1] + edges[1:]) / 2
        for k in WEIGHTS:
            cnt = np.histogram(terms[k], bins=edges)[0] / len(terms[k]) * 100
            fig.add_trace(go.Scatter(x=mids, y=cnt, name=k, mode="lines", line=dict(color=COLORS[k], width=2,
                                     shape="hvh", dash=None if k == "Current" else "dash"),
                                     hovertemplate="%{x:.1f}%: %{y:.1f}% of paths<extra>" + k + "</extra>"))
        C.finish(fig, f"Return after {horizon} months: distribution", 420, xtitle="% return",
                 ytitle="% of paths", hover="x unified")
        st.plotly_chart(fig, width="stretch", theme=None)

    st_tbl["Difference"] = st_tbl["Current"] - st_tbl["Baseline"]
    st.dataframe(st_tbl.style.format("{:+.2f}%").format("{:.1f}%", subset=pd.IndexSlice[["P(loss)"], :]),
                 width="stretch")

    # expected return decomposition
    parts = {}
    for k, w in WEIGHTS.items():
        s = portfolio_betas(B, w)
        a = float((w.reindex(model.alpha.index).fillna(0) * model.alpha.fillna(0)).sum()) * akeep * horizon
        d = {"T-bills": rf_now * horizon, "Alpha (kept share)": a}
        d.update({SHORT[k2]: float(s[k2] * mu_total[k2]) for k2 in model.keys})
        parts[k] = pd.Series(d)
    pdf = pd.DataFrame(parts)
    pdf = pdf[(pdf.abs() > 1e-9).any(axis=1)]
    fig = go.Figure()
    for k in WEIGHTS:
        C.bars(fig, list(pdf.index), pdf[k].values, k, COLORS[k], horizontal=True)
    fig.update_yaxes(autorange="reversed")
    C.finish(fig, f"Expected {horizon}-month return by source (simple sum, % pts)", 80 + 40 * len(pdf),
             xtitle="% pts")
    st.plotly_chart(fig, width="stretch", theme=None)
    st.caption(f"Current 3-month T-bill: {rf_now * 12:.2f}% a year. Factor views are applied as a steady drift "
               "over the horizon. The model is linear and uses historical volatility: it will understate "
               "crash risk unless you raise the volatility multiplier.")

# --------------------------------------------------------------------------- #
# Inflation (oil pass-through)
# --------------------------------------------------------------------------- #
with tabs[5]:
    if INFL is None:
        st.info("Inflation data isn't loaded. Click Refresh data in the sidebar.")
    else:
        fam = st.radio("Inflation measure", ["PCE", "CPI"], horizontal=True,
                       help="PCE is the Fed's target measure. CPI is what feeds the factor model's views.")
        pt = INFL[fam]
        st.markdown(f"Model of how oil (Brent, monthly average) feeds into **{fam}** inflation, estimated on "
                    f"{pt.start.strftime('%b %Y')} to {pt.end.strftime('%b %Y')}. Oil moves count in two parts: "
                    "every move, plus an extra term only when oil breaks to a **new 12-month high** "
                    "(Hamilton's net oil price increase). A rise back to a level oil has already been at only "
                    "gets the first part.")

        # ---- 1. second-wave question
        st.subheader("Does a second oil wave hit inflation less?")
        imp = INF.impulse_comparison(pt)
        pv = INF.new_high_pvalues(pt)
        nh, bk = imp.loc["New 12-month high"], imp.loc["Back to an earlier level"]
        m1, m2, m3 = st.columns(3)
        m1.metric("+10% oil to a NEW high: headline after 12m", f"{nh['Headline after 12m']:+.2f} pp")
        m2.metric("+10% oil BACK to an earlier level: headline after 12m", f"{bk['Headline after 12m']:+.2f} pp")
        share = bk["Headline after 12m"] / nh["Headline after 12m"] if nh["Headline after 12m"] > 0 else np.nan
        m3.metric("Second wave as a share of the first", "n/a" if pd.isna(share) else f"{share:.0%}")
        if pv["headline"] < 0.05:
            st.success(f"Yes. In the data, oil rising to a new high adds significantly more to headline {fam} "
                       f"than the same rise back to an old level (p = {pv['headline']:.3f}). "
                       f"Core: p = {pv['core']:.3f}" + (" (also significant)." if pv["core"] < 0.05 else
                                                        " (not significant: core barely tells the two apart)."))
        else:
            st.warning(f"The data can't reliably tell the two apart for headline {fam} (p = {pv['headline']:.2f}). "
                       "Treat the difference below as indicative.")
        fig = go.Figure()
        for lbl, color in (("New 12-month high", C.SERIES[2]), ("Back to an earlier level", C.SERIES[4])):
            C.bars(fig, list(imp.columns), imp.loc[lbl].values, lbl, color)
        C.finish(fig, f"Effect of a permanent +10% oil move on {fam} inflation (y/y, pp)", 360, ytitle="pp")
        st.plotly_chart(fig, width="stretch", theme=None)
        with st.expander("Estimated coefficients"):
            for nm, res in (("Headline", pt.head), ("Core", pt.core)):
                tbl = pd.DataFrame({"coef": res.params, "t-stat": res.tvalues})
                st.markdown(f"**{nm} {fam}** (monthly inflation, %). R² {res.rsquared:.2f}, {int(res.nobs)} months")
                st.dataframe(tbl.style.format({"coef": "{:+.4f}", "t-stat": "{:+.2f}"}), width="stretch")
            st.caption("'trend' = average of the last 3 months of core inflation (persistence). Oil terms are in "
                       "% moves of Brent. Newey-West standard errors.")

        # ---- 2. scenario paths
        st.subheader(f"{fam} inflation under each oil path")
        H = 18
        paths = INF.view_paths(LAST_BRENT, months=H)
        with st.expander("Add a custom oil path"):
            cc1, cc2, cc3 = st.columns(3)
            use_custom = cc1.checkbox("Include custom path", value=False)
            tgt = cc2.number_input("Brent target ($)", 30.0, 200.0, 100.0, 1.0)
            ramp = cc3.slider("Months to get there", 1, 12, 3)
            if use_custom:
                paths[f"Custom: ${tgt:.0f} in {ramp}m, then holds"] = INF.custom_path(LAST_BRENT, tgt, ramp, H)
        sel = st.multiselect("Paths", list(paths), default=list(paths))
        pal = {n: c for n, c in zip(paths, [C.SERIES[3], C.SERIES[2], C.SERIES[6], C.SERIES[4]])}

        @st.cache_data(show_spinner=False)
        def _proj(src, signature, fam_, path_tuple):
            return INF.project(INFL[fam_], np.array(path_tuple), n_sims=1000, seed=1)

        res = {n: _proj(SRC, SIGNATURE, fam, tuple(np.round(paths[n], 4))) for n in sel}
        hist_b = ds.levels["BRENT"].dropna().iloc[-24:]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            C.line(fig, hist_b.index.to_timestamp(), hist_b.values, "Actual", C.INK2, fmt=".0f")
            for n in sel:
                r = res[n]
                x = [hist_b.index[-1]] + list(r["path_index"])
                y = [hist_b.iloc[-1]] + list(paths[n])
                C.line(fig, pd.PeriodIndex(x).to_timestamp(), y, n, pal[n], dash="dot", fmt=".0f")
            C.finish(fig, "Brent ($, monthly average)", 380, hover="x unified")
            st.plotly_chart(fig, width="stretch", theme=None)
        for col, key, title in ((c2, "head", f"Headline {fam}, y/y %"), (None, "core", f"Core {fam}, y/y %")):
            fig = go.Figure()
            first = res[sel[0]] if sel else None
            if first is not None:
                act = first[f"{key}_yoy"].loc[:first["last_actual"]].dropna().iloc[-24:]
                C.line(fig, act.index.to_timestamp(), act.values, "Actual", C.INK2)
                for n in sel:
                    r = res[n]
                    proj = r[f"{key}_yoy"].loc[r["last_actual"]:]
                    C.line(fig, proj.index.to_timestamp(), proj.values, n, pal[n], dash="dot")
            C.finish(fig, title, 380, hover="x unified")
            if col is not None:
                with col:
                    st.plotly_chart(fig, width="stretch", theme=None)
            else:
                st.plotly_chart(fig, width="stretch", theme=None)

        if sel:
            rows = []
            for n in sel:
                r = res[n]
                fi = r["path_index"]
                a6, a12 = fi[5], fi[11]
                rows.append({
                    "Path": n,
                    "Brent in 12m": paths[n][11],
                    f"Headline now ({r['last_actual'].strftime('%b %y')})": r["head_yoy"].loc[r["last_actual"]],
                    f"Headline {a6.strftime('%b %y')}": r["head_yoy"].loc[a6],
                    f"Headline {a12.strftime('%b %y')}": r["head_yoy"].loc[a12],
                    "Headline 12m range (10-90%)": f"{r['head_yoy_p10'].loc[a12]:.1f} to {r['head_yoy_p90'].loc[a12]:.1f}",
                    f"Core {a12.strftime('%b %y')}": r["core_yoy"].loc[a12],
                    "Core 12m range (10-90%)": f"{r['core_yoy_p10'].loc[a12]:.1f} to {r['core_yoy_p90'].loc[a12]:.1f}",
                })
            tb = pd.DataFrame(rows).set_index("Path")
            num = [c for c in tb.columns if "range" not in c and c != "Brent in 12m"]
            st.dataframe(tb.style.format({**{c: "{:.2f}%" for c in num}, "Brent in 12m": "${:.0f}"}),
                         width="stretch")
        st.caption(f"Starts from the last {fam} print and uses actual Brent for months already known. 'Now' is "
                   "the last actual reading. The ranges add the model's historical month-to-month noise. "
                   "Only oil drives these paths: wages, shelter, tariffs and demand are held at their recent trend, "
                   "so treat them as the oil part of the inflation outlook, not a full forecast.")

        # ---- 3. link to factor-model views
        st.subheader("What this feeds into the 12-month views")
        rows = []
        for n in VIEWS:
            rows.append({"View": n, "CPI move: my original judgment": VIEWS[n].get("CPI", np.nan),
                         "CPI move: pass-through model (used now)": VIEWS_LIVE[n].get("CPI", np.nan)})
        st.dataframe(pd.DataFrame(rows).set_index("View").style.format("{:+.2f}"), width="stretch")
        st.caption("CPI factor = projected 12-month CPI inflation minus its historical average pace, in pp. "
                   "The Stress test and Forecast tabs now use the model's number. Other factor moves in the "
                   "views (2y, real yields, breakevens...) are still judgment.")

# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.subheader("Factor checks")
    fd = factor_diagnostics(ds.factors.loc[est_start:], model.keys)
    fd.index = [LABEL[k] for k in fd.index]
    st.dataframe(fd.style.format({"mean": "{:+.3f}", "stdev": "{:.3f}", "ADF p-value": "{:.3f}",
                                  "VIF": "{:.2f}", "months": "{:.0f}"}), width="stretch")
    st.caption("ADF p-value < 0.05 = stationary (good for regression). VIF > 5 = factor overlaps a lot with "
               "the others: its beta will be unstable; consider dropping one of the overlapping factors.")

    corr = ds.factors.loc[est_start:, model.keys].corr()
    lbls = [SHORT[k] for k in corr.index]
    fig = go.Figure(go.Heatmap(z=corr.values, x=lbls, y=lbls, zmin=-1, zmax=1, colorscale=C.DIVERGING,
                               text=corr.round(2).values, texttemplate="%{text}", xgap=2, ygap=2,
                               textfont=dict(color=C.INK),
                               hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
                               colorbar=dict(thickness=12)))
    fig.update_yaxes(autorange="reversed")
    C.finish(fig, "Factor correlations (monthly changes)", 520, legend=False)
    st.plotly_chart(fig, width="stretch", theme=None)

    st.subheader("Factor history")
    fk = st.selectbox("Factor", model.keys, format_func=lambda k: LABEL[k], key="diag_f")
    s = ds.factors[fk].dropna()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s.index.to_timestamp(), y=s.values, marker=dict(color=C.CURRENT),
                         hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>"))
    C.finish(fig, f"{LABEL[fk]}: monthly change ({UNIT[fk]})", 300, legend=False)
    st.plotly_chart(fig, width="stretch", theme=None)
    st.caption(FACTOR_BY_KEY[fk].desc + (f" FRED series: {FACTOR_BY_KEY[fk].fred}."))

    st.subheader("Download")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xw:
        model.betas.to_excel(xw, sheet_name="betas")
        model.tstats.to_excel(xw, sheet_name="tstats")
        pd.DataFrame({"r2": model.r2, "alpha_monthly": model.alpha, "nobs": model.nobs}).to_excel(xw, sheet_name="fit")
        f_out = ds.factors.copy(); f_out.index = f_out.index.to_timestamp(); f_out.to_excel(xw, sheet_name="factors")
        r_out = ds.returns.copy(); r_out.index = r_out.index.to_timestamp(); r_out.to_excel(xw, sheet_name="returns")
    st.download_button("Model output + data (Excel)", buf.getvalue(), "macro_factor_model.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
