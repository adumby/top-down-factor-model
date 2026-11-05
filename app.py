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
import oil_views as OV
import scenario_fill as SF
import data as D
import model as _model
import portfolio as _portfolio
for _m in (_config, C, INF, OV, SF, D, _model, _portfolio):
    importlib.reload(_m)

from config import (ASSETS, ASSET_BY_TICKER, CASH, EPISODES, FACTOR_BY_KEY, FACTOR_KEYS, FACTORS,
                    PRESETS, TICKERS)
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


# --------------------------------------------------------------------------- #
# Oil scenarios from the fund's oil-inflation model (top-down-oil repo -> oil_model/ folder)
# --------------------------------------------------------------------------- #
try:
    OM, OM_ERR = OV.load(), None
except OV.OilModelMissing as exc:
    OM, OM_ERR = None, str(exc)


def _flag_reapply():
    st.session_state["_reapply_oil"] = True


st.sidebar.markdown("---")
OIL_RULE = st.sidebar.radio("Fed reaction in oil scenarios", list(OV.RULE_LABELS), format_func=OV.RULE_LABELS.get,
                            key="oil_rule", on_change=_flag_reapply,
                            help="How the 2y and real yields respond in the oil-model scenarios. Core rule = what "
                                 "the Fed has done since 2016. Hawkish = the Fed doesn't look through energy.")
FILL_MODES = {"supply": "Supply-driven oil (demand held at average)",
              "history": "Historical mix (supply and demand)",
              "off": "Off: other factors stay at zero"}
FILL_MODE = st.sidebar.radio(
    "Other factors in oil scenarios", list(FILL_MODES), format_func=FILL_MODES.get, key="oil_fill_mode",
    on_change=_flag_reapply,
    help="The oil model sets oil, CPI, breakevens, the 2y and the real yield. The other factors are set to what they "
         "typically did alongside those moves (12-month changes, 2008-09 and 2020 crisis windows excluded). "
         "Supply-driven holds copper and industrial production at their average, so an oil move is read as supply "
         "(e.g. a war ending), not a global slowdown. Historical mix lets history decide, and history's big oil "
         "drops were mostly demand slumps.")
FILL_ON = FILL_MODE != "off"


def _current_levels() -> dict:
    """Latest actual levels the scenario moves are measured from."""
    L = ds.levels
    if L is None:
        return {}
    out = {}
    for k, col in (("T10YIEM", "BE10"), ("GS2", "Y2"), ("FII10", "REAL10"), ("BRENT", "BRENT")):
        if col in L.columns and L[col].notna().any():
            out[k] = float(L[col].dropna().iloc[-1])
    return out


CURRENT = _current_levels()
CURRENT_MONTH = ds.levels["BRENT"].dropna().index[-1] if ds.levels is not None else None
if OM is not None:
    _cpi_avg = float(ds.factors["CPI"].loc[est_start:].mean()) if "CPI" in ds.factors else 0.0
    VIEWS_LIVE = OV.views(OM, OIL_RULE, _cpi_avg, CURRENT)
    VIEW_NOTES_LIVE = OV.notes(OM, OIL_RULE)
    FILL_KEYS, FILL_R2 = [], None
    if FILL_ON:
        _S = SF.twelve_month_changes(ds.factors, FACTOR_KEYS, start=est_start)
        _spec = ["OIL", "CPI", "BREAKEVEN", "FED_EXPECT", "REAL_YIELD"]
        FILL_R2 = SF.explained_share(_S, _spec)
        _hold = SF.DEMAND_KEYS if FILL_MODE == "supply" else ()
        FILL_R2 = SF.explained_share(_S, _spec + [k for k in _hold if k in _S.columns])
        for _k in list(VIEWS_LIVE):
            VIEWS_LIVE[_k], FILL_KEYS = SF.fill(VIEWS_LIVE[_k], _S, hold_average=_hold)
        _filled_txt = (" Credit, VIX, the dollar, term premium and mortgage spread are set to what they typically did "
                       f"alongside these moves (12-month changes since {est_start.year}, crisis windows excluded)"
                       + ("; copper and industrial production are held at their average, so the oil move is read as "
                          "supply-driven." if _hold else "; growth and copper are filled too (historical mix of "
                          "supply and demand)."))
        VIEW_NOTES_LIVE = {k: v.replace(" Credit, VIX, dollar and growth are held flat because the oil model "
                                        "doesn't cover them.", _filled_txt) for k, v in VIEW_NOTES_LIVE.items()}
    if OV.staleness_months(OM) > 3:
        st.sidebar.warning(f"The oil-model export is from {OM.origin.strftime('%b %Y')}. Re-run it with new data "
                           "and upload a fresh export.")
else:
    VIEWS_LIVE, VIEW_NOTES_LIVE = {}, {}
    st.sidebar.error("Oil-model scenarios unavailable: " + OM_ERR)
OILP = "Oil model: "

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
        elif p.startswith(OILP):
            vals = VIEWS_LIVE.get(p.removeprefix(OILP), {})
        else:
            vals = PRESETS.get(p, {})
        for f in FACTORS:
            st.session_state[f"shock_{f.key}"] = float(round(vals.get(f.key, 0.0) / f.step) * f.step)

    opts = list(PRESETS) + [f"{OILP}{v}" for v in VIEWS_LIVE] + [f"Episode: {e}" for e in EPISODES]
    DEFAULT_PRESET = "Hawkish Fed surprise"
    st.session_state.setdefault("preset", DEFAULT_PRESET)
    if st.session_state.preset not in opts:          # e.g. a scenario renamed since the last run
        st.session_state.preset = DEFAULT_PRESET
        _apply_preset()
    if st.session_state.pop("_reapply_oil", False):  # Fed rule changed: refresh oil-scenario sliders
        st.session_state["_reapply_oil_view"] = True
        if st.session_state.preset.startswith(OILP):
            _apply_preset()
    st.selectbox("Scenario", opts, key="preset", on_change=_apply_preset,
                 help="Episodes use the actual cumulative factor moves from that period, applied as one shock.")
    if st.session_state.preset.startswith(OILP):
        st.info(VIEW_NOTES_LIVE.get(st.session_state.preset.removeprefix(OILP), "") +
                " Moves are measured from the latest actual levels and applied here as the full 12-month move at once (total impact, no alpha or T-bill return).")
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
        vals = VIEWS_LIVE.get(v.removeprefix(OILP), {}) if v.startswith(OILP) else PRESETS.get(v, {})
        if v.startswith(OILP):
            st.session_state["horizon"] = 12
        for f in FACTORS:
            st.session_state[f"view_{f.key}"] = float(vals.get(f.key, 0.0))

    with st.expander("Your macro view: expected total change in each factor over the horizon", expanded=True):
        vopts = (["Custom (all zero)"] + [f"{OILP}{v}" for v in VIEWS_LIVE] +
                 [p for p in PRESETS if p != "Custom (all zero)"])
        if st.session_state.get("view_preset") not in vopts:
            st.session_state["view_preset"] = "Custom (all zero)"
        if st.session_state.pop("_reapply_oil_view", False) and st.session_state["view_preset"].startswith(OILP):
            _apply_view()
        st.selectbox("Start from a scenario", vopts, key="view_preset", on_change=_apply_view,
                     help="Oil-model scenarios are 12-month paths. Other presets are treated as the expected total "
                          "move over the horizon.")
        if st.session_state.view_preset.startswith(OILP):
            st.info(VIEW_NOTES_LIVE.get(st.session_state.view_preset.removeprefix(OILP), ""))
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
    # ================= the fund's oil-inflation model =================
    st.subheader("Fund oil-inflation model")
    if OM is None:
        st.error(OM_ERR)
    else:
        st.markdown(
            f"Scenarios from the **top-down-oil** model: ARDL inflation equations anchored to the 10y breakeven, "
            f"oil → breakevens, a Fed reaction function for the 2y and the pass-through to the 10y real yield. "
            f"Data through **{OM.origin.strftime('%b %Y')}**, projected to **{OM.end.strftime('%b %Y')}**. Fed rule: "
            f"**{OV.RULE_LABELS[OIL_RULE]}** (change it in the sidebar).")
        labels = list(OM.labels.values())
        mine = [l for l in labels if l in ("Oil re-spikes to $110 and holds", "Oil eases to $90 and holds",
                                           "Brent slides steadily to $75", "Brent flat at $105")]
        sel = st.multiselect("Scenarios", labels, default=mine or labels[:4], key="om_sel")
        key_of = {v: k for k, v in OM.labels.items()}
        # fixed colour per scenario (follows the scenario, never its position in the list)
        _fixed = {"hold_110": C.SERIES[7], "hold_90": C.SERIES[2], "slide_75": C.SERIES[6], "flat_105": C.SERIES[3],
                  "flat_114": C.SERIES[0], "up20_nov": C.SERIES[1], "down25_nov": C.SERIES[5],
                  "normalize_75": C.SERIES[4]}
        pal = {l: _fixed.get(k, C.MUTED) for k, l in OM.labels.items()}

        def _legend_below(fig):
            fig.update_layout(legend=dict(orientation="h", y=-0.14, yanchor="top", x=0, xanchor="left"),
                              margin=dict(t=48, b=10))
            return fig

        def _chart(target, title):
            fig = go.Figure()
            h = OM.infl_hist[OM.infl_hist["target"] == target].set_index("date")["yoy"].iloc[-24:]
            C.line(fig, h.index.to_timestamp(), h.values, "Actual", C.INK2)
            for l in sel:
                q = OM.paths[(OM.paths["scenario"] == key_of[l]) & (OM.paths["target"] == target)].set_index("date")
                q = q.loc[q.index > OM.origin]["yoy"]
                x = [h.index[-1]] + list(q.index)
                y = [h.iloc[-1]] + list(q.values)
                C.line(fig, pd.PeriodIndex(x).to_timestamp(), y, l, pal[l], dash="dot")
            C.finish(fig, title, 430, hover="x unified")
            return _legend_below(fig)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            if ds.levels is not None:
                hb = ds.levels["BRENT"].dropna().iloc[-24:]
                C.line(fig, hb.index.to_timestamp(), hb.values, "Actual", C.INK2, fmt=".0f")
            for l in sel:
                q = OM.paths[(OM.paths["scenario"] == key_of[l]) & (OM.paths["target"] == "headline_pce")]
                q = q.set_index("date")["brent"]
                C.line(fig, q.index.to_timestamp(), q.values, l, pal[l], dash="dot", fmt=".0f")
            C.finish(fig, "Brent ($, monthly average)", 430, hover="x unified")
            st.plotly_chart(_legend_below(fig), width="stretch", theme=None)
        with c2:
            st.plotly_chart(_chart("headline_pce", "Headline PCE, y/y %"), width="stretch", theme=None)
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(_chart("core_pce", "Core PCE, y/y %"), width="stretch", theme=None)
        with c4:
            st.plotly_chart(_chart("headline_cpi", "Headline CPI, y/y %"), width="stretch", theme=None)

        summ = OV.summary(OM, OIL_RULE)
        summ = summ.loc[[l for l in sel if l in summ.index]]
        pct_cols = ["Headline PCE", "Core PCE", "Headline CPI", "Core CPI", "10y breakeven", "2y", "10y real"]
        st.markdown(f"**At {OM.end.strftime('%b %Y')}** (10-90% ranges for PCE)")
        st.dataframe(summ.style.format({**{c: "{:.2f}%" for c in pct_cols}, "Brent at end ($)": "${:.0f}",
                                        "XLK, real-yield channel": "{:+.1f}%"}), width="stretch")
        st.caption(f"Actual at {OM.origin.strftime('%b %Y')}: 2y {OM.origin_rate('GS2'):.2f}%, 10y real "
                   f"{OM.origin_rate('FII10'):.2f}%, 10y breakeven {OM.origin_rate('T10YIEM'):.2f}%, Brent "
                   f"${OM.meta['brent_origin']:.0f}. Paths from the oil model's own outputs; to update, re-run it and "
                   "upload a new export to the oil_model/ folder.")

        st.markdown("**What goes into the factor scenarios** (Stress test and Forecast tabs, 12-month moves)")
        base = (f"Measured from the latest actual levels ({CURRENT_MONTH.strftime('%b %Y')}): "
                + ", ".join(f"{k} {v:.2f}" for k, v in CURRENT.items())) if CURRENT else \
            "Measured from the oil model's origin month."
        vt = pd.DataFrame({l: VIEWS_LIVE[l] for l in sel if l in VIEWS_LIVE}).T
        vt = vt[[k for k in FACTOR_KEYS if k in vt.columns]]
        vt.columns = [f"{SHORT[k]} ({UNIT[k]})" + (" *" if k in FILL_KEYS else "") for k in vt.columns]
        st.dataframe(vt.style.format("{:+.2f}"), width="stretch")
        if FILL_KEYS:
            r2 = ", ".join(f"{SHORT[k]} {FILL_R2[k]:.0%}" for k in FILL_KEYS)
            st.caption(base + ". **\\*** = filled with its typical co-movement, not from the oil model. How much "
                       f"of each filled factor's 12-month moves the oil-model factors explain: {r2}. Low shares "
                       "mean history doesn't pin that factor down, so its filled value is close to its average.")
        else:
            st.caption(base + ". Credit, VIX, dollar, growth, copper, term premium and mortgage spread are held "
                       "flat (switch on 'Fill other factors' in the sidebar to use their typical co-movement).")

    # ================= extra check: second-wave test =================
    st.markdown("---")
    st.subheader("Extra check: does a second oil wave hit inflation less?")
    if INFL is None:
        st.info("Inflation data isn't loaded. Click Refresh data in the sidebar.")
    else:
        fam = st.radio("Inflation measure", ["PCE", "CPI"], horizontal=True, key="sw_fam")
        pt = INFL[fam]
        st.markdown(f"A separate quick test on live data ({pt.start.strftime('%b %Y')} to "
                    f"{pt.end.strftime('%b %Y')}): it splits oil moves into every move plus an extra term only when "
                    "Brent breaks a **new 12-month high** (Hamilton's net oil price increase). The fund model above "
                    "treats every move alike, so this tells you whether a rise back to an old level should count "
                    "for less.")
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
