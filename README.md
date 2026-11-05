# Top Down Fund: macro factor model

An interactive app that estimates how each holding responds to macro factors, then lets you
change the weights and see the effect on history, stress tests and forecasts.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens in the browser. The first load downloads about 20 years of data from FRED and Yahoo Finance
(takes ~30 s) and keeps it in memory for 6 hours. Use **Demo (synthetic)** in the sidebar to try it offline.

**Strict data policy:** if any FRED series or ETF fails to download, or a series has stopped updating
(no data in the last 3 months), the app stops and lists exactly what's missing. It never runs on partial
or old cached data. Changing factors or holdings in `config.py` forces a fresh download.

Optional: a free FRED API key (fred.stlouisfed.org > My Account > API Keys) makes downloads more reliable:

```bash
export FRED_API_KEY=your_key      # Windows: set FRED_API_KEY=your_key
```

Tests: `python -m pytest -q`

## What's in the app

| Tab | What it shows |
|---|---|
| Allocation (top) | Edit any weight. Save/load allocations as CSV. Blue = your weights, orange = baseline (the sheet). |
| Overview | Historical stats, model volatility, sensitivity to a 1σ move in each factor, where risk comes from. |
| History | Growth of $100, drawdowns, which factors drove returns, rolling portfolio betas, how the portfolio did in past episodes (2008, taper tantrum, 2022 hikes...). |
| Exposures | Beta of every holding to every factor (colored by t-stat), R², alpha, rolling betas per holding. |
| Stress test | Move the factor sliders (or pick a scenario / a historical episode) and see the one-month hit by holding and by factor. |
| Forecast | Monte Carlo over 1 to 36 months. Enter your macro view (e.g. 2y yield -0.75 pp over 12 months), raise the volatility multiplier for a stressed regime. |
| Inflation | Oil → inflation pass-through (PCE and CPI). Answers "does a second oil wave hit inflation less than the first?", projects headline and core inflation under the three oil paths (or your own), and feeds the CPI move into the 12-month views. |
| Diagnostics | Stationarity (ADF), collinearity (VIF), factor correlations, raw factor history, Excel export. |

## The model

For each holding: `return - T-bill = alpha + Σ beta_k × (monthly change in factor k) + noise`,
OLS with Newey-West standard errors, monthly data. Portfolio betas are weighted sums of holding betas,
so changing weights doesn't require re-estimating anything.

| Factor | FRED series | Monthly change in |
|---|---|---|
| Real yield | DFII10 | 10y TIPS yield (pp) |
| Inflation expectations | T10YIE | 10y breakeven (pp) |
| Fed hike/cut expectations | DGS2 | 2y Treasury yield (pp) |
| Credit spread | BAA10Y | Baa minus 10y (pp) |
| Real activity | INDPRO | Industrial production (%) |
| Dollar | DTWEXBGS | Broad dollar index (%) |
| Oil | DCOILWTICO | WTI (%) |
| Risk sentiment | VIXCLS | VIX (points) |
| Term premium | THREEFYTP10 | 10y Kim-Wright term premium (pp) |
| Copper | PCOPPUSDM | Global copper price (%) |
| Mortgage spread | MORTGAGE30US − DGS10 | 30y mortgage minus 10y Treasury (pp) |
| CPI inflation | CPIAUCSL | Headline CPI, actual (%) |

Note: a 10y−2y curve factor is deliberately *not* included. FRED's breakeven is defined as the 10y nominal minus
the 10y TIPS yield, so the curve equals real yield + breakeven − 2y exactly, and adding it would make the
regression singular. The existing three rate factors already capture curve moves.

## Changing things

- **Weights:** in the app. Save each version as a CSV so you can reload it.
- **Add or swap a holding:** add a line to `ASSETS` in `config.py` (name, ticker, group, default weight,
  and optionally older proxy tickers to extend its history).
- **Add a factor:** add a line to `FACTORS` in `config.py` with its FRED series ID. That's all it takes.
- **Scenarios / episodes:** `PRESETS` and `EPISODES` in `config.py`.

## Things to know

- **Short histories are backfilled with proxies:** XLC with VOX, SCHH with VNQ, IBID with STIP then TIP,
  AGZ with IEI, VGIT with IEF. The Exposures tab shows which ones were used. If a proxy isn't a good match
  for what you actually hold, change it in `config.py`.
- **VGIT is Vanguard's *intermediate*-term Treasury ETF** (~5-6 yr duration). If the sheet meant long-term
  Treasuries, the ticker is VGLT, which has roughly 3x the rate sensitivity.
- **Linear and backward-looking.** Betas come from history and change across regimes: check the rolling-beta
  charts, and set "Estimate betas from" to a later year if you think the recent regime matters more.
- **Factors overlap.** Real yield + breakeven ≈ nominal 10y, and the 2y moves with both. Fed shocks usually
  move several factors at once, so the presets move them together. Moving one slider alone is a
  "holding everything else equal" thought experiment.
- **Industrial production is published with a lag and revised.** Fine for explaining past returns,
  but a real-time backtest should use ALFRED vintages.
- **Monte Carlo uses normal distributions**, so it understates crash risk. Use the volatility multiplier and
  the stress tab for tails.

## Inflation module (`inflation.py`)

Monthly headline and core inflation are regressed on Brent moves split into two parts:
every move (`d`), plus Hamilton's **net oil price increase** (`N`), which is non-zero only when Brent
breaks above its highest level of the past 12 months. A rise back to a level oil has already seen gets
only the `d` effect, so the model measures how much smaller a "second wave" is. Core inflation also
carries its own 3-month trend (persistence), and headline follows core's trend.

The scenario paths start from the last inflation print, use actual Brent for months already known, and
simulate forward with the model's historical noise (10-90% ranges). Only oil drives the paths: wages,
shelter and demand stay at their recent trend, so read them as the oil part of the outlook.

The CPI move in each 12-month oil view (Stress test / Forecast) is now taken from this model instead of
judgment. Extra FRED series: DCOILBRENTEU, PCEPI, PCEPILFE, CPILFESL (strict mode applies to them too).

### Fed / yield links
The 2y yield (Fed path), 10y real yield and 10y breakeven are each regressed, over overlapping 12-month
windows, on the 12-month change in core PCE inflation (y/y) and the 12-month % change in Brent
(Newey-West(12) errors; for the 2y and real yield, windows starting with the 2y below 0.5% are dropped
because the Fed couldn't cut at the zero bound). For each oil view, the inflation module projects core PCE,
and these slopes turn that into the 2y, real-yield and breakeven moves the Stress test and Forecast use.
Note: core inflation drifts back toward its 2006-2026 average in every path, so all views share some
"baseline" easing; the Inflation tab shows a Brent-flat baseline so you can see the part due to oil.
