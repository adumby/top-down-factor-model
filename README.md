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
| Inflation | The fund's oil-inflation model (top-down-oil): Brent, headline/core PCE and CPI paths and the rates chain for every oil scenario, plus a second-wave test ("does a rise back to an old oil level hit inflation less?"). |
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
| Dollar | DTWEXBGS (before 2006: DTWEXB) | Broad dollar index (%). The older goods-only index fills 2003-05 so the factors start with TIPS in 2003 |
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

## Oil scenarios (from the fund's oil-inflation model)

The 12-month oil scenarios in the Stress test and Forecast tabs come from the **top-down-oil** model
(ARDL inflation equations anchored to the 10y breakeven, oil → breakevens, a Fed reaction function for
the 2y, and the 2y/breakeven pass-through to the 10y real yield). This app reads that model's exported
results from the `oil_model/` folder; `oil_views.py` turns each scenario into factor moves:

| Factor | From the oil model |
|---|---|
| Breakeven | change in the 10y breakeven to the end of the horizon |
| Fed path (2y) | change in the 2y, under the Fed rule chosen in the sidebar (core PCE rule or hawkish headline rule) |
| Real yield | change in the 10y real yield, same rule |
| CPI | projected 12-month headline CPI inflation minus its historical average pace |
| Oil | % change in Brent |

Moves are measured from the **latest actual levels** in this app's live data, so "$110 and holds" means
+15% from today's Brent, not from the oil model's starting month. Credit, VIX, dollar, growth, copper,
term premium and mortgage spread stay at zero because the oil model doesn't cover them.

**Updating:** in the top-down-oil repo run `python -m oilinfl`, then `python scripts/export_for_factor_app.py`,
and upload the 5 files in `factor_app_export/` to this repo's `oil_model/` folder. The sidebar warns when
the export is more than 3 months old; if the folder is missing, the oil scenarios are switched off and the
app says so.

### Second-wave test (`inflation.py`)
A separate check on live data: monthly inflation is regressed on Brent moves split into every move plus
Hamilton's **net oil price increase** (non-zero only when Brent breaks its 12-month high). It measures how
much less a rise back to an old level adds to inflation. The fund model treats every move alike, so this
is a cross-check on the $110 path, which stays below the spring 2026 peak.
