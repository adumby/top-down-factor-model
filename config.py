"""Static configuration: holdings, macro factors, stress scenarios, historical episodes.

Edit this file to change the universe, default weights or factor definitions.
"""
from __future__ import annotations

from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Holdings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Asset:
    name: str
    ticker: str
    group: str            # Equity / Fixed Income / Commodity / Cash
    weight: float         # default weight in percent
    proxy: tuple[str, ...] = ()   # tickers whose returns backfill history before `ticker` existed (in order)


ASSETS: list[Asset] = [
    Asset("Technology", "VGT", "Equity", 18.47),
    Asset("Consumer Discretionary", "XLY", "Equity", 3.70),
    Asset("Communication Services", "XLC", "Equity", 4.48, proxy=("VOX",)),
    Asset("REIT", "SCHH", "Equity", 1.14, proxy=("VNQ",)),
    Asset("Energy", "XLE", "Equity", 2.99),
    Asset("Materials", "XLB", "Equity", 3.70),
    Asset("Utilities", "XLU", "Equity", 1.42),
    Asset("Industrials", "XLI", "Equity", 3.24),
    Asset("Financials", "XLF", "Equity", 7.12),
    Asset("Health Care", "XLV", "Equity", 6.59),
    Asset("Consumer Staples", "XLP", "Equity", 5.08),
    Asset("Short-Term Corp (IG)", "IGSB", "Fixed Income", 13.50),
    Asset("Short-Term TIPS", "IBID", "Fixed Income", 10.00, proxy=("STIP", "TIP")),
    Asset("Agency Bonds", "AGZ", "Fixed Income", 6.00, proxy=("IEI",)),
    Asset("Treasury (VGIT, intermediate)", "VGIT", "Fixed Income", 5.50, proxy=("IEF",)),
    Asset("Gold", "IAU", "Commodity", 2.50),
    Asset("Cash", "CASH", "Cash", 4.57),
]

TICKERS = [a.ticker for a in ASSETS]
ASSET_BY_TICKER = {a.ticker: a for a in ASSETS}
CASH = "CASH"
RF_FRED_ID = "TB3MS"      # 3-month T-bill (annualised %), used as risk-free / cash return


# --------------------------------------------------------------------------- #
# Macro factors
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Factor:
    key: str
    label: str
    fred: str
    freq: str            # "D" daily (take month-end) or "M" monthly
    transform: str       # "diff" (level change) or "logdiff" (log change x100 = % change)
    unit: str            # unit of the monthly change, shown in the UI
    lo: float            # stress slider range
    hi: float
    step: float
    desc: str
    short: str = ""      # compact label for chart axes


FACTORS: list[Factor] = [
    Factor("REAL_YIELD", "Real yield (10y TIPS)", "DFII10", "D", "diff", "pp", -1.5, 1.5, 0.05,
           "Monthly change in the 10-year TIPS yield, in percentage points.", short="Real yield"),
    Factor("BREAKEVEN", "Inflation expectations (10y breakeven)", "T10YIE", "D", "diff", "pp", -1.0, 1.0, 0.05,
           "Monthly change in the 10-year breakeven inflation rate, in percentage points.", short="Breakeven infl."),
    Factor("FED_EXPECT", "Fed path (2y Treasury yield)", "DGS2", "D", "diff", "pp", -1.5, 1.5, 0.05,
           "Monthly change in the 2-year yield: the market's hike/cut expectations. "
           "Positive = more hikes / fewer cuts priced in.", short="2y yield (Fed path)"),
    Factor("CREDIT", "Credit spread (Baa - 10y)", "BAA10Y", "D", "diff", "pp", -1.5, 1.5, 0.05,
           "Monthly change in Moody's Baa corporate yield minus the 10-year Treasury, in percentage points.", short="Credit spread"),
    Factor("GROWTH", "Real activity (industrial production)", "INDPRO", "M", "logdiff", "%", -3.0, 3.0, 0.1,
           "Monthly % change in industrial production. Published with a lag and revised: "
           "see README on vintage data.", short="Ind. production"),
    Factor("USD", "US dollar (broad index)", "DTWEXBGS", "D", "logdiff", "%", -6.0, 6.0, 0.25,
           "Monthly % change in the Fed's broad trade-weighted dollar index.", short="Dollar"),
    Factor("OIL", "Oil (WTI)", "DCOILWTICO", "D", "logdiff", "%", -30.0, 30.0, 1.0,
           "Monthly % change in the WTI crude spot price.", short="Oil"),
    Factor("VIX", "Risk sentiment (VIX)", "VIXCLS", "D", "diff", "pts", -15.0, 25.0, 0.5,
           "Monthly change in the VIX, in index points.", short="VIX"),
    Factor("TERM_PREMIUM", "Term premium (10y, Kim-Wright)", "THREEFYTP10", "D", "diff", "pp", -1.0, 1.0, 0.05,
           "Monthly change in the 10-year term premium: the extra yield investors demand to hold long "
           "bonds instead of rolling short ones. Moves the long end independently of Fed expectations.",
           short="Term premium"),
    Factor("COPPER", "Copper (global price)", "PCOPPUSDM", "M", "logdiff", "%", -20.0, 20.0, 1.0,
           "Monthly % change in the global copper price (monthly average): a fast read on global "
           "industrial demand.", short="Copper"),
    Factor("MORTGAGE", "Mortgage spread (30y mortgage - 10y)", "MORTGAGE30US-DGS10", "D", "diff", "pp",
           -0.75, 0.75, 0.05,
           "Monthly change in the 30y fixed mortgage rate minus the 10y Treasury yield: housing-finance "
           "stress beyond what Treasury yields already capture.", short="Mortgage spread"),
    Factor("CPI", "CPI inflation (actual, m/m)", "CPIAUCSL", "M", "logdiff", "%", -1.0, 1.5, 0.05,
           "Monthly % change in headline CPI (seasonally adjusted). Actual inflation, versus breakevens "
           "which are expected inflation. Released mid-following-month.", short="CPI (actual)"),
]

FACTOR_KEYS = [f.key for f in FACTORS]
FACTOR_BY_KEY = {f.key: f for f in FACTORS}
# --------------------------------------------------------------------------- #
# Inflation pass-through module (Inflation tab): oil level + price indexes (monthly)
# --------------------------------------------------------------------------- #
OIL_LEVEL_FRED = "DCOILBRENTEU"          # Brent spot, daily -> monthly average
INFLATION_SERIES = {                     # key: (FRED id, label)
    "PCE_H": ("PCEPI", "Headline PCE"),
    "PCE_C": ("PCEPILFE", "Core PCE"),
    "CPI_H": ("CPIAUCSL", "Headline CPI"),
    "CPI_C": ("CPILFESL", "Core CPI"),
}

FRED_IDS = sorted({sid for f in FACTORS for sid in f.fred.split("-")} | {RF_FRED_ID, OIL_LEVEL_FRED}
                  | {v[0] for v in INFLATION_SERIES.values()})


# --------------------------------------------------------------------------- #
# Stress scenarios: one-month factor moves, in each factor's own unit.
# Illustrative magnitudes, not forecasts: tweak freely in the app.
# --------------------------------------------------------------------------- #
PRESETS: dict[str, dict[str, float]] = {
    "Custom (all zero)": {},
    "Hawkish Fed surprise": {"FED_EXPECT": 0.75, "REAL_YIELD": 0.40, "BREAKEVEN": -0.10,
                             "USD": 2.0, "CREDIT": 0.15, "VIX": 4.0},
    "Dovish Fed pivot": {"FED_EXPECT": -0.75, "REAL_YIELD": -0.40, "BREAKEVEN": 0.10,
                         "USD": -2.0, "CREDIT": -0.10, "VIX": -3.0},
    "Inflation scare": {"BREAKEVEN": 0.50, "REAL_YIELD": 0.30, "FED_EXPECT": 0.50, "OIL": 15.0,
                        "USD": 1.0, "GROWTH": -0.5, "VIX": 4.0},
    "Stagflation": {"OIL": 30.0, "BREAKEVEN": 0.60, "GROWTH": -1.5, "REAL_YIELD": 0.20,
                    "FED_EXPECT": 0.30, "CREDIT": 0.60, "VIX": 10.0, "USD": 2.0},
    "Recession / risk-off": {"GROWTH": -3.0, "CREDIT": 1.5, "VIX": 20.0, "FED_EXPECT": -1.0,
                             "REAL_YIELD": -0.70, "BREAKEVEN": -0.40, "OIL": -25.0, "USD": 3.0},
    "Oil spike": {"OIL": 30.0, "BREAKEVEN": 0.20, "USD": 1.0, "VIX": 2.0},
    "Dollar surge": {"USD": 6.0, "OIL": -8.0, "VIX": 3.0, "CREDIT": 0.2},
    "Goldilocks": {"GROWTH": 1.0, "CREDIT": -0.30, "VIX": -4.0, "FED_EXPECT": -0.10, "REAL_YIELD": -0.10},
}

# 12-month oil scenarios come from the fund's oil-inflation model (see oil_views.py and oil_model/).

# (start, end) inclusive, YYYY-MM. Limited by data availability (factors start ~2007).
EPISODES: dict[str, tuple[str, str]] = {
    "2008 financial crisis (Sep 2008 - Mar 2009)": ("2008-09", "2009-03"),
    "2013 taper tantrum (May - Aug 2013)": ("2013-05", "2013-08"),
    "2018 Q4 selloff (Oct - Dec 2018)": ("2018-10", "2018-12"),
    "2020 Covid crash (Feb - Mar 2020)": ("2020-02", "2020-03"),
    "2022 hiking cycle (Jan - Oct 2022)": ("2022-01", "2022-10"),
    "2023 regional bank stress (Mar 2023)": ("2023-03", "2023-03"),
    "Oil: 2011 Libya spike (Feb - Apr 2011)": ("2011-02", "2011-04"),
    "Oil: 2011-12 Brent holds above $100 (May 2011 - Jun 2012)": ("2011-05", "2012-06"),
    "Oil: 2022 Ukraine spike (Feb - Jun 2022)": ("2022-02", "2022-06"),
    "Oil: 2022 oil fades, Fed keeps hiking (Jul - Dec 2022)": ("2022-07", "2022-12"),
}
