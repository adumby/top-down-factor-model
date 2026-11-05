"""Scenario views from the Top Down oil-inflation model (repo: top-down-oil).

That model is the fund's inflation model: ARDL(6,12) inflation equations anchored to the 10y breakeven,
oil -> breakevens, an estimated Fed reaction function for the 2y (core rule, or a hawkish headline rule),
and the 2y/breakeven pass-through to the 10y real yield. Its results are exported with
`python scripts/export_for_factor_app.py` and uploaded to this app's `oil_model/` folder.

Each oil scenario becomes a set of 12-month factor moves (origin -> origin + 12 months):
  BREAKEVEN  = change in the 10y breakeven
  FED_EXPECT = change in the 2y yield (core or hawkish Fed rule)
  REAL_YIELD = change in the 10y real yield (same rule)
  CPI        = projected 12-month headline CPI inflation minus its historical average pace
  OIL        = % change in Brent (the factor model uses WTI; % moves track closely)
Every other factor is left at zero: the oil model doesn't cover credit, VIX, the dollar or growth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).parent / "oil_model"
FILES = ["meta.json", "rates_scenarios.csv", "rates_history.csv", "scenario_paths.csv", "inflation_history.csv"]
RULES = {"core": ("GS2", "FII10", "XLK_implied_pct"),
         "hawkish": ("GS2_hawk", "FII10_hawk", "XLK_implied_pct_hawk")}
RULE_LABELS = {"core": "Fed reacts to core PCE (estimated since 2016)",
               "hawkish": "Fed reacts to headline PCE (hawkish)"}


class OilModelMissing(RuntimeError):
    pass


@dataclass
class OilModel:
    meta: dict
    rates: pd.DataFrame          # date, scenario, T5YIEM, T10YIEM, GS2, FII10, ... (monthly path)
    rates_hist: pd.DataFrame     # actual rates history (date index)
    paths: pd.DataFrame          # date, scenario, target, brent, mom_pct, yoy, yoy_lo, yoy_hi
    infl_hist: pd.DataFrame      # date, target, mom_pct, yoy, ...

    @property
    def origin(self) -> pd.Period:
        return pd.Period(self.meta["origin"], "M")

    @property
    def end(self) -> pd.Period:
        return self.origin + int(self.meta["horizon"])

    @property
    def labels(self) -> dict[str, str]:
        return self.meta["labels"]

    def origin_rate(self, col: str) -> float:
        h = self.rates_hist
        return float(h.loc[h.index == self.origin, col].iloc[0])


def load(directory: Path | None = None) -> OilModel:
    """Looks in oil_model/ first, then next to app.py (GitHub's web upload can drop the folder)."""
    if directory is None:
        candidates = [DIR, DIR.parent]
        directory = next((d for d in candidates if all((d / f).exists() for f in FILES)), DIR)
    missing = [f for f in FILES if not (directory / f).exists()]
    if missing:
        raise OilModelMissing("Oil-model export not found: " + ", ".join(missing) +
                              f" (expected in {directory.name}/). Run scripts/export_for_factor_app.py "
                              "in the top-down-oil repo and upload the files.")
    meta = json.load(open(directory / "meta.json"))

    def per(df, col="date"):
        df[col] = pd.to_datetime(df[col]).dt.to_period("M")
        return df
    rates = per(pd.read_csv(directory / "rates_scenarios.csv"))
    rh = per(pd.read_csv(directory / "rates_history.csv")).set_index("date")
    paths = per(pd.read_csv(directory / "scenario_paths.csv"))
    ih = per(pd.read_csv(directory / "inflation_history.csv"))
    return OilModel(meta, rates, rh, paths, ih)


def scenario_end(om: OilModel, scenario: str) -> pd.Series:
    r = om.rates[om.rates["scenario"] == scenario].set_index("date")
    return r.loc[om.end]


def views(om: OilModel, rule: str, cpi_avg_mm: float, current: dict | None = None) -> dict[str, dict[str, float]]:
    """Factor moves for every scenario under the chosen Fed rule (keys = scenario labels).

    Moves are measured from `current` (latest actual levels: keys T10YIEM, GS2, FII10, BRENT) when given,
    otherwise from the oil model's origin month."""
    gs2, fii, _ = RULES[rule]
    cur = current or {}
    be0 = cur.get("T10YIEM", om.origin_rate("T10YIEM"))
    y20 = cur.get("GS2", om.origin_rate("GS2"))
    r0 = cur.get("FII10", om.origin_rate("FII10"))
    b0 = cur.get("BRENT", float(om.meta["brent_origin"]))
    out = {}
    for s, label in om.labels.items():
        if s not in set(om.rates["scenario"]):
            continue
        e = scenario_end(om, s)
        p = om.paths[(om.paths["scenario"] == s) & (om.paths["target"] == "headline_cpi")].set_index("date")
        fut = p.loc[(p.index > om.origin) & (p.index <= om.end)]
        cpi_12m = float(fut["mom_pct"].sum())
        out[label] = {
            "BREAKEVEN": round(float(e["T10YIEM"]) - be0, 3),
            "FED_EXPECT": round(float(e[gs2]) - y20, 3),
            "REAL_YIELD": round(float(e[fii]) - r0, 3),
            "CPI": round(cpi_12m - len(fut) * cpi_avg_mm, 3),
            "OIL": round(100.0 * np.log(float(fut["brent"].iloc[-1]) / b0), 2),
        }
    return out


def summary(om: OilModel, rule: str) -> pd.DataFrame:
    """One row per scenario: Brent and inflation at the horizon, and the rates chain under `rule`."""
    gs2, fii, xlk = RULES[rule]
    rows = []
    for s, label in om.labels.items():
        if s not in set(om.rates["scenario"]):
            continue
        e = scenario_end(om, s)
        row = {"Scenario": label}
        P = om.paths[om.paths["scenario"] == s]
        row["Brent at end ($)"] = float(P[(P["target"] == "headline_pce") & (P["date"] == om.end)]["brent"].iloc[0])
        for t, lbl in (("headline_pce", "Headline PCE"), ("core_pce", "Core PCE"),
                       ("headline_cpi", "Headline CPI"), ("core_cpi", "Core CPI")):
            q = P[(P["target"] == t) & (P["date"] == om.end)].iloc[0]
            row[lbl] = float(q["yoy"])
            if t in ("headline_pce", "core_pce"):
                row[f"{lbl} range"] = f"{q['yoy_lo']:.1f} to {q['yoy_hi']:.1f}"
        row["10y breakeven"] = float(e["T10YIEM"])
        row["2y"] = float(e[gs2])
        row["10y real"] = float(e[fii])
        row["XLK, real-yield channel"] = float(e[xlk])
        rows.append(row)
    return pd.DataFrame(rows).set_index("Scenario")


def notes(om: OilModel, rule: str) -> dict[str, str]:
    s = summary(om, rule)
    end = om.end.strftime("%b %Y")
    out = {}
    for label, r in s.iterrows():
        out[label] = (f"From the fund's oil-inflation model ({RULE_LABELS[rule].lower()}). By {end}: Brent "
                      f"${r['Brent at end ($)']:.0f}, headline PCE {r['Headline PCE']:.1f}%, core PCE "
                      f"{r['Core PCE']:.1f}%, 2y {r['2y']:.2f}%, 10y real {r['10y real']:.2f}%, 10y breakeven "
                      f"{r['10y breakeven']:.2f}%. Credit, VIX, dollar and growth are held flat because the "
                      "oil model doesn't cover them.")
    return out


def staleness_months(om: OilModel, today: pd.Timestamp | None = None) -> int:
    cur = (today or pd.Timestamp.today()).to_period("M")
    return (cur - om.origin).n
