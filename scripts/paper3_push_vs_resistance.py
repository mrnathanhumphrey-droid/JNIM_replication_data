"""
paper3_push_vs_resistance.py — SECONDARY BRIEF capstone dig: disentangle PUSH from RESISTANCE
(the primary brief's #1 confounder, §6.3).

Is Cote d'Ivoire's calm LOWER incoming JNIM push, or HIGHER resistance? Operationalize:
  PUSH_i   = JNIM fatalities in the SAHEL SOURCE states (Mali/BFA/Niger) whose event is nearest to
             littoral country i's border AND within PUSH_KM of it — the pressure at i's doorstep.
  OUTCOME_i= JNIM fatalities INSIDE littoral country i (2020-2025).
  LEAKAGE_i= OUTCOME_i / PUSH_i — the share of doorstep pressure that converts to violence inside.
             LOW leakage under HIGH push = strong resistance; ~0 push = can't tell (push asymmetry).

Public data: UCDP-GED v26.1 (JNIM by dyad, where_prec<=3), Natural Earth polygons. Joins GDP/km2 from
paper3_spending_density for the capacity axis. n is tiny — read the ordering.

    python _scripts/paper3_push_vs_resistance.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
UCDP = ROOT / "data" / "ucdp" / "GEDEvent_v26_1.csv"
NE = ROOT / "data" / "natural_earth" / "ne0.zip"
DENS = ROOT / "analysis" / "paper3_spending_density_2026_07_09.json"
OUT = ROOT / "analysis" / "paper3_push_vs_resistance_2026_07_09.json"
UTM = "EPSG:32631"
PUSH_KM = 250.0
JNIM_RE = r"JNIM|Jama'at Nasr al-Islam|Nusrat al-Islam|Group for Support of Islam"
LITTORAL = ["Benin", "Togo", "Ivory Coast", "Ghana"]
ALIAS = {"Ivory Coast": ["Ivory Coast", "Côte d'Ivoire", "Cote d'Ivoire"]}
SOURCE = ["Mali", "Burkina Faso", "Niger"]


def main():
    u = pd.read_csv(UCDP, low_memory=False,
                    usecols=["year", "country", "best", "where_prec", "latitude", "longitude",
                             "side_a", "side_b", "dyad_name"])
    nm = u.side_a.astype(str) + "|" + u.side_b.astype(str) + "|" + u.dyad_name.astype(str)
    j = u[nm.str.contains(JNIM_RE, case=False, regex=True, na=False)].copy()
    j = j.dropna(subset=["latitude", "longitude"])
    j = j[(j.where_prec <= 3) & (j.year.between(2020, 2025))]

    w = gpd.read_file(NE)
    ncol = "ADMIN" if "ADMIN" in w.columns else "NAME"
    w = w[[ncol, "geometry"]].rename(columns={ncol: "name"}).to_crs(UTM)

    def poly(disp):
        for cand in ALIAS.get(disp, [disp]):
            g = w[w.name == cand]
            if len(g):
                return g.geometry.iloc[0]
        return None
    litt = gpd.GeoDataFrame({"litt_country": LITTORAL},
                            geometry=[poly(c) for c in LITTORAL], crs=UTM)

    g = gpd.GeoDataFrame(j, geometry=gpd.points_from_xy(j.longitude, j.latitude),
                         crs="EPSG:4326").to_crs(UTM)

    # OUTCOME: JNIM fatalities inside each littoral country
    inside = gpd.sjoin(g, litt, how="inner", predicate="within")
    outcome = inside.groupby("litt_country")["best"].sum().to_dict()

    # PUSH: source-state JNIM events, nearest littoral border, within PUSH_KM
    src = g[g.country.isin(SOURCE)].copy()
    near = gpd.sjoin_nearest(src, litt, how="left", distance_col="d_m")
    near = near[~near.index.duplicated(keep="first")]
    near["d_km"] = near["d_m"] / 1000.0
    push_ev = near[near.d_km <= PUSH_KM]
    push = push_ev.groupby("litt_country")["best"].sum().to_dict()
    push_n = push_ev.groupby("litt_country").size().to_dict()

    dens = {r["country"]: r for r in json.load(open(DENS))["rows"]}

    rows = []
    for c in LITTORAL:
        p = float(push.get(c, 0)); o = float(outcome.get(c, 0))
        rows.append(dict(country=c, push_fatalities=p, push_events=int(push_n.get(c, 0)),
                         outcome_fatalities=o, leakage=round(o / p, 3) if p else None,
                         gdp_per_km2_k=round(dens[c]["gdp_per_km2"] / 1e3, 0)))
    df = pd.DataFrame(rows)

    print(f"PUSH vs RESISTANCE  (JNIM, UCDP prec<=3, 2020-2025; push = source fatalities within {PUSH_KM:.0f} km of border)")
    print(f"{'Country':<14}{'PUSH_fatal':>11}{'push_ev':>9}{'OUTCOME':>9}{'leakage':>9}{'GDP/km2$k':>11}")
    for _, r in df.iterrows():
        print(f"{r.country:<14}{r.push_fatalities:>11.0f}{r.push_events:>9}{r.outcome_fatalities:>9.0f}"
              f"{str(r.leakage):>9}{r.gdp_per_km2_k:>11.0f}")

    print("\nINTERPRETATION")
    civ = df[df.country == "Ivory Coast"].iloc[0]
    ben = df[df.country == "Benin"].iloc[0]
    print(f"  Cote d'Ivoire doorstep push: {civ.push_fatalities:.0f} fatalities / {civ.push_events} events; outcome inside: {civ.outcome_fatalities:.0f}")
    print(f"  Benin        doorstep push: {ben.push_fatalities:.0f} fatalities / {ben.push_events} events; outcome inside: {ben.outcome_fatalities:.0f}")
    if civ.push_fatalities >= 0.3 * ben.push_fatalities and civ.outcome_fatalities == 0:
        print("  -> CIV faces NON-TRIVIAL push yet 0 outcome => RESISTANCE is real (capacity holds under pressure).")
    elif civ.push_fatalities < 0.3 * ben.push_fatalities:
        print("  -> CIV's doorstep is much quieter => part of its calm is LOWER PUSH (asymmetry), not only resistance.")

    OUT.write_text(json.dumps({"push_km": PUSH_KM, "rows": rows}, indent=2, default=str), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
