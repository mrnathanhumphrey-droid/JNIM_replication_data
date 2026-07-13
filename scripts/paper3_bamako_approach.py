"""
paper3_bamako_approach.py — SECONDARY BRIEF dig (P3): is JNIM road violence CLOSING IN on Bamako,
and which FUEL CORRIDORS carry it?

Two tests on UCDP-GED v26.1 (Mali, JNIM, where_prec<=3, 2018-2025):
  A) distance-to-Bamako by year — does JNIM violence ring the capital tighter over time, peaking 2025
     (the fuel-blockade year, when tanker attacks concentrated on the Bamako approaches)?
  B) named-corridor split — assign each on-corridor JNIM event to its nearest national route (OSM `ref`)
     and see which routes carry the concentration, especially the two international FUEL-IMPORT highways:
        RN1  Bamako - Kayes - Senegal (Dakar port)
        RN7  Bamako - Sikasso - Cote d'Ivoire (Abidjan port)

    python _scripts/paper3_bamako_approach.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
UCDP = ROOT / "data" / "ucdp" / "GEDEvent_v26_1.csv"
ROADS = ROOT / "data" / "osm" / "mali_roads" / "gis_osm_roads_free_1.shp"
OUT = ROOT / "analysis" / "paper3_bamako_approach_2026_07_09.json"
UTM = "EPSG:32630"
BAMAKO = Point(-8.0029, 12.6392)                 # lon, lat
CORRIDOR_KM = 5.0
JNIM_RE = r"JNIM|Jama'at Nasr al-Islam|Nusrat al-Islam|Group for Support of Islam"
FUEL = {"RN1": "Bamako-Kayes-Senegal", "RN7": "Bamako-Sikasso-CotedIvoire"}


def main():
    u = pd.read_csv(UCDP, low_memory=False,
                    usecols=["year", "country", "best", "where_prec", "latitude", "longitude",
                             "side_a", "side_b", "dyad_name"])
    ml = u[(u.country == "Mali")].copy()
    nm = ml.side_a.astype(str) + "|" + ml.side_b.astype(str) + "|" + ml.dyad_name.astype(str)
    ml = ml[nm.str.contains(JNIM_RE, case=False, regex=True, na=False)]
    ml = ml.dropna(subset=["latitude", "longitude"])
    ml = ml[(ml.where_prec <= 3) & (ml.year.between(2018, 2025))]

    g = gpd.GeoDataFrame(ml, geometry=gpd.points_from_xy(ml.longitude, ml.latitude),
                         crs="EPSG:4326").to_crs(UTM)
    bam = gpd.GeoSeries([BAMAKO], crs="EPSG:4326").to_crs(UTM).iloc[0]
    g["km_to_bamako"] = g.geometry.distance(bam) / 1000.0

    roads = gpd.read_file(ROADS)
    corr = roads[roads.fclass.isin(["trunk", "primary", "trunk_link", "primary_link"])].to_crs(UTM)
    near = gpd.sjoin_nearest(g, corr[["ref", "geometry"]], how="left", distance_col="dist_m")
    near = near[~near.index.duplicated(keep="first")]
    g["route"] = near["ref"].values
    g["dist_km"] = near["dist_m"].values / 1000.0
    g["on_corridor"] = g["dist_km"] <= CORRIDOR_KM

    # ---- A) distance-to-Bamako by year ----
    print("A) JNIM violence distance to Bamako, by year")
    print(f"{'year':>5}{'n':>5}{'fatal':>7}{'med_km':>8}{'<=150km':>9}{'<=250km':>9}")
    yr = {}
    for y in range(2018, 2026):
        s = g[g.year == y]
        if not len(s):
            continue
        d = {"n": int(len(s)), "fatal": int(s.best.sum()),
             "median_km_to_bamako": round(float(s.km_to_bamako.median()), 0),
             "pct_within_150km": round(100 * (s.km_to_bamako <= 150).mean(), 1),
             "pct_within_250km": round(100 * (s.km_to_bamako <= 250).mean(), 1)}
        yr[str(y)] = d
        print(f"{y:>5}{d['n']:>5}{d['fatal']:>7}{d['median_km_to_bamako']:>8.0f}{d['pct_within_150km']:>9}{d['pct_within_250km']:>9}")

    # ---- B) named-corridor split (on-corridor events only) ----
    onc = g[g.on_corridor & g.route.notna()]
    print(f"\nB) On-corridor JNIM events by national route (n={len(onc)})")
    top = onc.route.value_counts().head(10)
    all_by_route = onc.groupby("route").size()
    r25 = onc[onc.year == 2025].route.value_counts()
    print(f"{'route':>7}{'all':>6}{'2025':>6}  (fuel corridor?)")
    routes = {}
    for rt in top.index:
        routes[rt] = {"all": int(all_by_route.get(rt, 0)), "y2025": int(r25.get(rt, 0)),
                      "fuel_corridor": FUEL.get(rt, "")}
        print(f"{rt:>7}{int(all_by_route.get(rt,0)):>6}{int(r25.get(rt,0)):>6}  {FUEL.get(rt,'')}")

    fuel_share_25 = round(100 * onc[(onc.year == 2025) & onc.route.isin(FUEL)].shape[0] / max(r25.sum(), 1), 1)
    fuel_share_all = round(100 * onc[onc.route.isin(FUEL)].shape[0] / max(len(onc), 1), 1)
    print(f"\nFuel-corridor (RN1+RN7) share of on-corridor JNIM events: all-years {fuel_share_all}%  |  2025 {fuel_share_25}%")

    OUT.write_text(json.dumps({"dist_to_bamako_by_year": yr, "on_corridor_by_route": routes,
                               "fuel_corridor_share_all": fuel_share_all, "fuel_corridor_share_2025": fuel_share_25},
                              indent=2), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
