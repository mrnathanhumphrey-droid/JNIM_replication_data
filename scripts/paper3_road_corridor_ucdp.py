"""
paper3_road_corridor_ucdp.py — SECONDARY BRIEF, Direction 3 on PUBLIC data (UCDP-GED v26.1).

Replaces the GDELT proxy with UCDP-GED (public, event-level, real lat/long + a geoprecision field
`where_prec`, clean JNIM dyad attribution, full 2025 coverage). Question unchanged: is JNIM violence
in Mali concentrated on the primary road CORRIDORS (fuel-tanker routes), and rising into the 2025
Bamako blockade?

Two upgrades over the GDELT pass:
  1. PRECISION FILTER — where_prec<=3 keeps only events geocoded to town/local level (not admin
     centroids), so distance-to-road is meaningful. (where_prec: 1=exact .. 7=country.)
  2. CLEAN ATTRIBUTION — JNIM via dyad/side regex (same match as the primary brief), vs GDELT's
     generic REBEL/MILITANT tags.
Still weaker than ACLED in ONE way: UCDP has no "attack on convoy/road" event-type tag, so this is
proximity-to-corridor, not confirmed road-target. Honest. Paired baseline = non-JNIM Mali violence
(same precision filter) nets out the residual geo-bias.

    python _scripts/paper3_road_corridor_ucdp.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
UCDP = ROOT / "data" / "ucdp" / "GEDEvent_v26_1.csv"
ROADS = ROOT / "data" / "osm" / "mali_roads" / "gis_osm_roads_free_1.shp"
OUT = ROOT / "analysis" / "paper3_road_corridor_ucdp_2026_07_09.json"
UTM = "EPSG:32630"
CORRIDOR_KM = 5.0
PREC_MAX = 3                       # keep where_prec 1-3 (exact / near / adm2-town)
JNIM_RE = r"JNIM|Jama'at Nasr al-Islam|Nusrat al-Islam|Group for Support of Islam"


def main():
    u = pd.read_csv(UCDP, low_memory=False,
                    usecols=["year", "country", "type_of_violence", "best", "where_prec",
                             "latitude", "longitude", "side_a", "side_b", "dyad_name"])
    ml = u[u["country"] == "Mali"].copy()
    names = (ml["side_a"].astype(str) + "|" + ml["side_b"].astype(str) + "|" + ml["dyad_name"].astype(str))
    ml["jnim"] = names.str.contains(JNIM_RE, case=False, regex=True, na=False)
    ml = ml.dropna(subset=["latitude", "longitude"])
    print(f"UCDP-GED Mali events (all years): {len(ml):,}   JNIM-attributed: {int(ml['jnim'].sum()):,}")
    print("where_prec distribution (JNIM):", ml[ml.jnim]["where_prec"].value_counts().sort_index().to_dict())

    ml = ml[(ml["where_prec"] <= PREC_MAX) & (ml["year"].between(2018, 2025))]
    print(f"after precision<= {PREC_MAX} + 2018-2025 filter: {len(ml):,} events "
          f"(JNIM {int(ml['jnim'].sum()):,} / baseline {int((~ml['jnim']).sum()):,})")

    roads = gpd.read_file(ROADS)
    corr = roads[roads["fclass"].isin(["trunk", "primary", "trunk_link", "primary_link"])].to_crs(UTM)
    g = gpd.GeoDataFrame(ml, geometry=gpd.points_from_xy(ml["longitude"], ml["latitude"]),
                         crs="EPSG:4326").to_crs(UTM)
    near = gpd.sjoin_nearest(g, corr[["geometry"]], how="left", distance_col="dist_m")
    near = near[~near.index.duplicated(keep="first")]
    g["dist_km"] = near["dist_m"].values / 1000.0
    g["on_corridor"] = g["dist_km"] <= CORRIDOR_KM

    def stats(sub):
        n = len(sub)
        return {"n": int(n), "fatalities": int(sub["best"].sum()),
                "pct_on_corridor": round(100 * sub["on_corridor"].mean(), 1) if n else None,
                "median_dist_km": round(float(sub["dist_km"].median()), 1) if n else None}

    jih, base = g[g["jnim"]], g[~g["jnim"]]
    overall = {"jnim": stats(jih), "baseline": stats(base),
               "differential_pct": (round(stats(jih)["pct_on_corridor"] - stats(base)["pct_on_corridor"], 1)
                                    if stats(jih)["pct_on_corridor"] is not None and stats(base)["pct_on_corridor"] is not None else None)}
    by_year = {}
    for y in range(2018, 2026):
        jy, by = jih[jih.year == y], base[base.year == y]
        sj, sb = stats(jy), stats(by)
        by_year[str(y)] = {"jnim_n": sj["n"], "jnim_fatal": sj["fatalities"], "jnim_pct": sj["pct_on_corridor"],
                           "base_pct": sb["pct_on_corridor"],
                           "diff": (round(sj["pct_on_corridor"] - sb["pct_on_corridor"], 1)
                                    if sj["pct_on_corridor"] is not None and sb["pct_on_corridor"] is not None else None)}

    OUT.write_text(json.dumps({"corridor_km": CORRIDOR_KM, "prec_max": PREC_MAX,
                               "overall": overall, "by_year": by_year}, indent=2), encoding="utf-8")
    print(f"\nOn-corridor = within {CORRIDOR_KM} km of a trunk/primary road")
    print(f"  JNIM     %on-corridor: {overall['jnim']['pct_on_corridor']}  (median {overall['jnim']['median_dist_km']} km, n={overall['jnim']['n']})")
    print(f"  BASELINE %on-corridor: {overall['baseline']['pct_on_corridor']}  (median {overall['baseline']['median_dist_km']} km, n={overall['baseline']['n']})")
    print(f"  >> DIFFERENTIAL: {overall['differential_pct']:+} pts")
    print("\nBy year (JNIM n | JNIM fatal | JNIM %corr | base %corr | diff):")
    for y, d in by_year.items():
        print(f"  {y}: {str(d['jnim_n']):>4} | {str(d['jnim_fatal']):>5} | {str(d['jnim_pct']):>5} | {str(d['base_pct']):>5} | {str(d['diff']):>6}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
