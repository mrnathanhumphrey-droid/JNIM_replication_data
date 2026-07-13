"""
paper3_road_corridor_gdelt.py — SECONDARY BRIEF first pass (Direction 3: road-infrastructure control).

Question: is JNIM-attributable violence in Mali disproportionately concentrated on the primary road
CORRIDORS (the fuel-tanker routes Bamako<->coast), and is that concentration RISING 2014-2024 — i.e. is
the Sept-2025 Bamako fuel blockade the culmination of a long road-targeting trend rather than a novelty?

Data (all on disk / free):
  - GDELT Events v1, Mali 2014-2024 (data/gdelt/gdelt-mali-2014_2024.csv). CAMEO-coded, admin-CENTROID
    geo -> a corridor-pressure PROXY, not road-precise. ACLED (blocked on key) upgrades this later.
  - OSM road network (Geofabrik Mali) trunk+primary = fuel corridors.

METHOD DISCIPLINE (Floor Battery / paired differential): GDELT geo is centroid-biased, so a raw
"% of jihadist events near a road" is not interpretable alone. We compute the SAME statistic for
NON-jihadist material-conflict events as a baseline and report the DIFFERENTIAL. If jihadist violence
is road-targeting above the ambient geo-bias, jihadist %-near-corridor > baseline %-near-corridor.

    python _scripts/paper3_road_corridor_gdelt.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
GDELT = ROOT / "data" / "gdelt" / "gdelt-mali-2014_2024.csv"
ROADS = ROOT / "data" / "osm" / "mali_roads" / "gis_osm_roads_free_1.shp"
OUT = ROOT / "analysis" / "paper3_road_corridor_gdelt_2026_07_09.json"
UTM = "EPSG:32630"                     # UTM 30N — meters over Mali
CORRIDOR_KM = 5.0                      # "on the corridor" threshold
JIHADIST = ("REBEL", "MILITANT", "QAEDA", "TERROR", "INSURGEN", "ISLAMIST", "EXTREMIST", "JIHAD")
MALI_BBOX = (-12.3, 10.0, 4.3, 25.0)   # lon/lat bounds


def load_events():
    cols = GDELT.open().readline().strip().split("|")
    df = pd.read_csv(GDELT, sep="|", names=cols, header=0, low_memory=False,
                     usecols=["sqldate", "year", "eventrootcode", "quadclass", "actor1name",
                              "actor2name", "actiongeo_countrycode", "actiongeo_lat", "actiongeo_long"])
    df = df[(df["quadclass"] == 4) & (df["actiongeo_countrycode"] == "ML")].copy()   # material conflict in Mali
    df = df.dropna(subset=["actiongeo_lat", "actiongeo_long"])
    lon, lat = df["actiongeo_long"], df["actiongeo_lat"]
    df = df[(lon.between(MALI_BBOX[0], MALI_BBOX[2])) & (lat.between(MALI_BBOX[1], MALI_BBOX[3]))]
    df = df[df["year"].between(2014, 2024)]
    # de-dup GDELT's re-reports: one event per (date, rounded loc, actor pair)
    df["k"] = (df["sqldate"].astype(str) + "_" + df["actiongeo_lat"].round(3).astype(str) + "_"
               + df["actiongeo_long"].round(3).astype(str) + "_"
               + df["actor1name"].astype(str) + "_" + df["actor2name"].astype(str))
    df = df.drop_duplicates("k")
    names = (df["actor1name"].astype(str) + " " + df["actor2name"].astype(str)).str.upper()
    df["jihadist"] = names.apply(lambda s: any(k in s for k in JIHADIST))
    return df


def main():
    ev = load_events()
    roads = gpd.read_file(ROADS)
    corr = roads[roads["fclass"].isin(["trunk", "primary", "trunk_link", "primary_link"])].to_crs(UTM)
    corr_union_idx = corr.sindex

    g = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev["actiongeo_long"], ev["actiongeo_lat"]),
                         crs="EPSG:4326").to_crs(UTM)
    near = gpd.sjoin_nearest(g, corr[["geometry"]], how="left", distance_col="dist_m")
    near = near[~near.index.duplicated(keep="first")]
    g["dist_km"] = near["dist_m"].values / 1000.0
    g["on_corridor"] = g["dist_km"] <= CORRIDOR_KM

    def stats(sub):
        n = len(sub)
        return {"n": int(n), "pct_on_corridor": round(100 * sub["on_corridor"].mean(), 1) if n else None,
                "median_dist_km": round(float(sub["dist_km"].median()), 1) if n else None}

    jih, base = g[g["jihadist"]], g[~g["jihadist"]]
    overall = {"jihadist": stats(jih), "baseline_nonjihadist": stats(base),
               "differential_pct_on_corridor": round(stats(jih)["pct_on_corridor"] - stats(base)["pct_on_corridor"], 1)}

    by_year = {}
    for y in range(2014, 2025):
        jy, by = jih[jih["year"] == y], base[base["year"] == y]
        sj, sb = stats(jy), stats(by)
        by_year[str(y)] = {"jihadist_n": sj["n"], "jihadist_pct_corridor": sj["pct_on_corridor"],
                           "baseline_pct_corridor": sb["pct_on_corridor"],
                           "differential": (round(sj["pct_on_corridor"] - sb["pct_on_corridor"], 1)
                                            if sj["pct_on_corridor"] is not None and sb["pct_on_corridor"] is not None else None)}

    res = {"corridor_km": CORRIDOR_KM, "n_trunk_primary_segments": int(len(corr)),
           "overall": overall, "by_year": by_year}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"GDELT Mali material-conflict events 2014-2024 (deduped): {len(g):,}")
    print(f"  jihadist-actor: {len(jih):,}   baseline: {len(base):,}")
    print(f"\nOn-corridor = within {CORRIDOR_KM} km of a trunk/primary road ({len(corr)} segments)")
    print(f"  JIHADIST   %on-corridor: {overall['jihadist']['pct_on_corridor']}   (median {overall['jihadist']['median_dist_km']} km)")
    print(f"  BASELINE   %on-corridor: {overall['baseline_nonjihadist']['pct_on_corridor']}   (median {overall['baseline_nonjihadist']['median_dist_km']} km)")
    print(f"  >> DIFFERENTIAL (jihadist - baseline): {overall['differential_pct_on_corridor']:+} pts")
    print("\nBy year (jihadist_n | jih %corr | base %corr | diff):")
    for y, d in by_year.items():
        print(f"  {y}: {str(d['jihadist_n']):>4} | {str(d['jihadist_pct_corridor']):>5} | {str(d['baseline_pct_corridor']):>5} | {str(d['differential']):>6}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
