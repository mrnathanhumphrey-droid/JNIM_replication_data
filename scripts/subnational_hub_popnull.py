"""
subnational_hub_popnull.py — the tight control: within-country POPULATION-WEIGHTED null.

The distance-to-hub test showed insurgents sit farther from urban centres than baseline violence.
Objection: insurgencies just happen in emptier regions. This nets that out entirely. Per country
we build the population-weighted distribution of distance-to-nearest-hub (where the PEOPLE are,
relative to urban centres), then locate each insurgent event's distance in that distribution.

  insurgent_pctile = share of the country's POPULATION that lives CLOSER to a hub than this event.
  mean pctile ~ 0.50  -> insurgents strike where people are (no avoidance).
  mean pctile  > 0.50 -> insurgents strike FARTHER from hubs than people live -> genuine avoidance.

    python _scripts/subnational_hub_popnull.py
"""
from __future__ import annotations
import json, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio, rasterio.mask
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from subnational_density_grid import POP, NE0, MOLL, COUNTRIES, NE_ALIAS
from subnational_hub_distance import hub_coords, load_events

OUT = Path("D:/IDP") / "analysis" / "subnational_hub_popnull.json"


def wmedian(d, w):
    o = np.argsort(d); d, w = d[o], w[o]
    c = np.cumsum(w)
    return float(d[np.searchsorted(c, 0.5 * c[-1])])


def country_pop_cdf(geom_moll, tree):
    """Population-weighted empirical CDF of distance-to-hub for one country's inhabited pixels."""
    with rasterio.open(POP) as src:
        arr, t = rasterio.mask.mask(src, [geom_moll], crop=True, filled=True, nodata=0)
    a = arr[0].astype("float64"); a[a < 0] = 0
    rows, cols = np.where(a > 0)
    if len(rows) < 50:
        return None
    x = t.c + (cols + 0.5) * t.a
    y = t.f + (rows + 0.5) * t.e
    w = a[rows, cols]
    d, _ = tree.query(np.column_stack([x, y])); d /= 1000.0
    o = np.argsort(d); d, w = d[o], w[o]
    cdf = np.cumsum(w) / w.sum()
    return d, cdf, wmedian(d, w)


def main():
    print("indexing hubs + country polygons ...")
    tree = cKDTree(hub_coords())
    ne = gpd.read_file(NE0)
    ncol = "ADMIN" if "ADMIN" in ne.columns else "NAME"
    want = {a: c for c in COUNTRIES for a in ([c] + NE_ALIAS.get(c, []))}
    ne = ne[ne[ncol].isin(want)].copy()
    ne["country"] = ne[ncol].map(want)
    ne = ne.dissolve("country").to_crs(MOLL)

    u = load_events()
    tf = Transformer.from_crs("EPSG:4326", MOLL, always_xy=True)
    ex, ey = tf.transform(u.longitude.values, u.latitude.values)
    d_km, _ = tree.query(np.column_stack([ex, ey])); u["hub_km"] = d_km / 1000.0

    print("\n" + "=" * 92)
    print("WITHIN-COUNTRY POPULATION-WEIGHTED NULL  (insurgent pctile in the population's own")
    print("distance-to-hub distribution; >50% = strikes farther from hubs than people live)")
    print("=" * 92)
    print(f"  {'country':<15}{'pop-wtd med km':>15}{'insurgent med km':>18}{'insurgent pctile':>18}{'n_ins':>7}")
    rows = []; all_pct = []
    for c, g in ne.groupby(level=0):
        cdf = country_pop_cdf(g.geometry.iloc[0], tree)
        ins = u[(u.country == c) & (u.insurgent)]
        if cdf is None or len(ins) < 20:
            continue
        dd, cc, wmed = cdf
        pct = np.interp(ins.hub_km.values, dd, cc)     # share of pop closer than each event
        rows.append(dict(country=c, pop_med_km=round(wmed, 1),
                         ins_med_km=round(float(ins.hub_km.median()), 1),
                         ins_pctile=round(float(pct.mean()) * 100, 1), n=int(len(ins))))
        all_pct.append(pct)
        print(f"  {c:<15}{wmed:>15.1f}{ins.hub_km.median():>18.1f}{pct.mean()*100:>17.1f}%{len(ins):>7}")

    pooled = np.concatenate(all_pct)
    # test: are insurgent percentiles centered above 0.5? (Wilcoxon on pctile-0.5)
    w, p = wilcoxon(pooled - 0.5, alternative="greater")
    share_above = float((pooled > 0.5).mean()) * 100
    print("-" * 92)
    print(f"  POOLED insurgent percentile = {pooled.mean()*100:.1f}%  (median {np.median(pooled)*100:.1f}%)")
    print(f"  share of insurgent events farther from a hub than the median local resident: {share_above:.1f}%")
    print(f"  Wilcoxon (pctile > 50%): p = {p:.2e}")
    verdict = ("AVOIDANCE SURVIVES the population-weighted control — insurgents strike farther from"
               "\n            hubs than people live, within country" if pooled.mean() > 0.5 and p < 0.01
               else "no avoidance beyond where people are")
    print(f"  -> {verdict}")

    # per-conflict pooled percentile
    print("\n  by conflict (pooled insurgent percentile in population-weighted null):")
    for conf in ("JNIM", "BokoHaram", "alShabaab"):
        pcs = []
        for c, g in ne.groupby(level=0):
            ins = u[(u.country == c) & (u.conflict == conf)]
            if len(ins) < 20:
                continue
            cdf = country_pop_cdf(g.geometry.iloc[0], tree)
            if cdf is None:
                continue
            dd, cc, _ = cdf
            pcs.append(np.interp(ins.hub_km.values, dd, cc))
        if pcs:
            allp = np.concatenate(pcs)
            print(f"    {conf:<11} pctile = {allp.mean()*100:.1f}%  (n={len(allp)})")

    OUT.write_text(json.dumps({"per_country": rows, "pooled_pctile": round(float(pooled.mean()), 4),
                               "share_above_median_resident": round(share_above, 1),
                               "wilcoxon_p": float(p)}, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
