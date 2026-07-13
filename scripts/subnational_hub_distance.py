"""
subnational_hub_distance.py — the resolution-correct test of "insurgents AVOID economic hubs".

Event-level (no grid/MAUP): distance from each insurgent event to the nearest ECONOMIC HUB
(GHS-SMOD urban centre, class 30), compared to NON-insurgent organized violence in the same
countries (nets out "violence happens near towns anyway" — exactly the JNIM road-corridor logic,
but distance-to-hub instead of distance-to-road). If insurgents avoid hubs, their events sit
FARTHER from urban centres than baseline violence.

    python _scripts/subnational_hub_distance.py
"""
from __future__ import annotations
import json, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from subnational_density_grid import (UCDP, MOLL, YEARS, LAKE_CHAD, COUNTRIES, JNIM_RE, SHAB_RE,
                                      africa_window)

ROOT = Path(__file__).resolve().parents[1]
SMOD = ROOT / "data" / "ghs" / "GHS_SMOD_1km" / "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0.tif"
OUT = ROOT / "analysis" / "subnational_hub_distance.json"
UCDP_NAMES = {"Ivory Coast": "Ivory Coast (Cote d'Ivoire)"}   # UCDP uses "Ivory Coast"
HUB_CLASS = 30            # SMOD 30 = Urban Centre (economic hub); 23=dense cluster


def hub_coords():
    with rasterio.open(SMOD) as src:
        r0, r1, c0, c1 = africa_window(src)
        arr = src.read(1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0))
        x0, y0 = src.xy(r0, c0, offset="ul")
    ii, jj = np.where(arr >= HUB_CLASS)
    xs = x0 + (jj + 0.5) * 1000.0
    ys = y0 - (ii + 0.5) * 1000.0
    return np.column_stack([xs, ys])


def load_events():
    u = pd.read_csv(UCDP, low_memory=False,
                    usecols=["year", "country", "best", "where_prec", "latitude", "longitude",
                             "side_a", "side_b", "dyad_name"]).dropna(subset=["latitude", "longitude"])
    u = u[(u.where_prec <= 4) & (u.year.between(*YEARS)) & (u.country.isin(COUNTRIES))].copy()
    nm = u.side_a.astype(str) + "|" + u.side_b.astype(str) + "|" + u.dyad_name.astype(str)
    is_j = nm.str.contains(JNIM_RE, case=False, regex=True, na=False)
    is_s = nm.str.contains(SHAB_RE, case=False, regex=True, na=False)
    is_b = (u.country.isin(LAKE_CHAD)) & (u.side_b.isin(["JAS", "IS"]))
    u["insurgent"] = is_j | is_s | is_b
    u["conflict"] = np.where(is_j, "JNIM", np.where(is_b, "BokoHaram", np.where(is_s, "alShabaab", "other")))
    return u


def main():
    print("indexing economic hubs (GHS-SMOD urban centres) ...")
    hubs = hub_coords()
    tree = cKDTree(hubs)
    print(f"  {len(hubs):,} urban-centre km-pixels in the Africa window")

    u = load_events()
    tf = Transformer.from_crs("EPSG:4326", MOLL, always_xy=True)
    ex, ey = tf.transform(u.longitude.values, u.latitude.values)
    d_km, _ = tree.query(np.column_stack([ex, ey]))
    u["hub_km"] = d_km / 1000.0

    ins = u[u.insurgent]; base = u[~u.insurgent]
    print(f"\n  insurgent events: {len(ins):,} | baseline (other violence, same countries): {len(base):,}")

    def summ(d):
        return dict(n=int(len(d)), median_km=round(float(d.median()), 1),
                    mean_km=round(float(d.mean()), 1),
                    within10=round(float((d <= 10).mean()) * 100, 1),
                    within25=round(float((d <= 25).mean()) * 100, 1),
                    within50=round(float((d <= 50).mean()) * 100, 1))
    si, sb = summ(ins.hub_km), summ(base.hub_km)

    print("\n" + "=" * 84)
    print("DISTANCE TO NEAREST ECONOMIC HUB (urban centre) — insurgent vs baseline violence")
    print("=" * 84)
    print(f"  {'':<22}{'median km':>11}{'mean km':>10}{'<=10km':>9}{'<=25km':>9}{'<=50km':>9}")
    print(f"  {'INSURGENT':<22}{si['median_km']:>11}{si['mean_km']:>10}{si['within10']:>8}%{si['within25']:>8}%{si['within50']:>8}%")
    print(f"  {'baseline violence':<22}{sb['median_km']:>11}{sb['mean_km']:>10}{sb['within10']:>8}%{sb['within25']:>8}%{sb['within50']:>8}%")
    u_stat, p = mannwhitneyu(ins.hub_km, base.hub_km, alternative="greater")
    verdict = ("INSURGENTS AVOID HUBS (sit farther from urban centres than other violence)"
               if si["median_km"] > sb["median_km"] and p < 0.01 else
               "no avoidance (insurgents are not farther from hubs)")
    print(f"\n  Mann-Whitney (insurgent farther?): p={p:.2e}  ->  {verdict}")

    print("\n  by conflict (median km to nearest hub; baseline = same-country other violence):")
    per = {}
    for c in ("JNIM", "BokoHaram", "alShabaab"):
        d = u[u.conflict == c]
        per[c] = dict(median_km=round(float(d.hub_km.median()), 1), n=int(len(d)))
        print(f"    {c:<11} median={per[c]['median_km']:>6} km (n={per[c]['n']})")
    print(f"    {'baseline':<11} median={sb['median_km']:>6} km (n={sb['n']})")

    OUT.write_text(json.dumps({"hub_class": HUB_CLASS, "insurgent": si, "baseline": sb,
                               "mannwhitney_p": float(p), "by_conflict": per}, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
