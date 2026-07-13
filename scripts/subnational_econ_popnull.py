"""
subnational_econ_popnull.py — the refinement: continuous ECONOMIC OUTPUT (gridded GDP) instead
of the binary urban-centre flag, with the same within-country population-weighted null.

Metric = GDP density (GDP/km2) at each event's location, from Kummu et al. downscaled gridded GDP
(gdpTot 5-arcmin, year 2020). Per country we build the POPULATION-WEIGHTED distribution of GDP
density (the economic value where people actually live), then locate each insurgent event in it.

  insurgent GDP-percentile = share of the population that lives at LOWER GDP density than the event.
  ~50%  -> insurgents strike where the economy is (no avoidance).
  <50%  -> insurgents strike at LOWER economic value than people live -> they AVOID economic output.

Direct continuous-economic analogue of subnational_hub_popnull (which used distance-to-urban-centre).

    python _scripts/subnational_econ_popnull.py
"""
from __future__ import annotations
import json, warnings, sys, math
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio, rasterio.mask
from pyproj import Transformer
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from subnational_density_grid import POP, NE0, MOLL, COUNTRIES, NE_ALIAS
from subnational_hub_distance import load_events

ROOT = Path(__file__).resolve().parents[1]
GDP = ROOT / "data" / "kummu_gdp" / "gdpTot_1990_2024_5arcmin.tif"
OUT = ROOT / "analysis" / "subnational_econ_popnull.json"
GDP_YEAR = 2020
GDP_BAND = GDP_YEAR - 1990 + 1
_DEG = 0.08333333333333333
_KM_PER_DEG = 111.319

# load the GDP band once (global, EPSG:4326)
with rasterio.open(GDP) as _g:
    _GDP = _g.read(GDP_BAND).astype("float64")     # total GDP (PPP) per ~9km cell
    _T = _g.transform
_NY, _NX = _GDP.shape


def gdp_density(lon, lat):
    """GDP density (PPP $ / km2) at lon/lat arrays; nan where no data / uninhabited."""
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    col = ((lon - _T.c) / _T.a).astype(int)
    row = ((lat - _T.f) / _T.e).astype(int)
    ok = (col >= 0) & (col < _NX) & (row >= 0) & (row < _NY)
    out = np.full(lon.shape, np.nan)
    g = np.full(lon.shape, np.nan)
    g[ok] = _GDP[row[ok], col[ok]]
    area = (_DEG * _KM_PER_DEG) ** 2 * np.cos(np.radians(lat))   # cell area km2 (lat-dependent)
    dens = g / area
    dens[~np.isfinite(dens) | (g <= 0)] = np.nan
    return dens


def wmedian(x, w):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def country_pop_gdp_cdf(geom_moll):
    """Population-weighted CDF of GDP density for one country's inhabited pixels."""
    with rasterio.open(POP) as src:
        arr, t = rasterio.mask.mask(src, [geom_moll], crop=True, filled=True, nodata=0)
    a = arr[0].astype("float64"); a[a < 0] = 0
    rows, cols = np.where(a > 0)
    if len(rows) < 50:
        return None
    mx = t.c + (cols + 0.5) * t.a
    my = t.f + (rows + 0.5) * t.e
    w = a[rows, cols]
    tf = Transformer.from_crs(MOLL, "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(mx, my)
    d = gdp_density(lon, lat)
    m = np.isfinite(d)
    if m.sum() < 50:
        return None
    d, w = d[m], w[m]
    o = np.argsort(d); d, w = d[o], w[o]
    cdf = np.cumsum(w) / w.sum()
    return d, cdf, wmedian(d, w)


def main():
    print(f"gridded GDP: {GDP.name} band {GDP_BAND} (year {GDP_YEAR})")
    ne = gpd.read_file(NE0)
    ncol = "ADMIN" if "ADMIN" in ne.columns else "NAME"
    want = {a: c for c in COUNTRIES for a in ([c] + NE_ALIAS.get(c, []))}
    ne = ne[ne[ncol].isin(want)].copy(); ne["country"] = ne[ncol].map(want)
    ne = ne.dissolve("country").to_crs(MOLL)

    u = load_events()
    u["gdp_dens"] = gdp_density(u.longitude.values, u.latitude.values)

    print("\n" + "=" * 96)
    print("WITHIN-COUNTRY POPULATION-WEIGHTED NULL — economic value (GDP density) at insurgent events")
    print("(<50% = insurgents strike at LOWER GDP density than the population lives at = avoidance)")
    print("=" * 96)
    print(f"  {'country':<15}{'pop-wtd med GDP/km2':>20}{'insurgent med':>15}{'ins GDP-pctile':>16}{'n':>7}")
    rows = []; all_pct = []
    for c, g in ne.groupby(level=0):
        cdf = country_pop_gdp_cdf(g.geometry.iloc[0])
        ins = u[(u.country == c) & (u.insurgent) & (u.gdp_dens.notna())]
        if cdf is None or len(ins) < 20:
            continue
        dd, cc, wmed = cdf
        pct = np.interp(ins.gdp_dens.values, dd, cc)
        rows.append(dict(country=c, pop_med_gdpdens=round(wmed, 0),
                         ins_med_gdpdens=round(float(ins.gdp_dens.median()), 0),
                         ins_gdp_pctile=round(float(pct.mean()) * 100, 1), n=int(len(ins))))
        all_pct.append(pct)
        print(f"  {c:<15}{wmed:>20,.0f}{ins.gdp_dens.median():>15,.0f}{pct.mean()*100:>15.1f}%{len(ins):>7}")

    pooled = np.concatenate(all_pct)
    w, p = wilcoxon(pooled - 0.5, alternative="less")     # test pctile < 0.5
    share_below = float((pooled < 0.5).mean()) * 100
    print("-" * 96)
    print(f"  POOLED insurgent GDP-percentile = {pooled.mean()*100:.1f}%  (median {np.median(pooled)*100:.1f}%)")
    print(f"  share of insurgent events at LOWER GDP density than the median local resident: {share_below:.1f}%")
    print(f"  Wilcoxon (pctile < 50%): p = {p:.2e}")
    verdict = ("AVOIDANCE CONFIRMED with continuous GDP — insurgents strike at lower economic output"
               "\n            than where people live, within country"
               if pooled.mean() < 0.5 and p < 0.01 else "no economic-output avoidance")
    print(f"  -> {verdict}")

    print("\n  by conflict (pooled insurgent GDP-percentile in population-weighted null):")
    perconf = {}
    for conf in ("JNIM", "BokoHaram", "alShabaab"):
        pcs = []
        for c, g in ne.groupby(level=0):
            ins = u[(u.country == c) & (u.conflict == conf) & (u.gdp_dens.notna())]
            if len(ins) < 20:
                continue
            cdf = country_pop_gdp_cdf(g.geometry.iloc[0])
            if cdf is None:
                continue
            dd, cc, _ = cdf
            pcs.append(np.interp(ins.gdp_dens.values, dd, cc))
        if pcs:
            allp = np.concatenate(pcs); perconf[conf] = round(float(allp.mean()) * 100, 1)
            print(f"    {conf:<11} GDP-pctile = {allp.mean()*100:.1f}%  (n={len(allp)})")

    OUT.write_text(json.dumps({"gdp_year": GDP_YEAR, "per_country": rows,
                               "pooled_gdp_pctile": round(float(pooled.mean()), 4),
                               "share_below_median_resident": round(share_below, 1),
                               "wilcoxon_p": float(p), "by_conflict": perconf}, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
