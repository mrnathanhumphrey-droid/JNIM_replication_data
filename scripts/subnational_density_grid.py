"""
subnational_density_grid.py — the powered, coverage-filling test of the density-shield law.

National pool = n9 across 3 density clusters (Boko Haram owned the low-density arm). This drops
to a 50 km GRID over the 13 source+receiving countries of the three insurgencies, so every
country supplies its FULL density range (empty periphery -> dense core). Tests insurgent violence
vs population density WITHIN country (fixed effects), n in the thousands.

Density = GHS-POP 2020 (1km, Mollweide) summed to 50km cells / area  (population/km2 — the on-disk
proxy; economic-density via nightlights/gridded-GDP is the next upgrade). Violence = UCDP-GED
fatalities from JNIM / Boko Haram-ISWAP (JAS,IS) / al-Shabaab, 2015-2025, binned to the same grid.

    python _scripts/subnational_density_grid.py
"""
from __future__ import annotations
import json, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from pyproj import Transformer
from scipy.stats import spearmanr, rankdata
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
POP = ROOT / "data" / "ghs" / "GHS_POP_1km" / "GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.tif"
NE0 = ROOT / "data" / "natural_earth" / "ne0.zip"
UCDP = ROOT / "data" / "ucdp" / "GEDEvent_v26_1.csv"
OUT = ROOT / "analysis" / "subnational_density_grid.json"
MOLL = "ESRI:54009"
BLOCK = 50            # 1km pixels per cell edge -> 50 km cells (2500 km2)
CELL_M = BLOCK * 1000.0
YEARS = (2015, 2025)
LAKE_CHAD = ["Nigeria", "Niger", "Chad", "Cameroon"]
COUNTRIES = ["Mali", "Burkina Faso", "Niger", "Benin", "Togo", "Ivory Coast", "Ghana",
             "Nigeria", "Chad", "Cameroon", "Somalia", "Kenya", "Ethiopia"]
NE_ALIAS = {"Ivory Coast": ["Ivory Coast", "Côte d'Ivoire", "Cote d'Ivoire"]}
JNIM_RE = r"JNIM|Nusrat al-Islam|Group for Support of Islam|Jama'at Nasr al-Islam"
SHAB_RE = r"Shabaab|al-Shabab"


def africa_window(src):
    """Pixel window of the Africa extent in the Mollweide raster."""
    lon = np.linspace(-20, 55, 60); lat = np.linspace(-38, 40, 60)
    tf = Transformer.from_crs("EPSG:4326", MOLL, always_xy=True)
    LO, LA = np.meshgrid(lon, lat)
    X, Y = tf.transform(LO.ravel(), LA.ravel())
    X, Y = X[np.isfinite(X)], Y[np.isfinite(Y)]
    r0, c0 = src.index(X.min(), Y.max()); r1, c1 = src.index(X.max(), Y.min())
    r0, r1 = max(0, min(r0, r1)), min(src.height, max(r0, r1))
    c0, c1 = max(0, min(c0, c1)), min(src.width, max(c0, c1))
    return r0, r1, c0, c1


def build_grid():
    with rasterio.open(POP) as src:
        r0, r1, c0, c1 = africa_window(src)
        # trim to a whole number of BLOCK-sized cells
        H = ((r1 - r0) // BLOCK) * BLOCK; W = ((c1 - c0) // BLOCK) * BLOCK
        win = rasterio.windows.Window(c0, r0, W, H)
        arr = src.read(1, window=win).astype("float64")
        arr[arr < 0] = 0.0                                  # nodata -> 0
        x0, y0 = src.xy(r0, c0, offset="ul")                # Mollweide UL of window
    ny, nx = H // BLOCK, W // BLOCK
    pop = arr.reshape(ny, BLOCK, nx, BLOCK).sum(axis=(1, 3))   # population per 50km cell
    j = np.arange(nx); i = np.arange(ny)
    cx = x0 + (j + 0.5) * CELL_M                             # cell-center Mollweide x
    cy = y0 - (i + 0.5) * CELL_M                             # cell-center Mollweide y
    CX, CY = np.meshgrid(cx, cy)
    df = pd.DataFrame({"i": np.repeat(i, nx), "j": np.tile(j, ny),
                       "mx": CX.ravel(), "my": CY.ravel(), "pop": pop.ravel()})
    df["x0"], df["y0"] = x0, y0                              # grid origin (for event binning)
    return df


def assign_country(df):
    tf = Transformer.from_crs(MOLL, "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(df.mx.values, df.my.values)
    g = gpd.GeoDataFrame(df.assign(lon=lon, lat=lat),
                         geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    ne = gpd.read_file(NE0)
    ncol = "ADMIN" if "ADMIN" in ne.columns else "NAME"
    ne = ne[[ncol, "geometry"]].rename(columns={ncol: "country"})
    want = set(sum(([c] + NE_ALIAS.get(c, []) for c in COUNTRIES), []))
    ne = ne[ne.country.isin(want)]
    j = gpd.sjoin(g, ne, how="inner", predicate="within").drop(columns="index_right")
    j["country"] = j["country"].replace({a: "Ivory Coast" for a in NE_ALIAS["Ivory Coast"]})
    return pd.DataFrame(j.drop(columns="geometry"))


def bin_violence(cells):
    u = pd.read_csv(UCDP, low_memory=False,
                    usecols=["year", "country", "best", "where_prec", "latitude", "longitude",
                             "side_a", "side_b", "dyad_name"]).dropna(subset=["latitude", "longitude"])
    u = u[(u.where_prec <= 4) & (u.year.between(*YEARS))]
    nm = u.side_a.astype(str) + "|" + u.side_b.astype(str) + "|" + u.dyad_name.astype(str)
    is_jnim = nm.str.contains(JNIM_RE, case=False, regex=True, na=False)
    is_shab = nm.str.contains(SHAB_RE, case=False, regex=True, na=False)
    is_boko = (u.country.isin(LAKE_CHAD)) & (u.side_b.isin(["JAS", "IS"]))
    u = u[is_jnim | is_shab | is_boko].copy()
    tf = Transformer.from_crs("EPSG:4326", MOLL, always_xy=True)
    ex, ey = tf.transform(u.longitude.values, u.latitude.values)
    x0, y0 = cells.x0.iloc[0], cells.y0.iloc[0]
    u["j"] = ((ex - x0) // CELL_M).astype("int64")
    u["i"] = ((y0 - ey) // CELL_M).astype("int64")
    agg = u.groupby(["i", "j"])["best"].sum().rename("ins_fatal").reset_index()
    agg2 = u.groupby(["i", "j"]).size().rename("ins_events").reset_index()
    out = cells.merge(agg, on=["i", "j"], how="left").merge(agg2, on=["i", "j"], how="left")
    out["ins_fatal"] = out.ins_fatal.fillna(0.0); out["ins_events"] = out.ins_events.fillna(0)
    return out


def within_country_fe(cells, ycol, xcol="density"):
    """Rank density and violence WITHIN each country, then Spearman on pooled within-ranks =
    a fixed-effects partial rank correlation (nets out the national level)."""
    d = cells[cells[xcol].notna()].copy()
    d = d.groupby("country").filter(lambda g: len(g) >= 15)
    xr = d.groupby("country")[xcol].transform(lambda s: rankdata(s))
    yr = d.groupby("country")[ycol].transform(lambda s: rankdata(s))
    rho, p = spearmanr(xr, yr)
    per = {}
    for c, g in d.groupby("country"):
        if len(g) >= 15 and g[ycol].sum() > 0:
            r, _ = spearmanr(g[xcol], g[ycol]); per[c] = (round(float(r), 3), int(len(g)))
    return float(rho), float(p), int(len(d)), per


def main():
    print("building 50km grid from GHS-POP ...")
    cells = build_grid()
    cells = assign_country(cells)
    cells = bin_violence(cells)
    cells["area_km2"] = (CELL_M / 1000.0) ** 2
    cells["density"] = cells["pop"] / cells["area_km2"]        # population per km2
    print(f"  {len(cells):,} land cells across {cells.country.nunique()} countries; "
          f"{int((cells.ins_fatal>0).sum())} cells with insurgent violence; "
          f"total insurgent fatalities binned: {cells.ins_fatal.sum():,.0f}")

    print("\n" + "=" * 90)
    print("WITHIN-COUNTRY (fixed-effects) rank correlation: insurgent violence vs population density")
    print("=" * 90)
    res = {}
    for ycol, lbl in (("ins_fatal", "insurgent FATALITIES"), ("ins_events", "insurgent EVENTS")):
        rho, p, n, per = within_country_fe(cells, ycol)
        print(f"\n  {lbl}:  FE rho = {rho:+.3f}  (p={p:.2e}, n={n:,} cells)")
        res[ycol] = dict(fe_rho=round(rho, 3), p=p, n=n, per_country=per)
        print("  per-country (rho, n cells):")
        for c, (r, nn) in sorted(per.items(), key=lambda kv: kv[1][0]):
            print(f"    {c:<16} rho={r:+.3f} (n={nn})")

    # descriptive: within-country density quintile -> violence
    print("\n" + "=" * 90)
    print("Within-country density QUINTILE -> share of cells with any insurgent violence")
    print("=" * 90)
    d = cells.copy()
    d["dq"] = d.groupby("country")["density"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]))
    tab = d.groupby("dq").agg(any_viol=("ins_fatal", lambda s: (s > 0).mean()),
                              mean_fatal=("ins_fatal", "mean")).reset_index()
    print(f"  {'density quintile':<18}{'% cells w/ violence':>20}{'mean fatalities':>18}")
    for _, r in tab.iterrows():
        print(f"  Q{int(r.dq)} ({'lowest' if r.dq==1 else 'highest' if r.dq==5 else '   '}){'':<6}"
              f"{100*r.any_viol:>18.1f}%{r.mean_fatal:>18.1f}")
    res["quintile"] = tab.to_dict("records")

    # negative-binomial FE GLM (corroboration) if statsmodels present
    try:
        import statsmodels.formula.api as smf, statsmodels.api as sm
        d2 = cells.copy(); d2["logdens"] = np.log1p(d2.density)
        m = smf.glm("ins_fatal ~ logdens + C(country)", data=d2,
                    family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        b, se = m.params["logdens"], m.bse["logdens"]
        print(f"\nNB GLM (FE): log1p(density) coef = {b:+.3f} ± {se:.3f}  (z={b/se:+.1f}) "
              f"-> {'NEG (denser=less violence)' if b<0 else 'POS'}")
        res["nb_glm"] = dict(coef=round(float(b), 4), se=round(float(se), 4), z=round(float(b/se), 2))
    except Exception as e:
        print(f"\n(statsmodels NB skipped: {type(e).__name__})")

    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
