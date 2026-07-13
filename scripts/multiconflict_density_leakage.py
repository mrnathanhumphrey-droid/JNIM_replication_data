"""
multiconflict_density_leakage.py — does the JNIM density-leakage law GENERALIZE across African
jihadist diffusions? Pools the receiving states of THREE independent insurgencies and tests the
same scale-free statistic the JNIM secondary brief used (leakage vs GDP/km2).

For each conflict: SOURCE core -> neighboring RECEIVING states.
  PUSH_i    = source-group fatalities whose event is nearest receiving state i's border AND within
              PUSH_KM of it (the pressure at i's doorstep).
  OUTCOME_i = group fatalities INSIDE receiving state i.
  LEAKAGE_i = OUTCOME_i / PUSH_i  (nets out push -> POOLABLE across conflicts of different intensity).
  Test: does LEAKAGE fall with GDP/km2 across all receiving states, pooled?

Conflicts (UCDP-GED v26.1, where_prec<=3, 2015-2025):
  JNIM        source Mali/BFA/Niger   -> Benin, Togo, Cote d'Ivoire, Ghana   (actor by dyad/side regex)
  Boko/ISWAP  source Nigeria          -> Niger, Chad, Cameroon               (side_b in {JAS, IS})
  al-Shabaab  source Somalia          -> Kenya, Ethiopia                     (actor by regex)

Distances per conflict in a LOCAL UTM zone (W.Africa 31N / Lake Chad 33N / Horn 37N) so the 250 km
doorstep threshold is metrically honest across the continent. Areas from Natural Earth in Africa
Albers equal-area. National GDP/km2 (WDI mean 2020-24) — the SAME crude national operationalization
the brief used, kept identical for a fair replication (caveat: it averages rich cores with empty
periphery; that is the next, subnational, refinement).

    python _scripts/multiconflict_density_leakage.py
"""
from __future__ import annotations
import json, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
UCDP = ROOT / "data" / "ucdp" / "GEDEvent_v26_1.csv"
NE = ROOT / "data" / "natural_earth" / "ne0.zip"
WDI = ROOT / "data" / "wb_wdi" / "extracted" / "WDICSV.csv"
OUT = ROOT / "analysis" / "multiconflict_density_leakage.json"
ALBERS = "ESRI:102022"           # Africa Albers Equal Area — for national areas
PUSH_KM = 250.0
YEARS = (2015, 2025)
WDI_YRS = [str(y) for y in range(2020, 2025)]

ALIAS = {"Ivory Coast": ["Ivory Coast", "Côte d'Ivoire", "Cote d'Ivoire"]}
ISO3 = {"Benin": "BEN", "Togo": "TGO", "Ivory Coast": "CIV", "Ghana": "GHA",
        "Niger": "NER", "Chad": "TCD", "Cameroon": "CMR", "Kenya": "KEN", "Ethiopia": "ETH"}

# conflict -> (event selector, source countries, receiving countries, local UTM epsg)
JNIM_RE = r"JNIM|Nusrat al-Islam|Group for Support of Islam|Jama'at Nasr al-Islam"
SHAB_RE = r"Shabaab|al-Shabab"
LAKE_CHAD = ["Nigeria", "Niger", "Chad", "Cameroon"]

def sel_jnim(u, nm):
    return u[nm.str.contains(JNIM_RE, case=False, regex=True, na=False)]
def sel_boko(u, nm):
    return u[(u.country.isin(LAKE_CHAD)) & (u.side_b.isin(["JAS", "IS"]))]
def sel_shab(u, nm):
    return u[nm.str.contains(SHAB_RE, case=False, regex=True, na=False)]

CONFLICTS = {
    "JNIM":       dict(sel=sel_jnim, source=["Mali", "Burkina Faso", "Niger"],
                       recv=["Benin", "Togo", "Ivory Coast", "Ghana"], utm="EPSG:32631"),
    "BokoHaram":  dict(sel=sel_boko, source=["Nigeria"],
                       recv=["Niger", "Chad", "Cameroon"], utm="EPSG:32633"),
    "alShabaab":  dict(sel=sel_shab, source=["Somalia"],
                       recv=["Kenya", "Ethiopia"], utm="EPSG:32637"),
}


def load_events():
    u = pd.read_csv(UCDP, low_memory=False,
                    usecols=["year", "country", "best", "where_prec", "latitude", "longitude",
                             "side_a", "side_b", "dyad_name"])
    u = u.dropna(subset=["latitude", "longitude"])
    u = u[(u.where_prec <= 3) & (u.year.between(*YEARS))]
    nm = u.side_a.astype(str) + "|" + u.side_b.astype(str) + "|" + u.dyad_name.astype(str)
    return u, nm


def national_density():
    """GDP/km2 ($k) and GDP/capita from WDI mean 2020-24 and Albers equal-area."""
    wdi = pd.read_csv(WDI, low_memory=False)
    def wb(iso, ind):
        r = wdi[(wdi["Country Code"] == iso) & (wdi["Indicator Code"] == ind)]
        if not len(r):
            return np.nan
        v = pd.to_numeric(r[WDI_YRS].iloc[0], errors="coerce").dropna()
        return float(v.mean()) if len(v) else np.nan
    w = gpd.read_file(NE)
    ncol = "ADMIN" if "ADMIN" in w.columns else "NAME"
    w = w[[ncol, "geometry"]].rename(columns={ncol: "name"}).to_crs(ALBERS)
    def area_km2(disp):
        for cand in ALIAS.get(disp, [disp]):
            g = w[w.name == cand]
            if len(g):
                return float(g.geometry.iloc[0].area) / 1e6
        return np.nan
    out = {}
    for disp, iso in ISO3.items():
        gdp, pop, a = wb(iso, "NY.GDP.MKTP.CD"), wb(iso, "SP.POP.TOTL"), area_km2(disp)
        out[disp] = dict(gdp_per_km2_k=round(gdp / a / 1e3, 1) if a else np.nan,
                         gdp_per_capita=round(gdp / pop, 0) if pop else np.nan)
    return out


def poly_ne(utm):
    w = gpd.read_file(NE)
    ncol = "ADMIN" if "ADMIN" in w.columns else "NAME"
    return w[[ncol, "geometry"]].rename(columns={ncol: "name"}).to_crs(utm)


def run_conflict(name, cfg, u, nm):
    ev = cfg["sel"](u, nm).copy()
    w = poly_ne(cfg["utm"])
    def poly(disp):
        for cand in ALIAS.get(disp, [disp]):
            g = w[w.name == cand]
            if len(g):
                return g.geometry.iloc[0]
        return None
    recv = gpd.GeoDataFrame({"recv_country": cfg["recv"]},
                            geometry=[poly(c) for c in cfg["recv"]], crs=cfg["utm"])
    g = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev.longitude, ev.latitude),
                         crs="EPSG:4326").to_crs(cfg["utm"])
    # OUTCOME: fatalities inside each receiving polygon
    inside = gpd.sjoin(g, recv, how="inner", predicate="within")
    outcome = inside.groupby("recv_country")["best"].sum().to_dict()
    # PUSH: source-country events, nearest receiving border, within PUSH_KM
    src = g[g.country.isin(cfg["source"])].copy()
    near = gpd.sjoin_nearest(src, recv, how="left", distance_col="d_m")
    near = near[~near.index.duplicated(keep="first")]
    near["d_km"] = near.d_m / 1000.0
    pe = near[near.d_km <= PUSH_KM]
    push = pe.groupby("recv_country")["best"].sum().to_dict()
    push_n = pe.groupby("recv_country").size().to_dict()
    rows = []
    for c in cfg["recv"]:
        p, o = float(push.get(c, 0)), float(outcome.get(c, 0))
        rows.append(dict(conflict=name, country=c, push_fatalities=p, push_events=int(push_n.get(c, 0)),
                         outcome_fatalities=o, leakage=round(o / p, 3) if p else None))
    return rows


def sp(x, y, label, out):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        print(f"  {label:<44}: n<3 skip"); return
    rho, p = spearmanr(x[m], y[m])
    print(f"  {label:<44}: rho={rho:+.2f} (p={p:.3f}, n={int(m.sum())})")
    out[label] = dict(rho=round(float(rho), 3), p=round(float(p), 4), n=int(m.sum()))


def main():
    u, nm = load_events()
    dens = national_density()
    allrows = []
    for name, cfg in CONFLICTS.items():
        allrows += run_conflict(name, cfg, u, nm)
    df = pd.DataFrame(allrows)
    df["gdp_per_km2_k"] = df.country.map(lambda c: dens[c]["gdp_per_km2_k"])
    df["gdp_per_capita"] = df.country.map(lambda c: dens[c]["gdp_per_capita"])

    print("=" * 96)
    print(f"MULTI-CONFLICT DENSITY–LEAKAGE  (UCDP prec<=3, {YEARS[0]}-{YEARS[1]}; push = source fatal within {PUSH_KM:.0f} km of border)")
    print("=" * 96)
    print(f"{'conflict':<11}{'receiving':<15}{'PUSH_fat':>9}{'push_ev':>8}{'OUTCOME':>9}{'leakage':>9}{'GDP/km2$k':>11}{'GDP/cap$':>10}")
    for _, r in df.iterrows():
        print(f"{r.conflict:<11}{r.country:<15}{r.push_fatalities:>9.0f}{r.push_events:>8}"
              f"{r.outcome_fatalities:>9.0f}{str(r.leakage):>9}{r.gdp_per_km2_k:>11.1f}{str(r.gdp_per_capita):>10}")

    print("\nPOOLED across all 3 conflicts:")
    res = {}
    lk = df[df.leakage.notna()]
    sp(lk.gdp_per_km2_k, lk.leakage, "LEAKAGE vs GDP/km2 (the density-shield law)", res)
    sp(lk.gdp_per_capita, lk.leakage, "LEAKAGE vs GDP/capita (wealth, not density)", res)
    sp(df.gdp_per_km2_k, df.outcome_fatalities, "raw OUTCOME vs GDP/km2 (no push-netting)", res)
    sp(df.gdp_per_km2_k, df.push_fatalities, "PUSH vs GDP/km2 (should be ~flat: density!=less push)", res)

    print("\nWITHIN-conflict leakage ordering vs GDP/km2 (each n small — read direction):")
    for name in CONFLICTS:
        d = lk[lk.conflict == name]
        order = d.sort_values("gdp_per_km2_k")[["country", "gdp_per_km2_k", "leakage"]].values.tolist()
        print(f"  {name:<11}: " + "  ".join(f"{c}({g:.0f}k→{l})" for c, g, l in order))

    OUT.write_text(json.dumps({"push_km": PUSH_KM, "years": YEARS,
                               "rows": allrows, "density": dens, "pooled_spearman": res},
                              indent=2, default=str), encoding="utf-8")
    print(f"\nNOTE: national GDP/km2 (crude — averages rich core with empty periphery). n small; read the ordering.")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
