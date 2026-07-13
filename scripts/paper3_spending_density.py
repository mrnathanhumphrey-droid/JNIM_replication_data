"""
paper3_spending_density.py — SECONDARY BRIEF, Direction 2 (tractable half): SPENDING-DENSITY refinement.

The colleague endorsed "GDP per square km" and asked to quantify capacity as a DENSITY, not a topline.
The primary brief found ABSOLUTE military budget orders JNIM containment (Spearman rho=-0.95, n=4) while
spend-per-border-km FAILED (rho=-0.32). This tests the missing density operationalizations:

  - milex_per_km2  = military $ / national land area  (GARRISON DENSITY: force per unit terrain to hold)
  - gdp_per_km2    = GDP / area                        (the colleague's economic-density model)
  - milex_per_capita, gdp_per_capita                   (per-head capacity)

against the SAME outcome (JNIM-attributable fatalities 2020-2025) and the SAME 4-country littoral core
(Benin/Togo/CIV/Ghana; Nigeria excluded for the Boko-Haram/ISWAP confound), so it is a clean paired
comparison with the primary brief's absolute-budget result. n=4 is descriptive — read the ordering.
An EXTENDED panel adds the 3 Sahel-source states as context (low capacity-density, high violence end).

Data: Natural Earth 10m (area), World Bank API (GDP/milex/pop, mean 2020-2024). Reproducible.

    python _scripts/paper3_spending_density.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
NE = ROOT / "data" / "natural_earth" / "ne0.zip"
WDI = ROOT / "data" / "wb_wdi" / "extracted" / "WDICSV.csv"
OUT = ROOT / "analysis" / "paper3_spending_density_2026_07_09.json"
_WDI_DF = None
YEARS = [str(y) for y in range(2020, 2025)]

CODES = {"Benin": "BJ", "Togo": "TG", "Ivory Coast": "CI", "Ghana": "GH", "Nigeria": "NG",
         "Mali": "ML", "Burkina Faso": "BF", "Niger": "NE"}
ISO2_TO_3 = {"BJ": "BEN", "TG": "TGO", "CI": "CIV", "GH": "GHA", "NG": "NGA",
             "ML": "MLI", "BF": "BFA", "NE": "NER"}
ALIAS = {"Ivory Coast": ["Ivory Coast", "Côte d'Ivoire", "Cote d'Ivoire"]}
LITTORAL = ["Benin", "Togo", "Ivory Coast", "Ghana", "Nigeria"]
SAHEL = ["Mali", "Burkina Faso", "Niger"]
# JNIM-attributable fatalities 2020-2025 (littoral, from v26.1 fit); Sahel = total organized-violence
# cumulative 2021-2025 (primary brief §2.1) as the violence level for the extended-panel context.
JNIM_FATAL = {"Benin": 333, "Togo": 275, "Ivory Coast": 0, "Ghana": 0, "Nigeria": None}
VIOL_TOTAL = {"Mali": 11119, "Burkina Faso": 19212, "Niger": None}


def load_areas():
    w = gpd.read_file(NE)
    ncol = "ADMIN" if "ADMIN" in w.columns else "NAME"
    w = w[[ncol, "geometry"]].rename(columns={ncol: "name"}).to_crs(32631)  # UTM31N, meters
    area = {}
    for disp in CODES:
        for cand in ALIAS.get(disp, [disp]):
            g = w[w.name == cand]
            if len(g):
                area[disp] = float(g.geometry.iloc[0].area) / 1e6  # m2 -> km2
                break
    return area


def wb(code, ind):
    """Mean 2020-2024 of a WDI indicator for an ISO2 country, from the local WDICSV.csv (robust)."""
    global _WDI_DF
    if _WDI_DF is None:
        _WDI_DF = pd.read_csv(WDI, low_memory=False)
    iso3 = ISO2_TO_3[code]
    row = _WDI_DF[(_WDI_DF["Country Code"] == iso3) & (_WDI_DF["Indicator Code"] == ind)]
    if not len(row):
        return np.nan
    vals = pd.to_numeric(row[YEARS].iloc[0], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else np.nan


def main():
    area = load_areas()
    rows = []
    for c in LITTORAL + SAHEL:
        gdp = wb(CODES[c], "NY.GDP.MKTP.CD")
        mil = wb(CODES[c], "MS.MIL.XPND.CD")
        pop = wb(CODES[c], "SP.POP.TOTL")
        a = area.get(c, np.nan)
        rows.append(dict(country=c, area_km2=a, gdp=gdp, milex=mil, pop=pop,
                         milex_per_km2=mil / a if a else np.nan,
                         gdp_per_km2=gdp / a if a else np.nan,
                         milex_per_capita=mil / pop if pop else np.nan,
                         gdp_per_capita=gdp / pop if pop else np.nan,
                         outcome=JNIM_FATAL.get(c, VIOL_TOTAL.get(c))))
    df = pd.DataFrame(rows)

    print("=" * 92)
    print("SPENDING-DENSITY CROSS-SECTION")
    print("=" * 92)
    print(f"{'Country':<15}{'area_kkm2':>10}{'milex$M':>9}{'GDP$B':>8}{'milex/km2$':>12}{'GDP/km2$k':>11}{'milex/cap$':>11}{'outcome':>9}")
    for _, r in df.iterrows():
        print(f"{r.country:<15}{r.area_km2/1e3:>10.0f}{r.milex/1e6:>9.0f}{r.gdp/1e9:>8.0f}"
              f"{r.milex_per_km2:>12.0f}{r.gdp_per_km2/1e3:>11.0f}{r.milex_per_capita:>11.1f}{str(r.outcome):>9}")

    core = df[df.country.isin(LITTORAL)].copy()
    core = core[core.outcome.notna()]  # drop Nigeria
    print(f"\nCORE littoral cross-section (n={len(core)}, Nigeria excluded) — density vs JNIM fatalities:")

    def sp(x, y, label):
        m = x.notna() & y.notna()
        if m.sum() < 3:
            print(f"  {label:<34}: n<3, skip"); return None
        rho, p = spearmanr(x[m], y[m])
        print(f"  {label:<34}: Spearman rho={rho:+.2f} (p={p:.2f}, n={int(m.sum())})")
        return rho

    res_core = {
        "absolute_milex": sp(core.milex, core.outcome, "ABSOLUTE milex (primary-brief ref)"),
        "milex_per_km2": sp(core.milex_per_km2, core.outcome, "milex per km2 (garrison density)"),
        "gdp_per_km2": sp(core.gdp_per_km2, core.outcome, "GDP per km2 (colleague's model)"),
        "milex_per_capita": sp(core.milex_per_capita, core.outcome, "milex per capita"),
        "gdp_per_capita": sp(core.gdp_per_capita, core.outcome, "GDP per capita"),
    }

    # extended panel: littoral core + Sahel-source (context; different role, flagged)
    ext = df[df.outcome.notna()].copy()
    print(f"\nEXTENDED panel (n={len(ext)}, +Sahel source states as context — mixes source/receiving roles):")
    res_ext = {
        "milex_per_km2": sp(ext.milex_per_km2, ext.outcome, "milex per km2 vs violence"),
        "gdp_per_km2": sp(ext.gdp_per_km2, ext.outcome, "GDP per km2 vs violence"),
        "absolute_milex": sp(ext.milex, ext.outcome, "ABSOLUTE milex vs violence"),
    }

    OUT.write_text(json.dumps({"rows": rows, "core_spearman": res_core, "extended_spearman": res_ext},
                              indent=2, default=str), encoding="utf-8")
    print(f"\nNOTE: n is tiny — descriptive ordering, not powered inference.")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
