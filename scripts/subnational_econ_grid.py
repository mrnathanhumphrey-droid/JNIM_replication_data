"""
subnational_econ_grid.py — the CORRECT subnational test: do insurgents AVOID economic hubs,
controlling for target-availability (population)?

Fix of the population-only test (which just recovered "violence follows people"). Economic
density = GHS-BUILT-S 2020 (built-up surface m2 = density of physical/economic capital), on the
SAME 50km grid as GHS-POP. The hypothesis (user): insurgents avoid economic hubs and wage economic
warfare on mobility, so CONDITIONAL ON POPULATION, more economically-dense cells get LESS violence.

Tests, within-country (fixed effects), 13 source+receiving countries, UCDP insurgent fatalities
2015-2025:
  (A) built density vs violence, unconditioned   (expect still +: built tracks people)
  (B) built density | population  (NB GLM)        <-- THE TEST: negative = hubs avoided
  (C) built PER CAPITA vs violence                (capital-per-person; expect -)
  (D) built-per-capita quintile -> violence share (the picture)

    python _scripts/subnational_econ_grid.py
"""
from __future__ import annotations
import json, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from pyproj import Transformer
from scipy.stats import spearmanr, rankdata
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from subnational_density_grid import (POP, NE0, UCDP, MOLL, BLOCK, CELL_M, YEARS, LAKE_CHAD,
                                      COUNTRIES, NE_ALIAS, JNIM_RE, SHAB_RE, africa_window,
                                      assign_country, bin_violence)

ROOT = Path(__file__).resolve().parents[1]
BUILT = ROOT / "data" / "ghs" / "GHS_BUILT_S_1km" / "GHS_BUILT_S_E2020_GLOBE_R2023A_54009_1000_V1_0.tif"
OUT = ROOT / "analysis" / "subnational_econ_grid.json"


def block_sum(tif, r0, r1, c0, c1):
    with rasterio.open(tif) as src:
        H = ((r1 - r0) // BLOCK) * BLOCK; W = ((c1 - c0) // BLOCK) * BLOCK
        arr = src.read(1, window=rasterio.windows.Window(c0, r0, W, H)).astype("float64")
        arr[arr < 0] = 0.0
        x0, y0 = src.xy(r0, c0, offset="ul")
    ny, nx = H // BLOCK, W // BLOCK
    return arr.reshape(ny, BLOCK, nx, BLOCK).sum(axis=(1, 3)), x0, y0, ny, nx


def build_grid_both():
    with rasterio.open(POP) as src:
        r0, r1, c0, c1 = africa_window(src)
    pop, x0, y0, ny, nx = block_sum(POP, r0, r1, c0, c1)
    built, *_ = block_sum(BUILT, r0, r1, c0, c1)                  # built-up m2 per 50km cell
    j = np.arange(nx); i = np.arange(ny)
    CX, CY = np.meshgrid(x0 + (j + 0.5) * CELL_M, y0 - (i + 0.5) * CELL_M)
    df = pd.DataFrame({"i": np.repeat(i, nx), "j": np.tile(j, ny),
                       "mx": CX.ravel(), "my": CY.ravel(),
                       "pop": pop.ravel(), "built": built.ravel()})
    df["x0"], df["y0"] = x0, y0
    return df


def fe_spearman(d, ycol, xcol):
    d = d[d[xcol].notna() & np.isfinite(d[xcol])].copy()
    d = d.groupby("country").filter(lambda g: len(g) >= 15)
    xr = d.groupby("country")[xcol].transform(lambda s: rankdata(s))
    yr = d.groupby("country")[ycol].transform(lambda s: rankdata(s))
    rho, p = spearmanr(xr, yr)
    return float(rho), float(p), int(len(d))


def main():
    print("building 50km grid (population + built-up) ...")
    cells = build_grid_both()
    cells = assign_country(cells)
    cells = bin_violence(cells)
    A = (CELL_M / 1000.0) ** 2
    cells["pop_density"] = cells["pop"] / A
    cells["built_density"] = cells["built"] / A                  # built-up m2 per km2
    cells["built_per_cap"] = np.where(cells["pop"] > 50, cells["built"] / cells["pop"], np.nan)  # m2/person
    print(f"  {len(cells):,} cells; {(cells.ins_fatal>0).sum()} with insurgent violence; "
          f"built-up present in {(cells.built>0).sum()} cells")

    print("\n" + "=" * 92)
    print("A. UNCONDITIONED — built-up density vs insurgent violence, within country (FE)")
    print("=" * 92)
    r, p, n = fe_spearman(cells, "ins_fatal", "built_density")
    print(f"   FE rho = {r:+.3f} (p={p:.2e}, n={n:,})   [built tracks people, so + expected]")

    print("\n" + "=" * 92)
    print("B. THE TEST — built-up density CONTROLLING FOR POPULATION (NB fixed-effects GLM)")
    print("=" * 92)
    res = {"A_uncond_built_fe": dict(rho=round(r, 3), p=p, n=n)}
    try:
        import statsmodels.formula.api as smf, statsmodels.api as sm
        d = cells.copy()
        d["lbuilt"] = np.log1p(d.built_density); d["lpop"] = np.log1p(d.pop_density)
        m = smf.glm("ins_fatal ~ lbuilt + lpop + C(country)", data=d,
                    family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        bb, seb = m.params["lbuilt"], m.bse["lbuilt"]
        bp, sep = m.params["lpop"], m.bse["lpop"]
        print(f"   log(built density) | population : coef = {bb:+.3f} ± {seb:.3f}  (z={bb/seb:+.1f})")
        print(f"   log(population)    | built      : coef = {bp:+.3f} ± {sep:.3f}  (z={bp/sep:+.1f})")
        verdict = ("HUBS AVOIDED (economic density protective once targets are held fixed)"
                   if bb < 0 and bb / seb < -2 else
                   "no avoidance (built still + after pop control)" if bb > 0 else "weak/ambiguous")
        print(f"   -> {verdict}")
        res["B_built_given_pop"] = dict(built_coef=round(float(bb), 4), built_z=round(float(bb/seb), 2),
                                        pop_coef=round(float(bp), 4), pop_z=round(float(bp/sep), 2),
                                        verdict=verdict)
    except Exception as e:
        print(f"   (statsmodels NB failed: {type(e).__name__}: {e})")

    print("\n" + "=" * 92)
    print("C. built-up PER CAPITA (capital per person) vs violence, within country (FE)")
    print("=" * 92)
    rc, pc, nc = fe_spearman(cells, "ins_fatal", "built_per_cap")
    print(f"   FE rho = {rc:+.3f} (p={pc:.2e}, n={nc:,})   [neg = insurgents avoid capital-rich cells]")
    res["C_built_per_cap_fe"] = dict(rho=round(rc, 3), p=pc, n=nc)

    print("\n" + "=" * 92)
    print("D. within-country built-per-capita QUINTILE -> insurgent violence")
    print("=" * 92)
    d = cells[cells.built_per_cap.notna()].copy()
    d = d.groupby("country").filter(lambda g: len(g) >= 25)
    d["q"] = d.groupby("country")["built_per_cap"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]))
    tab = d.groupby("q").agg(any_viol=("ins_fatal", lambda s: (s > 0).mean()),
                             mean_fatal=("ins_fatal", "mean"),
                             mean_pop=("pop", "mean")).reset_index()
    print(f"   {'built/capita quintile':<24}{'% w/ violence':>15}{'mean fatal':>12}{'mean pop':>12}")
    for _, x in tab.iterrows():
        lab = "lowest capital" if x.q == 1 else "highest capital" if x.q == 5 else ""
        print(f"   Q{int(x.q)} {lab:<20}{100*x.any_viol:>14.1f}%{x.mean_fatal:>12.1f}{x.mean_pop:>12.0f}")
    res["D_quintile"] = tab.to_dict("records")

    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
