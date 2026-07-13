"""
Paper 3 P1 test — fiscal capacity per Sahel-facing border km vs JNIM fatalities.
Computes shared border lengths reproducibly from Natural Earth admin_0 (10m),
joins World Bank military spend, and runs the cross-section (Spearman; n is tiny — honest).
"""
import geopandas as gpd, pandas as pd, numpy as np, json, zipfile, io
from pathlib import Path
from scipy.stats import spearmanr
import sys
sys.stdout.reconfigure(encoding="utf-8")

NE = Path("data/natural_earth/ne0.zip")
w = gpd.read_file(NE)
namecol = "ADMIN" if "ADMIN" in w.columns else "NAME"
w = w[[namecol, "geometry"]].rename(columns={namecol: "name"})

# UTM 31N — metric CRS for West Africa; few-% edge distortion, immaterial vs the 130–1140 km spread
wm = w.to_crs(32631)
def geom(n):
    g = wm[wm.name == n]
    return g.geometry.iloc[0] if len(g) else None

# Sahel-source (JNIM staging/transit) countries
SAHEL_SRC = ["Burkina Faso", "Niger", "Mali"]
LITTORAL = ["Benin", "Togo", "Ivory Coast", "Ghana", "Nigeria"]
# Natural Earth naming fixes
ALIAS = {"Ivory Coast": ["Ivory Coast", "Côte d'Ivoire", "Cote d'Ivoire"]}
def resolve(n):
    for cand in ALIAS.get(n, [n]):
        if geom(cand) is not None: return cand
    return n

def border_km(a, b):
    ga, gb = geom(resolve(a)), geom(resolve(b))
    if ga is None or gb is None: return 0.0
    shared = ga.boundary.intersection(gb.boundary)
    return shared.length / 1000.0  # m -> km

print("=" * 78)
print("SAHEL-FACING BORDER LENGTHS (Natural Earth 10m, UTM31N) — km with BFA/NER/MLI")
print("=" * 78)
border = {}
print(f"{'Country':<14}{'BurkinaFaso':>13}{'Niger':>10}{'Mali':>10}{'SahelTotal':>13}")
for c in LITTORAL:
    parts = {s: border_km(c, s) for s in SAHEL_SRC}
    tot = sum(parts.values())
    border[c] = tot
    print(f"{c:<14}{parts['Burkina Faso']:>13.0f}{parts['Niger']:>10.0f}{parts['Mali']:>10.0f}{tot:>13.0f}")

# CIA Factbook reference (validation)
FB = {"Benin": 386+277, "Togo": 131, "Ivory Coast": 545+599, "Ghana": 602, "Nigeria": 1608}
print("\nValidation vs CIA Factbook bilateral sums (km):")
for c in LITTORAL:
    print(f"  {c:<14} computed={border[c]:>6.0f}   factbook={FB[c]:>6}   diff={border[c]-FB[c]:>+6.0f}")

# ---- World Bank military spend (mean 2020-2024, USD) + outcome + controls ----
import urllib.request
codes = {"Benin":"BJ","Togo":"TG","Ivory Coast":"CI","Ghana":"GH","Nigeria":"NG"}
def wb(code, ind):
    url=f"https://api.worldbank.org/v2/country/{code}/indicator/{ind}?format=json&date=2020:2024&per_page=100"
    d=json.load(urllib.request.urlopen(url,timeout=40))[1]
    vals=[r['value'] for r in d if r['value'] is not None]
    return np.mean(vals) if vals else np.nan
milex_mean = {c: wb(codes[c], "MS.MIL.XPND.CD") for c in LITTORAL}

# Outcome: cumulative JNIM-attributable fatalities 2020-2025 (from v26.1 fit)
jnim_fatal = {"Benin":333, "Togo":275, "Ivory Coast":0, "Ghana":0, "Nigeria":None}  # NGA: Boko Haram/ISWAP confound, exclude from JNIM count
# Years since first JNIM border contact (approx, from literature)
contact_yr = {"Benin":2021, "Togo":2022, "Ivory Coast":2020, "Ghana":None, "Nigeria":2024}

print("\n" + "=" * 78)
print("CAPACITY CROSS-SECTION (P1)")
print("=" * 78)
print(f"{'Country':<14}{'milex_mean$M':>13}{'%spend/km($k)':>15}{'JNIM_fatal':>12}{'yrs_contact':>12}")
rows=[]
for c in LITTORAL:
    perkm = milex_mean[c]/border[c]/1e3 if border[c] else np.nan  # $k per km
    yrs = (2025-contact_yr[c]) if contact_yr[c] else np.nan
    rows.append(dict(country=c, milex_mean=milex_mean[c], border_km=border[c],
                     perkm_k=perkm, jnim_fatal=jnim_fatal[c], yrs_contact=yrs))
    print(f"{c:<14}{milex_mean[c]/1e6:>13.0f}{perkm:>15.0f}{str(jnim_fatal[c]):>12}{str(yrs):>12}")

df = pd.DataFrame(rows)
core = df[df.jnim_fatal.notna()].copy()  # drop Nigeria (Boko Haram confound)
print(f"\nCore JNIM-corridor set (n={len(core)}, Nigeria excluded — BH/ISWAP confound):")

def sp(x, y, label):
    m = x.notna() & y.notna()
    if m.sum() < 3: print(f"  {label}: n<3, skip"); return
    rho, p = spearmanr(x[m], y[m])
    print(f"  {label}: Spearman rho={rho:+.2f} (p={p:.2f}, n={m.sum()})")

sp(core.perkm_k, core.jnim_fatal, "per-border-km spend  vs JNIM fatalities")
sp(core.milex_mean, core.jnim_fatal, "ABSOLUTE milex       vs JNIM fatalities")

print("\nNOTE: n=4 — these are descriptive, not powered. Read the ordering, not the p-value.")
Path("analysis/paper3_border_capacity_2026_06_30.json").write_text(
    json.dumps({"border_km":border,"milex_mean":milex_mean,"rows":rows}, indent=2, default=str), encoding="utf-8")
print("\nsaved -> analysis/paper3_border_capacity_2026_06_30.json")
