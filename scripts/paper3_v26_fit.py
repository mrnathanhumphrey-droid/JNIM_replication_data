"""
Paper 3 v26.1 fit — PRE_REG_001 strife epicenter diffusion, REFRESHED on UCDP-GED v26.1 (1989-2025).
Adds 2025 final year (the forward-watch tripwire data) and PROMOTES Benin to a focus country.
Outputs the empirical backbone for the JNIM fiscal-circuit-breaker blog follow-up.
"""
import pandas as pd, numpy as np, json
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")

UCDP = Path("data/ucdp/GEDEvent_v26_1.csv")
OUT = Path("analysis/paper3_v26_fit_2026_06_30.json")

u = pd.read_csv(UCDP, usecols=["country","year","type_of_violence","best",
                               "adm_1","side_a","side_b","dyad_name"], low_memory=False)
print("="*84)
print(f"PAPER 3 v26.1 FIT — UCDP-GED {len(u):,} events, {u.year.min()}-{u.year.max()}")
print("="*84)

strife = u[u.type_of_violence==2].copy()
cyt = strife.groupby(["country","year"])["best"].sum().reset_index()  # type-2 only

focus = ["Togo","Ivory Coast","Ghana","Benin"]
anchors = ["Mali","Burkina Faso","Niger"]
TYPE = {1:"state-based",2:"strife",3:"one-sided"}

def first_ge50(c):
    r = cyt[cyt.country==c]
    a = r[r.best>=50]
    return int(a.year.min()) if len(a) else None

# ---- 1. Type-2 tripwire status ----
print("\n[1] TYPE-2 STRIFE TRIPWIRE (the formal PRE_REG_001 operational test, >=50/yr)")
print(f"{'Country':<16}{'first >=50 yr':<15}{'lifetime t2':<13}{'2020-24 t2':<12}{'2025 t2':<10}")
emergence = {}
for c in focus+anchors:
    r = cyt[cyt.country==c]
    life = int(r.best.sum()) if len(r) else 0
    acute = int(r[r.year.between(2020,2024)].best.sum()) if len(r) else 0
    y25 = int(r[r.year==2025].best.sum()) if len(r) else 0
    fy = first_ge50(c)
    emergence[c] = dict(first_ge50=fy, lifetime_t2=life, t2_2020_24=acute, t2_2025=y25)
    print(f"{c:<16}{str(fy):<15}{life:<13}{acute:<12}{y25:<10}")

# ---- 2. Year-by-year ALL-TYPE fatalities (the substantive front, not just t2) ----
print("\n[2] ALL-TYPE FATALITIES 2018-2025 (total organized-violence deaths)")
allcy = u.groupby(["country","year"])["best"].sum().reset_index()
yrs = list(range(2018,2026))
print(f"{'Country':<16}" + "".join(f"{y:<8}" for y in yrs))
for c in focus+anchors:
    row = [int(allcy[(allcy.country==c)&(allcy.year==y)].best.sum() or 0) for y in yrs]
    print(f"{c:<16}" + "".join(f"{v:<8}" for v in row))

# ---- 3. Per-country 2024 vs 2025 breakdown by type + top dyads ----
print("\n[3] FOCUS COUNTRIES — 2024 vs 2025 by violence type + top dyads")
detail = {}
for c in focus:
    d = {}
    for yr in (2024,2025):
        rows = u[(u.country==c)&(u.year==yr)]
        by_type = {TYPE[t]: int(rows[rows.type_of_violence==t].best.sum()) for t in (1,2,3)}
        d[yr] = dict(events=int(len(rows)), total_fatal=int(rows.best.sum()), by_type=by_type)
    # top dyads 2025
    r25 = u[(u.country==c)&(u.year==2025)]
    top = r25.groupby("dyad_name")["best"].sum().sort_values(ascending=False).head(6)
    d["top_dyads_2025"] = {k:int(v) for k,v in top.items()}
    # JNIM presence (any year/type)
    rc = u[u.country==c]
    jnim = bool(rc[["side_a","side_b","dyad_name"]].apply(
        lambda s: s.astype(str).str.contains("JNIM|Jama'at Nasr al-Islam", case=False, regex=True, na=False)).any().any())
    d["jnim_present"] = jnim
    detail[c] = d
    print(f"\n--- {c} ---  JNIM present: {jnim}")
    for yr in (2024,2025):
        e=d[yr]; print(f"  {yr}: {e['events']} events, {e['total_fatal']} fatal  | "
                        f"state-based {e['by_type']['state-based']}, strife {e['by_type']['strife']}, one-sided {e['by_type']['one-sided']}")
    print("  Top dyads 2025:")
    for k,v in d["top_dyads_2025"].items():
        print(f"    {v:>5}  {k}")

# ---- 4. Sahel-anchor 2025 totals for context ----
print("\n[4] SAHEL-CORE 2025 all-type fatalities (context):")
for c in anchors:
    print(f"  {c}: {int(allcy[(allcy.country==c)&(allcy.year==2025)].best.sum() or 0)}")

# ---- 5. JNIM-attributable fatalities in the littoral (any type) 2020-2025 ----
print("\n[5] JNIM-ATTRIBUTABLE fatalities by country, 2020-2025 (dyad/side name match)")
mask = u[["side_a","side_b","dyad_name"]].apply(
    lambda s: s.astype(str).str.contains("JNIM|Jama'at Nasr al-Islam", case=False, regex=True, na=False)).any(axis=1)
jn = u[mask & u.year.between(2020,2025)]
jnsum = jn.groupby("country")["best"].sum().sort_values(ascending=False)
jnim_by_country = {}
for c in focus+anchors:
    v = int(jnsum.get(c,0)); jnim_by_country[c]=v
    print(f"  {c:<16}{v}")

out = dict(version="UCDP-GED v26.1 (1989-2025)", fired="2026-06-30",
           emergence=emergence, focus_detail=detail, jnim_2020_25=jnim_by_country)
OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print(f"\nsaved -> {OUT}")
