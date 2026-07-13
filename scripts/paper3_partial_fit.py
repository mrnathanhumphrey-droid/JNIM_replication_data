"""
Paper 3 partial fit — PRE_REG_001 strife epicenter diffusion.
In-sample confirms: MLI 2012 → BFA 2020 → NER 2021.
Forward-watch with current data (UCDP-GED v25 through 2024):
- TGO/CIV/GHA: have they crossed 50-fatality strife threshold yet?
- Actor-overlap: JNIM/IS-Sahel dyad presence?
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")

UCDP = Path("data/ucdp/GEDEvent_v25_1.csv")
print("=" * 80)
print("PAPER 3 PARTIAL FIT — Strife Epicenter Diffusion (PRE_REG_001)")
print("=" * 80)

u = pd.read_csv(UCDP, usecols=["country", "year", "type_of_violence", "best",
                                "adm_1", "side_a", "side_b", "dyad_name"])
print(f"\n[1/3] UCDP-GED v25: {len(u)} events, years {u['year'].min()}-{u['year'].max()}")

# Filter type 2 = non-state strife
strife = u[u["type_of_violence"] == 2].copy()
print(f"  Non-state strife events (type=2): {len(strife)}")

# Countries of interest
predicted_countries = ["Togo", "Ivory Coast", "Ghana"]
sahel_anchors = ["Mali", "Burkina Faso", "Niger", "Benin"]
car_subcluster = ["Central African Republic", "Cameroon", "Chad", "Congo"]
hti_subcluster = ["Haiti", "Dominican Republic"]

all_countries = predicted_countries + sahel_anchors + car_subcluster + hti_subcluster

# ============================================================================
# In-sample diffusion confirmation
# ============================================================================
print("\n" + "=" * 80)
print("In-sample diffusion confirmation (MLI 2012 → BFA 2020 → NER 2021)")
print("=" * 80)

print(f"\n{'Country':<28} {'First ≥50 strife year':<22} {'Lifetime strife fatal':<22} {'2020-2024 strife'}")
country_year_strife = strife.groupby(["country", "year"])["best"].sum().reset_index()
results_emergence = {}
for c in all_countries:
    rows = country_year_strife[country_year_strife["country"] == c]
    if len(rows) == 0:
        first_year = "NO STRIFE EVER"
        lifetime = 0
        acute = 0
    else:
        # First year with >= 50
        above_50 = rows[rows["best"] >= 50]
        first_year = int(above_50["year"].min()) if len(above_50) > 0 else "<50 ever"
        lifetime = int(rows["best"].sum())
        acute_rows = rows[rows["year"].between(2020, 2024)]
        acute = int(acute_rows["best"].sum()) if len(acute_rows) > 0 else 0
    results_emergence[c] = {"first_year": first_year, "lifetime": lifetime, "acute": acute}
    print(f"{c:<28} {str(first_year):<22} {lifetime:<22} {acute}")

# ============================================================================
# Forward-watch test: have TGO/CIV/GHA emerged yet?
# ============================================================================
print("\n" + "=" * 80)
print("FORWARD-WATCH: TGO / CIV / GHA emergence status (PRE_REG_001 H3)")
print("=" * 80)

# Year-by-year detail for predicted countries
print("\nYear-by-year strife fatalities 2018-2024:")
print(f"{'Country':<20} {'2018':<8} {'2019':<8} {'2020':<8} {'2021':<8} {'2022':<8} {'2023':<8} {'2024':<8}")
for c in predicted_countries + sahel_anchors:
    row = []
    for yr in range(2018, 2025):
        d = country_year_strife[(country_year_strife["country"] == c) & (country_year_strife["year"] == yr)]
        val = int(d["best"].sum()) if len(d) > 0 else 0
        row.append(f"{val:<8}")
    print(f"{c:<20} {' '.join(row)}")

# F1 falsifier check (TGO/CIV/GHA show ZERO by end-2025)
print("\n[F1 check] TGO/CIV/GHA strife status:")
for c in predicted_countries:
    r = results_emergence[c]
    if r["first_year"] == "NO STRIFE EVER" or r["first_year"] == "<50 ever":
        status = "NOT YET CROSSED 50 THRESHOLD"
    else:
        status = f"FIRST ≥50 YEAR: {r['first_year']}"
    print(f"  {c}: lifetime={r['lifetime']}, acute={r['acute']}, {status}")

# Predicted window: TGO 2022-2025, CIV 2023-2026, GHA 2024-2027
print("\nPredicted window comparison:")
for c, window in [("Togo", "2022-2025"), ("Ivory Coast", "2023-2026"), ("Ghana", "2024-2027")]:
    r = results_emergence[c]
    print(f"  {c}: predicted {window}; observed first ≥50 year = {r['first_year']}")

# ============================================================================
# Actor-overlap test (PRE_REG_001 Prediction set B)
# ============================================================================
print("\n" + "=" * 80)
print("ACTOR-OVERLAP TEST (H2 mechanism a: JNIM / IS-Sahel cross-border)")
print("=" * 80)

# Search for JNIM, IS-Sahel, and related actors across all violence types
target_actors = ["JNIM", "Jama'at Nasr al-Islam wal Muslimin", "IS-Sahel", "IS Sahel",
                 "Islamic State Sahel", "Dozos", "Koglweogo", "VDP"]

print(f"\nActor presence across countries (any year, any violence type):")
print(f"{'Actor pattern':<35} {'MLI':<5} {'BFA':<5} {'NER':<5} {'BEN':<5} {'TGO':<5} {'CIV':<5} {'GHA':<5}")
country_codes = {"Mali": "MLI", "Burkina Faso": "BFA", "Niger": "NER", "Benin": "BEN",
                 "Togo": "TGO", "Ivory Coast": "CIV", "Ghana": "GHA"}

# Use any-string-match for actor names
for actor_pattern in target_actors:
    row = [f"{actor_pattern:<35}"]
    for c, code in country_codes.items():
        rows_c = u[u["country"] == c]
        # Check if actor name appears in side_a, side_b, or dyad_name
        found = (
            rows_c["side_a"].astype(str).str.contains(actor_pattern, case=False, na=False, regex=False).any() or
            rows_c["side_b"].astype(str).str.contains(actor_pattern, case=False, na=False, regex=False).any() or
            rows_c["dyad_name"].astype(str).str.contains(actor_pattern, case=False, na=False, regex=False).any()
        )
        row.append(f"{'YES' if found else 'no':<5}")
    print(" ".join(row))

# More granular: full dyad list for TGO/CIV/GHA
print("\n--- TGO/CIV/GHA UCDP events 2020-2024 (any type_of_violence) ---")
for c in predicted_countries:
    rows = u[(u["country"] == c) & (u["year"].between(2020, 2024))]
    if len(rows) == 0:
        print(f"\n{c}: NO UCDP EVENTS 2020-2024")
        continue
    print(f"\n{c}: {len(rows)} events 2020-2024")
    print(f"  Total fatalities: {rows['best'].sum():,.0f}")
    print(f"  By type: state-based={rows[rows['type_of_violence']==1]['best'].sum():,.0f}, "
          f"strife={rows[rows['type_of_violence']==2]['best'].sum():,.0f}, "
          f"one-sided={rows[rows['type_of_violence']==3]['best'].sum():,.0f}")
    top_dyads = rows.groupby("dyad_name")["best"].sum().sort_values(ascending=False).head(5)
    print(f"  Top 5 dyads:")
    for dyad, fat in top_dyads.items():
        print(f"    {dyad}: {fat:,.0f}")

# ============================================================================
# CAR sub-cluster check (PRE_REG_001 Prediction set C)
# ============================================================================
print("\n" + "=" * 80)
print("CAR sub-cluster (PRE_REG_001 Prediction set C)")
print("=" * 80)

for c, expected in [("Central African Republic", "epicenter 2011"),
                     ("Cameroon", "≥50 by 2018"),
                     ("Chad", "≥50 by 2015"),
                     ("Congo", "NOT predicted to diffuse")]:
    r = results_emergence.get(c, {})
    print(f"  {c}: predicted {expected}; observed first ≥50 = {r.get('first_year', 'NO DATA')}, lifetime={r.get('lifetime', 0)}")

# ============================================================================
# HTI sub-cluster (PRE_REG_001 Prediction set D)
# ============================================================================
print("\n" + "=" * 80)
print("HTI sub-cluster (PRE_REG_001 Prediction set D)")
print("=" * 80)

for c in hti_subcluster:
    r = results_emergence.get(c, {})
    print(f"  {c}: first ≥50 year = {r.get('first_year', 'NO DATA')}, lifetime={r.get('lifetime', 0)}, acute={r.get('acute', 0)}")

# Save
import json
out = {
    "in_sample_anchors": {c: results_emergence[c] for c in sahel_anchors},
    "predicted_countries": {c: results_emergence[c] for c in predicted_countries},
    "car_subcluster": {c: results_emergence[c] for c in car_subcluster},
    "hti_subcluster": {c: results_emergence[c] for c in hti_subcluster},
}
Path("analysis/paper3_partial_fit_2026_05_27.json").write_text(
    json.dumps(out, indent=2, default=str), encoding="utf-8"
)
print(f"\nResults saved to: analysis/paper3_partial_fit_2026_05_27.json")
