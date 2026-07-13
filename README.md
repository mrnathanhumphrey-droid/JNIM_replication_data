# JNIM Containment — Replication Code

Analysis code behind two research briefs on how West African states contain the
southward spread of JNIM (Jama'at Nusrat al-Islam wal-Muslimin) from the Sahel
toward the coast, 2018–2025 — plus a follow-on **extension** that tests whether the
same economic-density mechanism generalizes to other African insurgencies.

This repository holds only the scripts and their computed outputs, so the results
can be reproduced and audited independently. The written briefs and their full
argument are published separately:

- **A Fiscal Circuit-Breaker on Jihadist Diffusion: Why Côte d'Ivoire Contained JNIM and Benin Did Not** — https://doi.org/10.5281/zenodo.21184277
- **Density and Denial: Economic Density as State Capacity and Road Control as Insurgent Strategy in the Containment of JNIM Diffusion, West Africa, 2018–2025** — https://doi.org/10.5281/zenodo.21300874

Author: Nathan Humphrey (Resolve Research).

## Contents

1. [What each script does](#what-each-script-does)
2. [Extension — generalization across three insurgencies](#extension--generalization-across-three-insurgencies)
3. [Repository layout](#repository-layout)
4. [Data (not included)](#data-not-included)
5. [Running the scripts](#running-the-scripts)
6. [Auditing the results](#auditing-the-results)
7. [Citation](#citation)
8. [License](#license)

## What each script does

Every script reads public source data, computes a result, and writes a JSON file
to `analysis/`. The samples are small national cross-sections; the briefs report
them as orderings, not powered statistical estimates.

| Script | Purpose | Output |
|---|---|---|
| `paper3_partial_fit.py` | In-sample diffusion fit on UCDP-GED v25.1: Mali 2012 → Burkina Faso 2020 → Niger 2021. | `paper3_partial_fit_2026_05_27.json` |
| `paper3_v26_fit.py` | Refresh of the diffusion fit on UCDP-GED v26.1, adding final-year 2025 and the coastal receiving states. | `paper3_v26_fit_2026_06_30.json` |
| `paper3_border_capacity.py` | Shared Sahel-facing border lengths from Natural Earth, joined to World Bank military spend; spend-per-border-km cross-section. | `paper3_border_capacity_2026_06_30.json` |
| `paper3_spending_density.py` | GDP-per-km² (economic density) versus military topline as predictors of JNIM fatalities. | `paper3_spending_density_2026_07_09.json` |
| `paper3_road_corridor_ucdp.py` | Distance from JNIM events to the primary road network in Mali (UCDP-GED, event-level coordinates). | `paper3_road_corridor_ucdp_2026_07_09.json` |
| `paper3_road_corridor_gdelt.py` | Earlier road-corridor pass on GDELT, superseded by the UCDP version above. | `paper3_road_corridor_gdelt_2026_07_09.json` |
| `paper3_bamako_approach.py` | Whether JNIM road violence closes in on Bamako, split by national fuel corridor. | `paper3_bamako_approach_2026_07_09.json` |
| `paper3_push_vs_resistance.py` | Separates incoming JNIM pressure (push) from state resistance using a leakage rate. | `paper3_push_vs_resistance_2026_07_09.json` |

## Extension — generalization across three insurgencies

> **Status: working analysis, not yet a published brief.** These scripts test whether
> the density-as-state-capacity mechanism from the JNIM briefs holds beyond West Africa.
> They are provided for transparency and reproduction; treat the results as exploratory
> until written up. Sample sizes are small at the national level — read the orderings.

The extension pools the receiving states of **three** African jihadist diffusions —
JNIM (Sahel → littoral), Boko Haram/ISWAP (NE Nigeria → Niger/Chad/Cameroon), and
al-Shabaab (Somalia → Kenya/Ethiopia) — and asks the same question at two scales.

**National (n = 9 receiving states).** The scale-free *leakage rate* (violence inside ÷
doorstep push) falls with GDP/km² at ρ = −0.83, pooled across the three conflicts —
economic density beats GDP/capita, and the relationship is not an artifact of push,
size, or reverse causality. Its main limit: the low-density arm rests on Boko Haram, so
dropping that conflict weakens the pooled estimate (a coverage, not a size, problem).

**Subnational (50 km grid + event level).** A coarse grid cannot separate "avoid the
hub" from "attack its road-approaches," but at event resolution insurgent violence sits
markedly **farther from economic hubs** (urban centres) than other violence in the same
countries. Controlling for where the population actually lives (a within-country,
population-weighted null), insurgent events fall at the **71st percentile** of the
population's own distance-to-hub distribution — they strike farther from the cores than
the people do. This holds across all three insurgencies.

| Script | Purpose | Output |
|---|---|---|
| `multiconflict_density_leakage.py` | Pools JNIM / Boko Haram-ISWAP / al-Shabaab receiving states; leakage vs GDP/km². | `multiconflict_density_leakage.json` |
| `multiconflict_robustness.py` | Hardening battery: leave-one-out (state + conflict), permutation, spec sweep, alternative predictors and leakage definitions, actor-coding variants. | `multiconflict_robustness.json` |
| `multiconflict_size_gravity.py` | Conflict-size census; tests whether the leakage result is a size or gravity artifact. | `multiconflict_size_gravity.json` |
| `subnational_density_grid.py` | 50 km grid; population density vs insurgent violence within country (fixed effects). | `subnational_density_grid.json` |
| `subnational_econ_grid.py` | 50 km grid; built-up (economic) density vs violence, controlling for population. | `subnational_econ_grid.json` |
| `subnational_hub_distance.py` | Event-level distance to nearest urban centre, insurgent vs baseline violence. | `subnational_hub_distance.json` |
| `subnational_hub_popnull.py` | Within-country, population-weighted null for the hub-avoidance test. | `subnational_hub_popnull.json` |

## Repository layout

```
JNIM/
├─ scripts/     the analysis scripts (paper3_* = the two briefs; multiconflict_*/subnational_* = the extension)
├─ analysis/    the JSON outputs they produced (the numbers behind the tables)
├─ requirements.txt
├─ LICENSE
└─ README.md
```

## Data (not included)

The scripts read public datasets that are too large to store here and are better
obtained from their maintainers directly. Download each one and place it at the path
shown, relative to the repository root:

| Dataset | Path | Source |
|---|---|---|
| UCDP-GED v25.1 | `data/ucdp/GEDEvent_v25_1.csv` | https://ucdp.uu.se/downloads/ |
| UCDP-GED v26.1 | `data/ucdp/GEDEvent_v26_1.csv` | https://ucdp.uu.se/downloads/ |
| Natural Earth admin-0 countries, 10m | `data/natural_earth/ne0.zip` | https://www.naturalearthdata.com/ |
| OSM Mali roads (Geofabrik) | `data/osm/mali_roads/gis_osm_roads_free_1.shp` | https://download.geofabrik.de/africa/mali.html |
| GDELT Mali events | `data/gdelt/gdelt-mali-2014_2024.csv` | https://www.gdeltproject.org/ |
| World Bank WDI (bulk CSV) | `data/wb_wdi/extracted/WDICSV.csv` | https://datatopics.worldbank.org/world-development-indicators/ |
| GHS-POP 2020, 1 km (population) | `data/ghs/GHS_POP_1km/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.tif` | https://ghsl.jrc.ec.europa.eu/download.php |
| GHS-SMOD 2020, 1 km (settlement model / urban centres) | `data/ghs/GHS_SMOD_1km/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0.tif` | https://ghsl.jrc.ec.europa.eu/download.php |
| GHS-BUILT-S 2020, 1 km (built-up surface) | `data/ghs/GHS_BUILT_S_1km/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_1000_V1_0.tif` | https://ghsl.jrc.ec.europa.eu/download.php |

`paper3_border_capacity.py` and `paper3_spending_density.py` also read the World
Bank indicator API directly over HTTP (GDP, military expenditure, population), so
they need network access. The GHS layers are only needed for the subnational
extension scripts.

## Running the scripts

Requires Python 3.10 or newer.

```
pip install -r requirements.txt
```

Run each script from the repository root so the relative data paths resolve:

```
python scripts/paper3_v26_fit.py
python scripts/multiconflict_density_leakage.py
```

Each run rewrites its JSON file in `analysis/`. The extension scripts import shared
helpers from one another, so keep them together in `scripts/`.

## Auditing the results

The JSON files already in `analysis/` are the exact outputs used in the briefs (and,
for the extension, the exact numbers reported in the working analysis). To check them,
download the source data to the paths above, re-run a script, and compare its JSON
against the committed copy. Any difference is a discrepancy worth raising.

## Citation

If you use this code or its results, please cite the briefs:

> Humphrey, N. (2026). *A Fiscal Circuit-Breaker on Jihadist Diffusion: Why Côte d'Ivoire Contained JNIM and Benin Did Not.* Zenodo. https://doi.org/10.5281/zenodo.21184277

> Humphrey, N. (2026). *Density and Denial: Economic Density as State Capacity and Road Control as Insurgent Strategy in the Containment of JNIM Diffusion, West Africa, 2018–2025.* Zenodo. https://doi.org/10.5281/zenodo.21300874

## License

Code is released under the MIT License (see `LICENSE`). The source datasets keep
their own licenses, held by their respective maintainers.
