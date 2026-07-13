# JNIM Containment — Replication Code

Analysis code behind two research briefs on how West African states contain the
southward spread of JNIM (Jama'at Nusrat al-Islam wal-Muslimin) from the Sahel
toward the coast, 2018–2025.

This repository holds only the scripts and their computed outputs, so the results
can be reproduced and audited independently. The written briefs and their full
argument are published separately:

- **A Fiscal Circuit-Breaker on Jihadist Diffusion: Why Côte d'Ivoire Contained JNIM and Benin Did Not** — https://doi.org/10.5281/zenodo.21184277
- **Density and Denial: Economic Density as State Capacity and Road Control as Insurgent Strategy in the Containment of JNIM Diffusion, West Africa, 2018–2025** — https://doi.org/10.5281/zenodo.21300874

Author: Nathan Humphrey (Resolve Research).

## Contents

1. [What each script does](#what-each-script-does)
2. [Repository layout](#repository-layout)
3. [Data (not included)](#data-not-included)
4. [Running the scripts](#running-the-scripts)
5. [Auditing the results](#auditing-the-results)
6. [Citation](#citation)
7. [License](#license)

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

## Repository layout

```
JNIM/
├─ scripts/     the eight analysis scripts
├─ analysis/    the JSON outputs they produced (the numbers behind the briefs' tables)
├─ requirements.txt
├─ LICENSE
└─ README.md
```

## Data (not included)

The scripts read public datasets that are too large to store here (about 1.2 GB
total) and are better obtained from their maintainers directly. Download each one
and place it at the path shown, relative to the repository root:

| Dataset | Path | Source |
|---|---|---|
| UCDP-GED v25.1 | `data/ucdp/GEDEvent_v25_1.csv` | https://ucdp.uu.se/downloads/ |
| UCDP-GED v26.1 | `data/ucdp/GEDEvent_v26_1.csv` | https://ucdp.uu.se/downloads/ |
| Natural Earth admin-0 countries, 10m | `data/natural_earth/ne0.zip` | https://www.naturalearthdata.com/ |
| OSM Mali roads (Geofabrik) | `data/osm/mali_roads/gis_osm_roads_free_1.shp` | https://download.geofabrik.de/africa/mali.html |
| GDELT Mali events | `data/gdelt/gdelt-mali-2014_2024.csv` | https://www.gdeltproject.org/ |
| World Bank WDI (bulk CSV) | `data/wb_wdi/extracted/WDICSV.csv` | https://datatopics.worldbank.org/world-development-indicators/ |

`paper3_border_capacity.py` and `paper3_spending_density.py` also read the World
Bank indicator API directly over HTTP (GDP, military expenditure, population), so
they need network access.

## Running the scripts

Requires Python 3.10 or newer.

```
pip install -r requirements.txt
```

Run each script from the repository root so the relative data paths resolve:

```
python scripts/paper3_v26_fit.py
```

Each run rewrites its JSON file in `analysis/`.

## Auditing the results

The JSON files already in `analysis/` are the exact outputs used in the briefs.
To check them, download the source data to the paths above, re-run a script, and
compare its JSON against the committed copy. Any difference is a discrepancy worth
raising.

## Citation

If you use this code or its results, please cite the briefs:

> Humphrey, N. (2026). *A Fiscal Circuit-Breaker on Jihadist Diffusion: Why Côte d'Ivoire Contained JNIM and Benin Did Not.* Zenodo. https://doi.org/10.5281/zenodo.21184277

> Humphrey, N. (2026). *Density and Denial: Economic Density as State Capacity and Road Control as Insurgent Strategy in the Containment of JNIM Diffusion, West Africa, 2018–2025.* Zenodo. https://doi.org/10.5281/zenodo.21300874

## License

Code is released under the MIT License (see `LICENSE`). The source datasets keep
their own licenses, held by their respective maintainers.
