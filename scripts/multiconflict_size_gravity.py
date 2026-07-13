"""
multiconflict_size_gravity.py — are the three conflicts equitable in size, and does the pooled
density-leakage result need gravity/size normalization?

Answers three things:
  A. SIZE census per conflict (source mass, receiving mass, density RANGE of receiving states).
  B. Is leakage size-confounded? (rank-corr of leakage vs push magnitude & vs conflict total size).
     If leakage is independent of magnitude, the ratio already nets out size.
  C. GRAVITY normalization: recompute push with (i) distance-decay weighting and (ii) shared-
     border-length normalization, redo pooled rho AND leave-one-conflict-out. Does the finding —
     and the Boko-Haram dependence — change?

    python _scripts/multiconflict_size_gravity.py
"""
from __future__ import annotations
import json, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from scipy.stats import spearmanr
from shapely.ops import unary_union
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiconflict_robustness import (load_ucdp, densities, CONFLICTS, ne_in, poly_of,
                                      select, recv_gdf)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "multiconflict_size_gravity.json"
DECAY_L = 100.0   # km e-folding length for gravity distance-decay

def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3: return (np.nan, np.nan, int(m.sum()))
    r, p = spearmanr(x[m], y[m]); return (float(r), float(p), int(m.sum()))

def compute(u, D):
    """Per receiving state: push (250km cut), push_decay, push_per_border_km, outcome, 3 leakages."""
    rows = []
    for name, cfg in CONFLICTS.items():
        ev = select(u, name, ["JAS", "IS"])
        ev = ev[(ev.where_prec <= 3) & (ev.year.between(2015, 2025))]
        recv = recv_gdf(name); w = ne_in(cfg["utm"])
        src_union = unary_union([poly_of(s, w) for s in cfg["source"] if poly_of(s, w) is not None])
        g = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev.longitude, ev.latitude),
                             crs="EPSG:4326").to_crs(cfg["utm"])
        inside = gpd.sjoin(g, recv, how="inner", predicate="within")
        outcome = inside.groupby("recv_country")["best"].sum().to_dict()
        src = g[g.country.isin(cfg["source"])]
        near = gpd.sjoin_nearest(src, recv, how="left", distance_col="d_m")
        near = near[~near.index.duplicated(keep="first")].copy()
        near["d_km"] = near.d_m / 1000.0
        near["w"] = np.exp(-near.d_km / DECAY_L)                 # gravity distance-decay weight
        cut = near[near.d_km <= 250]
        push = cut.groupby("recv_country")["best"].sum().to_dict()
        push_decay = (near.assign(wb=near.best * near.w).groupby("recv_country")["wb"].sum()).to_dict()
        # source-conflict totals (size)
        src_fatal, src_ev = float(src.best.sum()), int(len(src))
        for c in cfg["recv"]:
            poly_c = poly_of(c, w)
            border_km = poly_c.boundary.intersection(src_union.boundary).length / 1000.0 if src_union else np.nan
            p, pd_, o = float(push.get(c, 0)), float(push_decay.get(c, 0)), float(outcome.get(c, 0))
            rows.append(dict(conflict=name, country=c, gdp_km2=D[c]["gdp_km2"],
                             src_fatal=src_fatal, src_events=src_ev,
                             push=p, push_decay=round(pd_, 1), border_km=round(border_km, 0),
                             outcome=o,
                             leak=(o / p) if p else np.nan,
                             leak_decay=(o / pd_) if pd_ else np.nan,
                             leak_borderpush=(o / (p / border_km)) if (p and border_km) else np.nan))
    return pd.DataFrame(rows)

def loo_conflict(df, ycol):
    d = df.dropna(subset=[ycol])
    out = {}
    for name in CONFLICTS:
        dd = d[d.conflict != name]; r, p, n = rho(dd.gdp_km2, dd[ycol]); out[name] = (round(r, 3), n)
    r0, p0, n0 = rho(d.gdp_km2, d[ycol])
    return r0, out

def main():
    u = load_ucdp(); D = densities()
    df = compute(u, D)

    print("=" * 100)
    print("A. SIZE CENSUS — per receiving state (source mass is the whole conflict's source total)")
    print("=" * 100)
    print(f"{'conflict':<11}{'recv':<14}{'GDP/km2$':>9}{'srcFatal':>9}{'push':>7}{'push~grav':>10}{'border_km':>10}{'outcome':>9}{'leak':>7}")
    for _, r in df.iterrows():
        print(f"{r.conflict:<11}{r.country:<14}{r.gdp_km2/1e3:>9.0f}{r.src_fatal:>9.0f}{r.push:>7.0f}"
              f"{r.push_decay:>10.0f}{str(r.border_km):>10}{r.outcome:>9.0f}{('%.3f'%r.leak) if r.leak==r.leak else '  nan':>7}")

    print("\nConflict-level totals (are they equitable?):")
    g = df.groupby("conflict").agg(src_fatal=("src_fatal", "first"), src_events=("src_events", "first"),
                                   recv_push=("push", "sum"), recv_outcome=("outcome", "sum"),
                                   n_recv=("country", "size"),
                                   dens_min=("gdp_km2", lambda s: s.min()/1e3),
                                   dens_max=("gdp_km2", lambda s: s.max()/1e3))
    for name, r in g.iterrows():
        print(f"  {name:<11} source={r.src_fatal:>7.0f} fatal / {r.src_events:>5} ev | "
              f"recv push={r.recv_push:>6.0f} outcome={r.recv_outcome:>6.0f} | "
              f"n_recv={int(r.n_recv)} | GDP/km2 range {r.dens_min:.0f}-{r.dens_max:.0f}k")
    sizes = g.src_fatal
    print(f"  -> source-size ratio  max/min = {sizes.max()/sizes.min():.1f}x  (JNIM vs Boko vs Shabaab are NOT equitable)")

    print("\n" + "=" * 100)
    print("B. IS LEAKAGE SIZE-CONFOUNDED?  (if leak is independent of magnitude, the ratio nets out size)")
    print("=" * 100)
    d = df.dropna(subset=["leak"])
    for lbl, x in (("leak vs push magnitude", d.push),
                   ("leak vs source-conflict size", d.src_fatal),
                   ("leak vs outcome magnitude", d.outcome),
                   ("leak vs border_km", d.border_km)):
        r, p, n = rho(x, d.leak)
        print(f"  {lbl:<32} rho={r:+.3f} (p={p:.3f}, n={n})   {'<- size leaks in' if p<0.1 else 'independent of size ✓'}")

    print("\n" + "=" * 100)
    print("C. GRAVITY NORMALIZATION — recompute pooled rho + leave-one-conflict-out under 3 push defs")
    print("=" * 100)
    for ycol, lbl in (("leak", "push = fatal within 250km (BASE)"),
                      ("leak_decay", f"push = distance-decay weighted (L={DECAY_L:.0f}km)"),
                      ("leak_borderpush", "push = per shared-border-km (exposure-normalized)")):
        r0, loo = loo_conflict(df, ycol)
        print(f"\n  [{lbl}]")
        print(f"     pooled rho = {r0:+.3f}")
        for name, (r, n) in loo.items():
            flag = "  <- collapses" if abs(r) < 0.6 else ""
            print(f"     drop {name:<11} rho={r:+.3f} (n={n}){flag}")

    print("\n" + "=" * 100)
    print("D. WHY DOES DROPPING BOKO HARAM COLLAPSE IT? — density RANGE per conflict")
    print("=" * 100)
    for name in CONFLICTS:
        dd = df[df.conflict == name]
        print(f"  {name:<11} GDP/km2 (k): {sorted((dd.gdp_km2/1e3).round(0).tolist())}")
    print("  -> if Boko Haram holds the ONLY low-density observations, the collapse is a RANGE/coverage")
    print("     problem (fixable only with more low-density points, i.e. subnational), NOT a size artifact.")

    OUT.write_text(json.dumps({"rows": df.to_dict("records")}, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved -> {OUT}")

if __name__ == "__main__":
    main()
