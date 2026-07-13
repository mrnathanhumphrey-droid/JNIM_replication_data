"""
multiconflict_robustness.py — harden the pooled density–leakage finding (rho=-0.85, n=9).
Loads UCDP once, then runs a full battery: leave-one-out (state + conflict), cluster-honest
permutation p, within-conflict sign concordance, spec sweep (push_km / window / precision),
alternative predictors (pop density, GDP/capita, pre-conflict GDP), alternative leakage defs,
and actor-coding variants. Read-only; writes a summary JSON.

    python _scripts/multiconflict_robustness.py
"""
from __future__ import annotations
import json, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from scipy.stats import spearmanr, kendalltau
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
rng = np.random.default_rng(7)

ROOT = Path(__file__).resolve().parents[1]
UCDP = ROOT / "data" / "ucdp" / "GEDEvent_v26_1.csv"
NE = ROOT / "data" / "natural_earth" / "ne0.zip"
WDI = ROOT / "data" / "wb_wdi" / "extracted" / "WDICSV.csv"
OUT = ROOT / "analysis" / "multiconflict_robustness.json"
ALBERS = "ESRI:102022"
ALIAS = {"Ivory Coast": ["Ivory Coast", "Côte d'Ivoire", "Cote d'Ivoire"]}
ISO3 = {"Benin": "BEN", "Togo": "TGO", "Ivory Coast": "CIV", "Ghana": "GHA",
        "Niger": "NER", "Chad": "TCD", "Cameroon": "CMR", "Kenya": "KEN", "Ethiopia": "ETH"}
JNIM_RE = r"JNIM|Nusrat al-Islam|Group for Support of Islam|Jama'at Nasr al-Islam"
SHAB_RE = r"Shabaab|al-Shabab"
LAKE_CHAD = ["Nigeria", "Niger", "Chad", "Cameroon"]

# ---- one-time loads -------------------------------------------------------
def load_ucdp():
    u = pd.read_csv(UCDP, low_memory=False,
                    usecols=["year", "country", "best", "where_prec", "latitude", "longitude",
                             "side_a", "side_b", "dyad_name"]).dropna(subset=["latitude", "longitude"])
    u = u[u.year.between(2013, 2025)]
    nm = u.side_a.astype(str) + "|" + u.side_b.astype(str) + "|" + u.dyad_name.astype(str)
    u["is_jnim"] = nm.str.contains(JNIM_RE, case=False, regex=True, na=False)
    u["is_shab"] = nm.str.contains(SHAB_RE, case=False, regex=True, na=False)
    return u

def poly_of(disp, w):
    for cand in ALIAS.get(disp, [disp]):
        g = w[w.name == cand]
        if len(g):
            return g.geometry.iloc[0]
    return None

def ne_in(utm):
    w = gpd.read_file(NE)
    ncol = "ADMIN" if "ADMIN" in w.columns else "NAME"
    return w[[ncol, "geometry"]].rename(columns={ncol: "name"}).to_crs(utm)

def densities():
    wdi = pd.read_csv(WDI, low_memory=False)
    def wb(iso, ind, yrs):
        r = wdi[(wdi["Country Code"] == iso) & (wdi["Indicator Code"] == ind)]
        if not len(r): return np.nan
        v = pd.to_numeric(r[[str(y) for y in yrs]].iloc[0], errors="coerce").dropna()
        return float(v.mean()) if len(v) else np.nan
    w = ne_in(ALBERS)
    d = {}
    for disp, iso in ISO3.items():
        a = None
        for cand in ALIAS.get(disp, [disp]):
            g = w[w.name == cand]
            if len(g): a = float(g.geometry.iloc[0].area) / 1e6; break
        gdp = wb(iso, "NY.GDP.MKTP.CD", range(2020, 2025))
        gdp_early = wb(iso, "NY.GDP.MKTP.CD", range(2010, 2015))
        pop = wb(iso, "SP.POP.TOTL", range(2020, 2025))
        d[disp] = dict(gdp_km2=gdp / a, gdp_km2_early=gdp_early / a, pop_km2=pop / a,
                       gdp_cap=gdp / pop)
    return d

CONFLICTS = {
    "JNIM":      dict(source=["Mali", "Burkina Faso", "Niger"],
                      recv=["Benin", "Togo", "Ivory Coast", "Ghana"], utm="EPSG:32631"),
    "BokoHaram": dict(source=["Nigeria"], recv=["Niger", "Chad", "Cameroon"], utm="EPSG:32633"),
    "alShabaab": dict(source=["Somalia"], recv=["Kenya", "Ethiopia"], utm="EPSG:32637"),
}

def select(u, conflict, bh_actors):
    if conflict == "JNIM":  return u[u.is_jnim]
    if conflict == "alShabaab": return u[u.is_shab]
    return u[(u.country.isin(LAKE_CHAD)) & (u.side_b.isin(bh_actors))]

# cache receiving GeoDataFrames per conflict
_RECV = {}
def recv_gdf(conflict):
    if conflict not in _RECV:
        cfg = CONFLICTS[conflict]; w = ne_in(cfg["utm"])
        _RECV[conflict] = gpd.GeoDataFrame({"recv_country": cfg["recv"]},
                                           geometry=[poly_of(c, w) for c in cfg["recv"]], crs=cfg["utm"])
    return _RECV[conflict]

def rows_for(u, push_km, prec, y0, y1, bh_actors=("JAS", "IS"), include_eth=True):
    out = []
    for name, cfg in CONFLICTS.items():
        ev = select(u, name, list(bh_actors))
        ev = ev[(ev.where_prec <= prec) & (ev.year.between(y0, y1))]
        recv = recv_gdf(name)
        g = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev.longitude, ev.latitude),
                             crs="EPSG:4326").to_crs(cfg["utm"])
        inside = gpd.sjoin(g, recv, how="inner", predicate="within")
        outcome = inside.groupby("recv_country")["best"].sum().to_dict()
        src = g[g.country.isin(cfg["source"])]
        near = gpd.sjoin_nearest(src, recv, how="left", distance_col="d_m")
        near = near[~near.index.duplicated(keep="first")]
        pe = near[near.d_m / 1000.0 <= push_km]
        push = pe.groupby("recv_country")["best"].sum().to_dict()
        for c in cfg["recv"]:
            if c == "Ethiopia" and not include_eth: continue
            p, o = float(push.get(c, 0)), float(outcome.get(c, 0))
            out.append(dict(conflict=name, country=c, push=p, outcome=o,
                            leakage=(o / p) if p else np.nan))
    return pd.DataFrame(out)

def rho(x, y, method="spearman"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3: return (np.nan, np.nan, int(m.sum()))
    r, p = (spearmanr if method == "spearman" else kendalltau)(x[m], y[m])
    return (float(r), float(p), int(m.sum()))

# ---- battery --------------------------------------------------------------
def main():
    u = load_ucdp()
    D = densities()
    res = {}

    base = rows_for(u, 250, 3, 2015, 2025)
    base["gdp_km2"] = base.country.map(lambda c: D[c]["gdp_km2"])
    lk = base.dropna(subset=["leakage"])
    r0, p0, n0 = rho(lk.gdp_km2, lk.leakage)
    rk, pk, _ = rho(lk.gdp_km2, lk.leakage, "kendall")
    print(f"BASELINE  leakage~GDP/km2 : Spearman rho={r0:+.3f} (p={p0:.4f}, n={n0}) | Kendall tau={rk:+.3f} (p={pk:.4f})")
    res["baseline"] = dict(spearman=r0, p=p0, n=n0, kendall=rk)

    # 1. leave-one-STATE-out
    print("\n[1] LEAVE-ONE-STATE-OUT (rho recomputed dropping each state):")
    loo = []
    for i in range(len(lk)):
        d = lk.drop(lk.index[i]); r, _, _ = rho(d.gdp_km2, d.leakage)
        loo.append((lk.iloc[i]["country"], round(r, 3)))
    for c, r in loo: print(f"    -{c:<14} rho={r:+.3f}")
    rr = [r for _, r in loo]
    print(f"    range: [{min(rr):+.3f}, {max(rr):+.3f}]  -> {'STABLE' if max(rr) < -0.6 else 'FRAGILE'}")
    res["loo_state"] = dict(rows=loo, min=min(rr), max=max(rr))

    # 2. leave-one-CONFLICT-out
    print("\n[2] LEAVE-ONE-CONFLICT-OUT:")
    lco = {}
    for name in CONFLICTS:
        d = lk[lk.conflict != name]; r, p, n = rho(d.gdp_km2, d.leakage)
        lco[name] = dict(rho=round(r, 3), p=round(p, 4), n=n)
        print(f"    drop {name:<11} rho={r:+.3f} (p={p:.4f}, n={n})")
    res["loo_conflict"] = lco

    # 3. cluster-honest permutation p
    print("\n[3] PERMUTATION p (|rho|>=obs):")
    obs = abs(r0); N = 20000
    # naive: shuffle leakage across all states
    cnt = 0; L = lk.leakage.values; G = lk.gdp_km2.values
    for _ in range(N):
        cnt += abs(spearmanr(G, rng.permutation(L))[0]) >= obs
    p_naive = (cnt + 1) / (N + 1)
    # within-conflict: permute leakage only within each conflict
    cntw = 0; idx = {name: lk.index[lk.conflict == name].to_numpy() for name in CONFLICTS}
    for _ in range(N):
        Lp = lk.leakage.copy()
        for name, ix in idx.items():
            Lp.loc[ix] = rng.permutation(Lp.loc[ix].values)
        cntw += abs(spearmanr(G, Lp.values)[0]) >= obs
    p_within = (cntw + 1) / (N + 1)
    # cluster-level: 3 conflict means
    cm = lk.groupby("conflict").agg(gk=("gdp_km2", "mean"), lk=("leakage", "mean"))
    r_cl, p_cl, _ = rho(cm.gk, cm.lk)
    print(f"    naive (9 indep)     p={p_naive:.4f}")
    print(f"    within-conflict     p={p_within:.4f}   (conservative — removes between-conflict signal)")
    print(f"    cluster-level means rho={r_cl:+.3f} (n=3 conflicts)")
    res["permutation"] = dict(p_naive=p_naive, p_within=p_within, cluster_rho=r_cl)

    # 4. within-conflict sign concordance
    print("\n[4] WITHIN-CONFLICT direction (leakage vs GDP/km2):")
    conc = {}
    for name in CONFLICTS:
        d = lk[lk.conflict == name]
        if len(d) >= 2:
            r, _, _ = rho(d.gdp_km2, d.leakage) if len(d) >= 3 else (np.sign(np.corrcoef(d.gdp_km2, d.leakage)[0, 1]) * 1.0, np.nan, len(d))
            sign = "neg✓" if r < 0 else "POS✗"
            conc[name] = round(float(r), 3)
            print(f"    {name:<11} rho={r:+.3f} ({sign}, n={len(d)})")
    res["within_concordance"] = conc

    # 5. spec sweep
    print("\n[5] SPEC SWEEP (rho stays negative?):")
    sweep = {}
    for pk_ in (150, 200, 250, 300, 350):
        d = rows_for(u, pk_, 3, 2015, 2025); d["g"] = d.country.map(lambda c: D[c]["gdp_km2"])
        r, p, n = rho(d.g, d.leakage); sweep[f"push_{pk_}km"] = round(r, 3)
        print(f"    push={pk_}km   rho={r:+.3f} (n={n})")
    for (a, b) in ((2015, 2025), (2018, 2025), (2020, 2025)):
        d = rows_for(u, 250, 3, a, b); d["g"] = d.country.map(lambda c: D[c]["gdp_km2"])
        r, p, n = rho(d.g, d.leakage); sweep[f"yr_{a}_{b}"] = round(r, 3)
        print(f"    yr={a}-{b} rho={r:+.3f} (n={n})")
    for pr in (2, 3, 4):
        d = rows_for(u, 250, pr, 2015, 2025); d["g"] = d.country.map(lambda c: D[c]["gdp_km2"])
        r, p, n = rho(d.g, d.leakage); sweep[f"prec_le{pr}"] = round(r, 3)
        print(f"    prec<={pr}     rho={r:+.3f} (n={n})")
    res["spec_sweep"] = sweep

    # 6. alternative predictors
    print("\n[6] ALTERNATIVE PREDICTORS (leakage vs ...):")
    altp = {}
    for key, lbl in (("gdp_km2", "GDP/km2 (base)"), ("pop_km2", "population/km2"),
                     ("gdp_cap", "GDP/capita"), ("gdp_km2_early", "PRE-conflict GDP(2010-14)/km2")):
        x = lk.country.map(lambda c: D[c][key]); r, p, n = rho(x, lk.leakage)
        altp[key] = dict(rho=round(r, 3), p=round(p, 4)); print(f"    {lbl:<32} rho={r:+.3f} (p={p:.4f})")
    res["alt_predictors"] = altp

    # 7. alternative leakage definitions
    print("\n[7] ALTERNATIVE OUTCOME/LEAKAGE DEFS (vs GDP/km2):")
    altl = {}
    g = lk.gdp_km2
    defs = {
        "ratio o/p (base)": lk.leakage,
        "o/(p+o)": lk.outcome / (lk.push + lk.outcome),
        "log ratio": np.log((lk.outcome + 1) / (lk.push + 1)),
        "raw outcome": lk.outcome,
    }
    for lbl, y in defs.items():
        r, p, n = rho(g, y); altl[lbl] = round(r, 3); print(f"    {lbl:<20} rho={r:+.3f} (p={p:.4f})")
    res["alt_leakage"] = altl

    # 8. actor-coding variants
    print("\n[8] ACTOR-CODING VARIANTS:")
    av = {}
    for lbl, actors, eth in (("BH=JAS+IS, +Eth", ("JAS", "IS"), True),
                             ("BH=JAS only", ("JAS",), True),
                             ("BH=IS only", ("IS",), True),
                             ("BH=JAS+IS, -Eth", ("JAS", "IS"), False)):
        d = rows_for(u, 250, 3, 2015, 2025, bh_actors=actors, include_eth=eth)
        d = d.dropna(subset=["leakage"]); d["g"] = d.country.map(lambda c: D[c]["gdp_km2"])
        r, p, n = rho(d.g, d.leakage); av[lbl] = dict(rho=round(r, 3), p=round(p, 4), n=n)
        print(f"    {lbl:<20} rho={r:+.3f} (p={p:.4f}, n={n})")
    res["actor_variants"] = av

    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
