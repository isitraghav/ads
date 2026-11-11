"""Twenty exploratory hypotheses beyond H1-H3.

Each hypothesis states a direction before the test. Cross-sectional county tests run on Jan-Oct 2025
AND Jan-Oct 2026 (same ten months, so seasonality cannot differ), and a pattern counts as SUPPORTED only
if it holds in both years. Calendar and store-behaviour tests use the full Jan 2025 - Oct 2026 panel.
These are exploratory (not pre-registered like H1-H3), so the p-values are reported next to a
Holm-adjusted p across all 20.

Run:  .venv/Scripts/python scripts/05_twenty_hypotheses.py
Out:  reports/tables/twenty_hypotheses.csv, reports/tables/twenty_hypotheses_detail.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abd_demand import config as C  # noqa: E402

RNG = np.random.default_rng(C.SEED)
P = C.PROCESSED
OUT = ROOT / "reports" / "tables"
RESULTS, DETAIL = [], {}


def log(*a):
    print(*a, flush=True)


def record(hid, theme, hypothesis, test, effect, p, verdict, detail=None):
    RESULTS.append(dict(id=hid, theme=theme, hypothesis=hypothesis, test=test, effect=effect,
                        p_value=p, verdict=verdict))
    DETAIL[hid] = detail or {}
    log(f"\n[{hid}] {verdict} | {effect} | p={p}")


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def ols(df, y, xs, w=None, cluster=None):
    d = df[[y] + xs + ([w] if w else []) + ([cluster] if cluster else [])].dropna()
    X = sm.add_constant(d[xs].astype(float))
    m = sm.WLS(d[y].astype(float), X, weights=d[w]) if w else sm.OLS(d[y].astype(float), X)
    if cluster:
        return m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    return m.fit(cov_type="HC3")


def one_sided(fit, x, sign):
    p2, b = fit.pvalues[x], fit.params[x]
    return p2 / 2 if np.sign(b) == sign else 1 - p2 / 2


def perm_p(df, y, x, ctrl, sign, n=2000):
    """Freedman-Lane permutation p-value (one-sided) for coefficient x."""
    d = df[[y, x] + ctrl].dropna().astype(float)
    Xr = sm.add_constant(d[ctrl]).values
    beta_r = np.linalg.lstsq(Xr, d[y].values, rcond=None)[0]
    fit_r = Xr @ beta_r
    res = d[y].values - fit_r
    Xf = sm.add_constant(d[[x] + ctrl]).values
    XtXi = np.linalg.inv(Xf.T @ Xf)

    def tstat(yy):
        b = XtXi @ Xf.T @ yy
        e = yy - Xf @ b
        s2 = e @ e / (len(yy) - Xf.shape[1])
        return b[1] / np.sqrt(s2 * XtXi[1, 1])

    t0 = tstat(d[y].values)
    ts = np.array([tstat(fit_r + RNG.permutation(res)) for _ in range(n)])
    return float((1 + np.sum(sign * ts >= sign * t0)) / (n + 1))


def county_two_years(cty, y, x, ctrl, sign, w="cty_population_21_plus"):
    """Run the same county regression in 2025 and 2026; return per-year stats and verdict."""
    out = {}
    for yr in (2025, 2026):
        d = cty[cty.yr == yr]
        f = ols(d, y, [x] + ctrl)
        fw = ols(d, y, [x] + ctrl, w=w)
        out[yr] = dict(coef=float(f.params[x]), p=float(one_sided(f, x, sign)),
                       p_perm=perm_p(d, y, x, ctrl, sign), coef_w=float(fw.params[x]),
                       p_w=float(one_sided(fw, x, sign)), r2=float(f.rsquared), n=int(f.nobs))
    ok = [out[yr]["p"] < 0.05 and out[yr]["p_perm"] < 0.05 for yr in (2025, 2026)]
    rev = [np.sign(out[yr]["coef"]) == -sign and out[yr]["p"] > 0.95 for yr in (2025, 2026)]
    if all(ok):
        v = "SUPPORTED (both years)" + ("" if all(out[yr]["p_w"] < 0.05 for yr in (2025, 2026)) else "; not under pop. weights")
    elif any(ok):
        v = "PARTIAL (one year only)"
    elif all(rev):
        v = "REJECTED - opposite sign, both years"
    else:
        v = "NOT SUPPORTED"
    return out, v, max(out[2025]["p"], out[2026]["p"])


def wsd(x, w):
    m = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - m) ** 2, weights=w)))


# ----------------------------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------------------------
log("loading SSOT ...")
cols = ["ordered_on", "store_no", "county_fips_code", "category_name", "family", "item_no", "vendor_name",
        "bottle_volume_ml", "state_bottle_cost", "sales_bottles", "liters", "dollars", "month",
        "price_per_liter", "store_type", "is_return", "mini_multipack_suspect"]
S = pd.read_parquet(C.SSOT, columns=cols)
for c in ["store_no", "county_fips_code", "category_name", "family", "item_no", "vendor_name", "month", "store_type"]:
    S[c] = S[c].astype(str)
S["yr"] = S.month.str[:4].astype(int)
S["mo"] = S.month.str[5:].astype(int)
S["in_p"] = S.mo <= 10                        # Jan-Oct, comparable across 2025 and 2026
S["date"] = S.ordered_on.dt.normalize()

prof = pd.read_parquet(C.COUNTY_PROFILE)
prof["county_fips_code"] = prof.county_fips_code.astype(str)
prof["log_pop"] = np.log(prof.cty_total_population)
prof["income_k"] = prof.cty_median_household_income / 1000
prof["log_density"] = np.log(prof.geo_pop_density)
CTRL = ["log_pop", "income_k", "cty_median_age", "cty_pct_bachelors_or_higher"]
stores = pd.read_parquet(C.STORE_DIM)
sm_ = pd.read_parquet(P / "store_month.parquet")
MONTHS = sorted(S.month.unique())
MIDX = {m: i for i, m in enumerate(MONTHS)}

# county x year table (Jan-Oct)
SP = S[S.in_p]
g = SP.groupby(["county_fips_code", "yr"])
cty = pd.DataFrame({"liters": g.liters.sum(), "dollars": g.dollars.sum(), "lines": g.size(),
                    "stores": g.store_no.nunique()}).reset_index()
fam = SP.pivot_table(index=["county_fips_code", "yr"], columns="family", values="liters", aggfunc="sum").fillna(0)
fam = fam.div(fam.sum(1), axis=0).add_suffix("_sh").reset_index()
cty = cty.merge(fam, on=["county_fips_code", "yr"]).merge(prof, on="county_fips_code")
cty["lpa"] = cty.liters / cty.cty_population_21_plus
cty["log_lpa"] = np.log(cty.lpa)
cty["log_stores_pc"] = np.log(cty.stores / cty.cty_population_21_plus * 1e4)
log(f"county-year table: {cty.shape}")

# ==============================================================================================
# THEME A - geography & culture
# ==============================================================================================
# H4 - Wisconsin "brandy belt" spills across the Mississippi
def seg_dist_miles(lat, lon, a, b):
    """distance (miles) from points to segment a-b, equirectangular approximation."""
    k = np.cos(np.radians(42.5))
    P_ = np.c_[lon * k * 69, lat * 69]
    A, B = np.array([a[1] * k * 69, a[0] * 69]), np.array([b[1] * k * 69, b[0] * 69])
    t = np.clip(((P_ - A) @ (B - A)) / ((B - A) @ (B - A)), 0, 1)
    return np.linalg.norm(P_ - (A + t[:, None] * (B - A)), axis=1)


# Iowa-Wisconsin border: the Mississippi from the MN line (~43.50 N, 91.22 W) to Dubuque/IL line (~42.50 N, 90.64 W)
cty["miles_to_wi"] = seg_dist_miles(cty.geo_lat.values, cty.geo_lon.values, (43.50, -91.22), (42.50, -90.64))
cty["near_wi"] = -np.log1p(cty.miles_to_wi)
res, v, p = county_two_years(cty, "brandy_sh", "near_wi", CTRL, +1)
c26 = cty[cty.yr == 2026]
wi = c26[c26.geo_borders_wi == 1].brandy_sh.mean()
rest = c26[(c26.geo_borders_wi == 0)].brandy_sh.mean()
rho = stats.spearmanr(c26.miles_to_wi, c26.brandy_sh)[0]
record("H4", "Geography & culture",
       "Wisconsin's brandy culture crosses the Mississippi: brandy share rises the closer an Iowa county is to Wisconsin.",
       "OLS of brandy share on -log(1+miles to WI border) + 4 controls, HC3 + Freedman-Lane permutation, 2025 & 2026",
       f"WI-border counties {wi:.1%} brandy vs {rest:.1%} elsewhere ({wi / rest:.1f}x); Spearman(distance, share) = {rho:.2f}",
       round(p, 4), v, res)

# H5 - spatial contagion of taste (Moran's I on demographic residuals)
adj = pd.read_csv(C.ADJACENCY, sep="|", dtype=str)
adj = adj[adj["County GEOID"].str.startswith("19") & adj["Neighbor GEOID"].str.startswith("19")
          & (adj["County GEOID"] != adj["Neighbor GEOID"])]


def morans_i(values, ids, n_perm=2000):
    idx = {c: i for i, c in enumerate(ids)}
    W = np.zeros((len(ids), len(ids)))
    for a, b in zip(adj["County GEOID"], adj["Neighbor GEOID"]):
        if a in idx and b in idx:
            W[idx[a], idx[b]] = 1
    W = W / W.sum(1, keepdims=True)
    z = values - values.mean()
    I0 = (z @ W @ z) / (z @ z)
    sims = np.array([(zz @ W @ zz) / (zz @ zz) for zz in (RNG.permutation(z) for _ in range(n_perm))])
    return float(I0), float((1 + np.sum(sims >= I0)) / (n_perm + 1))


mor = {}
for f_ in ["vodka", "whiskey", "liqueur", "rum", "agave", "rtd", "brandy", "gin"]:
    mor[f_] = {}
    for yr in (2025, 2026):
        d = cty[cty.yr == yr]
        fit = ols(d, f"{f_}_sh", CTRL + ["cty_pct_hispanic", "geo_borders_mo"])
        resid = fit.resid.values
        I0, pI = morans_i(resid, d.county_fips_code.values)
        mor[f_][yr] = dict(I=round(I0, 3), p=round(pI, 4))
sig_both = [f_ for f_ in mor if all(mor[f_][y]["p"] < 0.05 for y in (2025, 2026))]
record("H5", "Geography & culture",
       "Taste is contagious: after removing demographics, neighbouring counties still drink alike (spatially clustered residuals).",
       "Moran's I of OLS residuals (family share ~ 4 controls + %Hispanic + MO border), queen adjacency, 2,000 permutations, each year",
       f"{len(sig_both)}/8 families clustered in both years: "
       + ", ".join(f"{f_} I={mor[f_][2026]['I']}" for f_ in sig_both),
       max([mor[f_][2026]["p"] for f_ in sig_both] or [1.0]),
       f"SUPPORTED for {len(sig_both)}/8 families" if sig_both else "NOT SUPPORTED", mor)

# H6 - distance decay of Missouri leakage: where does the border stop mattering?
def hinge_scan(d, ks):
    y = d.log_lpa.values
    Xc = sm.add_constant(d[CTRL].astype(float)).values
    r2 = []
    for k in ks:
        X = np.c_[Xc, np.minimum(d.geo_miles_to_mo_border.values, k)]
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ b
        r2.append(1 - e @ e / ((y - y.mean()) @ (y - y.mean())))
    return np.array(r2)


ks = np.arange(10, 205, 5)
h6 = {}
for yr in (2025, 2026):
    d = cty[cty.yr == yr].reset_index(drop=True)
    r2 = hinge_scan(d, ks)
    kbest = int(ks[r2.argmax()])
    boots = []
    for _ in range(500):
        s = d.iloc[RNG.integers(0, len(d), len(d))].reset_index(drop=True)
        boots.append(ks[hinge_scan(s, ks).argmax()])
    d["hinge"] = np.minimum(d.geo_miles_to_mo_border, kbest)
    f = ols(d, "log_lpa", ["hinge"] + CTRL)
    lin = ols(d.assign(m=d.geo_miles_to_mo_border), "log_lpa", ["m"] + CTRL)
    h6[yr] = dict(breakpoint_miles=kbest, boot_ci=[int(np.percentile(boots, 5)), int(np.percentile(boots, 95))],
                  pct_per_10mi=float(np.expm1(10 * f.params["hinge"])), p=float(one_sided(f, "hinge", +1)),
                  r2_hinge=float(f.rsquared), r2_linear=float(lin.rsquared))
v = ("SUPPORTED (both years)" if all(h6[y]["p"] < 0.05 and h6[y]["breakpoint_miles"] < 150 for y in h6)
     else "PARTIAL" if any(h6[y]["p"] < 0.05 for y in h6) else "NOT SUPPORTED")
record("H6", "Geography & culture",
       "Border leakage has a travel-cost horizon: liters per adult rise with distance from Missouri, then flatten beyond a measurable break.",
       "Hinge regression log(L/adult) ~ min(miles to MO, k) + controls; k grid-searched 10-200 mi; bootstrap (500) for k",
       f"break ~{h6[2026]['breakpoint_miles']} mi in 2026 (90% CI {h6[2026]['boot_ci']}), {h6[2025]['breakpoint_miles']} mi in 2025; "
       f"+{h6[2026]['pct_per_10mi']:.1%} liters/adult per 10 mi inside the break",
       round(max(h6[y]["p"] for y in h6), 4), v, h6)

# H7 - Sioux County: the cultural outlier demographics cannot explain
h7 = {}
for yr in (2025, 2026):
    d = cty[cty.yr == yr].reset_index(drop=True)
    X = sm.add_constant(d[CTRL + ["geo_borders_mo", "log_stores_pc"]].astype(float))
    f = sm.OLS(d.log_lpa, X).fit()
    sr = f.get_influence().resid_studentized_external
    d["sr"] = sr
    d["rank"] = d.sr.rank()
    srow = d[d.county_name == "Sioux"].iloc[0]
    h7[yr] = dict(sioux_studentized_resid=round(float(srow.sr), 2), sioux_rank_of_99=int(srow["rank"]),
                  sioux_gap_pct=float(np.expm1(f.resid[d.county_name == "Sioux"].iloc[0])),
                  bottom5=d.nsmallest(5, "sr")[["county_name", "sr"]].round(2).values.tolist())
p7 = max(2 * stats.t.sf(abs(h7[y]["sioux_studentized_resid"]), 99 - 8) for y in h7)
v = ("SUPPORTED (bottom-3 in both years)" if all(h7[y]["sioux_rank_of_99"] <= 3 for y in h7)
     else "PARTIAL" if all(h7[y]["sioux_rank_of_99"] <= 10 for y in h7) else "NOT SUPPORTED")
record("H7", "Geography & culture",
       "Religion leaves a fingerprint: Dutch-Reformed Sioux County buys far less than its demographics, store count and border position predict.",
       "Externally studentised residual of Sioux County in log(L/adult) ~ controls + MO border + log(stores/10k adults), both years",
       f"Sioux buys {h7[2026]['sioux_gap_pct']:+.0%} vs prediction (2026), rank {h7[2026]['sioux_rank_of_99']}/99 "
       f"(2025 rank {h7[2025]['sioux_rank_of_99']}); studentised resid {h7[2026]['sioux_studentized_resid']}",
       round(p7, 4), v, h7)

# ==============================================================================================
# THEME B - the cultural calendar
# ==============================================================================================
Sp = S[S.liters > 0]
st_m = Sp.groupby(["store_no", "month"]).liters.sum()

# H8 - St Patrick's Day: Irish whiskey stocked in Feb-Mar (Scotch as placebo)
def store_window_lift(mask_cat, months_in, months_out):
    cat = Sp[mask_cat].groupby(["store_no", "month"]).liters.sum()
    sh = (cat / st_m).fillna(0).rename("sh").reset_index()
    a = sh[sh.month.isin(months_in)].groupby("store_no").sh.mean()
    b = sh[sh.month.isin(months_out)].groupby("store_no").sh.mean()
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    j = j[(j.a > 0) | (j.b > 0)]
    return j


h8 = {}
for yr in (2025, 2026):
    inm = [f"{yr}-02", f"{yr}-03"]
    out_ = [f"{yr}-{m:02d}" for m in range(4, 11)]
    ir = store_window_lift(Sp.category_name == "IRISH WHISKIES", inm, out_)
    sc = store_window_lift(Sp.category_name.isin(["SCOTCH WHISKIES", "SINGLE MALT SCOTCH"]), inm, out_)
    h8[yr] = dict(irish_lift=float(ir.a.mean() / ir.b.mean() - 1), irish_p=float(stats.wilcoxon(ir.a, ir.b, alternative="greater").pvalue),
                  scotch_lift=float(sc.a.mean() / sc.b.mean() - 1), scotch_p=float(stats.wilcoxon(sc.a, sc.b, alternative="greater").pvalue),
                  n_stores=len(ir))
mar = Sp[Sp.category_name == "IRISH WHISKIES"].groupby("month").liters.sum() / Sp.groupby("month").liters.sum()
v = ("SUPPORTED (both years; placebo flat)" if all(h8[y]["irish_p"] < 0.05 and h8[y]["irish_lift"] > 2 * max(h8[y]["scotch_lift"], 0.01) for y in h8)
     else "PARTIAL" if any(h8[y]["irish_p"] < 0.05 for y in h8) else "NOT SUPPORTED")
record("H8", "Cultural calendar",
       "St Patrick's Day is visible in wholesale data: Irish whiskey's share jumps in Feb-Mar orders, while Scotch (same shelf, no holiday) does not.",
       "Store-level paired Wilcoxon: Irish share Feb-Mar vs Apr-Oct, each year; Scotch as placebo",
       f"Irish share +{h8[2026]['irish_lift']:.0%} (2026), +{h8[2025]['irish_lift']:.0%} (2025) vs Scotch {h8[2026]['scotch_lift']:+.0%}; "
       f"peak month share {mar.max():.2%} ({mar.idxmax()}) vs median {mar.median():.2%}",
       float(max(h8[y]["irish_p"] for y in h8)), v, h8)

# H9 - Cinco de Mayo: is the agave bump a Hispanic or an Anglo celebration?
ag_daily = Sp.assign(ag=Sp.liters.where(Sp.family == "agave", 0))
h9 = {}
for yr in (2025, 2026):
    pre = ag_daily[(ag_daily.date >= f"{yr}-04-14") & (ag_daily.date <= f"{yr}-05-04")]
    flank = ag_daily[((ag_daily.date >= f"{yr}-03-17") & (ag_daily.date <= f"{yr}-04-06"))
                     | ((ag_daily.date >= f"{yr}-05-12") & (ag_daily.date <= f"{yr}-06-01"))]
    a = pre.groupby("county_fips_code")[["ag", "liters"]].sum()
    b = flank.groupby("county_fips_code")[["ag", "liters"]].sum()
    u = pd.DataFrame({"uplift": np.log((a.ag / a.liters) / (b.ag / b.liters))}).dropna()
    u = u[np.isfinite(u.uplift)].reset_index().merge(prof, on="county_fips_code")
    f = ols(u, "uplift", ["cty_pct_hispanic"] + CTRL)
    state_up = (pre.ag.sum() / pre.liters.sum()) / (flank.ag.sum() / flank.liters.sum()) - 1
    u["w"] = u.county_fips_code.map(a.liters)
    fb = sm.WLS(u.uplift, np.ones(len(u)), weights=u.w).fit(cov_type="HC3")   # volume-weighted mean bump
    h9[yr] = dict(state_uplift=float(state_up), mean_cty_uplift_p=float(fb.pvalues.iloc[0] / 2 if fb.params.iloc[0] > 0 else 1),
                  coef_per_10pt_hisp=float(10 * f.params["cty_pct_hispanic"]), p_hisp_pos=float(one_sided(f, "cty_pct_hispanic", +1)),
                  p_hisp_neg=float(one_sided(f, "cty_pct_hispanic", -1)))
bump = all(h9[y]["mean_cty_uplift_p"] < 0.05 for y in h9)
hisp = all(h9[y]["p_hisp_pos"] < 0.05 for y in h9)
neg = all(h9[y]["p_hisp_neg"] < 0.05 for y in h9)
v = ("SUPPORTED" if bump and hisp else
     f"REJECTED - bump exists but is {'smaller' if neg else 'no larger'} in Hispanic counties" if bump else "NOT SUPPORTED - no bump")
record("H9", "Cultural calendar",
       "The Cinco de Mayo agave bump is bigger in counties with more Hispanic residents.",
       "County log-uplift of agave share (Apr 14-May 4 vs flanking 3-week windows) regressed on %Hispanic + controls, each year",
       f"statewide agave-share bump {h9[2026]['state_uplift']:+.1%} (2026), {h9[2025]['state_uplift']:+.1%} (2025); "
       f"Hispanic slope {h9[2026]['coef_per_10pt_hisp']:+.3f} log-pts per +10 pts",
       float(max(h9[y]["p_hisp_pos"] for y in h9)), v, h9)

# H10 - the college-town calendar (student exodus in summer)
cm = Sp.groupby(["county_fips_code", "month"]).liters.sum().unstack()
h10 = {}
COLLEGE = {"19169": "Story (ISU)", "19103": "Johnson (U. Iowa)"}
for yr in (2025, 2026):
    r = np.log((cm[f"{yr}-06"] + cm[f"{yr}-07"]) / (cm[f"{yr}-09"] + cm[f"{yr}-10"]))
    rel = r - r.median()
    rk = r.rank()
    h10[yr] = {COLLEGE[c]: dict(summer_vs_fall=float(np.expm1(rel[c])), rank_of_99=int(rk[c])) for c in COLLEGE}
    # exact p: chance two specific counties both rank this low under random ranks
    worst = max(rk[c] for c in COLLEGE)
    h10[yr]["p_both_this_low"] = float((worst / 99) ** 2)
p10 = max(h10[y]["p_both_this_low"] for y in (2025, 2026))
v = "SUPPORTED (both years)" if p10 < 0.05 else "NOT SUPPORTED"
record("H10", "Cultural calendar",
       "College towns empty out: Ames and Iowa City order far less in Jun-Jul relative to Sep-Oct than any normal county (students leave, football returns).",
       "County ratio log[(Jun+Jul)/(Sep+Oct) liters]; rank of Story & Johnson among 99; P(both ranks this low by chance)",
       f"2026: Story {h10[2026]['Story (ISU)']['summer_vs_fall']:+.0%}, Johnson {h10[2026]['Johnson (U. Iowa)']['summer_vs_fall']:+.0%} vs median county; "
       f"ranks {h10[2026]['Story (ISU)']['rank_of_99']} & {h10[2026]['Johnson (U. Iowa)']['rank_of_99']}/99",
       round(p10, 4), v, h10)

# H11 - the gift economy: December premiumisation is a rich-county phenomenon
Spr = Sp[Sp.price_per_liter.notna()].copy()
cut = float((P / "premium_cut.txt").read_text())
Spr["prem_l"] = Spr.liters.where(Spr.price_per_liter >= cut, 0)
base = Spr[(Spr.yr == 2025) & Spr.mo.between(1, 10)].groupby("county_fips_code")[["prem_l", "liters"]].sum()
hol = Spr[(Spr.yr == 2025) & Spr.mo.isin([11, 12])].groupby("county_fips_code")[["prem_l", "liters"]].sum()
u = pd.DataFrame({"d_prem": 100 * (hol.prem_l / hol.liters - base.prem_l / base.liters)}).reset_index().merge(prof, on="county_fips_code")
f = ols(u, "d_prem", ["income_k", "log_pop", "cty_median_age", "cty_pct_bachelors_or_higher"])
pp = perm_p(u, "d_prem", "income_k", ["log_pop", "cty_median_age", "cty_pct_bachelors_or_higher"], +1)
famhol = (Sp[(Sp.yr == 2025) & Sp.mo.isin([11, 12])].groupby("family").liters.sum() / Sp[(Sp.yr == 2025) & Sp.mo.isin([11, 12])].liters.sum()) / \
         (Sp[(Sp.yr == 2025) & Sp.mo.between(1, 10)].groupby("family").liters.sum() / Sp[(Sp.yr == 2025) & Sp.mo.between(1, 10)].liters.sum()) - 1
p11 = float(one_sided(f, "income_k", +1))
v = "SUPPORTED" if p11 < 0.05 and pp < 0.05 else ("REJECTED - opposite sign" if f.params["income_k"] < 0 and p11 > 0.95 else "NOT SUPPORTED")
record("H11", "Cultural calendar",
       "The holiday gift economy is a rich-county thing: the Nov-Dec jump in premium share grows with county income.",
       "County change in premium share (Nov-Dec 2025 minus Jan-Oct 2025, pp) ~ income + controls, HC3 + permutation (one holiday season available)",
       f"statewide premium share {base.prem_l.sum() / base.liters.sum():.1%} -> {hol.prem_l.sum() / hol.liters.sum():.1%}; "
       f"+{f.params['income_k'] * 10:.2f} pp per +$10k income; holiday mix: brandy {famhol['brandy']:+.0%}, liqueur {famhol['liqueur']:+.0%}, rtd {famhol['rtd']:+.0%}",
       round(p11, 4), v, dict(coef=float(f.params["income_k"]), p=p11, p_perm=pp, family_holiday_lift=famhol.round(3).to_dict()))

# H12 - seasonal palate rotation
summer = ["2025-06", "2025-07", "2025-08", "2026-06", "2026-07", "2026-08"]
winter = ["2025-11", "2025-12", "2026-01"]
sf = Sp.pivot_table(index=["store_no", "month"], columns="family", values="liters", aggfunc="sum").fillna(0)
sf = sf.div(sf.sum(1), axis=0).reset_index()
a = sf[sf.month.isin(summer)].groupby("store_no").mean(numeric_only=True)
b = sf[sf.month.isin(winter)].groupby("store_no").mean(numeric_only=True)
idx = a.index.intersection(b.index)
pred = {"rtd": +1, "rum": +1, "agave": +1, "vodka": +1, "liqueur": -1, "brandy": -1, "whiskey": -1, "gin": +1}
h12 = {}
for f_, sgn in pred.items():
    lift = a.loc[idx, f_].mean() / b.loc[idx, f_].mean() - 1
    pw = stats.wilcoxon(a.loc[idx, f_], b.loc[idx, f_], alternative="greater" if sgn > 0 else "less").pvalue
    h12[f_] = dict(summer_vs_winter=float(lift), predicted=("summer" if sgn > 0 else "winter"), p=float(pw))
hits = [f_ for f_ in h12 if h12[f_]["p"] < 0.05]
record("H12", "Cultural calendar",
       "Iowa rotates its palate with the seasons: RTD, rum, agave, gin and vodka gain share in summer; whiskey, liqueur and brandy in winter.",
       "Store-level paired Wilcoxon of family shares, summer (Jun-Aug 25/26) vs winter (Nov 25-Jan 26), direction pre-stated per family",
       ", ".join(f"{f_} {h12[f_]['summer_vs_winter']:+.0%}" for f_ in h12),
       float(max(h12[f_]["p"] for f_ in hits)) if hits else 1.0,
       f"SUPPORTED for {len(hits)}/8 predicted directions" + ("" if len(hits) == 8 else f" (fails: {', '.join(sorted(set(h12) - set(hits)))})"), h12)

# ==============================================================================================
# THEME C - prices & economics
# ==============================================================================================
# H13 - forward buying: state price cuts shift orders in time more than they create demand
# DATA QUALITY: state_bottle_cost is blank for Jan-Jun 2025 and truncated to whole dollars for Nov 2025-Jan 2026
# (every row is an integer there; 6.74 -> 6.00). Only Jul-Oct 2025 and Feb-Oct 2026 carry usable prices.
COST_OK = [m for m in MONTHS if ("2025-07" <= m <= "2025-10") or ("2026-02" <= m <= "2026-10")]
im = S[~S.is_return.astype(bool)].groupby(["item_no", "month"]).agg(b=("sales_bottles", "sum"), cost=("state_bottle_cost", "median")).reset_index()
im = im[im.month.isin(COST_OK)]
nmon = im.groupby("item_no").month.nunique()
reg_items = nmon[nmon >= len(COST_OK) - 1].index
im = im[im.item_no.isin(reg_items)]
grid = pd.MultiIndex.from_product([reg_items, COST_OK], names=["item_no", "month"])
im = im.set_index(["item_no", "month"]).reindex(grid).reset_index()
im["b"] = im.b.fillna(0)
im["cost"] = im.groupby("item_no").cost.transform(lambda s: s.ffill().bfill())
modal = im.groupby("item_no").cost.agg(lambda s: s.round(2).mode().iloc[0])
im["rel"] = im.cost / im.item_no.map(modal)
im["promo"] = (im.rel <= 0.95).astype(int)
im["t"] = im.month.map(MIDX)
im = im.sort_values(["item_no", "t"])
# leads/lags only across consecutive calendar months (the Nov-Jan gap breaks the chain)
nxt = im.groupby("item_no").t.shift(-1) - im.t
prv = im.t - im.groupby("item_no").t.shift(1)
im["promo_lag"] = im.groupby("item_no").promo.shift(1).where(prv == 1)
im["promo_lead"] = im.groupby("item_no").promo.shift(-1).where(nxt == 1)
im = im.dropna(subset=["promo_lag", "promo_lead"])
im["after"] = ((im.promo_lag == 1) & (im.promo == 0)).astype(int)    # first month back at full price
im["before"] = ((im.promo_lead == 1) & (im.promo == 0)).astype(int)  # month before a cut (stores wait)
im["y"] = np.log1p(im.b)


def twfe(df, y, xs, a="item_no", b="month", iters=50):
    d = df[[y, *xs, a, b]].copy()
    cols_ = [y, *xs]
    for _ in range(iters):
        d[cols_] = d[cols_] - d.groupby(a)[cols_].transform("mean")
        d[cols_] = d[cols_] - d.groupby(b)[cols_].transform("mean")
    return sm.OLS(d[y], d[xs]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[a])[0]})


f = twfe(im, "y", ["promo", "after", "before"])
cut_size = 1 - im.loc[im.promo == 1, "rel"].median()
lift, hang, pre_ = (float(np.expm1(f.params[k])) for k in ["promo", "after", "before"])
p13 = float(f.pvalues["after"] / 2 if f.params["after"] < 0 else 1 - f.pvalues["after"] / 2)
p_after_pos = float(f.pvalues["after"] / 2 if f.params["after"] > 0 else 1 - f.pvalues["after"] / 2)
v = ("SUPPORTED - promo spike followed by a hangover" if p13 < 0.05 and lift > 0 else
     f"REJECTED - timing is reversed: the surge comes the month AFTER the cut (p={p_after_pos:.1e}), no hangover"
     if p_after_pos < 0.05 else "NOT SUPPORTED - no post-promo dip")
record("H13", "Prices & economics",
       "State price cuts mostly move orders in time: retailers stock up during a temporary cost cut and then under-order once it ends.",
       f"Item x month two-way fixed effects, {len(reg_items):,} regularly stocked items, clean-price months only (Jul-Oct 25, Feb-Oct 26), SEs clustered by item; promo = cost >=5% below item's modal cost",
       f"orders {lift:+.0%} in promo months (median cut {cut_size:.0%}, implied arc elasticity {np.log1p(lift) / np.log(1 - cut_size):.1f}); "
       f"{hang:+.0%} the month after; {pre_:+.0%} the month before",
       round(p13, 4), v, dict(params=f.params.round(4).to_dict(), p=f.pvalues.round(5).to_dict(), n=int(f.nobs),
                              promo_item_months_by_month=im.groupby("month").promo.sum().to_dict()))

# H14 - tariff-era price pass-through: imports got dearer than domestic
def origin(cat):
    c = cat.upper()
    if "CANADIAN" in c: return "Canada"
    if "TEQUILA" in c or "MEZCAL" in c: return "Mexico"
    if "SCOTCH" in c or "IRISH" in c: return "UK/Ireland"
    if c.startswith("IMPORTED"): return "Other import"
    if c.startswith("AMERICAN") or any(k in c for k in ["BOURBON", "TENNESSEE", "RYE", "CORN WHISK", "BLENDED WHISK"]): return "USA"
    return None


jo = S[S.mo.between(7, 10) & (S.liters > 0) & S.state_bottle_cost.notna()]   # Jul-Oct: the only clean-price window in both years
ic = jo.groupby(["item_no", "yr"]).agg(cost=("state_bottle_cost", lambda s: s.round(2).mode().iloc[0]),
                                       L=("liters", "sum"), cat=("category_name", "first")).reset_index()
ic = ic.pivot(index="item_no", columns="yr").dropna()
pc = pd.DataFrame({"dlog": np.log(ic["cost"][2026] / ic["cost"][2025]), "w": ic["L"][2025],
                   "origin": ic["cat"][2025].map(origin)}).dropna()
pc = pc[pc.dlog.abs() < 0.5]
grp = pc.groupby("origin").apply(lambda d: pd.Series({"wmean_pct": np.expm1(np.average(d.dlog, weights=d.w)), "items": len(d),
                                                      "share_items_up": (d.dlog > 0.001).mean()}))
imp = pc[pc.origin != "USA"]
us = pc[pc.origin == "USA"]
p14 = float(stats.mannwhitneyu(imp.dlog, us.dlog, alternative="greater").pvalue)
diff = grp.loc[grp.index != "USA", "wmean_pct"].mean() - grp.loc["USA", "wmean_pct"]
v = "SUPPORTED" if p14 < 0.05 and diff > 0 else ("REJECTED - domestic rose as much or more" if diff <= 0 else "NOT SUPPORTED")
record("H14", "Prices & economics",
       "The 2025-26 tariff era shows up on the shelf: same-item wholesale costs rose more for imported spirits than for American ones.",
       "Same-item modal state cost Jul-Oct 2026 vs Jul-Oct 2025 by origin (from category); Mann-Whitney imports vs USA (cost is blank Jan-Jun 2025)",
       "; ".join(f"{o} {r.wmean_pct:+.1%} (n={int(r['items'])})" for o, r in grp.iterrows()),
       round(p14, 4), v, grp.round(4).to_dict())

# H15 - inequality builds a barbell
pr = Spr[Spr.in_p]
pr = pr.assign(lp=np.log(pr.price_per_liter))
disp = pr.groupby(["county_fips_code", "yr"]).apply(lambda d: pd.Series({"price_sd": wsd(d.lp.values, d.liters.values),
                                                                       "price_mean": np.average(d.lp, weights=d.liters)})).reset_index()
cty = cty.merge(disp, on=["county_fips_code", "yr"])
cty["gini100"] = 100 * cty.cty_gini_index
res15, v15, p15 = county_two_years(cty, "price_sd", "gini100", CTRL, +1)
res15m, v15m, _ = county_two_years(cty, "price_mean", "gini100", CTRL, +1)
record("H15", "Prices & economics",
       "Inequality builds a barbell: high-Gini counties buy a wider spread of price tiers (more bottom shelf AND more top shelf), not a higher average.",
       "County liter-weighted SD of log price/L ~ Gini + controls (and mean log price as the contrast), both years",
       f"SD: {res15[2026]['coef']:+.4f} per Gini point (2026), {res15[2025]['coef']:+.4f} (2025); mean price: {res15m[2026]['coef']:+.4f} ({v15m})",
       round(p15, 4), v15.replace("SUPPORTED", "PARTIAL - spread widens, but the average price rises too", 1)
       if v15m.startswith("SUPPORTED") else v15, dict(sd=res15, mean=res15m))

# H16 - the mini-bottle index tracks poverty
S50 = S[S.in_p & (S.dollars > 0)]
mini = S50.assign(md=S50.dollars.where(S50.bottle_volume_ml <= 50, 0))
mc = mini.groupby(["county_fips_code", "yr"])[["md", "dollars"]].sum()
cty = cty.merge((mc.md / mc.dollars).rename("mini_dsh").reset_index(), on=["county_fips_code", "yr"])
conv = S50.assign(cv=S50.dollars.where(S50.store_type.str.startswith("convenience"), 0)).groupby(["county_fips_code", "yr"])[["cv", "dollars"]].sum()
cty = cty.merge((conv.cv / conv.dollars).rename("conv_sh").reset_index(), on=["county_fips_code", "yr"])
ctrl16 = ["log_pop", "cty_median_age", "cty_pct_bachelors_or_higher", "conv_sh"]
res16, v16, p16 = county_two_years(cty, "mini_dsh", "cty_poverty_rate", ctrl16, +1)
# within store type: store-level mini share on county poverty with store-type dummies, clustered by county
ss = mini[mini.yr == 2026].groupby(["store_no", "county_fips_code", "store_type"])[["md", "dollars"]].sum().reset_index()
ss = ss[ss.dollars > 5000].merge(prof, on="county_fips_code")
ss["mini_sh"] = ss.md / ss.dollars
dm = pd.get_dummies(ss.store_type, prefix="st", drop_first=True).astype(float)
ss = pd.concat([ss, dm], axis=1)
fs = ols(ss, "mini_sh", ["cty_poverty_rate", "log_pop", "cty_median_age", "cty_pct_bachelors_or_higher"] + list(dm.columns), cluster="county_fips_code")
record("H16", "Prices & economics",
       "The 50 ml 'nip' is a poverty index: poorer counties spend a larger share on mini bottles, even comparing stores of the same type.",
       "County dollar share of <=50 ml bottles ~ poverty rate + controls + convenience-store share (both years); store-level with store-type FE, clustered by county",
       f"+{res16[2026]['coef'] * 10:.2%} mini share per +10 pts poverty (county, 2026); store-level within type +{fs.params['cty_poverty_rate'] * 10:.2%} (p={one_sided(fs, 'cty_poverty_rate', +1):.3f})",
       round(p16, 4), ("PARTIAL - holds within store type (store level), not across counties" if v16 == "NOT SUPPORTED" and one_sided(fs, "cty_poverty_rate", +1) < 0.05 else v16), dict(county=res16, store_level=dict(coef=float(fs.params["cty_poverty_rate"]), p=float(one_sided(fs, "cty_poverty_rate", +1)), n=int(fs.nobs))))

# H17 - outlet density: supply creates (some) demand, consistent with H3's 60% diversion
res17, v17, p17 = county_two_years(cty, "log_lpa", "log_stores_pc", CTRL + ["geo_borders_mo"], +1)
el = res17[2026]["coef"]
record("H17", "Prices & economics",
       "More outlets, more drinking - but only partly: liters per adult rise with store density with an elasticity well below 1, matching H3's finding that ~60% of a new store's volume is stolen.",
       "County log(L/adult) ~ log(stores per 10k adults) + controls + MO border, both years; compare elasticity to 1 - 0.60 = 0.40 implied by H3",
       f"elasticity {el:.2f} (2026), {res17[2025]['coef']:.2f} (2025); H3-implied 0.40",
       round(p17, 4), v17, res17)

# ==============================================================================================
# THEME D - market structure & store behaviour
# ==============================================================================================
# H18 - the long tail lives in cities (rarefied SKU richness)
lines26 = S[(S.yr == 2026) & (S.liters > 0)][["county_fips_code", "item_no", "vendor_name", "liters"]]
nmin = int(lines26.groupby("county_fips_code").size().min())
nsamp = min(3000, nmin)
rich = {}
for c, d in lines26.groupby("county_fips_code"):
    items = d.item_no.values
    rich[c] = np.mean([len(np.unique(RNG.choice(items, nsamp, replace=False))) for _ in range(20)])
vh = lines26.groupby(["county_fips_code", "vendor_name"]).liters.sum()
vh = (vh / vh.groupby(level=0).transform("sum")) ** 2
r18 = pd.DataFrame({"rich": pd.Series(rich), "vendor_hhi": vh.groupby(level=0).sum()}).reset_index(names="county_fips_code").merge(prof, on="county_fips_code")
f18 = ols(r18, "rich", ["log_density", "income_k", "cty_median_age", "cty_pct_bachelors_or_higher"])
f18h = ols(r18, "vendor_hhi", ["log_density", "income_k", "cty_median_age", "cty_pct_bachelors_or_higher"])
p18 = float(one_sided(f18, "log_density", +1))
pp18 = perm_p(r18, "rich", "log_density", ["income_k", "cty_median_age", "cty_pct_bachelors_or_higher"], +1)
q = r18.sort_values("geo_pop_density")
record("H18", "Market structure",
       "The long tail lives in cities: for the same number of order lines, dense counties buy many more distinct products and are less dominated by a few suppliers.",
       f"Rarefaction: distinct SKUs in {nsamp:,} random 2026 order lines per county (20 draws) ~ log density + controls; vendor HHI as second outcome",
       f"+{f18.params['log_density']:.0f} distinct SKUs per log-unit of density; densest decile {q.tail(10).rich.mean():.0f} vs sparsest {q.head(10).rich.mean():.0f} SKUs; "
       f"vendor HHI slope {f18h.params['log_density']:+.4f}",
       round(max(p18, pp18), 4), "SUPPORTED" if p18 < 0.05 and pp18 < 0.05 else "NOT SUPPORTED",
       dict(coef=float(f18.params["log_density"]), p=p18, p_perm=pp18, hhi_coef=float(f18h.params["log_density"]),
            hhi_p=float(one_sided(f18h, "log_density", -1))))

# H19 - Zipf's law for spirits
il = S[(S.yr == 2026) & (S.liters > 0)].groupby("item_no").liters.sum().sort_values(ascending=False)
rk = np.arange(1, len(il) + 1)
sel = (rk >= 10) & (rk <= 1000)
fz = sm.OLS(np.log(il.values[sel]), sm.add_constant(np.log(rk[sel]))).fit()
top1 = il.head(int(np.ceil(0.01 * len(il)))).sum() / il.sum()
top20 = il.head(int(0.2 * len(il))).sum() / il.sum()
il25 = S[(S.yr == 2025) & (S.liters > 0)].groupby("item_no").liters.sum()
top100 = il.head(100).index
stab = stats.spearmanr(il.loc[top100], il25.reindex(top100).fillna(0))[0]
slope = float(fz.params[1])
ci = list(fz.conf_int()[1])
record("H19", "Market structure",
       "Spirits obey Zipf's law: product volume falls as 1/rank, so a tiny head of SKUs carries the state and the head is stable year to year.",
       "OLS log(liters) ~ log(rank), ranks 10-1000, 2026; Pareto shares; Spearman of top-100 volumes 2025 vs 2026",
       f"slope {slope:.2f} (95% CI {ci[0]:.2f} to {ci[1]:.2f}; Zipf = -1); top 1% of {len(il):,} SKUs = {top1:.0%} of liters, top 20% = {top20:.0%}; top-100 stability rho = {stab:.2f}",
       None, "SUPPORTED (Zipf-like)" if -1.3 <= slope <= -0.7 else f"NOT SUPPORTED - slope {slope:.2f} is {'steeper' if slope < -1.3 else 'flatter'} than Zipf",
       dict(slope=slope, ci=ci, top1=float(top1), top20=float(top20), stability=float(stab), n_items=len(il)))

# H20 - new-store honeymoon: opening stock-up spike
smx = sm_.copy()
smx["first_t"] = smx.first_month.map(MIDX)
smx["k"] = smx.t - smx.first_t
new = stores[(stores.entry_type == "new_store") & stores.first_month.between("2025-03", "2026-04")].store_no
nm = smx[smx.store_no.isin(new) & smx.k.between(0, 6)].pivot_table(index="store_no", columns="k", values="liters", aggfunc="sum").fillna(0).clip(lower=0)  # returns can make a month negative
nm = nm[(nm[[3, 4, 5]].mean(1) > 0)]
ss_ = nm[[3, 4, 5]].mean(1)
ratio = nm.div(ss_, axis=0)
p20 = float(stats.wilcoxon(np.log((nm[0] + 1) / (ss_ + 1)), alternative="greater").pvalue)   # opening order vs steady state
p20_1 = float(stats.wilcoxon(np.log((nm[1] + 1) / (ss_ + 1)), alternative="less").pvalue)     # month after opening dips
p20_0 = p20
record("H20", "Store behaviour",
       "New stores have a honeymoon: the opening order (empty shelves being filled) overshoots the store's steady-state monthly level, so a forecast built on it overshoots.",
       "Liters in months 0-6 after first order / mean of months 3-5, for genuinely new stores (first order Mar 2025-Apr 2026); one-sided Wilcoxon",
       "median ratio by month since opening: " + ", ".join(f"m{k}={ratio[k].median():.2f}" for k in range(0, 7)) + f" (n={len(nm)})",
       round(p20, 6), ("SUPPORTED - and followed by a dip below steady state in month 1" if p20_1 < 0.05 else "SUPPORTED") if p20 < 0.05 else "NOT SUPPORTED - entrants ramp up rather than overshoot",
       dict(median_ratio={int(k): float(ratio[k].median()) for k in ratio}, p_month0_gt=p20, p_month1_lt=p20_1, n=len(nm)))

# H21 - the death spiral: exiting stores shrink volume and range months before they stop
sk = S[S.liters > 0].groupby(["store_no", "month"]).item_no.nunique().rename("skus").reset_index()
smx = smx.merge(sk, on=["store_no", "month"], how="left").fillna({"skus": 0})
wide_l = smx.pivot_table(index="store_no", columns="t", values="liters", aggfunc="sum").reindex(columns=range(len(MONTHS))).fillna(0)
wide_s = smx.pivot_table(index="store_no", columns="t", values="skus", aggfunc="sum").reindex(columns=range(len(MONTHS))).fillna(0)
stores["last_t"] = stores.last_month.map(MIDX)
stores["first_t"] = stores.first_month.map(MIDX)
ex = stores[(stores.last_t <= MIDX["2026-07"]) & (stores.last_t - stores.first_t >= 9)]
state_l = wide_l.sum()
rows = []
for _, r in ex.iterrows():
    L = r.last_t
    late, early = list(range(L - 3, L)), list(range(L - 9, L - 3))
    if wide_l.loc[r.store_no, early].mean() <= 0:
        continue
    vol = (wide_l.loc[r.store_no, late].mean() / wide_l.loc[r.store_no, early].mean()) / (state_l[late].mean() / state_l[early].mean())
    sku = wide_s.loc[r.store_no, late].mean() / max(wide_s.loc[r.store_no, early].mean(), 1e-9)
    rows.append(dict(store_no=r.store_no, vol_ratio=vol, sku_ratio=sku))
dsp = pd.DataFrame(rows)
p21 = float(stats.wilcoxon(np.log(dsp.vol_ratio.clip(lower=1e-3)), alternative="less").pvalue)
# early-warning AUC: feature = log(last 3 / previous 6 months, seasonally adjusted) for every store-month; label = stops within 3 months
feat, lab = [], []
for t in range(9, MIDX["2026-07"] + 1):
    late, early = list(range(t - 3, t)), list(range(t - 9, t - 3))
    act = wide_l.index[(wide_l[early].mean(1) > 0) & (wide_l[late].sum(1) > 0)]
    adj_ = state_l[late].mean() / state_l[early].mean()
    x = np.log((wide_l.loc[act, late].mean(1) + 1) / (wide_l.loc[act, early].mean(1) + 1) / adj_)
    lt = stores.set_index("store_no").last_t.reindex(act)
    feat += list(-x.values)
    lab += list(((lt >= t) & (lt <= t + 2) & (lt <= MIDX["2026-07"])).astype(int).values)
auc = roc_auc_score(lab, feat)
record("H21", "Store behaviour",
       "Stores die slowly, not suddenly: in the 3 months before their last order, exiting stores' volume and product range shrink well below their own seasonal norm - an early-warning signal.",
       f"{len(dsp)} stores whose last order was <= Jul 2026: season-adjusted volume (last 3 full months vs previous 6), SKU count ratio; AUC of the decline as a 3-month exit predictor",
       f"median volume {dsp.vol_ratio.median() - 1:+.0%}, median SKU range {dsp.sku_ratio.median() - 1:+.0%} before exit; early-warning AUC = {auc:.2f}",
       round(p21, 4), "SUPPORTED" if p21 < 0.05 and auc > 0.7 else "MOSTLY REJECTED - the decline is small and useless as a warning: stores die suddenly" if p21 < 0.05 and auc < 0.6 else "PARTIAL" if p21 < 0.05 else "NOT SUPPORTED",
       dict(n=len(dsp), vol_median=float(dsp.vol_ratio.median()), sku_median=float(dsp.sku_ratio.median()), auc=float(auc), p=p21))

# H22 - the customers belong to the address, not the owner
fv = Sp.pivot_table(index="store_no", columns="family", values="liters", aggfunc="sum").fillna(0)
fv = fv.div(fv.sum(1), axis=0)
sd = stores.set_index("store_no")


def jsd(a, b):
    from scipy.spatial.distance import jensenshannon
    return float(jensenshannon(a, b, base=2))


pairs = []
for s_new, r in sd[sd.entry_type == "ownership_change"].iterrows():
    cand = sd[(sd.addr_key == r.addr_key) & (sd.index != s_new) & (sd.last_t <= r.first_t + 1)]
    if len(cand) and s_new in fv.index and cand.index[0] in fv.index:
        pairs.append((cand.index[0], s_new, r.city_key, r.store_type))
same = [jsd(fv.loc[a], fv.loc[b]) for a, b, _, _ in pairs]
null = []
for a, b, city, stype in pairs:
    pool = sd[(sd.store_type == stype) & (sd.index != a) & (sd.index != b) & sd.index.isin(fv.index)].index
    null += [jsd(fv.loc[a], fv.loc[c]) for c in RNG.choice(pool, min(20, len(pool)), replace=False)]
vol_pairs = []
for a, b, _, _ in pairs:
    va = wide_l.loc[a][wide_l.loc[a] > 0].mean() if a in wide_l.index else np.nan
    vb = wide_l.loc[b][wide_l.loc[b] > 0].mean() if b in wide_l.index else np.nan
    vol_pairs.append(vb / va)
p22 = float(stats.mannwhitneyu(same, null, alternative="less").pvalue)
record("H22", "Store behaviour",
       "Customers belong to the address, not the owner: when a shop changes hands, the new licensee's product mix is far closer to its predecessor's than to any random store of the same type.",
       f"Jensen-Shannon distance of family mix: {len(pairs)} predecessor->successor pairs at the same address vs random same-type pairs; Mann-Whitney",
       f"median JS distance {np.median(same):.3f} at same address vs {np.median(null):.3f} random same-type ({np.median(same) / np.median(null):.0%} of it); "
       f"successor's monthly volume = {np.nanmedian(vol_pairs):.0%} of predecessor's",
       round(p22, 6), "SUPPORTED" if p22 < 0.05 else "NOT SUPPORTED",
       dict(n_pairs=len(pairs), js_same=float(np.median(same)), js_null=float(np.median(null)), vol_ratio=float(np.nanmedian(vol_pairs))))

# H23 - Iowa's generational whiskey divide: Canadian whisky is the old-Iowa pour, bourbon the young one
BOURBON = ["STRAIGHT BOURBON WHISKIES", "BOTTLED IN BOND BOURBON", "SINGLE BARREL BOURBON WHISKIES"]
wk = SP[SP.family == "whiskey"]
wc = wk.assign(can=wk.liters.where(wk.category_name == "CANADIAN WHISKIES", 0),
               bou=wk.liters.where(wk.category_name.isin(BOURBON), 0)).groupby(["county_fips_code", "yr"])[["can", "bou", "liters"]].sum()
wc = pd.DataFrame({"can_sh": wc.can / wc.liters, "bou_sh": wc.bou / wc.liters}).reset_index()
cty = cty.merge(wc, on=["county_fips_code", "yr"])
ctrl23 = ["log_density", "income_k", "cty_pct_bachelors_or_higher"]
res23c, v23c, p23c = county_two_years(cty, "can_sh", "cty_median_age", ctrl23, +1)
res23b, v23b, p23b = county_two_years(cty, "bou_sh", "cty_median_age", ctrl23, -1)
# same split at store level inside the same county (removes county-wide taste): no age variable there, so use
# the national picture as context only
cat_rank = SP.groupby("category_name").liters.sum().sort_values(ascending=False)
c26 = cty[cty.yr == 2026]
q_age = c26.cty_median_age.quantile([0.2, 0.8])
old, young = c26[c26.cty_median_age >= q_age[0.8]], c26[c26.cty_median_age <= q_age[0.2]]
both = v23c.startswith("SUPPORTED") and v23b.startswith("SUPPORTED")
# what actually drives the split once age is controlled: density (Canadian) and education (bourbon)
drv = {yr: dict(can_density=float(ols(cty[cty.yr == yr], "can_sh", ["log_density", "cty_median_age", "income_k", "cty_pct_bachelors_or_higher"]).params["log_density"]),
                bou_bach=float(ols(cty[cty.yr == yr], "bou_sh", ["cty_pct_bachelors_or_higher", "cty_median_age", "income_k", "log_density"]).params["cty_pct_bachelors_or_higher"]),
                r_can_density=float(cty[cty.yr == yr][["can_sh", "log_density"]].corr().iloc[0, 1]),
                r_can_age_raw=float(cty[cty.yr == yr][["can_sh", "cty_median_age"]].corr().iloc[0, 1])) for yr in (2025, 2026)}
age_proxy = all(drv[y]["r_can_age_raw"] > 0 for y in drv) and not (v23c.startswith(("SUPPORTED", "PARTIAL")) or v23b.startswith(("SUPPORTED", "PARTIAL")))
v23 = ("SUPPORTED - both halves of the divide, both years" if both else
       "REJECTED - age is only a proxy: rurality drives Canadian, education drives bourbon" if age_proxy else
       f"PARTIAL - Canadian: {v23c}; bourbon: {v23b}" if (v23c.startswith(("SUPPORTED", "PARTIAL")) or v23b.startswith(("SUPPORTED", "PARTIAL")))
       else f"NOT SUPPORTED (Canadian: {v23c}; bourbon: {v23b})")
record("H23", "Geography & culture",
       "Iowa's whiskey divide is generational: within the whiskey aisle, Canadian whisky's share rises with a county's median age and bourbon's share falls, after controlling for density, income and education.",
       "County share of whiskey liters (Canadian; bourbon) ~ median age + log density + income + % bachelor's, HC3 + permutation, both years",
       f"Canadian is Iowa's #{list(cat_rank.index).index('CANADIAN WHISKIES') + 1} category overall; oldest-quintile counties {old.can_sh.mean():.0%} Canadian / {old.bou_sh.mean():.0%} bourbon "
       f"vs youngest {young.can_sh.mean():.0%} / {young.bou_sh.mean():.0%}; with controls, +5 yrs median age: Canadian {5 * res23c[2026]['coef']:+.1%} pts, bourbon {5 * res23b[2026]['coef']:+.1%} pts (2026); "
       f"instead 10x density = {np.log(10) * drv[2026]['can_density']:+.0%} pts Canadian (r={drv[2026]['r_can_density']:.2f}), +10 pts bachelor's = {10 * drv[2026]['bou_bach']:+.0%} pts bourbon",
       round(max(p23c, p23b), 4), v23, dict(canadian=res23c, bourbon=res23b, drivers=drv))

# ----------------------------------------------------------------------------------------------
# write
# ----------------------------------------------------------------------------------------------
out = pd.DataFrame(RESULTS)
pv = out.p_value.astype(float)
order = pv.fillna(1).argsort().values
holm = np.ones(len(pv))
m = pv.notna().sum()
running = 0
for rank, i in enumerate(order):
    if np.isnan(pv.iloc[i]):
        holm[i] = np.nan
        continue
    running = max(running, min(1, (m - rank) * pv.iloc[i]))
    holm[i] = running
out["p_holm"] = holm
OUT.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT / "twenty_hypotheses.csv", index=False)
(OUT / "twenty_hypotheses_detail.json").write_text(json.dumps(DETAIL, indent=2, default=str))
pd.set_option("display.width", 250, "display.max_colwidth", 140)
log("\n", out[["id", "verdict", "effect", "p_value", "p_holm"]].to_string())
