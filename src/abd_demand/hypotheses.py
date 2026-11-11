"""Hypothesis tests (course unit: hypothesis testing, causal inference / difference-in-differences).

county_test   OLS with HC3 robust SEs + Freedman-Lane permutation p-value + robustness checks
stacked_did   stacked difference-in-differences event study for store entry (H3)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config as C


# --------------------------------------------------------------------------------------------
# County-level tests (H1, H2)
# --------------------------------------------------------------------------------------------
def _fit(d, x, y, controls, weights=None):
    X = sm.add_constant(d[[x] + controls].astype(float))
    m = sm.WLS(d[y].astype(float), X, weights=weights) if weights is not None else sm.OLS(d[y].astype(float), X)
    return m.fit(cov_type="HC3")


def one_sided(p_two: float, coef: float, sign: int) -> float:
    return p_two / 2 if np.sign(coef) == sign else 1 - p_two / 2


def freedman_lane(d, x, y, controls, sign, n_perm=5000, seed=C.SEED):
    """Permutation test for one coefficient in a regression with controls (Freedman & Lane 1983):
    permute the residuals of the controls-only model, rebuild y, refit, compare t-statistics."""
    rng = np.random.default_rng(seed)
    Xr = sm.add_constant(d[controls].astype(float))
    red = sm.OLS(d[y].astype(float), Xr).fit()
    Xf = sm.add_constant(d[[x] + controls].astype(float)).values
    t_obs = sm.OLS(d[y].astype(float).values, Xf).fit().tvalues[1]
    fitted, resid = red.fittedvalues.values, red.resid.values
    ts = np.empty(n_perm)
    for i in range(n_perm):
        ys = fitted + rng.permutation(resid)
        ts[i] = sm.OLS(ys, Xf).fit().tvalues[1]
    return float((1 + np.sum(sign * ts >= sign * t_obs)) / (n_perm + 1))


def county_test(d: pd.DataFrame, x: str, y: str, controls: list[str], sign: int,
                weight_col: str = "cty_population_21_plus", n_perm=5000, n_boot=2000, seed=C.SEED) -> dict:
    d = d[[x, y, weight_col, "county_name"] + controls].dropna() if "county_name" in d else d[[x, y, weight_col] + controls].dropna()
    fit = _fit(d, x, y, controls)
    coef = fit.params[x]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        if s[x].nunique() < 2:
            continue
        boots.append(_fit(s, x, y, controls).params[x])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    cooks = sm.OLS(d[y].astype(float), sm.add_constant(d[[x] + controls].astype(float))).fit().get_influence().cooks_distance[0]
    keep = cooks <= 4 / len(d)
    fit_trim = _fit(d[keep], x, y, controls)
    fit_w = _fit(d, x, y, controls, weights=d[weight_col])
    res = dict(
        n=len(d), coef=coef, se=fit.bse[x], p_hc3=one_sided(fit.pvalues[x], coef, sign),
        p_perm=freedman_lane(d, x, y, controls, sign, n_perm, seed), r2=fit.rsquared,
        boot_ci_low=lo, boot_ci_high=hi,
        coef_trimmed=fit_trim.params[x], p_trimmed=one_sided(fit_trim.pvalues[x], fit_trim.params[x], sign),
        n_dropped=int((~keep).sum()),
        coef_weighted=fit_w.params[x], p_weighted=one_sided(fit_w.pvalues[x], fit_w.params[x], sign),
    )
    res["supported"] = bool(res["p_hc3"] < 0.05 and res["p_perm"] < 0.05
                            and res["p_trimmed"] < 0.05 and res["p_weighted"] < 0.05)
    return res


# --------------------------------------------------------------------------------------------
# Stacked difference-in-differences (H3)
# --------------------------------------------------------------------------------------------
def build_stacks(sm_: pd.DataFrame, stores: pd.DataFrame, entry_type: str, window: int = 6,
                 first_g: int = 3, last_g: int = 17, clean_types=("new_store", "ownership_change")) -> pd.DataFrame:
    """One stack per treatment month g. Treated units: incumbent stores in cities whose first
    `entry_type` arrival is at g. Controls: incumbents in cities with no arrival of any kind within
    [g-window, g+window]. Incumbent = first order at least 3 months before g and an order in g-3..g-1
    (defined from pre-period data only, so no conditioning on what happens after entry)."""
    s = stores.copy()
    months = sorted(sm_.month.unique())
    m_idx = {m: i for i, m in enumerate(months)}
    s["g"] = s.first_month.map(m_idx)
    arrivals = s[s.entry_type.isin(clean_types)]
    treat_city = (arrivals[arrivals.entry_type == entry_type].groupby("city_key").g.min())
    treat_city = treat_city[(treat_city >= first_g) & (treat_city <= last_g)]
    # a treated city must not have had a different arrival in the 6 months before its event
    other = arrivals.groupby("city_key").g.apply(list)
    ok = [c for c, g in treat_city.items()
          if not any((g - window) <= x < g for x in other.get(c, []))]
    treat_city = treat_city[ok]

    # A store that stops ordering at the address where a new licence then appears is the *same shop*
    # under a new owner: its drop to zero is mechanical, so it is never used as an incumbent.
    predecessors = set(s[(s.is_exiter == 1) & s.addr_key.isin(set(arrivals.addr_key))
                         & ~s.store_no.isin(arrivals.store_no)].store_no)

    active = sm_.pivot_table(index="store_no", columns="t", values="liters", aggfunc="sum", fill_value=0)
    stacks, entrants = [], []
    for g, cities in treat_city.groupby(treat_city):
        lo, hi = max(0, g - window), min(len(months) - 1, g + window)
        busy = set(arrivals.loc[(arrivals.g >= g - window) & (arrivals.g <= g + window), "city_key"])
        pre = [t for t in (g - 3, g - 2, g - 1) if t >= 0]
        inc = s[(s.g <= g - 3) & s.store_no.isin(active.index) & ~s.store_no.isin(predecessors)]
        inc = inc[(active.loc[inc.store_no, pre].sum(axis=1) > 0).values]
        tr = inc[inc.city_key.isin(cities.index)]
        co = inc[~inc.city_key.isin(busy)]
        unit = pd.concat([tr.assign(treated=1), co.assign(treated=0)])
        d = sm_[sm_.store_no.isin(unit.store_no) & sm_.t.between(lo, hi)].merge(
            unit[["store_no", "treated"]], on="store_no")
        d["stack"], d["event_time"] = g, d.t - g
        stacks.append(d)
        ent = s[(s.entry_type == entry_type) & s.city_key.isin(cities.index) & s.g.between(g, hi)]
        e = sm_[sm_.store_no.isin(ent.store_no) & sm_.t.between(g, hi)].assign(stack=g, event_time=lambda x: x.t - g)
        entrants.append(e)
    out = pd.concat(stacks, ignore_index=True)
    out["unit"] = out["stack"].astype(str) + "|" + out["store_no"]
    out["stack_month"] = out["stack"].astype(str) + "|" + out["t"].astype(str)
    pre_mean = out[out.event_time < 0].groupby("unit").liters.mean()
    out["pre_mean"] = out.unit.map(pre_mean)
    out["liters_norm"] = out.liters / out.pre_mean.where(out.pre_mean > 0)  # 1.0 = own pre-entry level
    out.attrs["entrants"] = pd.concat(entrants, ignore_index=True)
    out.attrs["n_predecessors_excluded"] = len(predecessors)
    return out


def city_level(st: pd.DataFrame) -> pd.DataFrame:
    """Collapse an incumbent stack to city x month and add the entrants' liters, giving the city's
    total purchases. Used to ask whether the market grows by the entrant's volume."""
    inc = st.groupby(["stack", "city_key", "t", "event_time", "treated"], as_index=False).liters.sum()
    ent = st.attrs["entrants"].groupby(["stack", "city_key", "t"], as_index=False).liters.sum().rename(
        columns={"liters": "entrant_liters"})
    c = inc.merge(ent, on=["stack", "city_key", "t"], how="left").fillna({"entrant_liters": 0.0})
    c["total_liters"] = c.liters + c.entrant_liters
    c["unit"] = c["stack"].astype(str) + "|" + c.city_key
    c["stack_month"] = c["stack"].astype(str) + "|" + c.t.astype(str)
    pm = c[c.event_time < 0].groupby("unit").liters.mean()
    c["pre_mean"] = c.unit.map(pm)
    c = c[c.pre_mean > 0]
    c["incumbent_norm"] = c.liters / c.pre_mean
    c["total_norm"] = c.total_liters / c.pre_mean
    c["entrant_norm"] = c.entrant_liters / c.pre_mean
    return c


def _demean(df: pd.DataFrame, cols: list[str], fe: list[str], tol=1e-8, max_iter=200) -> pd.DataFrame:
    """Absorb several fixed effects by alternating projections (method of Guimaraes & Portugal)."""
    out = df[cols].astype(float).copy()
    for _ in range(max_iter):
        prev = out.copy()
        for f in fe:
            out = out - out.groupby(df[f]).transform("mean")
        if (out - prev).abs().to_numpy().max() < tol:
            break
    return out


def stacked_did(st: pd.DataFrame, outcome: str = "log_liters", window: int = 6, ref: int = -1) -> dict:
    """Event-study and pooled DiD on a stacked panel. FE: unit (store x stack) and stack x month.
    SEs clustered by city."""
    st = st.copy()
    st["log_liters"] = np.log1p(st.liters.clip(lower=0))
    st["post"] = (st.event_time >= 0).astype(int)
    st["treat_post"] = st.treated * st.post
    ev = [e for e in range(-window, window + 1) if e != ref]
    for e in ev:
        st[f"ev_{e}"] = ((st.event_time == e) & (st.treated == 1)).astype(float)
    ev_cols = [f"ev_{e}" for e in ev]
    fe = ["unit", "stack_month"]
    dm = _demean(st, [outcome, "treat_post"] + ev_cols, fe)
    groups = st.city_key.astype("category").cat.codes
    pooled = sm.OLS(dm[outcome], dm[["treat_post"]]).fit(cov_type="cluster", cov_kwds={"groups": groups})
    es = sm.OLS(dm[outcome], dm[ev_cols]).fit(cov_type="cluster", cov_kwds={"groups": groups})
    leads = [c for c in ev_cols if int(c.split("_")[1]) < ref]
    wald = es.wald_test(np.eye(len(ev_cols))[[ev_cols.index(c) for c in leads]], scalar=True)
    coefs = pd.DataFrame({"event_time": ev, "coef": es.params.values, "se": es.bse.values})
    coefs = pd.concat([coefs, pd.DataFrame({"event_time": [ref], "coef": [0.0], "se": [0.0]})]).sort_values("event_time")
    return dict(
        att=float(pooled.params["treat_post"]), se=float(pooled.bse["treat_post"]),
        p=float(pooled.pvalues["treat_post"]), ci=tuple(pooled.conf_int().loc["treat_post"]),
        pretrend_F=float(wald.statistic), pretrend_p=float(wald.pvalue), event_study=coefs,
        n_obs=int(len(st)), n_treated_units=int(st.loc[st.treated == 1, "unit"].nunique()),
        n_control_units=int(st.loc[st.treated == 0, "unit"].nunique()),
        n_treated_cities=int(st.loc[st.treated == 1, "city_key"].nunique()),
        n_clusters=int(groups.nunique()),
    )
