"""Feature layer: SSOT -> analysis tables.

county_period_table  one row per county x period, for the county hypotheses (H1, H2)
store_month          store x month liters (zeros filled between a store's first month and the end), for H3
build_store_panel    store x spirits family x month modelling table for the forecast. Every feature for
                     target month t uses only information available at the end of month t-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

SSOT_COLS = ["ordered_on", "month", "year", "store_no", "county_fips_code", "family", "bottle_volume_ml",
             "sales_bottles", "liters", "dollars", "price_per_liter", "city_key"]

COUNTY_FEATURES = ["cty_total_population", "cty_population_21_plus", "cty_median_household_income",
                   "cty_pct_hispanic", "cty_pct_bachelors_or_higher", "cty_median_age", "cty_poverty_rate",
                   "cty_pct_age_21_34", "geo_pop_density", "geo_borders_mo", "geo_border_tax_gap",
                   "geo_log_hub_ratio", "geo_miles_to_mo_border", "geo_is_border_county"]

PERIODS = {
    "FY2025": ("2025-01", "2025-12"),
    "JanOct2025": ("2025-01", "2025-10"),
    "JanOct2026": ("2026-01", "2026-10"),
}


def load_ssot(cols=SSOT_COLS) -> pd.DataFrame:
    d = pd.read_parquet(C.SSOT, columns=cols)
    for c in ("month", "store_no", "county_fips_code", "family", "city_key"):
        if c in d:
            d[c] = d[c].astype(str)
    return d


def month_range(a: str, b: str) -> list[str]:
    return [str(p) for p in pd.period_range(a, b, freq="M")]


# --------------------------------------------------------------------------------------------
# County x period (H1, H2)
# --------------------------------------------------------------------------------------------
def county_period_table(sales: pd.DataFrame, profile: pd.DataFrame, premium_cut: float | None = None) -> pd.DataFrame:
    """Volume and mix outcomes per county and period. premium_cut: USD/liter threshold for the
    'premium' tier; fixed from 2025 (liter-weighted top quartile) so 2026 is measured on the same scale."""
    pos = sales[(sales.liters > 0) & (sales.dollars > 0)]
    if premium_cut is None:
        p25 = pos[(pos.year == 2025) & pos.price_per_liter.notna()]  # multi-pack minis have no valid price
        order = np.argsort(p25.price_per_liter.values)
        cum = np.cumsum(p25.liters.values[order]) / p25.liters.sum()
        premium_cut = float(p25.price_per_liter.values[order][np.searchsorted(cum, 0.75)])
    out = []
    for name, (a, b) in PERIODS.items():
        months = (b[:4] == a[:4]) and (int(b[5:]) - int(a[5:]) + 1)
        d = sales[(sales.month >= a) & (sales.month <= b)]
        dp = pos[(pos.month >= a) & (pos.month <= b)]
        g = d.groupby("county_fips_code", observed=True)
        L = dp.groupby("county_fips_code", observed=True).liters.sum()

        def share(mask, col="liters"):
            return dp[mask].groupby("county_fips_code", observed=True)[col].sum().reindex(L.index, fill_value=0)

        f = pd.DataFrame({
            "liters": g.liters.sum(),
            "dollars": g.dollars.sum(),
            "lines": g.size(),
            "stores": g.store_no.nunique(),
        })
        f["handle_share"] = share(dp.bottle_volume_ml >= 1750) / L
        f["mini_bottle_share"] = (share(dp.bottle_volume_ml <= 375, "sales_bottles")
                                  / dp.groupby("county_fips_code", observed=True).sales_bottles.sum())
        f["premium_share"] = share(dp.price_per_liter >= premium_cut) / L
        for fam in C.FAMILIES:
            f[f"{fam}_share"] = share(dp.family == fam) / L
        f["period"], f["months"] = name, months
        out.append(f.reset_index())
    t = pd.concat(out, ignore_index=True)
    t["county_fips_code"] = t.county_fips_code.astype(str)
    t = t.merge(profile, on="county_fips_code", how="left", validate="many_to_one")
    t["liters_per_adult"] = t.liters * (12 / t.months) / t.cty_population_21_plus  # annualised
    t["dollars_per_adult"] = t.dollars * (12 / t.months) / t.cty_population_21_plus
    t["log_liters_per_adult"] = np.log(t.liters_per_adult)
    t["stores_per_10k_adults"] = t.stores / t.cty_population_21_plus * 1e4
    t["log_population"] = np.log(t.cty_total_population)
    t["income_k"] = t.cty_median_household_income / 1000
    t.attrs["premium_cut"] = premium_cut
    return t


# --------------------------------------------------------------------------------------------
# Store x month (H3)
# --------------------------------------------------------------------------------------------
def store_month(sales: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    m = (sales.groupby(["store_no", "month"], observed=True)
         .agg(liters=("liters", "sum"), dollars=("dollars", "sum"), lines=("liters", "size")).reset_index())
    m["store_no"] = m.store_no.astype(str)
    months = month_range("2025-01", "2026-10")
    s = stores.set_index("store_no")
    grid = [(sid, mo) for sid, first in s.first_month.items() for mo in months if mo >= first]
    g = pd.DataFrame(grid, columns=["store_no", "month"])
    g = g.merge(m, on=["store_no", "month"], how="left").fillna({"liters": 0.0, "dollars": 0.0, "lines": 0})
    g = g.merge(stores[["store_no", "city_key", "county_fips_code", "store_type", "entry_type",
                        "first_month", "last_month"]], on="store_no", how="left")
    g["t"] = g.month.map({m_: i for i, m_ in enumerate(months)})
    return g


# --------------------------------------------------------------------------------------------
# Store x family x month modelling table (forecast)
# --------------------------------------------------------------------------------------------
def business_days(month: str) -> int:
    p = pd.Period(month, "M")
    return int(np.busday_count(p.start_time.date(), (p.end_time + pd.Timedelta(days=1)).date()))


def weekday_counts(month: str) -> np.ndarray:
    p = pd.Period(month, "M")
    days = pd.date_range(p.start_time, p.end_time.normalize(), freq="D")
    return np.bincount(days.day_of_week, minlength=7)


def order_day_features(sales: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """Stores order from ABD on fixed weekdays (the median store puts 89% of its liters on one
    weekday), so a month with five of a store's order days carries ~25% more volume than a month with
    four. For each store and target month t: its weekday mix learned from orders *before* t, times the
    number of each weekday in t. Ratios against t-1 and t-12 let the model rescale its lag features."""
    m_idx = {m: i for i, m in enumerate(months)}
    s = sales[sales.liters > 0]
    w = (s.assign(wd=s.ordered_on.dt.day_of_week, t=s.month.map(m_idx))
         .groupby(["store_no", "t", "wd"], observed=True).liters.sum().unstack(fill_value=0.0)
         .reindex(columns=range(7), fill_value=0.0))
    stores = w.index.get_level_values(0).unique()
    full = pd.MultiIndex.from_product([stores, range(len(months))], names=["store_no", "t"])
    w = w.reindex(full, fill_value=0.0)
    cum = w.groupby(level=0).cumsum().groupby(level=0).shift(1)  # history up to t-1 only
    share = cum.div(cum.sum(axis=1), axis=0).to_numpy()
    K = np.vstack([weekday_counts(m) for m in months]).astype(float)
    t = full.get_level_values(1).to_numpy()
    cur = (share * K[t]).sum(axis=1)
    prev = np.where(t >= 1, (share * K[np.maximum(t - 1, 0)]).sum(axis=1), np.nan)
    yago = np.where(t >= 12, (share * K[np.maximum(t - 12, 0)]).sum(axis=1), np.nan)
    out = pd.DataFrame({"store_no": full.get_level_values(0).astype(str), "t": t, "order_days": cur,
                        "cal_ratio_lag1": cur / prev, "cal_ratio_lag12": cur / yago})
    return out


def build_store_panel(sales: pd.DataFrame, stores: pd.DataFrame, profile: pd.DataFrame,
                      last_month: str = "2026-10", forecast_month: str | None = None) -> pd.DataFrame:
    """One row per (store, family, target month). If forecast_month is given, a row for that
    future month is added (target unknown) so the API can score next month."""
    months = month_range("2025-01", last_month)
    if forecast_month:
        months = months + [forecast_month]
    m_idx = {m: i for i, m in enumerate(months)}

    sf = (sales.groupby(["store_no", "family", "month"], observed=True)
          .agg(liters=("liters", "sum"), lines=("liters", "size")).reset_index())
    sf["store_no"] = sf.store_no.astype(str)
    sf["family"] = sf.family.astype(str)
    pairs = sf[["store_no", "family"]].drop_duplicates()
    first = stores.set_index("store_no").first_month
    pairs = pairs.assign(first=pairs.store_no.map(first))
    grid = pairs.merge(pd.DataFrame({"month": months}), how="cross")
    grid = grid[grid.month >= grid["first"]].drop(columns="first")
    p = grid.merge(sf, on=["store_no", "family", "month"], how="left").fillna({"liters": 0.0, "lines": 0})
    p["t"] = p.month.map(m_idx)
    p = p.sort_values(["store_no", "family", "t"]).reset_index(drop=True)
    p["y"] = p.liters.clip(lower=0)  # target: net liters, returns clipped at zero
    if forecast_month:
        p.loc[p.month == forecast_month, ["y", "liters", "lines"]] = np.nan

    grp = p.groupby(["store_no", "family"], sort=False)
    for k in (1, 2, 3, 6, 12):
        p[f"lag{k}"] = grp.y.shift(k)
    shifted = grp.y.shift(1)
    for w in (3, 6):
        p[f"roll{w}_mean"] = shifted.groupby([p.store_no, p.family]).transform(lambda s: s.rolling(w, min_periods=1).mean())
    p["roll6_std"] = shifted.groupby([p.store_no, p.family]).transform(lambda s: s.rolling(6, min_periods=2).std())
    p["active_share6"] = (shifted > 0).astype(float).groupby([p.store_no, p.family]).transform(
        lambda s: s.rolling(6, min_periods=1).mean())
    p["lines_lag1"] = grp.lines.shift(1)

    # store-level totals (all families), lagged
    st = p.groupby(["store_no", "t"]).y.sum().rename("store_y").reset_index()
    st = st.sort_values(["store_no", "t"])
    st["store_lag1"] = st.groupby("store_no").store_y.shift(1)
    st["store_roll3"] = st.groupby("store_no").store_y.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    st["store_lag12"] = st.groupby("store_no").store_y.shift(12)
    p = p.merge(st.drop(columns="store_y"), on=["store_no", "t"], how="left")
    p["family_share_roll3"] = p.roll3_mean / (p.store_roll3.replace(0, np.nan))

    # statewide family momentum: last month vs same month a year earlier (known at t-1)
    sw = p.groupby(["family", "t"]).y.sum().rename("state_y").reset_index().sort_values(["family", "t"])
    sw["state_fam_lag1"] = sw.groupby("family").state_y.shift(1)
    sw["state_fam_yoy"] = sw.state_fam_lag1 / sw.groupby("family").state_y.shift(13)
    p = p.merge(sw[["family", "t", "state_fam_lag1", "state_fam_yoy"]], on=["family", "t"], how="left")

    # store attributes and local competition (from the H3 store dimension)
    s = stores.set_index("store_no")
    p["store_type"] = p.store_no.map(s.store_type)
    p["county_fips_code"] = p.store_no.map(s.county_fips_code).astype(str)
    p["city_key"] = p.store_no.map(s.city_key)
    p["store_age_months"] = p.t - p.store_no.map(s.first_month).map(m_idx)
    active = (p.groupby(["store_no", "t"]).y.sum() > 0).rename("active").reset_index()
    active["city_key"] = active.store_no.map(s.city_key)
    city_active = active.groupby(["city_key", "t"]).active.sum().rename("city_active").reset_index()
    city_active["city_active_stores_lag1"] = city_active.groupby("city_key").city_active.shift(1)
    entr = s[s.entry_type == "new_store"].assign(t=lambda d: d.first_month.map(m_idx))
    ec = entr.groupby(["city_key", "t"]).size().rename("n_new").reset_index()
    city_active = city_active.merge(ec, on=["city_key", "t"], how="left").fillna({"n_new": 0})
    city_active["city_new_entrants_last3"] = city_active.groupby("city_key").n_new.transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum())
    p = p.merge(city_active[["city_key", "t", "city_active_stores_lag1", "city_new_entrants_last3"]],
                on=["city_key", "t"], how="left")

    # calendar
    p["month_of_year"] = p.month.str[5:].astype(int)
    bd = {m: business_days(m) for m in months}
    p["business_days"] = p.month.map(bd)
    p = p.merge(order_day_features(sales, months), on=["store_no", "t"], how="left")
    p["lag1_cal_adj"] = p.lag1 * p.cal_ratio_lag1      # last month, rescaled to this month's order days
    p["lag12_cal_adj"] = p.lag12 * p.cal_ratio_lag12   # same month last year, rescaled likewise

    # county profile
    prof = profile.set_index("county_fips_code")[COUNTY_FEATURES]
    p = p.join(prof, on="county_fips_code")
    p["log_population"] = np.log(p.cty_total_population)
    p = p.drop(columns=["cty_total_population", "liters"])
    return p


NUMERIC_FEATURES = [
    "lag1", "lag2", "lag3", "lag6", "lag12", "roll3_mean", "roll6_mean", "roll6_std", "active_share6",
    "lines_lag1", "store_lag1", "store_roll3", "store_lag12", "family_share_roll3",
    "state_fam_lag1", "state_fam_yoy", "store_age_months", "city_active_stores_lag1",
    "city_new_entrants_last3", "month_of_year", "business_days", "order_days", "cal_ratio_lag1",
    "cal_ratio_lag12", "lag1_cal_adj", "lag12_cal_adj",
    "log_population", "cty_median_household_income", "cty_pct_hispanic", "cty_pct_bachelors_or_higher",
    "cty_median_age", "cty_poverty_rate", "cty_pct_age_21_34", "geo_pop_density", "geo_borders_mo",
    "geo_border_tax_gap", "geo_log_hub_ratio",
]
CATEGORICAL_FEATURES = ["family", "store_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
