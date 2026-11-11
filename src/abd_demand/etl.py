"""ETL: raw sources -> single source of truth (SSOT).

Sources
  1. Iowa Liquor Sales 2025 (Iowa Data Hub dataset 1262) and 2026 (dataset 1263), transaction lines
  2. US Census ACS 5-year 2020-2024, 16 tables, Iowa counties
  3. US Census county adjacency file 2024 + 2024 Gazetteer (land area, centroid)
  4. Tax Foundation 2026 state spirits excise rates (Iowa + 6 neighbouring states)

Every cleaning step records what it changed in a data-quality log, saved as JSON next to the outputs.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import config as C

SALES_COLS = [
    "invoice_id", "ordered_on", "store_no", "store_name", "store_address", "store_city",
    "store_zip_code", "county_fips_code", "county_name", "category_code", "category_name",
    "vendor_number", "vendor_name", "item_no", "im_desc", "pack", "bottle_volume_ml",
    "state_bottle_cost", "state_bottle_retail", "sales_bottles", "sales_dollars", "sales_liters",
    "sales_gallons",
]
STR_COLS = ["invoice_id", "store_no", "store_name", "store_address", "store_city", "store_zip_code",
            "county_fips_code", "county_name", "category_code", "category_name", "vendor_number",
            "vendor_name", "item_no", "im_desc"]
# Months where the publisher truncated liters, dollars and prices to whole numbers (found in profiling).
TRUNCATED_MONTHS = {"2025-11", "2025-12", "2026-01"}
# Closest clean month whose item prices are used to rebuild truncated dollars.
PRICE_DONOR_MONTH = {"2025-11": "2025-10", "2025-12": "2025-10", "2026-01": "2026-02"}


# --------------------------------------------------------------------------------------------
# Census county profile
# --------------------------------------------------------------------------------------------
def _load_acs(table: str) -> dict[str, dict[int, float | None]]:
    path = C.ACS_DIR / f"census_acs5y2024_{table}_iowa_counties.dat"
    df = pd.read_csv(path, sep="|", dtype=str)
    out = {}
    for _, row in df.iterrows():
        vals = {}
        for k, v in row.items():
            if k == "GEO_ID" or "_E" not in k:
                continue  # keep estimates (_E), skip margins of error (_M)
            num = float(v) if isinstance(v, str) and v != "" else None
            vals[int(k.split("_E")[1])] = num if num is not None and num >= 0 else None  # negative = ACS sentinel
        out[row["GEO_ID"][-5:]] = vals
    return out


def _pct(num, den):
    return round(100 * num / den, 2) if num is not None and den else None


def build_county_profile() -> pd.DataFrame:
    """99 Iowa counties x census, geography and border-tax variables."""
    t = {name: _load_acs(name) for name in (
        "b01001", "b01002", "b01003", "b02001", "b03003", "b11001", "b15003", "b17001",
        "b19013", "b19083", "b19301", "b23025", "b25003", "b25010", "b25064", "b25077")}
    rows = []
    for fips in sorted(t["b01003"]):
        sa = t["b01001"][fips]  # sex by age: 2 = male total, 3-25 male ages, 26 = female total, 27-49 female ages
        pop = t["b01003"][fips][1]
        pop_21 = sum(sa[i] for i in list(range(9, 26)) + list(range(33, 50)))   # 21 years and over
        pop_21_34 = sum(sa[i] for i in (9, 10, 11, 12) + (33, 34, 35, 36))        # 21-34
        pop_65 = sum(sa[i] for i in list(range(20, 26)) + list(range(44, 50)))    # 65 years and over
        edu, emp = t["b15003"][fips], t["b23025"][fips]
        rows.append({
            "county_fips_code": fips,
            "cty_total_population": pop,
            "cty_population_21_plus": pop_21,
            "cty_pct_population_21_plus": _pct(pop_21, pop),
            "cty_pct_age_21_34": _pct(pop_21_34, pop),
            "cty_pct_population_65_plus": _pct(pop_65, pop),
            "cty_pct_male": _pct(sa[2], sa[1]),
            "cty_median_age": t["b01002"][fips][1],
            "cty_households": t["b11001"][fips][1],
            "cty_avg_household_size": t["b25010"][fips][1],
            "cty_owner_occupied_rate": _pct(t["b25003"][fips][2], t["b25003"][fips][1]),
            "cty_median_household_income": t["b19013"][fips][1],
            "cty_per_capita_income": t["b19301"][fips][1],
            "cty_gini_index": t["b19083"][fips][1],
            "cty_poverty_rate": _pct(t["b17001"][fips][2], t["b17001"][fips][1]),
            "cty_labor_force": emp[3],
            "cty_unemployment_rate": _pct(emp[5], emp[3]),
            "cty_pct_bachelors_or_higher": _pct(sum(edu[i] for i in (22, 23, 24, 25)), edu[1]),
            "cty_median_home_value": t["b25077"][fips][1],
            "cty_median_gross_rent": t["b25064"][fips][1],
            "cty_pct_white_alone": _pct(t["b02001"][fips][2], t["b02001"][fips][1]),
            "cty_pct_black_alone": _pct(t["b02001"][fips][3], t["b02001"][fips][1]),
            "cty_pct_hispanic": _pct(t["b03003"][fips][3], t["b03003"][fips][1]),
        })
    prof = pd.DataFrame(rows).set_index("county_fips_code")

    # Geography: land area, density, centroid (2024 Gazetteer)
    gz = pd.read_csv(C.GAZETTEER, sep="\t", dtype={"GEOID": str})
    gz.columns = [c.strip() for c in gz.columns]
    gz = gz[gz.USPS == "IA"].set_index("GEOID")
    prof["county_name"] = gz.NAME.str.replace(" County", "", regex=False)
    prof["geo_land_sq_mi"] = gz.ALAND_SQMI
    prof["geo_pop_density"] = (prof.cty_total_population / prof.geo_land_sq_mi).round(2)
    prof["geo_lat"], prof["geo_lon"] = gz.INTPTLAT, gz.INTPTLONG
    prof["geo_miles_to_mo_border"] = ((prof.geo_lat - C.MO_BORDER_LAT) * C.MILES_PER_DEG_LAT).round(1)

    # Adjacency: which neighbouring states touch the county, and the biggest Iowa neighbour
    adj = pd.read_csv(C.ADJACENCY, sep="|", dtype=str)
    adj = adj[adj["County GEOID"].str.startswith("19") & (adj["County GEOID"] != adj["Neighbor GEOID"])]
    adj = adj.assign(nb_state=adj["Neighbor GEOID"].str[:2])
    out_of_state = adj[adj.nb_state != "19"]
    tax = pd.read_csv(C.TAX, dtype={"state_fips": str}).set_index("state_fips").spirits_excise_usd_per_gallon
    iowa_tax = tax["19"]
    for fips_state, abbr in C.STATE_FIPS.items():
        if abbr == "IA":
            continue
        touching = set(out_of_state.loc[out_of_state.nb_state == fips_state, "County GEOID"])
        prof[f"geo_borders_{abbr.lower()}"] = prof.index.isin(touching).astype(int)
    cheapest = out_of_state.assign(t=out_of_state.nb_state.map(tax)).groupby("County GEOID").t.min()
    prof["geo_border_tax_gap"] = (iowa_tax - prof.index.map(cheapest)).fillna(0.0).round(2)  # $/gal; 0 = interior
    prof["geo_is_border_county"] = prof.index.isin(out_of_state["County GEOID"]).astype(int)
    in_state = adj[adj.nb_state == "19"]
    max_nb = in_state.assign(p=in_state["Neighbor GEOID"].map(prof.cty_total_population)).groupby("County GEOID").p.max()
    prof["geo_log_hub_ratio"] = np.log(prof.index.map(max_nb) / prof.cty_total_population).round(4)
    prof["geo_n_neighbours"] = prof.index.map(adj.groupby("County GEOID").size())
    return prof.reset_index()


# --------------------------------------------------------------------------------------------
# Sales
# --------------------------------------------------------------------------------------------
def spirits_family(category_name: str) -> str:
    """Map ~48 ABD category names onto 9 families. Keyword-based, so it survives the 2026 renames
    (e.g. 'IMPORTED CORDIALS & LIQUEURS' -> 'IMPORTED CORDIALS & LIQUEUR'). Order matters:
    whiskey liqueurs (e.g. cinnamon whisky) are liqueurs, not whiskey."""
    c = str(category_name).upper()
    if "COCKTAIL" in c or "RTD" in c:
        return "rtd"
    if any(k in c for k in ("LIQUEUR", "SCHNAPPS", "CORDIAL", "TRIPLE SEC")):
        return "liqueur"
    if "VODKA" in c:
        return "vodka"
    if any(k in c for k in ("WHISK", "BOURBON", "SCOTCH")):
        return "whiskey"
    if "RUM" in c:
        return "rum"
    if "TEQUILA" in c or "MEZCAL" in c:
        return "agave"
    if re.search(r"\bGINS?\b", c):
        return "gin"
    if "BRAND" in c:
        return "brandy"
    return "other"


STORE_TYPE_RULES = [
    ("big_box_club", r"WAL-?MART|WALMART|TARGET|SAM'?S CLUB|SAMS CLUB|COSTCO"),
    ("pharmacy", r"WALGREEN|CVS|OSCO|DRUG|PHARMACY"),
    ("grocery", r"HY-?VEE|FAREWAY|FOODS?\b|FOODLAND|GROCER|SUPER ?MARKET|MARKET\b|\bIGA\b|SUPER SAVER|CASH SAVER|SAVE MORE|FAMILY FARE|MEATS?\b|ALDI|PRICE CHOPPER|NEW PIONEER|TRADER JOE"),
    ("convenience_chain", r"CASEY|KUM & GO|KUM AND GO|KWIK|QUIK ?TRIP|JET STOP|PUMP PANTRY|YESWAY|CIRCLE K|FAS MART|BREW OIL|\bBREW\b|MAVERIK|GIT N GO|GIT-N-GO|SPEEDWAY|SINCLAIR"),
    ("liquor_tobacco", r"LIQUOR|SPIRITS|WINE|TOBACCO|SMOKE|SMOKIN|VAPE|BEER|CELLAR|BOTTLE|CIGAR|\bBWS\b|BEVERAGE"),
    ("convenience_other", r"MART|STOP|GAS|FUEL|PANTRY|EXPRESS|XPRESS|CONVENIENCE|QUICK|QUIK|SHOP|STATION|OIL|BP\b|SHELL|AMOCO|PHILLIPS|CONOCO|JIFFY|BUCKY|KORNER|CORNER|COUNTRY STORE|SAVER|EASYGO"),
]


def store_type(name: str) -> str:
    n = str(name).upper()
    for label, pattern in STORE_TYPE_RULES:
        if re.search(pattern, n):
            return label
    return "other"


def _norm_city(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.replace(r"[^A-Z]", "", regex=True)


def read_sales(log: dict) -> pd.DataFrame:
    frames = []
    for dataset, pattern in ((1262, C.SALES_2025_GLOB), (1263, C.SALES_2026_GLOB)):
        files = sorted(C.RAW.glob(pattern))
        n = 0
        for f in files:
            d = pd.read_csv(f, engine="pyarrow", dtype={c: "string" for c in STR_COLS})
            assert list(d.columns) == SALES_COLS, f"schema changed in {f.name}"
            d["source_dataset"] = dataset
            d["source_file"] = f.name
            n += len(d)
            frames.append(d)
        log[f"rows_read_dataset_{dataset}"] = n
        log[f"rows_published_dataset_{dataset}"] = C.PUBLISHED_ROWS[dataset]
        assert n == C.PUBLISHED_ROWS[dataset], f"dataset {dataset}: read {n:,}, publisher lists {C.PUBLISHED_ROWS[dataset]:,}"
    df = pd.concat(frames, ignore_index=True)
    log["excluded_files"] = {
        "iowa_liquor_sales_2025_1262_rows_part_0005.csv":
            "429,989 rows (2025-08-27..2025-10-30) that duplicate rows in parts 3-4; an artefact of the "
            "download zip. Parts 1-4 hold exactly the 2,490,620 rows the publisher lists (checked above)."}
    return df


def clean_sales(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    # 1. exact duplicates (identical in all 23 published columns)
    h = pd.util.hash_pandas_object(df[SALES_COLS], index=False)
    dup = h.duplicated()
    log["exact_duplicates_removed"] = {int(k): int(v) for k, v in df.loc[dup, "source_dataset"].value_counts().items()}
    df = df.loc[~dup].reset_index(drop=True)

    # 2. types and calendar
    df["ordered_on"] = pd.to_datetime(df["ordered_on"])
    df["month"] = df["ordered_on"].dt.to_period("M").astype(str)
    df["year"] = df["ordered_on"].dt.year.astype("int16")

    # 3. blank category name -> fill from category_code seen elsewhere
    blank_cat = df.category_name.isna() | (df.category_name.str.strip() == "")
    code_map = df.loc[~blank_cat].groupby("category_code").category_name.agg(lambda s: s.mode().iat[0])
    df.loc[blank_cat, "category_name"] = df.loc[blank_cat, "category_code"].map(code_map)
    log["blank_category_filled_from_code"] = int(blank_cat.sum())
    log["blank_category_unrecoverable_labelled_unclassified"] = int((df.category_name.isna()).sum())
    df["category_name"] = df.category_name.fillna("UNCLASSIFIED")  # -> family 'other'

    # 4. blank county -> fill from the same store when the store maps to exactly one county
    blank_cty = df.county_fips_code.isna() | (df.county_fips_code.str.strip() == "")
    known = df.loc[~blank_cty].groupby("store_no").county_fips_code.agg(["nunique", "first"])
    one = known.loc[known["nunique"] == 1, "first"]
    fill = df.loc[blank_cty, "store_no"].map(one)
    df["county_imputed_from_store"] = 0
    df.loc[fill.index[fill.notna()], "county_fips_code"] = fill[fill.notna()]
    df.loc[fill.index[fill.notna()], "county_imputed_from_store"] = 1
    log["blank_county_rows"] = int(blank_cty.sum())
    log["blank_county_imputed_from_store"] = int(fill.notna().sum())
    no_cty = df.county_fips_code.isna() | (df.county_fips_code.str.strip() == "")
    log["rows_dropped_no_county"] = int(no_cty.sum())
    df = df.loc[~no_cty].reset_index(drop=True)

    # 5. liters: recompute everywhere from bottles x bottle size (source truncates 3 months)
    df["liters"] = df.sales_bottles * df.bottle_volume_ml / 1000
    clean_m = ~df.month.isin(TRUNCATED_MONTHS)
    agree = (df.loc[clean_m, "liters"] - df.loc[clean_m, "sales_liters"]).abs() < 0.011
    log["liters_recomputed_matches_published_in_clean_months"] = round(float(agree.mean()), 5)
    trunc = df.month.isin(TRUNCATED_MONTHS)
    log["liters_understated_by_source_in_truncated_months_pct"] = round(
        float(100 * (1 - df.loc[trunc, "sales_liters"].sum() / df.loc[trunc, "liters"].sum())), 2)

    # 6. dollars: rebuild truncated months from each item's unit price in the nearest clean month
    df["unit_price"] = np.where(df.sales_bottles != 0, df.sales_dollars / df.sales_bottles, np.nan)
    price = (df.loc[clean_m & (df.sales_bottles > 0)]
             .groupby(["month", "item_no"]).unit_price.median())
    df["dollars"] = df.sales_dollars.astype(float)
    df["dollars_rebuilt"] = 0
    for m, donor in PRICE_DONOR_MONTH.items():
        rows = df.month == m
        p = df.loc[rows, "item_no"].map(price.xs(donor, level="month"))
        ok = p.notna()
        idx = p.index[ok]
        df.loc[idx, "dollars"] = df.loc[idx, "sales_bottles"] * p[ok]
        df.loc[idx, "dollars_rebuilt"] = 1
    # validation of the method on a clean month: rebuild Oct-2025 dollars from Sep-2025 prices
    oct_ = df[(df.month == "2025-10")]
    p = oct_.item_no.map(price.xs("2025-09", level="month"))
    v = oct_[p.notna()]
    log["dollar_rebuild_validation_oct25_from_sep25_prices"] = {
        "rows_with_donor_price_pct": round(float(100 * p.notna().mean()), 2),
        "total_error_pct": round(float(100 * ((v.sales_bottles * p[p.notna()]).sum() / v.sales_dollars.sum() - 1)), 3)}
    rebuilt = df.dollars_rebuilt == 1
    log["dollars_rebuilt_rows"] = int(rebuilt.sum())
    log["dollars_source_understated_in_truncated_months_pct"] = round(
        float(100 * (1 - df.loc[rebuilt, "sales_dollars"].sum() / df.loc[rebuilt, "dollars"].sum())), 2)

    # 7. returns and derived fields
    df["is_return"] = (df.sales_bottles < 0).astype("int8")
    log["return_rows_kept_as_negative_volume"] = int(df.is_return.sum())

    # 8. multi-pack minis: for many <=200 ml items one recorded "bottle" is a sleeve, dispenser,
    #    carrier or party bucket of minis (e.g. '99 STRAWBERRIES MINI', 50 ml, $81 = $1,620/L). Dollars
    #    are right, liters are understated. The pack size is not recorded and estimating it from price
    #    is too noisy, so the rows are flagged, kept at the recorded liters (ABD's own convention,
    #    consistent over time) and excluded from price-per-liter metrics.
    small = df[(df.bottle_volume_ml <= 200) & (df.sales_bottles > 0)]
    item = small.groupby("item_no").agg(D=("dollars", "sum"), L=("liters", "sum"))
    suspect_items = set(item.index[(item.D / item.L) > 250])
    df["mini_multipack_suspect"] = df.item_no.isin(suspect_items).astype("int8")
    sus = df.mini_multipack_suspect == 1
    clean_small = small[~small.item_no.isin(suspect_items)]
    ref_ppl = clean_small.dollars.sum() / clean_small.liters.sum()
    log["mini_multipack_suspect"] = {
        "items": len(suspect_items), "rows": int(sus.sum()),
        "share_of_dollars_pct": round(float(100 * df.loc[sus, "dollars"].sum() / df.dollars.sum()), 2),
        "share_of_recorded_liters_pct": round(float(100 * df.loc[sus, "liters"].sum() / df.liters.sum()), 3),
        "reference_usd_per_liter_single_minis": round(float(ref_ppl), 2),
        "implied_missing_liters_pct_of_total": round(float(
            100 * (df.loc[sus, "dollars"].sum() / ref_ppl - df.loc[sus, "liters"].sum()) / df.liters.sum()), 2),
    }
    pos = (df.liters > 0) & (df.dollars > 0) & ~sus
    df["price_per_liter"] = np.where(pos, df.dollars / df.liters.where(pos), np.nan)
    df["family"] = df.category_name.map({c: spirits_family(c) for c in df.category_name.dropna().unique()})
    df["bottle_size_class"] = pd.cut(df.bottle_volume_ml, [0, 375, 999, 1500, 1e9],
                                     labels=["mini_<=375ml", "standard_750ml", "liter_1L", "handle_1.75L+"])
    log["family_other_share_of_liters_by_year"] = (
        df[df.family == "other"].groupby("year").liters.sum() / df.groupby("year").liters.sum()).round(4).to_dict()

    # columns dropped: sales_gallons duplicates liters and is also truncated
    df = df.drop(columns=["sales_gallons", "unit_price"])
    df = df.rename(columns={"sales_liters": "sales_liters_published", "sales_dollars": "sales_dollars_published"})
    return df


def build_store_dim(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """One row per store: type, market (city), first/last order month, entry classification."""
    g = df.sort_values("ordered_on").groupby("store_no", observed=True)
    s = pd.DataFrame({
        "store_name": g.store_name.last(),
        "store_address": g.store_address.last(),
        "store_city": g.store_city.agg(lambda x: x.mode().iat[0]),
        "store_zip_code": g.store_zip_code.agg(lambda x: x.mode().iat[0] if x.notna().any() else pd.NA),
        "county_fips_code": g.county_fips_code.agg(lambda x: x.mode().iat[0]),
        "first_month": g.month.min(),
        "last_month": g.month.max(),
        "active_months": g.month.nunique(),
        "liters_total": g.liters.sum(),
    })
    s["city_key"] = s.county_fips_code.astype(str) + "|" + _norm_city(s.store_city).astype(str)
    s["store_type"] = s.store_name.map(store_type)
    s["addr_key"] = (s.store_address.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
                     + "|" + s.city_key)
    all_months = sorted(df.month.unique())
    first, last = all_months[0], all_months[-1]
    m_idx = {m: i for i, m in enumerate(all_months)}
    s["first_idx"], s["last_idx"] = s.first_month.map(m_idx), s.last_month.map(m_idx)
    s["months_since_first"] = len(all_months) - 1 - s.first_idx
    # Entry: first order at least 3 months after the data starts (rules out sparse orderers seen in Jan-Mar).
    s["is_entrant"] = (s.first_idx >= 3).astype(int)
    s["is_exiter"] = (s.last_idx <= len(all_months) - 3).astype(int)
    # Ownership change: an entrant at the same street address as a store that stopped ordering
    # within 3 months either side of the entrant's first order. Not new competition.
    exiters = s[s.is_exiter == 1]
    s["entry_type"] = "incumbent"
    for sid, r in s[s.is_entrant == 1].iterrows():
        same = exiters[(exiters.addr_key == r.addr_key) & (exiters.index != sid)
                       & ((exiters.last_idx - r.first_idx).abs() <= 3)]
        s.loc[sid, "entry_type"] = "ownership_change" if len(same) else "new_store"
    s.loc[s.is_entrant == 0, "entry_type"] = np.where(s.loc[s.is_entrant == 0, "first_idx"] == 0, "incumbent", "early_entrant")
    log["stores_total"] = int(len(s))
    log["store_entry_types"] = s.entry_type.value_counts().to_dict()
    log["store_types"] = s.store_type.value_counts().to_dict()
    log["months_covered"] = f"{first}..{last} ({len(all_months)} months)"
    return s.drop(columns=["first_idx", "last_idx"]).reset_index()


def run() -> None:
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    log: dict = {}
    prof = build_county_profile()
    assert len(prof) == 99 and prof.county_fips_code.is_unique
    prof.to_parquet(C.COUNTY_PROFILE, index=False)
    log["county_profile"] = {"counties": len(prof), "columns": prof.shape[1]}
    print(f"county profile: {prof.shape}")

    df = read_sales(log)
    print(f"sales read: {len(df):,} rows")
    df = clean_sales(df, log)
    stores = build_store_dim(df, log)
    stores.to_parquet(C.STORE_DIM, index=False)
    print(f"stores: {len(stores):,}")

    # Integration: many-to-one join on 5-digit county FIPS, then store attributes on store_no
    n_before = len(df)
    # county_name already exists in the sales feed; take only the census/geography fields from the profile
    df = df.merge(prof.drop(columns="county_name"), on="county_fips_code", how="left",
                  validate="many_to_one", indicator=True)
    log["census_join_match_rate"] = round(float((df._merge == "both").mean()), 6)
    assert log["census_join_match_rate"] == 1.0, "every county FIPS must match the census profile"
    df = df.drop(columns="_merge")
    df = df.merge(stores[["store_no", "store_type", "city_key", "first_month", "entry_type"]]
                  .rename(columns={"first_month": "store_first_month"}),
                  on="store_no", how="left", validate="many_to_one")
    assert len(df) == n_before
    for c in df.columns:
        if df[c].dtype == "string" or df[c].dtype == object:
            df[c] = df[c].astype("category")
    df.to_parquet(C.SSOT, index=False)
    log["ssot"] = {"rows": int(len(df)), "columns": int(df.shape[1]),
                   "rows_by_year": {int(k): int(v) for k, v in df.year.value_counts().sort_index().items()},
                   "column_names": list(df.columns)}
    C.DQ_REPORT.write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps({k: v for k, v in log.items() if k != "ssot"}, indent=2, default=str))
    print(f"SSOT: {len(df):,} rows x {df.shape[1]} columns -> {C.SSOT.name}")
