"""Paths and constants shared by every script, notebook and the API."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"

SALES_2025_GLOB = "iowa_liquor_sales_2025/iowa_liquor_sales_2025_1262_rows_part_*.csv"
SALES_2026_GLOB = "iowa_liquor_sales_2026/iowa_liquor_sales_2026_1263_rows_part_*.csv"
# Row counts published by the Iowa Data Hub API (numRows), used as a completeness check.
PUBLISHED_ROWS = {1262: 2_490_620, 1263: 2_029_764}

ACS_DIR = RAW / "census_acs"
ADJACENCY = RAW / "census_geo" / "county_adjacency2024.txt"
GAZETTEER = RAW / "census_geo" / "2024_Gaz_counties_national.txt"
TAX = RAW / "state_tax" / "tax_foundation_spirits_excise_2026.csv"

# Outputs
SSOT = PROCESSED / "sales_ssot.parquet"                 # transaction-level integrated dataset
COUNTY_PROFILE = PROCESSED / "county_profile.parquet"   # 99 counties: census + geography + tax
STORE_DIM = PROCESSED / "store_dim.parquet"             # one row per store
STORE_PANEL = PROCESSED / "store_family_month.parquet"  # store x family x month (modelling table)
DQ_REPORT = PROCESSED / "data_quality_report.json"

STATE_FIPS = {"19": "IA", "29": "MO", "17": "IL", "27": "MN", "31": "NE", "46": "SD", "55": "WI"}
# Latitude of Iowa's straight southern border with Missouri (the 40°35' parallel, ~40.58 N).
MO_BORDER_LAT = 40.58
MILES_PER_DEG_LAT = 69.0

FAMILIES = ["vodka", "whiskey", "liqueur", "rum", "agave", "rtd", "brandy", "gin", "other"]
SEED = 2026
