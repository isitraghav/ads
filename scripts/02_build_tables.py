"""Step 2: build the analysis tables from the SSOT.
Outputs (data/processed/): county_period.parquet, store_month.parquet, store_family_month.parquet"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from abd_demand import config as C, features as F

t0 = time.time()
sales = F.load_ssot()
profile = pd.read_parquet(C.COUNTY_PROFILE)
stores = pd.read_parquet(C.STORE_DIM)

cp = F.county_period_table(sales, profile)
cp.to_parquet(C.PROCESSED / "county_period.parquet", index=False)
print(f"county_period: {cp.shape}, premium cut-off ${cp.attrs['premium_cut']:.2f}/L (2025 liter-weighted top quartile)")
(C.PROCESSED / "premium_cut.txt").write_text(f"{cp.attrs['premium_cut']:.4f}")

sm = F.store_month(sales, stores)
sm.to_parquet(C.PROCESSED / "store_month.parquet", index=False)
print(f"store_month: {sm.shape}")

panel = F.build_store_panel(sales, stores, profile, last_month="2026-10", forecast_month="2026-11")
panel.to_parquet(C.STORE_PANEL, index=False)
print(f"store_family_month: {panel.shape} ({panel.y.notna().sum():,} rows with a known target)")
print(f"done in {time.time() - t0:.0f}s")
