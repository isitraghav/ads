"""Step 4: monitoring report - data drift (PSI), concept drift (monthly error), schema drift, retrain rule.
Output: reports/tables/monitoring.json (read by the API dashboard), drift_psi.csv, stale_model_decay.csv
Usage:  .venv/Scripts/python scripts/04_monitor.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from abd_demand import config as C
from abd_demand import forecast as Fc
from abd_demand import monitoring as M
from abd_demand.features import FEATURES, NUMERIC_FEATURES, load_ssot

TABLES = C.ROOT / "reports" / "tables"
panel = Fc.load_panel()
model, meta = Fc.load_model(C.MODELS / "xgb_tweedie_final.json")

# data drift: training window vs the latest three months
# reference = training months once 6-month rolling windows are full (Apr-Jun 2025 have <6 months of history)
ref = panel[(panel.month >= "2025-07") & (panel.month <= "2026-03")]
cur = panel[(panel.month >= "2026-08") & (panel.month <= "2026-10")]
drift = M.drift_table(ref, cur, [f for f in NUMERIC_FEATURES if f in meta["features"]])

# concept drift on the out-of-time test months
pred = pd.read_parquet(TABLES / "test_predictions.parquet")
monthly = M.monthly_error(pred, {"model": "p_final", "status_quo": "p_seasonal_naive_last_year"})

# stale-model experiment: the same ensemble, but its ML part trained on 2025 only, scored on every 2026 month
tr25 = panel[(panel.month >= "2025-04") & (panel.month <= "2025-12") & panel.y.notna()]
p = dict(meta["params"])
stale = Fc.xgb_model("reg:tweedie", **p, n_estimators=meta["n_estimators"], early_stopping_rounds=None)
FEATS = meta["features"]
stale.fit(tr25[FEATS], tr25.y, verbose=False)
y26 = panel[(panel.month >= "2026-01") & panel.y.notna()].copy()
y26["p_stale"] = Fc.blend(stale.predict(y26[FEATS]), Fc.seasonal_anchor(y26), meta["weight"])
y26["p_sn"] = Fc.BASELINES["seasonal_naive_last_year"](y26)
decay = M.monthly_error(y26, {"stale_2025_model": "p_stale", "status_quo": "p_sn"})
decay.to_csv(TABLES / "stale_model_decay.csv", index=False)

# schema drift in the raw feed
sales = load_ssot(["category_name", "year", "liters"])
sales["category_name"] = sales.category_name.astype(str)
schema = M.schema_drift(sales)
schema.to_csv(TABLES / "schema_drift.csv", index=False)

valid_wape = meta["valid_metrics"]["wape_county"]
recent = monthly.tail(1).iloc[0]
imp = pd.read_csv(TABLES / "xgb_gain_importance.csv", index_col=0).iloc[:, 0]
watch = [f for f in imp.index if f not in M.EXPECTED_DRIFT][:10]  # 10 most important, non-calendar
drift["watched"] = drift.feature.isin(watch)
drift["expected_by_construction"] = drift.feature.isin(M.EXPECTED_DRIFT)
decision = M.retrain_decision(drift[drift.watched], recent.model, recent.status_quo, valid_wape)
drift.to_csv(TABLES / "drift_psi.csv", index=False)
report = {
    "generated_for": "2026-11", "reference_window": "2025-07..2026-03", "current_window": "2026-08..2026-10",
    "drift": json.loads(drift.round(4).to_json(orient="records")),
    "monthly_error": json.loads(monthly.round(4).to_json(orient="records")),
    "stale_model_decay": json.loads(decay.round(4).to_json(orient="records")),
    "schema_drift": json.loads(schema.round(1).to_json(orient="records")),
    "validation_wape_county": float(valid_wape), "retrain": decision,
}
(TABLES / "monitoring.json").write_text(json.dumps(report, indent=2))
print(drift.head(12).round(3).to_string())
print(monthly.round(4).to_string())
print(decay.round(4).to_string())
print(schema.to_string())
print(decision)
