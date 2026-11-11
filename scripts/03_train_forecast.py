"""Step 3: train, compare, tune and evaluate the next-month forecast; log everything to MLflow.

Final solution = ensemble of a tuned Tweedie XGBoost model and the calendar-adjusted seasonal naive,
with the ensemble weight and the prediction intervals both fitted on validation data only.

Outputs
  reports/tables/model_comparison.csv      every candidate (validation; test for the finalists)
  reports/tables/hpo_trials.csv            random-search trials
  reports/tables/test_predictions.parquet  out-of-time predictions (final ensemble, components, baselines)
  reports/tables/interval_calibration.csv  P10-P90 coverage on test, store and county level
  models/xgb_tweedie_final.json (+ .meta.json: features, weight, interval tables, metrics)
  data/processed/forecast_2026_11.parquet  next-month forecast served by the API
Usage:  .venv/Scripts/python scripts/03_train_forecast.py   (about 15-20 minutes)
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mlflow
import numpy as np
import pandas as pd

from abd_demand import config as C
from abd_demand import forecast as Fc
from abd_demand.features import FEATURES

T0 = time.time()
TABLES = C.ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{(C.ROOT / 'mlflow.db').as_posix()}")
mlflow.set_experiment("abd_next_month_forecast")

panel = Fc.load_panel()
train, valid, test = (Fc.split(panel, s) for s in ("train", "valid", "test"))
print(f"rows  train {len(train):,}  valid {len(valid):,}  test {len(test):,}", flush=True)
rows = []


def log_run(name, metrics_by_split, params=None, kind=""):
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("kind", kind)
        if params:
            mlflow.log_params(params)
        for split_name, m in metrics_by_split.items():
            mlflow.log_metrics({f"{split_name}_{k}": v for k, v in m.items()})
    for split_name, m in metrics_by_split.items():
        rows.append(dict(model=name, kind=kind, split=split_name, **m))


# 1. baselines ---------------------------------------------------------------------------------
for name, f in Fc.BASELINES.items():
    log_run(name, {"valid": Fc.evaluate(valid.assign(p=f(valid)), "p"),
                   "test": Fc.evaluate(test.assign(p=f(test)), "p")}, kind="baseline")
print("baselines done", flush=True)

# 2. candidate models (default settings), alone and blended with the seasonal anchor ---------------
for name in ("ridge_log", "random_forest", "xgb_squared_log", "xgb_tweedie"):
    t1 = time.time()
    _, (pv,) = Fc.fit_predict(name, train, valid, [valid])
    w, ev_blend = Fc.choose_weight(valid, pv)
    log_run(name, {"valid": Fc.evaluate(valid.assign(p=pv), "p")}, kind="candidate")
    log_run(f"{name}+seasonal (w={w})", {"valid": ev_blend}, params={"w": w}, kind="candidate_ensemble")
    print(f"{name}: alone {rows[-2]['wape_county']:.4f} | ensemble w={w} {ev_blend['wape_county']:.4f} "
          f"({time.time() - t1:.0f}s)", flush=True)

# 3. hyperparameter search (random search, 20 trials) for the ensemble's ML component --------------
rng = np.random.default_rng(C.SEED)
trials = []
for i, params in enumerate(Fc.sample_params(rng, 20)):
    m, (pv,) = Fc.fit_predict("xgb_tweedie", train, valid, [valid], **params)
    w, ev = Fc.choose_weight(valid, pv)
    alone = Fc.evaluate(valid.assign(p=pv), "p")
    trials.append(dict(trial=i, best_iteration=m.best_iteration, w=w, **params, **ev,
                       alone_wape_county=alone["wape_county"], alone_bias=alone["bias"]))
    with mlflow.start_run(run_name=f"hpo_{i:02d}"):
        mlflow.set_tag("kind", "hpo")
        mlflow.log_params(params | {"w": w})
        mlflow.log_metrics({f"valid_{k}": v for k, v in ev.items()} | {"best_iteration": m.best_iteration})
    print(f"  trial {i:02d}: ensemble county WAPE {ev['wape_county']:.4f} (w={w}, bias {ev['bias']:+.3f})", flush=True)
hpo = pd.DataFrame(trials).sort_values("wape_county")
hpo.to_csv(TABLES / "hpo_trials.csv", index=False)
top = hpo.iloc[0]
best = {k: (int(top[k]) if k in ("max_depth", "min_child_weight") else float(top[k])) for k in Fc.SEARCH_SPACE}
W = float(top.w)
n_trees = int(top.best_iteration * 1.1) + 1
print("best params:", best, "trees:", n_trees, "weight:", W, flush=True)

# 3b. responsible-AI feature decision, made on VALIDATION (never on test): keep the ethnicity feature
#     only if it improves the validation ensemble's county WAPE by more than 1% relative.
SENSITIVE = ["cty_pct_hispanic"]
with_eth = Fc.choose_weight(valid, Fc.fit_predict("xgb_tweedie", train, valid, [valid], **best)[1][0])[1]["wape_county"]
feats_wo = [f for f in FEATURES if f not in SENSITIVE]
m_wo = Fc.xgb_model("reg:tweedie", **best, n_estimators=2000)
m_wo.fit(train[feats_wo], train.y, eval_set=[(valid[feats_wo], valid.y)], verbose=False)
without_eth = Fc.choose_weight(valid, m_wo.predict(valid[feats_wo]))[1]["wape_county"]
KEEP_SENSITIVE = with_eth < 0.99 * without_eth
FEATS = FEATURES if KEEP_SENSITIVE else feats_wo
log_run("feature decision: with ethnicity", {"valid": {"wape_county": with_eth}}, kind="responsible_ai")
log_run("feature decision: without ethnicity", {"valid": {"wape_county": without_eth}}, kind="responsible_ai")
print(f"valid county WAPE with % Hispanic {with_eth:.4f} vs without {without_eth:.4f} -> "
      f"{'keep' if KEEP_SENSITIVE else 'DROP'} the ethnicity feature", flush=True)

# 4. intervals: fitted on validation predictions of the train-only tuned model ----------------------
m_tr, (pv,) = Fc.fit_predict("xgb_tweedie", train, valid, [valid], features=FEATS, **best)
valid_ens = valid.assign(p=Fc.blend(pv, Fc.seasonal_anchor(valid), W))
iv_store = Fc.fit_intervals(valid_ens, "p", None)
iv_county = Fc.fit_intervals(valid_ens, "p", ["county_fips_code", "family", "month"])

# 5. refit on train + valid, evaluate ONCE on the untouched test months ----------------------------
trval = pd.concat([train, valid])
final = Fc.xgb_model("reg:tweedie", **best, n_estimators=n_trees, early_stopping_rounds=None)
final.fit(trval[FEATS], trval.y, verbose=False)
tp = test.copy()
tp["p_xgb"] = final.predict(test[FEATS])
tp["p_anchor"] = Fc.seasonal_anchor(test)
tp["p_final"] = Fc.blend(tp.p_xgb, tp.p_anchor, W)
for name, f in Fc.BASELINES.items():
    tp[f"p_{name}"] = f(test)
_, (tp["p_ridge"],) = Fc.fit_predict("ridge_log", trval, valid, [test])
_, (tp["p_rf"],) = Fc.fit_predict("random_forest", trval, valid, [test])
ev_test = Fc.evaluate(tp, "p_final")
log_run("FINAL ensemble (tuned xgb + seasonal)", {"valid": {k: float(top[k]) for k in ev_test}, "test": ev_test},
        params=best | {"n_estimators": n_trees, "w": W}, kind="final")
for col, label in (("p_xgb", "tuned xgb alone"), ("p_ridge", "ridge_log"), ("p_rf", "random_forest")):
    log_run(f"{label} (test)", {"test": Fc.evaluate(tp, col)}, kind="final_comparison")
print("test (final):", {k: round(v, 4) for k, v in ev_test.items()}, flush=True)

# interval coverage on test (store x family and county x family)
tp["p10"], tp["p90"] = (Fc.apply_intervals(tp.p_final, iv_store, q) for q in (0.1, 0.9))
cty = tp.groupby(["county_fips_code", "family", "month"], observed=True)[["y", "p_final"]].sum()
cal = {"store_coverage_p10_p90": Fc.coverage(tp.y, tp.p10, tp.p90),
       "county_coverage_p10_p90": Fc.coverage(cty.y, Fc.apply_intervals(cty.p_final, iv_county, 0.1),
                                              Fc.apply_intervals(cty.p_final, iv_county, 0.9)),
       "county_coverage_below_p90": float((cty.y <= Fc.apply_intervals(cty.p_final, iv_county, 0.9)).mean()),
       "target": 0.8}
pd.Series(cal).to_csv(TABLES / "interval_calibration.csv")
with mlflow.start_run(run_name="interval_calibration"):
    mlflow.set_tag("kind", "intervals")
    mlflow.log_metrics(cal)
print("interval coverage:", {k: round(v, 3) for k, v in cal.items()}, flush=True)

# 6. ablations for responsible AI: does the model need the ethnicity / any census feature? ---------
county_feats = ["log_population", "cty_median_household_income", "cty_pct_hispanic", "cty_pct_bachelors_or_higher",
                "cty_median_age", "cty_poverty_rate", "cty_pct_age_21_34", "geo_pop_density", "geo_borders_mo",
                "geo_border_tax_gap", "geo_log_hub_ratio"]
for tag, drop in (("all_features", []), ("no_ethnicity", ["cty_pct_hispanic"]), ("no_county_features", county_feats)):
    feats = [f for f in FEATURES if f not in drop]  # ablations relative to the full set
    m = Fc.xgb_model("reg:tweedie", **best, n_estimators=n_trees, early_stopping_rounds=None)
    m.fit(trval[feats], trval.y, verbose=False)
    tp[f"p_{tag}"] = Fc.blend(m.predict(test[feats]), tp.p_anchor, W)
    log_run(f"ablation: {tag}", {"test": Fc.evaluate(tp, f"p_{tag}")}, kind="ablation")
    print(f"ablation {tag}: test county WAPE {rows[-1]['wape_county']:.4f}", flush=True)

# 7. save results ---------------------------------------------------------------------------------
pd.DataFrame(rows).to_csv(TABLES / "model_comparison.csv", index=False)
keep = ["store_no", "family", "month", "county_fips_code", "store_type", "store_age_months", "y", "lag12",
        "roll6_mean", "geo_borders_mo", "cty_pct_hispanic", "log_population", "cty_median_household_income"]
tp[keep + [c for c in tp.columns if c.startswith("p")]].to_parquet(TABLES / "test_predictions.parquet", index=False)
pd.Series(final.get_booster().get_score(importance_type="gain")).sort_values(ascending=False).to_csv(
    TABLES / "xgb_gain_importance.csv")

# 8. production refit on every known month, forecast November 2026 --------------------------------
known = panel[panel.y.notna() & (panel.month >= Fc.SPLITS["train"][0])]
prod = Fc.xgb_model("reg:tweedie", **best, n_estimators=n_trees, early_stopping_rounds=None)
prod.fit(known[FEATS], known.y, verbose=False)
meta = dict(features=FEATS, dropped_sensitive=[] if KEEP_SENSITIVE else SENSITIVE,
            sensitive_feature_decision={"valid_wape_with": with_eth, "valid_wape_without": without_eth}, categories=Fc.CATEGORY_LEVELS, params=best, n_estimators=n_trees, weight=W,
            trained_on=f"{Fc.SPLITS['train'][0]}..{known.month.max()}", rows=len(known), test_metrics=ev_test,
            valid_metrics={k: float(top[k]) for k in ev_test}, interval_store=iv_store,
            interval_county=iv_county, interval_calibration=cal)
Fc.save_model(prod, C.MODELS / "xgb_tweedie_final.json", meta)
nxt = panel[panel.month == "2026-11"].copy()
nxt["p_xgb"] = prod.predict(nxt[FEATS])
nxt["p_anchor"] = Fc.seasonal_anchor(nxt)
nxt["forecast_liters"] = Fc.blend(nxt.p_xgb, nxt.p_anchor, W)
nxt["p10"], nxt["p90"] = (Fc.apply_intervals(nxt.forecast_liters, iv_store, q) for q in (0.1, 0.9))
nxt.to_parquet(C.PROCESSED / "forecast_2026_11.parquet", index=False)
with mlflow.start_run(run_name="production_model_2026_11"):
    mlflow.set_tag("kind", "production")
    mlflow.log_params(best | {"n_estimators": n_trees, "w": W, "trained_on": meta["trained_on"]})
    mlflow.log_metrics({f"test_{k}": v for k, v in ev_test.items()})
    mlflow.log_artifact(str(C.MODELS / "xgb_tweedie_final.json"))
    mlflow.log_artifact(str(C.MODELS / "xgb_tweedie_final.meta.json"))
print(f"Nov-2026 forecast: {nxt.forecast_liters.sum():,.0f} liters statewide; done in {time.time() - T0:.0f}s")
