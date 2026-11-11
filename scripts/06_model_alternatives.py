"""Step 6: "why XGBoost and not LightGBM / CatBoost / ...?" - answered with an experiment, not opinion.

Every alternative gets the SAME treatment as the XGBoost candidate in script 03:
  * same 38 features, same time split (train Apr25-Mar26, valid Apr-Jun26, test Jul-Oct26)
  * comparable default settings (learning rate 0.05, depth ~6, early stopping on validation)
  * Tweedie p = 1.4 where the library supports Tweedie
  * ensemble weight with the seasonal naive chosen on VALIDATION (lowest county WAPE, |bias| <= 3%)
  * refit on train + valid with 1.1x the early-stopped tree count, scored once on the test months
A seed check (XGBoost, 3 seeds) shows how big a difference has to be before it is more than noise.

Outputs
  reports/tables/model_alternatives.csv
Usage:  .venv/Scripts/python scripts/06_model_alternatives.py   (about 5 minutes)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

from abd_demand import config as C
from abd_demand import forecast as Fc

TABLES = C.ROOT / "reports" / "tables"
META = json.loads((C.MODELS / "xgb_tweedie_final.meta.json").read_text())
FEATS = META["features"]  # the 38 features of the deployed model
CATS = [f for f in FEATS if f in ("family", "store_type")]
P = 1.4  # Tweedie variance power of the deployed model

panel = Fc.load_panel()
train, valid, test = (Fc.split(panel, s) for s in ("train", "valid", "test"))
trval = pd.concat([train, valid])
print(f"rows  train {len(train):,}  valid {len(valid):,}  test {len(test):,}", flush=True)


def as_str(d):
    """CatBoost wants categorical columns as plain strings."""
    d = d[FEATS].copy()
    for c in CATS:
        d[c] = d[c].astype(str)
    return d


# Each candidate: build(n_trees, early_stop) -> model;  fit(model, train, valid_or_None);  predict(model, df);
# trees(model) -> early-stopped tree count.
def xgb_build(objective, seed=C.SEED):
    def build(n, es):
        extra = {"tweedie_variance_power": P} if objective == "reg:tweedie" else {}
        return Fc.xgb_model(objective, n_estimators=n, early_stopping_rounds=100 if es else None,
                            random_state=seed, **extra)
    return build


CANDIDATES = {
    "XGBoost (Tweedie) - the chosen family": dict(
        build=xgb_build("reg:tweedie"),
        fit=lambda m, a, v: m.fit(a[FEATS], a.y, **({"eval_set": [(v[FEATS], v.y)]} if v is not None else {}),
                                  verbose=False),
        predict=lambda m, d: m.predict(d[FEATS]),
        trees=lambda m: m.best_iteration + 1),
    "XGBoost (Poisson)": dict(
        build=xgb_build("count:poisson"),
        fit=lambda m, a, v: m.fit(a[FEATS], a.y, **({"eval_set": [(v[FEATS], v.y)]} if v is not None else {}),
                                  verbose=False),
        predict=lambda m, d: m.predict(d[FEATS]),
        trees=lambda m: m.best_iteration + 1),
    "LightGBM (Tweedie)": dict(
        build=lambda n, es: lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=P, n_estimators=n, learning_rate=0.05, num_leaves=63,
            max_depth=-1, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=C.SEED, n_jobs=-1, verbose=-1),
        fit=lambda m, a, v: m.fit(a[FEATS], a.y, categorical_feature=CATS,
                                  **({"eval_set": [(v[FEATS], v.y)],
                                      "callbacks": [lgb.early_stopping(100, verbose=False)]} if v is not None else {})),
        predict=lambda m, d: m.predict(d[FEATS]),
        trees=lambda m: m.best_iteration_),
    "CatBoost (Tweedie)": dict(
        build=lambda n, es: cb.CatBoostRegressor(
            loss_function=f"Tweedie:variance_power={P}", iterations=n, learning_rate=0.05, depth=6,
            random_seed=C.SEED, thread_count=-1, verbose=False, allow_writing_files=False, cat_features=CATS,
            **({"early_stopping_rounds": 100} if es else {})),
        fit=lambda m, a, v: m.fit(as_str(a), a.y, **({"eval_set": (as_str(v), v.y)} if v is not None else {})),
        predict=lambda m, d: m.predict(as_str(d), prediction_type="Exponent"),
        trees=lambda m: m.get_best_iteration() + 1),
    "sklearn HistGradientBoosting (Poisson)": dict(
        build=lambda n, es: HistGradientBoostingRegressor(
            loss="poisson", max_iter=n, learning_rate=0.05, max_leaf_nodes=63, categorical_features="from_dtype",
            early_stopping=es, n_iter_no_change=100, validation_fraction=0.1, random_state=C.SEED),
        fit=lambda m, a, v: m.fit(a[FEATS], a.y),  # sklearn early-stops on an internal random 10% (not time-based)
        predict=lambda m, d: m.predict(d[FEATS]),
        trees=lambda m: m.n_iter_),
}

rows = []
for name, c in CANDIDATES.items():
    t0 = time.time()
    m = c["build"](3000, True)
    c["fit"](m, train, valid)
    fit_s = time.time() - t0
    pv = np.clip(c["predict"](m, valid), 0, None)
    alone_v = Fc.evaluate(valid.assign(p=pv), "p")
    w, blend_v = Fc.choose_weight(valid, pv)
    n_trees = int(c["trees"](m) * 1.1) + 1

    m2 = c["build"](n_trees, False)  # refit on train + valid, then score the untouched test months once
    c["fit"](m2, trval, None)
    pt = np.clip(c["predict"](m2, test), 0, None)
    alone_t = Fc.evaluate(test.assign(p=pt), "p")
    blend_t = Fc.evaluate(test.assign(p=Fc.blend(pt, Fc.seasonal_anchor(test), w)), "p")
    rows.append(dict(model=name, trees=n_trees, fit_seconds=round(fit_s, 1), weight=w,
                     valid_alone_county=alone_v["wape_county"], valid_alone_bias=alone_v["bias"],
                     valid_ens_county=blend_v["wape_county"], valid_ens_bias=blend_v["bias"],
                     test_alone_store=alone_t["wape_store"], test_alone_county=alone_t["wape_county"],
                     test_alone_bias=alone_t["bias"],
                     test_ens_store=blend_t["wape_store"], test_ens_county=blend_t["wape_county"],
                     test_ens_state=blend_t["wape_state"], test_ens_bias=blend_t["bias"]))
    print(f"{name:42s} trees {n_trees:5d}  fit {fit_s:6.0f}s  w={w:.2f}  valid ens {blend_v['wape_county']:.4f}"
          f"  test ens county {blend_t['wape_county']:.4f} state {blend_t['wape_state']:.4f}"
          f" bias {blend_t['bias']:+.3f}", flush=True)

# seed noise: the same XGBoost model with three different random seeds (row/column subsampling changes)
for seed in (1, 2, 3):
    m = xgb_build("reg:tweedie", seed)(3000, True)
    m.fit(train[FEATS], train.y, eval_set=[(valid[FEATS], valid.y)], verbose=False)
    w, ev = Fc.choose_weight(valid, m.predict(valid[FEATS]))
    rows.append(dict(model=f"seed check: XGBoost (Tweedie) seed {seed}", trees=m.best_iteration + 1, weight=w,
                     valid_ens_county=ev["wape_county"], valid_ens_bias=ev["bias"]))
    print(f"seed {seed}: valid ens county {ev['wape_county']:.4f}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(TABLES / "model_alternatives.csv", index=False)
print(out.round(4).to_string(index=False))
