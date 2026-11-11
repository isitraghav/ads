"""Next-month demand forecast: store x spirits family x month.

Course units: advanced feature engineering, ensemble learning, gradient boosting, hyperparameter
optimisation, model calibration (prediction-interval coverage), MLflow tracking.

Time-based split (no shuffling, so the model never sees the future):
  train  target months 2025-04 .. 2026-03   (lag-3 features exist from April 2025)
  valid  2026-04 .. 2026-06                 model selection, hyperparameter search, interval calibration
  test   2026-07 .. 2026-10                 untouched until the final evaluation
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as C
from .features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

SPLITS = {"train": ("2025-04", "2026-03"), "valid": ("2026-04", "2026-06"), "test": ("2026-07", "2026-10")}
CATEGORY_LEVELS = {
    "family": C.FAMILIES,
    "store_type": ["big_box_club", "convenience_chain", "convenience_other", "grocery", "liquor_tobacco",
                   "other", "pharmacy"],
}


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(C.STORE_PANEL)
    for c, levels in CATEGORY_LEVELS.items():
        p[c] = pd.Categorical(p[c].astype(str), categories=levels)
    return p


def split(p: pd.DataFrame, name: str) -> pd.DataFrame:
    a, b = SPLITS[name]
    return p[(p.month >= a) & (p.month <= b) & p.y.notna()]


# ---------------------------------------------------------------------------------------- metrics
def wape(y, yhat) -> float:
    """Weighted absolute percentage error = sum|error| / sum(actual). Volume-weighted, defined with
    zeros, and reads directly as 'share of liters mis-planned'."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / y.sum())


def bias(y, yhat) -> float:
    """Positive = over-forecast (overstock risk); negative = under-forecast (stock-out risk)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float((yhat - y).sum() / y.sum())


def evaluate(df: pd.DataFrame, pred_col: str) -> dict:
    """Errors at the three levels ABD plans at: store x family (delivery), county x family
    (regional capacity) and state x family (warehouse procurement)."""
    out = {"wape_store": wape(df.y, df[pred_col]), "bias": bias(df.y, df[pred_col]),
           "mae_store": float(np.abs(df.y - df[pred_col]).mean())}
    for lvl, keys in (("county", ["county_fips_code", "family", "month"]), ("state", ["family", "month"])):
        a = df.groupby(keys, observed=True)[["y", pred_col]].sum()
        out[f"wape_{lvl}"] = wape(a.y, a[pred_col])
    return out


# ---------------------------------------------------------------------------------------- baselines
BASELINES = {
    "naive_last_month": lambda d: d.lag1.fillna(0),
    "seasonal_naive_last_year": lambda d: d.lag12.fillna(d.roll3_mean).fillna(0),  # status quo
    "moving_avg_3m": lambda d: d.roll3_mean.fillna(0),
}


def seasonal_anchor(d: pd.DataFrame) -> pd.Series:
    """Same month last year rescaled to this month's order days; falls back to the 3-month mean for
    series without a year of history. This is the seasonal half of the ensemble."""
    return d.lag12_cal_adj.fillna(d.lag12).fillna(d.roll3_mean).fillna(0)


BASELINES["seasonal_naive_calendar_adj"] = seasonal_anchor


def blend(ml: np.ndarray, anchor: np.ndarray, w: float) -> np.ndarray:
    """Ensemble: w * gradient-boosting forecast + (1 - w) * calendar-adjusted seasonal naive.
    With one seasonal cycle of history the model under-reacts to seasonal swings (bias < 0) while the
    seasonal naive carries last year's level (bias > 0); averaging cancels part of both errors."""
    return w * np.asarray(ml) + (1 - w) * np.asarray(anchor)


def choose_weight(d: pd.DataFrame, ml: np.ndarray, max_abs_bias: float = 0.03,
                  grid=np.round(np.arange(0, 1.01, 0.05), 2)) -> tuple[float, dict]:
    """Pick the ensemble weight on validation: lowest county x family WAPE among weights whose
    absolute bias is within the pre-registered +/-3%."""
    anchor = seasonal_anchor(d).to_numpy()
    best = None
    for w in grid:
        ev = evaluate(d.assign(_b=blend(ml, anchor, w)), "_b")
        ok = abs(ev["bias"]) <= max_abs_bias
        key = (not ok, ev["wape_county"])
        if best is None or key < best[0]:
            best = (key, float(w), ev)
    return best[1], best[2]


# ---------------------------------------------------------------------------------------- intervals
INTERVAL_QS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]


def fit_intervals(d: pd.DataFrame, pred_col: str, keys: list[str] | None, min_obs: int = 30) -> dict:
    """Empirical (calibrated) predictive distribution: quantiles of actual / forecast, by forecast-size
    band, measured on validation data the model was not trained on. keys=None -> store x family
    level; otherwise forecasts and actuals are first summed over `keys` (e.g. county x family x month).
    Bands are log-spaced (1, 3, 10, 30 ... liters) because relative error shrinks with volume: a band
    with fewer than `min_obs` observations is merged into the band below it."""
    a = d if keys is None else d.groupby(keys, observed=True)[["y", pred_col]].sum().reset_index()
    a = a[a[pred_col] > 0]
    cuts = [c for c in (1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000) if c < a[pred_col].max()]
    edges = [0.0] + cuts + [np.inf]
    while True:  # merge sparse bands (from the top down)
        counts = np.histogram(a[pred_col], edges)[0]
        sparse = [i for i, c in enumerate(counts) if c < min_obs]
        if not sparse or len(edges) <= 2:
            break
        i = sparse[-1]
        del edges[i if i > 0 else 1]  # drop the lower edge of a sparse band (the upper edge for band 0)
    edges = np.array(edges, float)
    band = np.digitize(a[pred_col], edges[1:-1])
    ratio = (a.y / a[pred_col]).to_numpy()
    table = {int(b): np.quantile(ratio[band == b], INTERVAL_QS).tolist() for b in np.unique(band)}
    return {"edges": edges.tolist(), "qs": INTERVAL_QS, "ratio_quantiles": table}


def apply_intervals(pred: np.ndarray, tab: dict, q: float) -> np.ndarray:
    pred = np.asarray(pred, float)
    edges = np.array(tab["edges"])
    band = np.digitize(pred, edges[1:-1])
    out = np.empty_like(pred)
    for b, quants in tab["ratio_quantiles"].items():
        m = band == int(b)
        out[m] = pred[m] * np.interp(q, tab["qs"], quants)
    return np.clip(out, 0, None)


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    return float(((y >= lo) & (y <= hi)).mean())


# ---------------------------------------------------------------------------------------- models
VOLUME_FEATURES = ["lag1", "lag2", "lag3", "lag6", "lag12", "roll3_mean", "roll6_mean", "roll6_std",
                   "lines_lag1", "store_lag1", "store_roll3", "store_lag12", "state_fam_lag1"]


def ridge_model(alpha=10.0):
    """Linear benchmark. Volumes are right-skewed over 4 orders of magnitude, so they enter as
    log1p (the model is linear in log space, like its log1p target); missing lags become 0 plus a
    missing-indicator column."""
    other = [f for f in NUMERIC_FEATURES if f not in VOLUME_FEATURES]
    log_vol = make_pipeline(FunctionTransformer(lambda X: np.log1p(np.clip(X, 0, None))),
                            SimpleImputer(strategy="constant", fill_value=0, add_indicator=True), StandardScaler())
    pre = ColumnTransformer([
        ("vol", log_vol, VOLUME_FEATURES),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), other),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return make_pipeline(pre, Ridge(alpha=alpha))


def rf_model(seed=C.SEED):
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),  # sklearn forests accept NaN since 1.4
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    return make_pipeline(pre, RandomForestRegressor(n_estimators=200, min_samples_leaf=20, max_features=0.4,
                                                    n_jobs=-1, random_state=seed))


XGB_BASE = dict(tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, n_jobs=-1, random_state=C.SEED)


def xgb_model(objective="reg:tweedie", **params):
    p = dict(XGB_BASE, objective=objective, n_estimators=2000, learning_rate=0.05, max_depth=6,
             min_child_weight=10, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=100,
             eval_metric="mae" if objective != "reg:tweedie" else "tweedie-nloglik@1.3")
    if objective == "reg:tweedie":
        p["tweedie_variance_power"] = params.pop("tweedie_variance_power", 1.3)
        p["eval_metric"] = f"tweedie-nloglik@{p['tweedie_variance_power']}"
    p.update(params)
    return xgb.XGBRegressor(**p)


def fit_predict(name, train, valid, apply_to: list[pd.DataFrame], features: list[str] | None = None, **params):
    """Fit a candidate on `train` (early stopping on `valid` for XGBoost) and predict each frame.
    `features` (XGBoost only) overrides the default feature list."""
    F_ = features or FEATURES
    X, y = train[FEATURES], train.y
    if name == "ridge_log":
        m = ridge_model(**params).fit(X, np.log1p(y))
        cap = np.log1p(y.max())  # never predict beyond the largest volume seen in training
        return m, [np.expm1(np.clip(m.predict(d[FEATURES]), 0, cap)) for d in apply_to]
    if name == "random_forest":
        m = rf_model().fit(X, y)
        return m, [np.clip(m.predict(d[FEATURES]), 0, None) for d in apply_to]
    if name == "xgb_squared_log":  # least squares on log1p(liters)
        m = xgb_model("reg:squarederror", **params)
        m.fit(train[F_], np.log1p(y), eval_set=[(valid[F_], np.log1p(valid.y))], verbose=False)
        return m, [np.clip(np.expm1(m.predict(d[F_])), 0, None) for d in apply_to]
    if name == "xgb_tweedie":  # Tweedie: zero-inflated, right-skewed, non-negative volumes
        m = xgb_model("reg:tweedie", **params)
        m.fit(train[F_], y, eval_set=[(valid[F_], valid.y)], verbose=False)
        return m, [m.predict(d[F_]) for d in apply_to]
    raise ValueError(name)


SEARCH_SPACE = {
    "max_depth": [4, 5, 6, 7, 8, 10],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "min_child_weight": [1, 5, 10, 20, 50],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0, 10.0],
    "tweedie_variance_power": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
}


def sample_params(rng, n):
    return [{k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()} for _ in range(n)]


def save_model(model: xgb.XGBRegressor, path, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))


def load_model(path) -> tuple[xgb.XGBRegressor, dict]:
    m = xgb.XGBRegressor()
    m.load_model(path)
    return m, json.loads(path.with_suffix(".meta.json").read_text())
