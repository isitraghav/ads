"""ABD Spirits Supply Planner - FastAPI service + dashboard (course unit: API development, deployment).

Run locally:   .venv/Scripts/python -m uvicorn app.main:app --port 8000
Docker:        docker build -t abd-planner . && docker run -p 8000:8000 abd-planner
Open:          http://localhost:8000  (dashboard)   http://localhost:8000/docs  (API docs)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abd_demand import config as C  # noqa: E402
from abd_demand.forecast import CATEGORY_LEVELS, apply_intervals, blend, load_model, seasonal_anchor  # noqa: E402

MODEL, META = load_model(C.MODELS / "xgb_tweedie_final.json")
FEATURES = META["features"]
FC = pd.read_parquet(C.PROCESSED / "forecast_2026_11.parquet")
for c, levels in CATEGORY_LEVELS.items():
    FC[c] = pd.Categorical(FC[c].astype(str), categories=levels)
STORES = pd.read_parquet(C.STORE_DIM).set_index("store_no")
COUNTIES = pd.read_parquet(C.COUNTY_PROFILE).set_index("county_fips_code")
MONITOR_PATH = C.ROOT / "reports" / "tables" / "monitoring.json"
FORECAST_MONTH = "2026-11"

app = FastAPI(title="ABD Spirits Supply Planner", version="1.0.0",
              description="Next-month spirits demand per Iowa store and county, with uncertainty, "
                          "service-level planning, explanations and model monitoring.")


def _plan(forecast, level: str, service_level: float) -> np.ndarray:
    """Newsvendor planning quantity: the service-level quantile of the calibrated predictive
    distribution (forecast x empirical quantile of actual/forecast measured on validation data).
    service_level = stock-out cost / (stock-out cost + holding cost)."""
    tab = META["interval_store"] if level == "store" else META["interval_county"]
    return apply_intervals(np.atleast_1d(np.asarray(forecast, float)), tab, service_level)


def _family_table(d: pd.DataFrame, service_level: float) -> list[dict]:
    # County intervals come from county-level errors (which partly cancel across stores), not from
    # adding up store intervals - that is the risk-pooling effect in safety stock.
    g = d.groupby("family", observed=True).agg(forecast=("forecast_liters", "sum"),
                                               last_year=("lag12", "sum")).reset_index()
    g["p10"] = _plan(g.forecast, "county", 0.1)
    g["p90"] = _plan(g.forecast, "county", 0.9)
    g["plan_quantity"] = _plan(g.forecast, "county", service_level)
    g["safety_stock"] = g.plan_quantity - g.forecast
    g = g.sort_values("forecast", ascending=False)
    return json.loads(g.round(1).to_json(orient="records"))


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": "xgb_tweedie_final", "trained_on": META["trained_on"],
            "forecast_month": FORECAST_MONTH, "rows_scored": int(len(FC))}


@app.get("/counties")
def counties():
    tot = FC.groupby("county_fips_code").forecast_liters.sum()
    c = COUNTIES[["county_name", "cty_population_21_plus", "geo_borders_mo"]].join(tot.rename("forecast_liters"))
    return json.loads(c.reset_index().sort_values("county_name").round(1).to_json(orient="records"))


@app.get("/forecast/county/{fips}")
def county_forecast(fips: str, service_level: float = Query(0.9, ge=0.5, le=0.995)):
    d = FC[FC.county_fips_code == fips]
    if d.empty:
        raise HTTPException(404, f"no stores for county {fips}")
    stores = (d.groupby("store_no").agg(forecast=("forecast_liters", "sum"), last_year=("lag12", "sum"))
              .join(STORES[["store_name", "store_city", "store_type", "entry_type"]]).reset_index()
              .sort_values("forecast", ascending=False))
    prof = COUNTIES.loc[fips]
    return {
        "county": prof.county_name, "fips": fips, "month": FORECAST_MONTH, "service_level": service_level,
        "context": {"adults_21_plus": int(prof.cty_population_21_plus), "borders_missouri": bool(prof.geo_borders_mo),
                    "pct_hispanic": float(prof.cty_pct_hispanic)},
        "families": _family_table(d, service_level),
        "total_forecast": round(float(d.forecast_liters.sum()), 1),
        "total_last_year": round(float(d.lag12.sum()), 1),
        "stores": json.loads(stores.round(1).to_json(orient="records")),
    }


@app.get("/forecast/store/{store_no}")
def store_forecast(store_no: str, service_level: float = Query(0.9, ge=0.5, le=0.995), top_k: int = 5):
    d = FC[FC.store_no == store_no]
    if d.empty:
        raise HTTPException(404, f"unknown store {store_no}")
    # XGBoost's built-in TreeSHAP: per-feature contributions on the log scale of the Tweedie model,
    # so exp(contribution) is a multiplicative effect on the forecast.
    contrib = MODEL.get_booster().predict(xgb.DMatrix(d[FEATURES], enable_categorical=True), pred_contribs=True)
    fams = []
    for i, (_, r) in enumerate(d.iterrows()):
        c = pd.Series(contrib[i, :-1], index=FEATURES)
        top = c.reindex(c.abs().sort_values(ascending=False).index[:top_k])
        fams.append({
            "family": r.family, "forecast": round(float(r.forecast_liters), 1),
            "p10": round(float(r.p10), 1), "p90": round(float(r.p90), 1),
            "plan_quantity": round(float(_plan(r.forecast_liters, "store", service_level)[0]), 1),
            "ml_component": round(float(r.p_xgb), 1), "seasonal_component": round(float(r.p_anchor), 1),
            "last_month": None if pd.isna(r.lag1) else round(float(r.lag1), 1),
            "last_year": None if pd.isna(r.lag12) else round(float(r.lag12), 1),
            "drivers": [{"feature": k, "value": None if pd.isna(r[k]) else (str(r[k]) if not np.isreal(r[k]) else round(float(r[k]), 3)),
                         "effect_pct": round(100 * (np.exp(v) - 1), 1)} for k, v in top.items()],
        })
    s = STORES.loc[store_no]
    return {"store_no": store_no, "store_name": s.store_name, "city": s.store_city, "store_type": s.store_type,
            "entry_type": s.entry_type, "first_month": s.first_month, "month": FORECAST_MONTH,
            "service_level": service_level, "families": sorted(fams, key=lambda x: -x["forecast"])}


class FeatureRow(BaseModel):
    features: dict[str, float | str | None]


@app.post("/predict")
def predict(rows: list[FeatureRow]):
    """Score arbitrary feature rows (integration endpoint for ABD systems)."""
    X = pd.DataFrame([r.features for r in rows]).reindex(columns=FEATURES)
    for c, levels in CATEGORY_LEVELS.items():
        X[c] = pd.Categorical(X[c].astype(str), categories=levels)
    num = [f for f in FEATURES if f not in CATEGORY_LEVELS]
    X[num] = X[num].apply(pd.to_numeric, errors="coerce")
    ml = MODEL.predict(X)
    ens = blend(ml, seasonal_anchor(X), META["weight"])
    return {"forecast_liters": [round(float(v), 2) for v in ens], "ml_component": [round(float(v), 2) for v in ml],
            "ensemble_weight": META["weight"]}


@app.get("/monitoring")
def monitoring():
    if not MONITOR_PATH.exists():
        return {"status": "monitoring report not generated yet: run scripts/04_monitor.py"}
    return json.loads(MONITOR_PATH.read_text())


@app.get("/model")
def model_card():
    return {k: META[k] for k in ("trained_on", "rows", "params", "n_estimators", "weight", "valid_metrics",
                                 "test_metrics", "interval_calibration")}
