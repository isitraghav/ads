"""Model monitoring (course unit: MLOps, data drift, concept drift, automated retraining).

data drift     Population Stability Index (PSI) of every numeric feature, reference vs current window
concept drift  forecast error (WAPE) month by month on data the model was not trained on
schema drift   category names that appear or disappear between years in the raw feed
retraining     an explicit rule turning the three checks into a retrain / no-retrain decision
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_STABLE, PSI_MAJOR = 0.10, 0.25
# Features that shift by construction as time passes (calendar, store age, year-ago availability):
# reported, but never a retraining trigger on their own.
EXPECTED_DRIFT = {"store_age_months", "month_of_year", "business_days", "order_days", "cal_ratio_lag1",
                  "cal_ratio_lag12", "state_fam_yoy"}


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI = sum (a - e) * ln(a / e) over bins cut at the reference deciles. <0.10 stable,
    0.10-0.25 moderate shift, >0.25 major shift (standard credit-risk rule of thumb)."""
    e = pd.Series(expected).dropna().to_numpy(float)
    a = pd.Series(actual).dropna().to_numpy(float)
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant feature: compare share equal to the mode instead
        mode = pd.Series(e).mode().iat[0]
        pe, pa = np.array([(e == mode).mean(), (e != mode).mean()]), np.array([(a == mode).mean(), (a != mode).mean()])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        pe = np.histogram(e, edges)[0] / len(e)
        pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def drift_table(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        if not pd.api.types.is_numeric_dtype(ref[f]):
            continue
        v = psi(ref[f], cur[f])
        rows.append(dict(feature=f, psi=v, ref_mean=float(ref[f].mean()), cur_mean=float(cur[f].mean()),
                         status="major" if v > PSI_MAJOR else "moderate" if v > PSI_STABLE else "stable"))
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def monthly_error(pred: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """WAPE per month at county x family level for each prediction column."""
    out = []
    for month, d in pred.groupby("month"):
        a = d.groupby(["county_fips_code", "family"], observed=True)[["y"] + list(cols.values())].sum()
        row = {"month": month}
        for label, c in cols.items():
            row[label] = float(np.abs(a.y - a[c]).sum() / a.y.sum())
        out.append(row)
    return pd.DataFrame(out)


def schema_drift(sales: pd.DataFrame) -> pd.DataFrame:
    by = sales.groupby(["category_name", "year"], observed=True).liters.sum().unstack(fill_value=0)
    by.columns = [str(c) for c in by.columns]
    new = by[(by.get("2025", 0) == 0) & (by.get("2026", 0) > 0)].assign(change="new in 2026")
    gone = by[(by.get("2025", 0) > 0) & (by.get("2026", 0) == 0)].assign(change="absent in 2026")
    return pd.concat([new, gone]).reset_index()


def retrain_decision(drift: pd.DataFrame, recent_wape: float, baseline_wape: float, valid_wape: float,
                     tolerance: float = 1.20) -> dict:
    """Retrain when (a) any of the top-10 most important features shows a major PSI shift, or
    (b) live WAPE is more than 20% worse than at validation, or (c) the model stops beating the
    status-quo baseline. Otherwise keep the scheduled monthly refit only."""
    reasons = []
    major = drift[drift.status == "major"].feature.tolist()
    if major:
        reasons.append(f"major PSI shift in: {', '.join(major)}")
    if recent_wape > tolerance * valid_wape:
        reasons.append(f"live WAPE {recent_wape:.3f} > {tolerance:.0%} of validation WAPE {valid_wape:.3f}")
    if recent_wape >= baseline_wape:
        reasons.append("model no longer beats the seasonal-naive baseline")
    return {"retrain_now": bool(reasons), "reasons": reasons or ["no trigger fired: keep the scheduled monthly refit"]}
