# Matching spirits supply to demand in Iowa
**ADS end-term project, Group 1.** Darshita Yadav (240981) · Raghav Yadav (240598) · Rayna Manchanda (240601) · Prerit Shrivastava (240593)

Decision support for the **Iowa Alcoholic Beverages Division (ABD)**, the state's sole spirits wholesaler and
licensing authority: how many liters of each spirits family to plan for each county and store next month, and
how to treat border counties and newly licensed stores in that plan.

| Deliverable | Where |
|---|---|
| Final report | `reports/final_report.docx` / `.pdf` |
| Presentation (3-stage framework) | `reports/final_presentation.pptx` / `.pdf` |
| Code / notebooks | `src/abd_demand/` (library), `scripts/` (pipeline), `notebooks/01…05` (executed, written as readable reports) |
| Working demo | FastAPI + dashboard in `app/` (local or Docker) |
| Pre-registration of hypotheses | `reports/pre_registration.md` (written before any 2026 outcome was examined) |
| Viva preparation | `reports/viva.md` (full guide: every method explained simply + likely questions); `reports/viva_prep.md` (short version) |
| 20 exploratory hypotheses (H4–H23) | `notebooks/05_twenty_hypotheses.ipynb`, `reports/twenty_hypotheses.md`, slides 14–19 |

## Headline results
(Numbers are produced by the notebooks; see `reports/tables/`.)

- **H1, cross-border leakage: supported.** Counties bordering Missouri (spirits tax $2.00/gal vs Iowa's implied
  $15.14) buy **~44% fewer liters per adult** than comparable counties. Found in 2025, confirmed on unseen 2026
  data, and robust to permutation tests, trimming, population weights and measuring in dollars.
- **H2, agave mix: supported** out-of-time (carried from the pre-submission).
- **H3, new licences mostly move sales: supported** (stacked difference-in-differences). Incumbents lose ~13% of
  their volume when a genuinely new store opens; ~60% of the newcomer's volume is diverted from them.
- **Forecast:** an ensemble of tuned Tweedie XGBoost and a calendar-adjusted seasonal naive cuts county-level
  planning error by **~20%** against the status quo on an out-of-time test (Jul–Oct 2026), with bias inside ±3%
  and calibrated intervals.
- **Data-quality findings missed at the pre-submission:** liters *and dollars* truncated in Nov–Dec 2025 **and Jan
  2026**; multi-pack minis recorded as single 50 ml bottles (~2% of liters); 55 "new" stores that are ownership
  changes; stores ordering on one weekday (median 89% of liters), which makes the calendar a demand driver.

## Reproduce
Python 3.11 with [uv](https://docs.astral.sh/uv/). The raw sales CSVs (~1.1 GB) are not in git: download them from
the links in `data/links.txt` into `data/raw/iowa_liquor_sales_2025/` and `…_2026/`.

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv -r pyproject.toml
.venv/Scripts/python scripts/01_build_ssot.py        # raw -> SSOT (4.5M rows x 76 cols), ~1 min
.venv/Scripts/python scripts/02_build_tables.py      # county, store-month and modelling tables, ~1 min
.venv/Scripts/python scripts/03_train_forecast.py    # baselines, models, HPO, test, MLflow, ~8 min
.venv/Scripts/python scripts/04_monitor.py           # drift + retrain decision
.venv/Scripts/python scripts/05_twenty_hypotheses.py # 20 exploratory hypotheses, ~1 min
.venv/Scripts/python notebooks/_build/build.py       # rebuild and execute notebooks 01-05
```

**Demo:**
```bash
.venv/Scripts/python -m uvicorn app.main:app --port 8000     # http://localhost:8000  (API docs at /docs)
docker build -t abd-planner . && docker run -p 8000:8000 abd-planner
mlflow ui --backend-store-uri sqlite:///mlflow.db            # experiment tracking
```

## Structure
```
project/
├── data/
│   ├── links.txt                 provenance of every source
│   ├── raw/                      sales 2025 + 2026, ACS, county adjacency + Gazetteer, tax table
│   └── processed/                SSOT (parquet), county/store tables, data_quality_report.json, Nov-26 forecast
├── src/abd_demand/               etl, features, hypotheses (OLS/permutation/DiD), forecast, monitoring, plotting
├── scripts/                      01_build_ssot · 02_build_tables · 03_train_forecast · 04_monitor
├── notebooks/                    01 data & EDA · 02 hypotheses · 03 forecast & decision · 04 responsible AI & monitoring · 05 twenty more hypotheses
├── app/                          FastAPI service + dashboard
├── models/                       final model + model card (meta.json)
├── reports/                      report, slides, pre-registration, viva prep, figures/, tables/
├── Dockerfile, requirements-app.txt, pyproject.toml
```

## How the guideline requirements are met
| Requirement | Evidence |
|---|---|
| ≥100,000 rows, ≥20 (40+) variables | SSOT 4,518,330 rows × 76 columns (35 sales fields incl. derived/QA, 22 census, 15 geography/tax, 4 store) |
| Multiple credible, non-Kaggle sources | Iowa Data Hub (2 datasets), US Census ACS, Census geography, Tax Foundation |
| Provenance, SSOT, join keys, duplicates, missing values, units | `data/links.txt`, notebook 01, `data_quality_report.json` |
| BADIR | notebook headers; report structure |
| Course methods only | hypothesis testing, permutation/bootstrap, OLS, difference-in-differences (causal inference); feature engineering, ensembles, gradient boosting, HPO, calibration (advanced ML); newsvendor optimisation (decision intelligence); MLflow, FastAPI, Docker, drift monitoring (MLOps); SHAP, fairness audit (responsible AI) |
| Not used, and why | Spark (data fits in memory; it is the scale-up path for ABD's full 2012– history); deep learning / transformers (tabular monthly data, nothing to gain); A/B testing (no intervention under our control); K-means (planned in the pre-submission but not in the syllabus, and the forecast replaces it); RL (no sequential control problem); LIME (TreeSHAP is exact for trees) |
| Supported / unsupported / inconclusive hypotheses | notebook 02, `reports/tables/hypothesis_verdicts.csv` |
| Association vs causation | notebook 02 (H1/H2 associations; H3 DiD with its assumptions) |
| 5–10 recommendations, roadmap | report §8–9 |
| Production and responsible AI | notebook 04, `app/`, `Dockerfile`, `mlflow.db` |
