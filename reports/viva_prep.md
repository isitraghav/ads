# Viva and decision-challenge preparation, Group 1

Every member should be able to answer all of this without notes. Numbers come from the executed notebooks
(`reports/tables/`).

## 1. The questions the brief says will be asked

**Why did you choose this problem?** ABD, the state monopoly wholesaler, has to buy and truck stock *before* stores
order, and its simplest rule ("same month last year") over-buys in a market that fell ~3% in 2026. The data covers
every spirits sale in the state, not a sample, and ABD controls the levers we advise on (procurement, delivery
capacity, licensing).

**What decision does it support?** How many liters of each spirits family to plan for each county next month, at
what service level; how to plan border counties; how to plan, and judge, new licences.

**Why this data?** It is a census of transactions (2.49M + 2.03M lines), official, CC BY, and has a second year that
gave us a genuine out-of-time test. Census ACS, the county adjacency file and the Tax Foundation table explain
*where and why* demand differs. No Kaggle.

**What did your analysis establish?**
- H1: Missouri-border counties buy **44% fewer liters per adult** (CI 25–58%), replicated 2025 → 2026, robust
  in dollars. Up to ≈$5.9M sales / ≈$2.0M ABD margin a year at stake (upper bound).
- H2: agave share rises **+0.8 pts per +10 pts Hispanic population**, out-of-time.
- H3: a genuinely new store cuts incumbents' volume by **13%**; **~60%** of its volume is diverted (CI 28–129%);
  small incumbents −21%.
- Forecast: county WAPE **13.8% vs 17.3%** (−20%), state 5.2% vs 7.0% (−27%), bias +2.0% vs +6.5%; calibrated
  intervals; ≈25% less warehouse safety stock at 95% service.

**Why this model?** Volumes are non-negative, zero-inflated and skewed → **Tweedie** gradient boosting. With one
seasonal cycle the model under-forecast on validation, while last year's value over-forecasts in a shrinking market:
opposite errors → **ensemble**, weight 0.55 chosen on validation subject to |bias| ≤ 3%. It beat ridge, random forest,
XGBoost alone and all naive baselines at county and state level, the levels ABD plans at. XGBoost alone is better at
store level (36.6% vs 39.0%); we say so.

**What are its limitations?** One seasonal cycle; holidays and price changes only partly captured; store-located sales
and a 2020–24 census profile; new stores are under-forecast by ~24%; rural and border counties carry ~+7% bias (half
the status quo's); value estimates assume 1-month lead time and 20% holding cost.

**What would you change if business conditions changed?** See §3.

## 2. Methods, explained simply

| Question | Answer |
|---|---|
| Why WAPE and not MAPE or RMSE? | Volume-weighted (big series matter most), defined with zeros (MAPE divides by zero), and reads as "% of liters mis-planned". RMSE is dominated by a few giant stores. Bias is reported separately because over- and under-planning cost different things. |
| Why a time-ordered split? | Shuffling would let the model see the future. Train Apr-25–Mar-26, validate Apr–Jun-26, test Jul–Oct-26 once. |
| What is pre-registration and why? | We wrote the hypotheses and pass/fail rules before computing any 2026 outcome. It stops us picking hypotheses after seeing the answer, which was the weakness of the pre-submission (14 data-mined candidates confirmed on half the same year). |
| HC3 standard errors? | Heteroskedasticity-robust SEs, appropriate for 99 counties of very different sizes. |
| Freedman–Lane permutation test? | Permute the residuals of the controls-only model, rebuild y, refit, compare the t-statistic: a significance test for one coefficient with controls that needs no normality assumption. |
| Why population weights *and* unweighted? | Unweighted = the typical county; weighted = the typical adult. A result that holds only one way is fragile, which is why H1b and H1c are "inconclusive". |
| Difference-in-differences? | Compare incumbents' change before vs after a new store opens with the change for incumbents in cities with no arrivals. The difference removes common shocks (season, the −3% trend). |
| Why *stacked* DiD? | With entry at different dates, a plain two-way fixed-effects regression uses already-treated stores as controls and can be biased. Stacking builds one clean comparison per entry month. |
| Parallel trends? | Assumption: without entry, treated incumbents would have moved like the controls. Evidence: the pre-entry coefficients are flat (p = 0.97). |
| Placebo? | An ownership change at an existing address is not new competition, so it should show no effect. Post-change: none (−2%, p = 0.41). Its pre-period is noisy (20 cities), so it is supporting, not decisive, evidence. |
| Why exclude the predecessor store? | Otherwise the old licence's drop to zero at its own address looks like a huge competitive effect. Our first run made exactly this mistake (−70%), and we fixed it. |
| Tweedie loss? | A distribution for non-negative data with a point mass at zero and a long right tail: exactly monthly liters per store-family. |
| Why did HPO not help? | 20 random trials spanned only 14.3–14.8% WAPE and the defaults were as good. The gain is from features (calendar, history, competition) and the ensemble. Under a reduced budget, skip HPO. |
| Calibration? | Prediction intervals from validation residual quantiles by size band, checked on test: nominal 80% → 77.5% coverage; 92.4% below P90 (nominal 90%). |
| Newsvendor? | Order the service-level quantile of the forecast distribution. Service level = stock-out cost ÷ (stock-out cost + holding cost). |
| PSI? | Population Stability Index: how much a feature's distribution moved. < 0.1 stable, 0.1–0.25 moderate, > 0.25 major. |
| SHAP? | Exact additive feature contributions (TreeSHAP). Own history ≈78%, store ≈11%, demographics ≈1%. |
| Why drop % Hispanic? | Pre-stated rule: keep only if it improves validation WAPE by >1%. It improved it by 0.5%, so we dropped it. The test ablation agrees. |
| Why not Spark / deep learning / K-means / RL / LIME? | Data fit in memory; tabular monthly data; K-means is not in the handout; no sequential control problem; TreeSHAP is exact. The brief says not to force methods. |

## 3. Surprise scenarios

| Scenario | Response |
|---|---|
| **Budget cut** | Lower the service level (95% → 85%) on low-margin families first; the dashboard slider shows the stock saved. Skip HPO. |
| **Stock-outs become costlier** (e.g. holidays, penalties) | Raise the service level: newsvendor ratio cu ÷ (cu + co). The point forecast is unchanged. |
| **Over-stock becomes costlier** (storage full, capital cost up) | Lower the service level; the forecast's +2% bias also already helps vs the status quo's +6.5%. |
| **Data drift** | The monitor flags PSI > 0.25 on important features, WAPE > 1.2× validation, or losing to the status quo → retrain (~1 minute). Lesson already caught: a "major" PSI on `active_share6` was an artefact of rolling windows. |
| **Concept drift / an unseen season** | A model frozen in Dec 2025 fails only in January (never seen). The seasonal half of the ensemble and the monthly refit handle it. |
| **Another truncated month or renamed categories** | ETL gates: publisher row count, schema, whole-number-liters detector, unknown categories. The pipeline stops. |
| **Fairness complaint (rural counties)** | Audit: we beat the status quo in every group, but rural/border bias is ~+7% vs +1% urban (overstock). Apply a group bias correction learned on validation; audit monthly. |
| **Missouri raises its tax** | Expect border counties to recover volume; H1 bounds the upside; a natural experiment to measure causally. |
| **Wave of new licences** | Plan entrants at high service for 6 months and cut neighbours by most of the entrant's volume (~60% diversion). |
| **Missing census update** | Census features add <0.1 pt to the forecast (ablation); the model runs without them. |

## 4. What we fixed from the pre-submission (be ready to explain)

1. ABD does not stock stores, so the problem was reframed to procurement, delivery capacity and licensing.
2. H1 ("bigger counties drink more") ignored that sales are recorded at the store; we tested cross-border leakage.
3. Confirmation on half of the same year → pre-registered confirmation on new 2026 data.
4. 14 candidate hypotheses → 3, as the brief asks.
5. K-means (not in the handout) → forecast + SHAP.
6. "Public-health flag for high-volume counties" → high volume marks hubs and tourism, so it was replaced by the H3 licensing evidence.
7. New data defects: truncation also in Jan 2026 and in dollars; multi-pack minis; ownership changes.

## 5. Numbers to know cold

4,518,330 rows × 76 columns · 2,401 stores (281 new, 55 ownership changes) · 99 counties · 5 sources / 4 publishers ·
Missouri $2.00 vs Iowa $15.14 per gallon · H1 −44% (CI 25–58%) · H2 +0.8 pts per 10 pts · H3 −13%, diversion ~60% ·
county WAPE 13.8% vs 17.3% · state 5.2% vs 7.0% · bias +2.0% vs +6.5% · ensemble weight 0.55 · coverage 77.5% ·
safety stock −25% (≈$0.73M) · over-buying ≈84,000 L/month · median store 89% of liters on one weekday · market −3%.
