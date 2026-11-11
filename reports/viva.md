# Viva guide — Matching spirits supply to demand in Iowa

**Group 1:** Darshita Yadav (240981) · Raghav Yadav (240598) · Rayna Manchanda (240601) · Prerit Shrivastava (240593)

Everything here is explained simply, then in enough depth to survive follow-up questions. Every member should be
able to answer any section. Numbers come from the executed notebooks (`notebooks/01…05`) and `reports/tables/`.

**How to use this guide**
- Sections 1–2: the story. Learn the 60-second pitch by heart.
- Sections 3–9: what we did, step by step, with every method explained.
- Section 10: questions an instructor is likely to dig into, with answers, ending with **"Why X and not Y?"** for
  every major choice (model, loss, ensemble, metrics, tests, tools), each backed by a number.
- Sections 11–13: surprise scenarios, limitations, numbers to know.

---

## 1. The project in 60 seconds

> Iowa is a "control state": the Alcoholic Beverages Division (ABD) is the only wholesaler of spirits. It buys from
> suppliers, stores everything in one warehouse, trucks it to ~2,240 licensed stores, and also issues those stores'
> licences. Every month it has to buy and plan trucks **before** stores order. The simple rule today, "plan the same
> month as last year", over-buys, because the market shrank ~3% in 2026, and it ignores *where* and *why* demand differs.
>
> We built one clean table of **4.5 million order lines** from five official sources, found and fixed seven data
> defects, and tested three pre-registered hypotheses on data we had not seen. (1) Counties bordering Missouri,
> where spirits tax is \$2 vs Iowa's \$15 a gallon, buy **44% fewer liters per adult**. (2) Agave share follows the
> Hispanic population. (3) A new store takes about **60%** of its volume from existing stores. We then built a forecast
> that cuts planning error by **20–27%** against today's rule, with calibrated uncertainty, an explanation for every
> forecast, a fairness audit, monitoring and a working API. Twenty further exploratory hypotheses added findings such
> as "price-cut orders arrive a month late" and "a shop that changes owner keeps 92% of its customers".

**Three sentences if interrupted:** ABD must buy before demand is known. We explain where demand differs (border,
ethnicity, new stores) and forecast it 20–27% better than today. The result is less over-buying, ~25% less safety
stock, and evidence for licensing and tax policy.

---

## 2. The story in BADIR order

| BADIR step | What we did | Where |
|---|---|---|
| **B**usiness question | ABD's monthly buying/trucking decision and its licensing decisions; KPI = WAPE ≥10% better than status quo, bias within ±3% | report §1, notebook 01 |
| **A**nalysis plan | Pre-registered H1–H3 with pass/fail rules; time-ordered split; status-quo baseline | `pre_registration.md` |
| **D**ata | 5 sources → one SSOT (4.52M × 76); quality gates; EDA | notebook 01 |
| **I**nsights | H1–H3 tested on unseen 2026 data; 20 exploratory hypotheses | notebooks 02, 05 |
| **R**ecommendations | Forecast + service-level plan; 10 recommendations; API, monitoring, roadmap | notebooks 03, 04; slides 22–25 |

**What changed since the pre-submission (be ready to explain each):**
1. We thought ABD stocks stores; it does not (stores order what they want). The problem became procurement, delivery
   capacity and licensing.
2. Pre-submission H1 ("bigger counties drink more") ignored that sales are recorded at the *store*. Exploring 2025 showed
   the lowest-buying counties all touch Missouri, so we tested cross-border leakage instead.
3. Patterns were "confirmed" on half of the same year. Now: pre-registered, confirmed on new 2026 data.
4. 14 candidate hypotheses → 3, as the brief asks (plus 20 clearly labelled exploratory ones).
5. K-means (not in the course handout) → forecast + SHAP.
6. New data defects found: truncation in Jan 2026 too and in dollars; multi-pack minis; ownership changes.

---

## 3. The data

### 3.1 Sources (five, from four publishers, none from Kaggle)

| Source | Publisher | Size | Why we need it |
|---|---|---|---|
| Iowa Liquor Sales 2025 (dataset 1262) | Iowa Data Hub | 2,490,620 rows | every order a store placed with ABD (discovery sample) |
| Iowa Liquor Sales 2026 (dataset 1263) | Iowa Data Hub | 2,029,764 rows | same for Jan–Oct 2026 (confirmation sample + forecast test) |
| ACS 5-year 2020–24 (16 tables) | US Census Bureau | 99 counties, 22 variables | population, age, income, education, ethnicity, poverty |
| County adjacency + Gazetteer 2024 | US Census Bureau | 99 counties | which counties touch which state, area, location |
| State spirits excise, Jan 2026 | Tax Foundation | 7 states | the tax gap (Iowa \$15.14 vs Missouri \$2.00 per gallon) |

**Why this data is unusually good:** Iowa is a control state, so the sales file is *every* spirits sale in the state, not a
sample. Row counts match the publisher's API exactly.

**What one row means:** one product ordered by one store on one day (an *order line*). It is a **business purchase from
the wholesaler**, recorded where the **store** is, not where the drinker lives.

### 3.2 The SSOT (single source of truth)
One table at the finest grain (order line), 4,518,330 rows × 76 columns: 35 sales/derived/QA columns, 22 census, 15
geography/tax, 4 store attributes. Every analysis aggregates from it, so all numbers agree.

- **Join key = 5-digit county FIPS code**, not county name. Names vary in spelling across sources; FIPS is a federal
  standard. Join is many-to-one, validated, 100% matched.
- **Store key = `store_no`**, stable across years (2,078 stores appear in both years).
- Rebuild: `scripts/01_build_ssot.py`, about a minute; every step logged in `data_quality_report.json`.

### 3.3 Data-quality issues (know each one and why it matters)

| Issue | Size | What we did | Why it matters |
|---|---|---|---|
| Duplicated download file | 429,989 rows | excluded; parts 1–4 match the publisher count | would double-count Sep–Oct 2025 |
| Exact duplicate rows | 21 | removed | |
| Blank county | 4,135 rows | 2,102 filled from the store's county; 2,033 dropped (0.05%) | |
| **Liters truncated** to whole numbers in Nov–Dec 2025 **and** Jan 2026 | −3.4% | recomputed liters = bottles × bottle size, every row | otherwise a fake holiday slump; a 50 ml mini read as 0 L |
| Dollars and prices truncated in the same months | −0.4% | rebuilt from each item's price in the nearest clean month (tested: 0.08% error) | |
| Wholesale cost blank Jan–Jun 2025 | 1.2M rows | price per liter from dollars ÷ liters | affects price tests (H13, H14 use clean months only) |
| **Multi-pack minis**: one "bottle" = a sleeve of 50 ml minis | 313 items | flagged, excluded from prices; H1 re-tested in dollars | made the 99th-percentile price \$1,620/L |
| Returns (negative bottles) | 4,019 rows | kept as negative volume; forecast target clipped at 0 | they are real credits |
| **Ownership changes** recorded as new licences | 55 stores | classified separately from 281 genuinely new stores | without this, H3 finds "competition" where there is none |

---

## 4. EDA: techniques and findings (notebook 01)
Every finding is written as **Observation → Inference → Decision relevance**.

| Technique | What it is | What we found | So what |
|---|---|---|---|
| **Data profiling** | size, types, missing values per column | only price per liter has gaps (7.5%, by design) | safe to proceed |
| **Descriptive statistics** | mean, median, percentiles | median line 4.5 L, mean 8.8 L, max 13,200 L | heavy **right skew** → medians, log scales, Tweedie loss |
| **Histogram (log scale)** | counts per value range; log axis for data spanning orders of magnitude | small top-ups and bulk orders coexist | plan by store type / size |
| **Box plot + IQR rule** | middle 50% of data; outliers beyond 1.5 × IQR | ~1 in 8 priced lines are price "outliers" | they are real premium bottles → **kept** |
| **Frequency bar charts** | totals per category | vodka + whiskey ≈ 60% of liters; grocery ≈ 40% | forecast per family; store type as a feature |
| **Line chart + year-over-year** | value over time; compare with same month last year to remove seasonality | 2026 is ~3% below 2025, lower in 8 of 10 months | the status quo over-buys |
| **Seasonal index** | each month ÷ average month | Oct and Dec peaks (+10%), Jan–Feb lows (−11%) | plan by month, not a flat average |
| **Weekday concentration** | share of a store's liters on its main weekday | median store: **89%** on one weekday | 4 vs 5 order days = 25% swing → calendar feature |
| **Scatter plot** | two numeric variables, one dot per county | liters per adult collapse near Missouri | became **H1** |
| **Grouped box plot** | distributions side by side | border counties: median 2.9 L vs 7.0 L per adult | the whole distribution shifts, not one county |
| **Correlation heatmap (Spearman)** | rank correlation for every pair, −1 to +1 | predictors overlap (population, education, income); Missouri distance only 0.20 because the effect is local | use **regression with controls**; correlation misses threshold effects |
| **Pareto curve** | cumulative share of volume by store size | top 10% of stores buy **62%** of liters | volume-weighted metric (WAPE) |
| **Entry over time** | new licences per month | ~15 new stores/month + 55 ownership changes | became **H3** |

**Pearson vs Spearman:** Pearson measures straight-line association on raw values and is pulled by outliers. Spearman
works on ranks, so it captures any monotonic relationship and is robust to skew. We used Spearman because county sizes
range from 4,000 to 500,000 people.

---

## 5. The three pre-registered hypotheses (notebook 02)

### H1 — Cross-border leakage
- **Claim:** counties where cheaper out-of-state spirits are close buy fewer liters per adult from Iowa stores.
- **Test:** OLS of log(liters per adult) on "borders Missouri", controlling for log population, income, median age and
  % bachelor's. Four checks: HC3, permutation, without influential counties, population-weighted. 2025 and 2026.
- **Result:** **−44%** (95% CI −25% to −58%) in 2026; −43% in 2025. Passes all four checks. Also holds in **dollars**.
- **Sub-tests:** H1b tax-gap dose-response *inconclusive*; H1c fewer 1.75 L "stock-up" bottles *inconclusive* (both
  fail population weighting); H1d leakage to a bigger Iowa neighbour *not supported* (p ≈ 0.10).
- **Money:** ≈ \$5.9M of sales ≈ **\$2.0M of ABD margin a year**, an *upper bound* (part of the gap may be less drinking).
- **Why log?** Liters per adult is skewed and the effect is proportional; a log outcome makes the coefficient a
  percentage: exp(−0.58) − 1 = −44%.

### H2 — Ethnic composition and agave share
- **Claim:** counties with a larger Hispanic share buy a larger share of tequila/mezcal.
- **Result:** **+0.8 points** of agave share per +10 points of Hispanic population (average share ~3.6%); all four
  checks, both years.
- **Caveat:** ecological — it describes counties, not people. It informs regional assortment; ethnicity is kept **out**
  of the forecast model.

### H3 — New stores mostly move sales
- **Claim:** when a genuinely new store opens, incumbents in the same city order less, so the city grows by less than
  the newcomer sells.
- **Design:** stacked difference-in-differences (section 6.3).
- **Result:** incumbents **−13%** of their own pre-entry level (p = 0.01; pre-trend p = 0.97). **Diversion ≈ 60%** (CI
  28–129%): about 60% of the newcomer's volume is taken from existing stores, ~40% is new. Small incumbents −21%, large −4%.
- **Placebo:** a new owner at an existing shop shows no post-change effect (−2%, p = 0.41), but its pre-period is noisy
  (20 cities).
- **Disclosed deviation:** raw liters (pre-registered) were not significant because a few giant stores dominate the
  variance; we report liters relative to each store's own pre-entry average, and say so.

---

## 6. Statistics, explained simply

### 6.1 Basics
| Term | Simple explanation |
|---|---|
| **Hypothesis test** | Assume there is no effect (the *null*), then ask how surprising our data would be. |
| **p-value** | If there were truly no effect, the chance of seeing a result at least this strong. p < 0.05 = "less than 5% of the time" = significant. It is **not** the probability the hypothesis is true. |
| **One-sided test** | We test one direction stated in advance ("border counties buy *less*"). Fair because the direction was pre-registered. |
| **Confidence interval (95%)** | A range of plausible effect sizes. If we repeated the study many times, 95% of such ranges would contain the true value. |
| **Effect size vs significance** | Significance says "probably not zero"; effect size says "how big". We always report both. |
| **Type I / Type II error** | False alarm (finding an effect that isn't there) / missed effect. Multiple tests inflate Type I. |
| **Pre-registration** | Writing hypotheses and pass/fail rules *before* seeing the outcome data. Prevents choosing what to test after seeing the answer (p-hacking, HARKing). |
| **Discovery vs confirmation sample** | Find patterns in 2025, test them on 2026 data never looked at. |

### 6.2 County regressions (H1, H2, and H4–H23)
| Term | Simple explanation | Why we used it |
|---|---|---|
| **OLS regression** | Fits a straight-line relationship: outcome = a + b·x + c·controls. *b* is x's effect with controls held fixed. | separates the border effect from size, income, age, education |
| **Control variable** | Something that could affect both x and the outcome, included so it is held constant. | border counties are smaller and older; without controls we'd mix those in |
| **Confounder** | A third factor that drives both x and the outcome, creating a false link. | e.g. H23: age looked important only because old counties are rural |
| **Heteroskedasticity** | Noise that differs across units (small counties are noisier). | |
| **HC3 robust standard errors** | Uncertainty estimates that don't assume equal noise; HC3 is the version recommended for small samples (99 counties). | |
| **Permutation test (Freedman–Lane)** | Shuffle the residuals of the controls-only model 5,000 times to create "no-effect" worlds; the p-value is how often a shuffled world gives a t-statistic as extreme as the real one. | no normality assumption; works with controls |
| **Bootstrap CI** | Resample the 99 counties with replacement 2,000 times, refit, take the middle 95% of estimates. | effect-size range without normality |
| **Cook's distance** | How much the fit changes if one county is removed; > 4/n = influential. We refit without those. | result must not hinge on a few counties |
| **Population weighting** | Weight each county by its adults. Unweighted = typical county; weighted = typical person. | a robust result holds both ways (H1b/H1c failed this) |
| **Multicollinearity** | Predictors that move together (income, education, population) make each one's coefficient unstable. | why we interpret controls cautiously and focus on the variable of interest |
| **Ecological fallacy** | Wrongly concluding about individuals from group data. | H2 is about counties, never "Hispanic people drink tequila" |

### 6.3 Causal inference (H3)
| Term | Simple explanation |
|---|---|
| **Association vs causation** | Two things moving together does not mean one causes the other. H1/H2 are associations; H3 is designed to be causal. |
| **Difference-in-differences (DiD)** | Compare the change for the treated group (incumbents in a city with a new store) with the change for a control group (incumbents in cities without entry). The difference of the two changes removes everything common to both (season, the −3% market). |
| **Parallel-trends assumption** | Without the entry, both groups would have moved together. Can't be proven, but we check the months *before* entry: they should be flat (pre-trend p = 0.97). |
| **Fixed effects** | Remove each store's own average level and each month's statewide level, so only changes are compared. |
| **Why "stacked" DiD** | Entries happen in different months. A plain two-way fixed-effects regression then uses already-treated stores as controls and can be biased (even get the sign wrong). Stacking builds one clean comparison per entry month and pools them. |
| **Event study** | Estimate the treated–control gap separately for each month from −6 to +6. Before entry ≈ 0; after entry shows how the effect builds. |
| **Clustered standard errors (by city)** | Stores in the same city share shocks, so they are not independent; clustering accounts for that. |
| **Placebo test** | Run the design where no effect should exist (ownership change at an existing shop). An "effect" there would expose a flawed design. |
| **Diversion ratio** | Incumbents' lost liters ÷ newcomer's liters. 100% = pure reshuffling; 0% = all new demand. |
| **Cluster bootstrap** | Resample whole cities (not stores) 200 times to get the CI for the diversion ratio, respecting the clustering. |
| **Excluding predecessors** | A store that stops at the address where a new licence appears is the same shop. Our first run counted its drop to zero as competition and got −70%. We caught and fixed it. |
| **Selection bias direction** | New stores pick growing towns, which would make incumbents look *better*, biasing toward zero, so 60% is conservative. |

### 6.4 Methods used in the 20 exploratory hypotheses (notebook 05)
| Method | Simple explanation | Used for |
|---|---|---|
| **Moran's I** | Spatial autocorrelation: do neighbouring counties have similar values? +1 neighbours alike, 0 random. p-value by shuffling values across counties. | H5 (taste is contagious) |
| **Hinge (piecewise) regression** | Effect rises up to a break point *k*, then flat; *k* chosen by trying many values; bootstrap for its range. | H6 (~30-mile horizon) |
| **Studentised residual** | How far a county is from the model's prediction, in standard errors (beyond ±2 = unusual). | H7 (Sioux County) |
| **Wilcoxon signed-rank** | Paired comparison on the same units using ranks (e.g. each store in Feb–Mar vs Apr–Oct). | H8, H12, H20, H21 |
| **Placebo category** | Same test on a comparable item with no reason to move (Scotch for St Patrick's). | H8 |
| **Two-way fixed effects (item × month)** | Removes each product's own level and each month's statewide level, leaving "unusual for this product this month". | H13 (price cuts) |
| **Mann–Whitney U** | Compares two independent groups using ranks. | H14 (imports vs domestic), H22 |
| **Rarefaction** | Draw the same number of order lines (892, the smallest county's total) from every county, so variety is compared fairly. | H18 (long tail) |
| **Log-log rank regression (Zipf)** | log(volume) vs log(rank); slope −1 = Zipf's law. | H19 |
| **Jensen–Shannon distance** | Difference between two mixes, 0 (identical) to 1 (nothing shared). | H22 (same address, same customers) |
| **AUC** | How well a score separates two groups: 0.5 coin flip, 1.0 perfect. | H21 (exit early warning: 0.53) |
| **Holm correction** | For many tests: multiply the smallest p by 20, the next by 19, … Controls the chance of *any* false positive. | all 20 |

---

## 7. The 20 exploratory hypotheses (H4–H23)
**10 supported, 3 partial, 7 rejected.** Exploratory (not pre-registered); county patterns must hold in both years;
6 survive Holm (H6, H12, H18, H20, H21, H22).

| # | Hypothesis | Verdict | Key number |
|---|---|---|---|
| H4 | Wisconsin brandy culture crosses the river | ✅ | 1.7× brandy share on the WI border |
| H5 | Neighbouring counties drink alike beyond demographics | ✅ whiskey, rum | Moran's I 0.14–0.17 |
| H6 | Missouri pull has a travel horizon | 🟡 | ~30 mi in 2026; linear in 2025 |
| H7 | Dutch-Reformed Sioux County is "dry" | ❌ | rank 40/99; Louisa, Keokuk are the outliers |
| H8 | St Patrick's Day lifts Irish whiskey | ✅ | +29% share in Feb–Mar; Scotch +4% |
| H9 | Cinco de Mayo bump is Hispanic-led | ❌ | bump +16% but statewide |
| H10 | College towns empty in summer | ✅ | Ames −37%, Iowa City −25% |
| H11 | Holiday trade-up is for the rich | ❌ | same at every income |
| H12 | Palate rotates with seasons | ✅ 7/8 | RTD +48% in summer; rum is a *winter* drink |
| H13 | Price cuts → stock up, then dip | ❌ reversed | +49% the month **after** the cut |
| H14 | Tariffs raised import costs | ❌ | all origins within ±1% |
| H15 | Inequality → price barbell | 🟡 | spread widens but average rises too |
| H16 | 50 ml nips track poverty | 🟡 | only within store type (+3.3 pts per +10 pts poverty) |
| H17 | More outlets, more drinking, < 1:1 | ✅ | elasticity 0.4–0.5 ≈ H3's 40% |
| H18 | Long tail lives in cities | ✅ | 547 vs 322 products per equal sample |
| H19 | Zipf's law | ✅ | slope −1.10; top 1% of products = 38% |
| H20 | New-store opening honeymoon | ✅ | opening 1.86× normal, then 0.79× |
| H21 | Dying stores fade slowly | ❌ | −6% only; AUC 0.53 |
| H22 | Customers belong to the address | ✅ | 92% of volume kept |
| H23 | Whiskey divide is generational | ❌ | it is rurality (r = −0.69), not age |

**How to defend "isn't 20 tests p-hacking?"** (1) We label them exploratory and report Holm-corrected p-values.
(2) Every direction was stated before its test. (3) County patterns must replicate in **two independent years**, which
is stronger than a single p-value. (4) We report the rejections (7 of 20) as findings, not hide them.

**Best three to talk about:** H13 (counter-intuitive timing → plan promo stock for month t+1), H17 + H3 (two methods,
same answer), H23 (a textbook confounder).

---

## 8. The forecast (notebook 03)

### 8.1 Framing
| Choice | Answer | Why |
|---|---|---|
| Unit | store × spirits family × month | one bottom-up model serves stores (delivery), counties (trucks) and the state (buying) |
| Horizon | one month ahead | ABD's buying cycle |
| Target | net liters, returns clipped at 0 | |
| Status quo | same month last year (seasonal naive) | what a planner does today |
| Split | train Apr 2025–Mar 2026 · validation Apr–Jun 2026 · **test Jul–Oct 2026** | time-ordered; test used once |
| Main metric | **WAPE** at county × family | see 8.5 |

**Data leakage** = the model seeing information it would not have at prediction time. We avoid it by (a) a
time-ordered split, (b) every feature for month *t* using data up to the end of *t−1* only (lags and rolling windows are
shifted), (c) choosing the ensemble weight, tuning and the ethnicity decision on validation, never on test.

### 8.2 Features (38 used)
| Group | Examples | Why |
|---|---|---|
| Own history | lag 1/2/3/6/12, 3- and 6-month means, 6-month std, share of active months | demand persists |
| Store context | store total last month, family share of the store, store age, store type | Costco ≠ Casey's |
| Order calendar | expected order days this month (from the store's own past weekdays), ratios vs last month/year, day-adjusted lags | 89% of a store's liters on one weekday |
| Competition (H3) | active stores in the city, new entrants in last 3 months | entrants take volume |
| Market (H1/H2) | Missouri border, tax gap, population, income, education, age, density | level differences |
| Statewide | family volume last month, family YoY trend | the −3% market |

### 8.3 The algorithms
- **Naive / seasonal naive / moving average:** simple rules; the seasonal naive is the status quo.
- **Ridge regression:** linear regression with an L2 penalty (λ‖β‖²) that shrinks coefficients toward zero to reduce
  over-fitting and handle correlated features. On log(1+liters). Simple, but cannot capture interactions.
- **Decision tree:** splits the data by questions ("lag1 > 200?") into groups and predicts each group's average.
  Captures interactions; one tree over-fits.
- **Random forest (bagging):** many deep trees, each on a bootstrap sample and a random subset of features; average them.
  Averaging reduces variance. 200 trees, min 20 rows per leaf.
- **Gradient boosting (XGBoost):** trees built **one after another**; each new tree fits the remaining errors (the
  gradient of the loss) of the ensemble so far, and is added with a small weight (learning rate). Many small corrections
  → strong model. XGBoost adds regularisation (λ), handles missing values natively, and is fast (histogram method).
  Bagging reduces *variance*; boosting reduces *bias*.
- **Tweedie loss:** the error function for non-negative data with many exact zeros and a long right tail (a compound
  Poisson–gamma distribution, variance ∝ mean^p with 1 < p < 2). Monthly store-family liters are exactly that. Tuned
  p = 1.4. Least squares on log(liters) predicts a geometric mean and under-forecasts totals (−22% bias on validation).
- **Early stopping:** stop adding trees when validation error hasn't improved for 100 rounds.
- **Ensemble (blending):** final = **0.55 × XGBoost + 0.45 × day-adjusted seasonal naive**. Why: learned models
  under-forecast (one seasonal cycle; trees can't extrapolate above the biggest values they saw), while the seasonal
  naive over-forecasts (copies last year into a shrinking market). Opposite errors cancel. The weight is the lowest
  validation county WAPE with |bias| ≤ 3%.

**Final model settings:** depth 6, learning rate 0.02, min child weight 5, subsample 0.6, colsample 0.6, λ = 10,
Tweedie p = 1.4, 543 trees. For the test it was refit on train + validation; for production on all data to Oct 2026.

### 8.4 Hyperparameter tuning
**Random search**, 20 configurations, objective = validation county WAPE of the *ensemble* (the deployed thing), all
logged in MLflow. Result: every trial landed between ~14.3% and ~14.8%, and defaults were as good. Gains come from
features and the ensemble, not tuning. (Random search beats grid search for the same budget because only a few
hyperparameters usually matter.)

| Hyperparameter | Meaning |
|---|---|
| `max_depth` | how many questions deep each tree goes (complexity) |
| `learning_rate` | how much each new tree corrects (smaller = slower, more stable) |
| `min_child_weight` | minimum data in a leaf (prevents tiny, noisy leaves) |
| `subsample` / `colsample_bytree` | share of rows / features each tree sees (randomness reduces over-fitting) |
| `reg_lambda` | L2 penalty on leaf values |
| `tweedie_variance_power` | the Tweedie *p*: closer to 1 = Poisson-like, closer to 2 = gamma-like |

### 8.5 Metrics
| Metric | Formula | Verdict |
|---|---|---|
| **WAPE** | Σ\|actual − forecast\| ÷ Σ actual | **used**: reads as "% of liters mis-planned", volume-weighted, works with zeros |
| **Bias** | Σ(forecast − actual) ÷ Σ actual | **used**: + = over-buying, − = stock-outs; costs differ |
| MAPE | mean of \|error\| ÷ actual | not used: divides by zero for zero months; small series dominate |
| RMSE | √mean(error²) | not primary: dominated by a few giant stores; used for safety stock |
| MAE | mean \|error\| | reported at store level |

### 8.6 Results (out-of-time test, Jul–Oct 2026)
| Level | Status quo | Final | Change |
|---|---|---|---|
| State × family (warehouse buying) | 7.0% | **5.2%** | −27% |
| County × family (delivery regions) | 17.3% | **13.8%** | −20% |
| Store × family (store orders) | 48.5% | 39.0% | −20% |
| Bias | +6.5% | **+2.0%** | |

Beats the status quo in every test month (16–26% at county level). XGBoost alone is better at **store** level (36.6% vs
39.0%); the ensemble wins at county and state level, where ABD plans. We say so.

### 8.7 Prediction intervals and calibration
Empirical intervals: on validation, the quantiles of *actual ÷ forecast* within forecast-size bands; applied to new
forecasts. **Calibration check on test:** county P10–P90 covers **77.5%** (target 80%); 92% of actuals below P90
(target 90%). Well calibrated, so service levels mean what they say.

### 8.8 From forecast to decision: newsvendor and safety stock
- **Newsvendor rule:** order the forecast's **service-level quantile**, where service level = Cu ÷ (Cu + Co), Cu = cost
  of running short, Co = cost of holding extra. Stock-out loses ~33% margin + goodwill; holding ~1.7% a month → high
  service level justified.
- **Safety stock** ≈ z × RMSE of forecast error (z = 1.645 for 95%). Better forecast → less safety stock: **−25% at
  warehouse level ≈ \$0.73M** of working capital.
- **Over-buying removed:** the status quo plans ~**84,000 L/month** too much (≈ \$1.0M at cost).
- Assumptions to state: one-month lead time, 20% annual holding cost, stock at ABD cost (\$12.25/L).

### 8.9 Why XGBoost and not LightGBM or CatBoost? (we tested it: `scripts/06_model_alternatives.py`)
Every alternative got exactly the same treatment as our XGBoost: the same 38 features and time split, default settings
(learning rate 0.05, depth ≈ 6, early stopping on validation), Tweedie *p* = 1.4 where the library supports it, the
ensemble weight chosen on validation, then a refit on train + validation and **one** score on the test months.

| Model (untuned, blended with seasonal naive) | Fit time | Valid county WAPE | **Test county WAPE** | Test state WAPE | Test bias |
|---|---|---|---|---|---|
| **XGBoost, Tweedie** (our choice) | 7 s | 14.4% | **13.9%** | 5.1% | +1.7% |
| LightGBM, Tweedie | 2 s | 14.3% | **13.9%** | 5.1% | +1.7% |
| CatBoost, Tweedie | 92 s | 14.4% | 14.1% | 5.8% | +3.0% |
| XGBoost, Poisson | 14 s | 14.0% | 14.1% | 5.9% | +2.6% |
| sklearn HistGradientBoosting, Poisson | 13 s | **13.9%** | 14.3% | 5.1% | +1.8% |
| *Same XGBoost, 3 different random seeds* | | *14.5–14.6%* | | | |
| *Tuned final XGBoost (for reference)* | | *14.3%* | *13.8%* | *5.2%* | *+2.0%* |

**How to read it:**
1. **It is a tie.** All five land between 13.9% and 14.3% at county level. Changing only the random seed of the same
   XGBoost moves validation WAPE by 0.2 points, so differences of a few tenths are noise.
2. **LightGBM = XGBoost** (13.89% vs 13.92% on test, identical bias). LightGBM is 3× faster, but 7 seconds vs 2 seconds
   doesn't matter at 220,000 training rows. Switching would buy nothing.
3. **The validation "winner" lost on test.** HistGradientBoosting was best on validation (13.9%) and worst on test
   (14.3%). Picking among near-ties on three validation months chases noise, so we keep the model we committed to.
4. **CatBoost** is 13× slower and worse at the warehouse (state) level. Its strength, smart handling of
   high-cardinality categoricals, doesn't apply: we have only two categorical features (family, store type).
5. **The bottleneck is information, not the algorithm.** One seasonal cycle of history limits every model. The same
   story as tuning (8.4): settings and libraries move results by tenths of a point; features and the ensemble moved
   them by points.

**Why XGBoost in the first place (before we had this evidence):** native Tweedie loss, native missing values and
categoricals, built-in exact TreeSHAP (`pred_contribs`) for per-store explanations in the API, a single small model
file (4 MB JSON) for serving, mature documentation, and it is the standard taught gradient-boosting library.
**When LightGBM would be the right choice:** millions of rows (e.g. daily store × product forecasts), where its
leaf-wise growth and speed matter. On small data, leaf-wise trees over-fit more easily and need careful
`num_leaves` / `min_child_samples` settings.

---

## 9. Responsible AI and MLOps (notebook 04)

### 9.1 Explainability — SHAP
- **What:** splits each prediction into one contribution per feature so that base value + contributions = prediction.
  From game theory (Shapley values: each feature's average marginal contribution over all orders of adding features).
  **TreeSHAP** computes them exactly for tree models.
- **Global:** own history ~78%, store context ~11%, statewide trend and order calendar next, demographics ~1%.
- **Local:** every API store forecast returns its top drivers, so a planner can see *why* and override with a reason.
- **Why not LIME?** LIME approximates with a local linear model and is unstable; TreeSHAP is exact for trees.

### 9.2 Fairness
- **Audit:** WAPE and bias by group (urban/rural, income, % Hispanic, Missouri border, store type, store age), final vs
  status quo.
- **Result:** better than the status quo in **every** group. But rural/border counties carry ~+7% bias vs ~+1% urban
  (overstock risk; half the status quo's), and new stores are under-forecast ~24% (stock-out risk).
- **Mitigation:** group bias correction learned on validation; plan new stores from peer stores (H20); monthly audit;
  promote a new model only if no group has |bias| > 5%.

### 9.3 Sensitive attribute and ablation
- Rule set in advance: keep % Hispanic only if it improves validation WAPE by > 1%. It improved it by 0.5% → **dropped**.
- **Ablation** (remove features, retrain, compare): removing *all* census features changes test WAPE by < 0.1 pt.
  Demographics explain *levels* (already in each store's lags); they matter for planning norms and H1/H2, not next month.

### 9.4 Privacy and ethics
No personal data (business-to-government orders, CC BY). Ecological framing for H2. ABD is a regulator: the objective is
accurate supply (fewer stock-outs *and* less dead stock), not more sales. H1 revenue is reported with its causal caveat.

### 9.5 MLOps
| Piece | What | Detail |
|---|---|---|
| Pipeline | `scripts/01…05` + `notebooks/_build/build.py` | rebuild everything from raw files |
| **MLflow** | experiment tracking | 43 runs: baselines, candidates, 20 HPO trials, ablations, final, production |
| **FastAPI** | serving | `/forecast/county/{fips}`, `/forecast/store/{id}` (with SHAP drivers), `/predict`, `/monitoring`, `/model`, `/health`, dashboard at `/` |
| **Docker** | container | python:3.11-slim, health check; `docker build -t abd-planner . && docker run -p 8000:8000 abd-planner` |
| Model card | `models/xgb_tweedie_final.meta.json` | features, weight, intervals, metrics, dropped feature |
| Monitoring | `scripts/04_monitor.py` | PSI, monthly error, schema drift, retrain decision |

**Drift types:**
- **Data drift:** input distributions change. Measured by **PSI** = Σ (a − e) · ln(a/e) over 10 bins of the reference;
  < 0.10 stable, 0.10–0.25 moderate, > 0.25 major.
- **Concept drift:** the input → demand relationship changes. Watched via monthly error; a model frozen in Dec 2025 fails
  in **January**, the month it never saw → monthly refit + seasonal half in the ensemble.
- **Schema drift:** raw format changes (four category names were renamed in 2026; the family mapping absorbs them).
- **False alarm we caught:** `active_share6` looked like a major shift (PSI 0.33); it was an artefact of 6-month windows
  not yet full in early 2025. Lesson: a drift alert means investigate, not auto-retrain.

**Retraining rule:** monthly scheduled refit; immediate retrain if a watched feature has PSI > 0.25, live WAPE > 1.2 ×
validation WAPE, or the model stops beating the status quo. Promotion only if it beats the current model and the status
quo on the latest month and passes the fairness check.

---

## 10. Deep-dive questions and answers

### About the problem and data
**Q: Why is this a good decision-support problem?** A real stakeholder with a recurring, costly decision (buy before
demand is known), a measurable status quo to beat, and levers ABD actually controls (procurement, trucks, licences).

**Q: Sales are recorded at the store. Isn't that a flaw?** It is a feature we exploit: H1 measures exactly the gap
between where people live and where they buy. We never interpret county volume as residents' consumption.

**Q: Why not use the published liters column?** It is truncated in three months (a 50 ml bottle reads as 0 L), which
would create a fake holiday slump. We recompute liters for every row and verified it matches the source exactly in clean months.

**Q: How do you know your data is complete?** Row counts match the publisher's API exactly for both years, and the
census join matches 100% of rows.

**Q: Why 2025 and 2026 only?** They are the two published files with full order lines for the period we study. More
years would help seasonality (we have one full cycle); it is our first roadmap item. Spark would be the scale-up path.

### About the hypotheses
**Q: Is −44% causal?** No: an association across 99 counties, robust to controls, permutation, outliers, weighting,
dollars, and replicated out-of-time. A confounder tied to the southern border (rural poverty, religiosity) can't be
excluded. Store-level data from Missouri would allow a geographic regression discontinuity; it isn't public.

**Q: Why only 10 Missouri-border counties? Isn't n too small?** That is why we used a permutation test (exact for small
groups), bootstrap CIs and influence checks. The effect is large (−44%) and replicated in a second year.

**Q: Why HC3 and not ordinary standard errors?** County noise varies with size; ordinary SEs would be too small for small
counties. HC3 is the heteroskedasticity-robust version recommended for small samples.

**Q: Permutation vs bootstrap — what's the difference?** Permutation builds a world with *no* effect to get a p-value;
bootstrap resamples the real data to get a *range* for the effect. One answers "is it real?", the other "how big?".

**Q: Why one-sided tests?** The direction was pre-registered. A result in the opposite direction counts as failure, not
as a two-sided discovery.

**Q: What would make H3 wrong?** If incumbents in entry cities were already declining before entry (parallel trends
violated). The pre-entry months are flat (p = 0.97). If entrants pick growing towns, the bias goes toward zero, so
60% is conservative.

**Q: Why is the diversion CI so wide (28–129%)?** Only 116 entry cities, and city volumes are noisy. The central estimate
is ~60%; we can say "at least a quarter, plausibly all" is diverted, and plan conservatively.

**Q: Why did you change the H3 outcome?** Raw liters were dominated by a few giant stores (not significant). We kept the
pre-registered results in the table and labelled the normalised outcome as a deviation, which is the honest practice.

### About the forecast
**Q: Why not ARIMA or Prophet?** They model one series at a time; we have ~20,000 store-family series with only 22
months each (not enough for per-series seasonal models) and many shared drivers (calendar, store type, competition).
A global gradient-boosting model learns across all series at once. Our seasonal-naive half captures seasonality.

**Q: Why not deep learning (LSTM, transformers)?** Tabular monthly data with 22 points per series; trees are the
state of the art there, faster, and explainable with exact SHAP. Deep models would need far more history.

**Q: Why not K-means?** It isn't in the course handout and doesn't serve the decision; the forecast plus SHAP replaced the
segmentation idea from the pre-submission.

**Q: Why does XGBoost under-forecast?** One seasonal cycle of training data, and trees predict leaf averages, so they
cannot extrapolate above the largest values they saw; giant stores get pulled down. Rescaling series didn't fix it; the
ensemble did.

**Q: Why not stacking (a meta-model on top)?** With one weight to choose, a constrained grid on validation is simpler,
transparent, and less prone to over-fitting than a learned meta-model on three months of validation.

**Q: How do you handle a brand-new store (cold start)?** No history → under-forecast ~24%. Mitigation: forecast from peer
stores (same type, city size) until month 3, plan at high service, and remember the opening order is 1.86× normal (H20).

**Q: Why is store-level WAPE 39%?** A single store-family-month is noisy (an order can slip a week). That's why ABD plans
at county/state level (14% / 5%), where errors cancel (risk pooling).

**Q: Why did tuning not help?** The model's accuracy is limited by information (one seasonal cycle), not by settings. All
20 trials were within 0.5 points. Honest finding: skip tuning under a reduced budget.

**Q: How are the intervals calibrated?** Quantiles of actual ÷ forecast, measured on validation within size bands, then
checked on the untouched test: 77.5% coverage for a nominal 80%.

**Q: What is the business value, concretely?** ~25% less warehouse safety stock (≈ \$0.73M working capital) at 95%
service, and ~84,000 L/month of systematic over-buying removed (≈ \$1.0M of stock per month at cost). Planning estimates,
not audited savings.

### About responsible AI and deployment
**Q: Is the model fair?** It beats today's rule for every group, but errors aren't equal: rural/border counties are
over-forecast ~7% (vs 1% urban). We disclose it, mitigate it with a group bias correction, and audit monthly.

**Q: Why remove ethnicity if it was a real effect (H2)?** H2 is about *understanding* regional mix. In an operational model
it added only 0.5% (below our pre-set 1% bar), so the risk isn't justified. Decided on validation, confirmed by test ablation.

**Q: What happens when the data changes?** ETL checks stop bad loads (row counts, schema, truncation detector); PSI
watches inputs; monthly error watches performance; the retrain rule decides. A model frozen in 2025 already showed
concept drift in January.

**Q: How would ABD actually use this?** A planner opens the dashboard, picks a county, sets the service level, sees the
plan per family with P10–P90, opens a store to see its drivers, and overrides with a reason if needed.

**Q: What would you do with more time?** More years of history; weather and event calendars (football, State Fair,
RAGBRAI); peer-based cold-start model; a randomised regional pilot (A/B) to measure operational value causally.

### "Why X and not Y?" — every major choice, with the evidence behind it
The pattern for answering: **(1) what we chose, (2) the alternative, (3) the number that decided it, (4) when the
alternative would be better.** If we did *not* test an alternative, say so. That is stronger than pretending.

#### The model
| Question | Answer and evidence |
|---|---|
| **Why XGBoost, not LightGBM?** | Tested (8.9): **13.9% vs 13.9%** county WAPE on test, identical bias. A tie within seed noise (0.2 pts). LightGBM is faster, which matters at millions of rows, not at 220,000. |
| **Why not CatBoost?** | Tested: 14.1% county, 5.8% state (vs 5.1%), +3.0% bias, 13× slower. Its advantage is high-cardinality categoricals; we have two small ones. |
| **Why not Random Forest?** | Tested (script 03): test county WAPE alone **15.1% vs 14.6%** for XGBoost; blended on validation 14.29% vs 14.25% (a near-tie). XGBoost was kept because a forest only minimises squared error (no Tweedie loss for zero-heavy, skewed volumes), and boosting reduces *bias* where bagging reduces *variance*. Our problem is under-forecasting (bias). |
| **Why not linear regression (Ridge)?** | Tested: test county WAPE **18.4%**, worse than the status quo (17.3%), with −19% bias on validation. Demand is full of interactions (a lag means different things for Costco and Casey's) that a linear model can't represent without hand-made terms. |
| **Why not ARIMA / Prophet / ETS?** | One model per series: ~20,000 store-family series with only 22 months each is too short for per-series seasonal fitting. A global model learns across series and shares drivers (calendar, store type, competition). The seasonal naive half of the ensemble supplies the seasonality. *Not run*: reasoned from series length. |
| **Why not LSTM / transformers?** | Tabular monthly data, 22 time steps per series: trees are the state of the art there, train in seconds, and are exactly explainable with TreeSHAP. Deep models need far longer histories. *Not run.* |
| **Why one global model, not one per family or per county?** | A global model with `family` as a feature shares what is common (calendar, store size) and still learns family differences. Per-family models would have ~10× less data each. *Not run*: standard practice for many short series. |
| **Why forecast at store × family and add up (bottom-up), not forecast counties directly?** | One model serves all three decisions: store deliveries, county trucks, state buying. Adding up gives the state 5.2% WAPE. *A direct county model was not tested*: it's a fair follow-up. |
| **Why monthly, not weekly?** | ABD buys monthly. 89% of a typical store's liters arrive on one weekday, so a weekly series mostly measures which week the order fell in (noise), not demand. |

#### The loss, the ensemble and tuning
| Question | Answer and evidence |
|---|---|
| **Why Tweedie loss, not squared error on log(liters)?** | Tested: log-target XGBoost has **−22% bias** and 23.9% validation county WAPE, vs −9% and 16.4% for Tweedie. A log model predicts a geometric mean, which systematically under-shoots totals. Tweedie fits non-negative data with many exact zeros and a long tail. |
| **Why Tweedie, not Poisson?** | Tested: XGBoost-Poisson ties on county WAPE (14.1% vs 13.9% test) but is worse at state level (5.9% vs 5.1%) with higher bias (+2.6%). Tweedie also lets us tune *p* (1.4 won), which includes near-Poisson values (1.1). |
| **Why blend with the seasonal naive, not use XGBoost alone?** | Opposite errors: XGBoost under-forecasts (−9% validation bias), the seasonal naive over-forecasts (+8%). Blended: county 13.8% vs 14.6% alone, bias +2%. XGBoost alone *is* better at store level (36.6% vs 39.0%); ABD plans at county and state. |
| **Why the seasonal naive as the partner, not the 3-month moving average?** | The partner must err the *other* way. Moving-average bias is −1% to −6% (same direction as XGBoost), so it can't cancel anything. Seasonal naive bias is +5% to +8%. |
| **Why a weighted average, not stacking (a meta-model)?** | One weight chosen on three validation months is transparent and hard to over-fit. A meta-model would learn from the same three months with more freedom. |
| **Why random search, not grid search or Bayesian (Optuna)?** | All 20 random trials landed within 14.3–14.8%: the error surface is flat. A smarter search can't find a peak that isn't there. Random beats grid for a fixed budget because only a few settings matter. |
| **Why early stopping?** | It picks the number of trees from validation automatically (stops after 100 rounds without improvement), the main defence against over-fitting in boosting. |

#### Evaluation and decision
| Question | Answer and evidence |
|---|---|
| **Why WAPE, not MAPE or RMSE?** | MAPE divides by zero on zero months and lets tiny series dominate. RMSE is dominated by a few giant stores. WAPE reads as "% of liters mis-planned". RMSE is still used, for safety stock. |
| **Why a time-based split, not random k-fold?** | Random folds put future months in training, which is leakage that makes a forecast look better than it can be. We train on the past, tune on Apr–Jun 2026, test once on Jul–Oct 2026. |
| **Why empirical intervals, not quantile regression?** | One model, with intervals calibrated on held-out months (the idea behind split conformal prediction), checked on test: 77.5% coverage for a nominal 80%. Quantile regression needs a model per quantile and isn't guaranteed calibrated. |
| **Why the newsvendor rule, not "order the forecast"?** | Running short and holding extra cost different amounts (~33% lost margin vs ~1.7% a month). The best order is the quantile Cu ÷ (Cu + Co), not the mean. |

#### Hypothesis testing
| Question | Answer and evidence |
|---|---|
| **Why a permutation test, not an ordinary t-test?** | Only 10 Missouri-border counties and skewed volumes: the t-test's normality assumption is shaky. Permutation needs no distribution and is exact for small groups. We report both the HC3 and the permutation p-value; they agree (H1: 0.0003 vs 0.0002). |
| **Why HC3, not ordinary standard errors?** | Small counties are noisier. HC3 is the robust version recommended for small samples (99 counties). |
| **Why stacked DiD, not plain two-way fixed effects?** | Entries happen in different months. Plain TWFE then uses already-treated stores as controls and can be biased, even flip sign. Stacking builds a clean comparison per entry month. |
| **Why not synthetic control?** | It is built for one or a few treated units. We have 116 entry cities, which is what DiD is for. |
| **Why Holm, not Bonferroni or Benjamini–Hochberg?** | Holm controls the chance of *any* false positive like Bonferroni but is never less powerful. Here both keep 6. BH controls the *share* of false positives and would keep **12** (every p < 0.05). We report the strict 6 as findings and the other 6 as leads. |
| **Why Mann–Whitney / Wilcoxon, not t-tests, in H4–H23?** | Store volumes are heavily skewed (top 10% of stores = 62% of liters); rank tests aren't dragged by a few giants. |

#### Data, responsible AI and engineering
| Question | Answer and evidence |
|---|---|
| **Why Parquet, not CSV?** | The 938 MB of raw CSVs become a 147 MB Parquet (6× smaller) with column types stored, so dates and IDs can't silently change type between scripts. |
| **Why SHAP, not the model's built-in "gain" importance?** | Gain says how much a feature was used, not in which direction or for which prediction. SHAP splits *each* forecast into additive, signed contributions, so a planner sees why *this* store got *this* number. |
| **Why SHAP, not LIME?** | LIME fits a local linear approximation and is unstable from run to run. TreeSHAP is exact for tree models. |
| **Why PSI for drift, not a KS test?** | With tens of thousands of rows in each window, a KS test flags trivial differences as "significant". PSI measures the *size* of the shift with standard thresholds (0.10 / 0.25). |
| **Why drop % Hispanic from the model?** | Pre-set rule: keep it only if it improves validation WAPE by > 1%. It improved it by 0.5% (14.27% vs 14.34%), so it was dropped. The test ablation confirmed no loss. |
| **Why MLflow, not Weights & Biases?** | Runs locally on a SQLite file: free, offline, no account, and anyone can reproduce our 43 runs from the repo. |
| **Why FastAPI, not Flask or Streamlit?** | Automatic interactive API docs, input validation, and it serves both machines (JSON endpoints) and people (dashboard). Streamlit is a UI only; Flask needs that wiring added by hand. |
| **Why Docker?** | The same environment on any machine: `docker run` and the planner tool is up, with no "works on my laptop". |

---

## 11. Surprise scenarios (decision challenge)

| Scenario | Response |
|---|---|
| **Budget cut** | lower the service level (95% → 85%) on low-margin families first; skip tuning |
| **Stock-outs become costlier** | raise the service level (newsvendor ratio); forecast unchanged |
| **Overstock becomes costlier** (storage full, capital) | lower the service level; our +2% bias already beats +6.5% |
| **Data drift / new category (e.g. THC drinks)** | PSI + schema checks flag it; refit; family mapping updated |
| **Another truncated month** | the whole-number detector stops the ETL; liters recomputed from bottles |
| **Concept drift / unseen season** | seasonal half of the ensemble + monthly refit |
| **Fairness complaint from rural counties** | show the audit; apply group bias correction; monthly report |
| **Missouri raises its tax** | border counties should recover volume; H1 bounds it; a natural experiment to measure |
| **Wave of new licences** | plan entrants from peers, cut neighbours by ~60% of entrant volume (H3), expect 1.86× opening orders (H20) |
| **ABD runs a price promotion** | stock for the month **after** the cut (H13) |
| **Census update missing** | census features add < 0.1 pt; model runs without them |

---

## 12. Limitations (say them before you are asked)
1. 22 months = one seasonal cycle; holidays and promotions only partly captured.
2. Sales are recorded at the store; census profiles are 2020–24 averages (ecological data).
3. H1/H2 are associations; H3's CI is wide; revenue figures are upper bounds.
4. New stores are under-forecast; rural/border bias remains.
5. Value estimates assume one-month lead time, 20% holding cost.
6. The 20 extra hypotheses are exploratory; only 6 survive a multiple-testing correction.

---

## 13. Numbers to know cold
- **Data:** 4,518,330 rows × 76 columns · 5 sources, 4 publishers · 2,401 stores (281 new, 55 ownership changes) ·
  99 counties · Jan 2025 – Oct 2026 (22 months)
- **Tax:** Missouri \$2.00 vs Iowa \$15.14 per gallon
- **Market:** −3% in 2026 · median store 89% of liters on one weekday · top 10% of stores = 62% of liters
- **H1:** −44% (CI 25–58%) · ≈ \$2.0M margin/yr upper bound · **H2:** +0.8 pts per +10 pts · **H3:** −13%, diversion ~60%,
  small incumbents −21%
- **Forecast:** county WAPE 13.8% vs 17.3% (−20%) · state 5.2% vs 7.0% (−27%) · bias +2.0% vs +6.5% · weight 0.55 ·
  coverage 77.5% · safety stock −25% (≈ \$0.73M) · over-buying ≈ 84,000 L/month
- **SHAP:** own history 78% · store 11% · demographics ~1%
- **20 extra hypotheses:** 10 / 3 / 7 · H13 +49% month after · H20 1.86× · H22 92% · H23 r = −0.69
- **Model alternatives (test county WAPE, untuned + blend):** XGBoost 13.9% · LightGBM 13.9% · CatBoost 14.1% ·
  XGB-Poisson 14.1% · HistGB 14.3% · seed noise 0.2 pts · Random Forest alone 15.1% · Ridge 18.4%

## 14. Where everything is
| Need | File |
|---|---|
| Data, quality, EDA | `notebooks/01_data_pipeline_quality_eda.ipynb` |
| H1–H3 | `notebooks/02_hypothesis_testing.ipynb`, `reports/pre_registration.md` |
| Forecast and decision | `notebooks/03_forecast_model_and_decision.ipynb` |
| SHAP, fairness, monitoring | `notebooks/04_explainability_fairness_monitoring.ipynb` |
| 20 exploratory hypotheses | `notebooks/05_twenty_hypotheses.ipynb`, `reports/twenty_hypotheses.md` |
| LightGBM / CatBoost / Poisson comparison | `scripts/06_model_alternatives.py` → `reports/tables/model_alternatives.csv` |
| Report and slides | `reports/final_report.*`, `reports/final_presentation.*` |
| Code | `src/abd_demand/` (library), `scripts/` (pipeline), `app/` (API) |
