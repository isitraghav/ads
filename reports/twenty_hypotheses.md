# Twenty more hypotheses: what Iowa's spirits orders reveal

**Data:** 4.5 M ABD wholesale order lines (Jan 2025 – Oct 2026) × ACS 2024 × county adjacency × state tax.
**Code:** `scripts/05_twenty_hypotheses.py` → `reports/tables/twenty_hypotheses.csv` (+ `_detail.json`).
**Numbering:** H4–H23, continuing from the three pre-registered hypotheses H1–H3.

**Rules of the game.** Each hypothesis states its direction before it is tested. County-level patterns are
tested on Jan–Oct 2025 **and** Jan–Oct 2026 (the same ten months each year), using OLS with HC3 standard errors plus
a Freedman–Lane permutation test, controlling for log population, income, median age and % bachelor's degree. A
county pattern counts as **supported** only if it shows up in both years. These 20 are *exploratory*, not
pre-registered, so a Holm-adjusted p is reported too. Remember the unit throughout: these are **store orders from
the state wholesaler**, not checkout sales, so the data shows when retailers restock, not when people drink.

## Scorecard

| # | Hypothesis (one line) | Verdict |
|---|---|---|
| H4 | Wisconsin's brandy culture crosses the Mississippi | ✅ Supported |
| H5 | Taste is contagious between neighbouring counties | ✅ Supported for whiskey & rum |
| H6 | Missouri leakage has a ~30-mile travel horizon | 🟡 Partial |
| H7 | Dutch-Reformed Sioux County is a "dry" outlier | ❌ Rejected |
| H8 | St Patrick's Day shows up in wholesale data | ✅ Supported (placebo flat) |
| H9 | The Cinco de Mayo bump is bigger in Hispanic counties | ❌ Rejected (bump real, not ethnic) |
| H10 | College towns empty out in summer | ✅ Supported |
| H11 | Holiday premiumisation is a rich-county thing | ❌ Rejected |
| H12 | Iowa rotates its palate with the seasons | ✅ Supported (7 of 8 families) |
| H13 | Price cuts make retailers stock up, then under-order | ❌ Rejected: timing reversed |
| H14 | Tariff era: imports got dearer than domestic | ❌ Rejected |
| H15 | Inequality builds a price "barbell" | 🟡 Partial |
| H16 | The 50 ml "nip" is a poverty index | 🟡 Partial (within store type only) |
| H17 | More outlets → more drinking, elasticity < 1 | ✅ Supported |
| H18 | The long tail lives in cities | ✅ Supported |
| H19 | Spirits obey Zipf's law | ✅ Supported |
| H20 | New stores have an opening-order honeymoon | ✅ Supported (+ month-1 dip) |
| H21 | Dying stores fade slowly (early warning) | ❌ Mostly rejected: stores die suddenly |
| H22 | Customers belong to the address, not the owner | ✅ Supported |
| H23 | Iowa's whiskey divide is generational (Canadian = old, bourbon = young) | ❌ Rejected: it's rural vs urban, not old vs young |

**10 supported · 3 partial · 7 rejected.** Holm-robust across all 20: H6, H12, H18, H20, H21, H22.

---

## A. Geography & culture

### H4: The Wisconsin brandy belt crosses the river ✅
*Hypothesis:* Brandy share rises the closer an Iowa county is to Wisconsin, where the brandy old fashioned is the state cocktail.
*Test:* brandy share ~ −log(1 + miles to the IA–WI border) + controls, both years.
*Result:* WI-border counties take **2.7%** of liters in brandy vs **1.6%** elsewhere (**1.7×**). Spearman between distance and share = −0.31. Holds in 2025 and 2026 (p = 0.009).
*So what:* Drinking culture follows the river, not the state line. Brandy allocation should be weighted to NE Iowa, not spread pro-rata.

### H5: Taste is contagious ✅ (whiskey, rum)
*Hypothesis:* After removing demographics, neighbouring counties still drink alike.
*Test:* Moran's I on residuals of family share ~ controls + % Hispanic + MO border, queen adjacency, 2,000 permutations.
*Result:* Whiskey (I = 0.14) and rum (I = 0.17) residuals cluster in both years. Vodka clusters in 2025 only. Agave, gin, brandy and RTD do **not** cluster: their geography is fully explained by demographics.
*So what:* Some regional tastes cannot be predicted from census data and spread across county lines. A forecast should borrow strength from neighbouring counties (spatial lag) for whiskey and rum.

### H6: Border leakage has a ~30-mile horizon 🟡
*Hypothesis:* Liters per adult rise with distance from Missouri, then flatten past a travel-cost break point.
*Test:* hinge regression log(L/adult) ~ min(miles to MO, k) + controls, with k grid-searched and bootstrapped.
*Result:* In 2026 the best break is **~30 miles**: inside it, every 10 miles further from Missouri adds **+34%** liters per adult, and beyond it the effect is flat. In 2025 the gradient is linear (+2.7% per 10 mi) with no clear break. The distance effect itself is strong (Holm p = 0.005); the break point is unstable.
*So what:* This sharpens H1. Leakage is a ~30-mile band along the southern border, not a "border county" dummy.

### H7: The Sioux County "religion" outlier ❌
*Hypothesis:* Dutch-Reformed Sioux County buys far less than its demographics, store count and border position predict.
*Result:* It doesn't. Sioux sits at rank 40 of 99 (only −7% vs prediction). The real persistent outliers are **Louisa and Keokuk** (bottom 3 in both years).
*So what:* A good disproof. The cultural explanation people reach for first is wrong. Louisa and Keokuk are the counties worth investigating (for example, cross-border buying into Muscatine or Ottumwa).

## B. The cultural calendar

### H8: St Patrick's Day in the wholesale data ✅
*Test:* store-level paired Wilcoxon of Irish-whiskey share, Feb–Mar vs Apr–Oct, with Scotch (same shelf, no holiday) as a placebo.
*Result:* Irish share is **+29%** (2026) and **+24%** (2025) in Feb–Mar. Scotch moves only **+4%**. Irish peaks in **February** (1.84% vs a 1.16% median), because retailers stock up 3–6 weeks ahead.
*So what:* The calendar is visible in wholesale data a month early. Plan Irish-whiskey stock in January, not March.

### H9: Is Cinco de Mayo a Hispanic holiday or an American one? ❌
*Test:* agave share in the 3 weeks before 5 May vs flanking windows, by county, regressed on % Hispanic.
*Result:* The bump is real: **+16%** agave share in 2026 and **+8%** in 2025 (both p < 0.05). It is **not** larger in Hispanic counties (p = 0.29 in 2025, 0.06 in 2026).
*So what:* Cinco de Mayo works as a mainstream party occasion, not an ethnic one. H2 (Hispanic → agave) explains the **level** of agave demand; the **spike** is statewide.

### H10: College towns empty out ✅
*Test:* each county's (Jun+Jul)/(Sep+Oct) liters ratio, and where Story (ISU) and Johnson (U. Iowa) rank among 99.
*Result:* In 2026, Ames orders **−37%** and Iowa City **−25%** relative to the median county in summer vs autumn, ranking **2nd and 4th lowest of 99**. 2025 shows the same pattern (ranks 2 and 14). P(both this low by chance) = 0.002 in 2026.
*So what:* The academic calendar (students leaving, then football season) is a demand driver that census data cannot capture. Forecasts for these counties need a term-calendar feature.

### H11: The holiday gift economy is for the rich ❌
*Result:* Premium share rises statewide in Nov–Dec (24.8% → 27.1%), and brandy (+14%), rum (+18%) and liqueur (+9%) gain while RTD falls −33%. But the rise is **flat across income** (slope ≈ 0, p = 0.50).
*So what:* Holiday trading-up happens in every income band. Seasonal premium uplift should be applied uniformly.

### H12: Iowa rotates its palate with the seasons ✅ (7/8)
*Test:* store-level paired Wilcoxon of family share, summer (Jun–Aug) vs winter (Nov–Jan), with each family's direction stated in advance.
*Result:* Summer: **RTD +48%**, gin +14%, agave +9%, vodka +6%. Winter: brandy +7%, liqueur +6%, whiskey +6%. **Rum is the surprise:** it is a *winter* spirit in Iowa (spiced rum, holiday baking and punches), not a beach drink.
*So what:* Family mix needs month-specific shares. A trailing 3-month share (`family_share_roll3` in the forecast) will lag every turn of the season.

## C. Prices & economics

### H13: Price cuts make retailers stock up, then under-order ❌ (timing is reversed)
*Test:* item × month two-way fixed effects on 2,251 regularly stocked items. A "cut" means cost ≥5% below the item's modal cost.
*Result:* The opposite of forward-buying. Orders are only +14% during the cut month, then **+49% the month after the price returns to normal** (p < 1e-7), with no dip before. Among the 12 cleanest one-month cuts, the median order is −5% in the cut month and roughly **doubles** the month after.
*So what:* Price cuts appear to work through the shelf. Retailers pass the cut to consumers, sell through, and reorder the following month at full price. The volume response lags the price change by one order cycle, so promotion stock must be planned for month t+1. (The mechanism is inferred: n = 124 promotion spells in 2026.)
*⚠ Data quality:* the whole-dollar truncation of Nov 2025–Jan 2026 (slide 9) also hits `state_bottle_cost` (6.74 → 6.00), which makes ~900 items look like they were on sale. New finding: the field is also **blank for Jan–Jun 2025**. This test uses only clean-price months, and `data_quality_report.json` should record both issues.

### H14: The tariff era shows on the shelf ❌
*Result:* Same-item state cost, Jul–Oct 2026 vs 2025, barely moved for any origin: Canada −0.2%, Mexico −0.6%, UK/Ireland −0.7%, other imports +0.5%, USA −1.1% (Mann–Whitney p = 0.38).
*So what:* No tariff pass-through is visible in ABD's costs so far, so price is not why demand shifted between 2025 and 2026.

### H15: Inequality builds a barbell 🟡
*Result:* High-Gini counties do buy a wider spread of price tiers (SD of log $/L rises in both years). But their *average* price rises too, and neither effect survives population weighting.
*So what:* Unequal counties trade up rather than polarise. This is a weak signal.

### H16: The 50 ml nip is a poverty index 🟡
*Result:* Across counties: no effect (p = 0.58). Comparing stores of the **same type**, stores in poorer counties spend **+3.3 pts more of their dollars on minis per +10 pts poverty** (p = 0.02).
*So what:* Store format explains most of the nip market (convenience stores sell minis). Within a format, poverty shows through.

### H17: More outlets, more drinking, but less than 1:1 ✅
*Test:* log(L/adult) ~ log(stores per 10k adults) + controls + MO border.
*Result:* Elasticity **0.59 (2025) / 0.76 (2026)**, and **0.39 / 0.52 population-weighted**, against **0.40 implied by H3's 60% diversion ratio**.
*So what:* Two independent methods, a cross-sectional regression and a difference-in-differences of store entry, give the same answer: **each new licence adds roughly 40–50% of its volume as new demand.** That matters for licensing policy (public health) and warehouse planning alike.

### H18: The long tail lives in cities ✅
*Test:* rarefaction. Count distinct SKUs in 892 random order lines per county, the size of the smallest county, so every county is compared on an equal sample (removing the mechanical "bigger county = more SKUs" effect).
*Result:* For the same number of order lines, the densest decile buys **547 distinct products** vs **322** in the sparsest (+75 per log-unit of density, Holm p = 0.008). Surprise: vendor concentration is *not* lower in cities (HHI slope ≈ 0). Cities get more variety from the same big suppliers.
*So what:* Rural assortments can be cut to the head without losing sales. Urban warehouses need long-tail capacity.

### H19: Spirits obey Zipf's law ✅
*Result:* Log-volume vs log-rank slope = **−1.10** (Zipf = −1). The **top 1% of 4,792 SKUs carry 38% of liters** and the top 20% carry **91%**. The head barely moves: Spearman ρ = 0.93 for the top-100 between 2025 and 2026.
*So what:* About 50 SKUs drive more than a third of statewide volume. Forecast those individually; the tail can be modelled as a group.

## D. Store behaviour

### H20: The new-store honeymoon ✅
*Result:* A new store's **opening month is 1.86× its steady-state level** (shelf fill). It then **dips to 0.79×** in month 1 (p < 1e-7) and reaches ~0.98× by month 5–6 (n = 191 genuinely new stores).
*So what:* A naive forecast built on the opening order overshoots the next month by ~2.4×. The planner should cap entrants at month 0 and forecast them from peer stores until month 3.

### H21: Dying stores fade slowly ❌ (mostly rejected)
*Result:* In their last 3 full months, exiting stores are only −6% in volume and −5% in product range below their own seasonal norm. As a 3-month early-warning signal the decline is useless: **AUC = 0.53** (coin flip).
*So what:* Stores close suddenly (sale, licence loss, closure), not after a slow fade. ABD cannot see exits coming in order data, so the forecast should carry a small exit-hazard haircut rather than rely on a trend signal.

### H22: Customers belong to the address, not the owner ✅
*Result:* For 55 ownership changes, the new licensee's family mix is **half as far** (Jensen–Shannon 0.12 vs 0.25) from its predecessor's as from a random store of the same type, and it sells **92%** of the predecessor's monthly volume.
*So what:* An ownership change is not a new store. Forecast it with the predecessor's history; this validates the 55 store reclassifications in the data pipeline.

## E. The whiskey aisle

### H23: Iowa's whiskey divide is generational ❌ (age is only a proxy)
*Hypothesis:* Canadian whisky (Iowa's #2 category overall, unusual nationally) is the older generation's pour, and bourbon the younger one's. Within whiskey, Canadian share should rise with a county's median age and bourbon share should fall, after controlling for density, income and education.
*Test:* county share of whiskey liters (Canadian; bourbon) ~ median age + log density + income + % bachelor's, HC3 + permutation, both years.
*Result:* The raw pattern looks generational. The oldest fifth of counties pour **60% Canadian / 19% bourbon**, the youngest **51% / 23%**. But once density and education are held fixed, the age effect **reverses** (Canadian −2.3 pts per +5 years, p = 0.96). Two other variables do the work:
- **Rurality drives Canadian.** Density alone explains 48% of the variation in Canadian share (r = −0.69). A 10× denser county pours **21 pts less** Canadian whisky.
- **Education drives bourbon.** % bachelor's alone explains 45% of bourbon share. Each +10 pts of degree holders adds **+4 pts** bourbon.

*So what:* A good example of a confounder. "Old counties drink Canadian" is true but misleading, because old counties are also rural and less educated. Allocate Canadian whisky by rurality and bourbon by education level, not by age.

---

## Caveats to say out loud
- **Exploratory, not pre-registered.** With 20 tests, about one "significant" result is expected by chance. Six findings survive a Holm correction. H4, H8, H10 and H17 do not survive it by themselves, but each **replicates in two independent years**, which is the stronger test.
- **Orders ≠ consumption.** Every calendar effect is a *restocking* effect, typically 2–6 weeks ahead of the occasion.
- **Associations, not causes**, except where a design gives causal leverage (placebo in H8, within-item FE in H13, rarefaction in H18).
- **Thresholds** (5% price cut, equal-size rarefaction samples, 3-month windows) were fixed before the results were seen. Sensitivity to them is not yet reported.
