# Pre-registration of hypotheses and decision rules

**Written:** 2026-11-11, *before* any 2026 sales outcome was aggregated or inspected.
At this point the 2026 file (Iowa Data Hub dataset 1263, Jan–Oct 2026) had only been profiled for
row counts, blank fields, store-ID format and first/last order dates. No county, category or
store-level outcome from 2026 had been computed.

**Discovery data:** 2025 (dataset 1262) — already used in the pre-submission and for exploration.
**Confirmation data:** Jan–Oct 2026 (dataset 1263), never seen before these rules were fixed.

This fixes the pre-submission's main caveat: there, patterns spotted in the full-year 2025 EDA were
"confirmed" on half of the same year. Here the confirmation is a genuinely new sample.

---

## Why the pre-submission hypotheses changed

| Pre-submission | Status in final project | Reason |
|---|---|---|
| H1: larger population → more liters per adult and more orders per person | **Re-interpreted as H1** | Supported in the pre-submission, but sales are recorded where the *store* is. Exploration on 2025 showed the lowest-buying counties all sit on the Missouri border, where spirits are taxed at $2.00/gal against Iowa's implied $15.14/gal. So "big counties drink more" may really be "shoppers travel to buy". H1 tests that mechanism directly. |
| H2: % Hispanic → tequila/mezcal share | **Kept as H2**, confirmed out-of-time | Supported in the pre-submission; it directly informs the category mix ABD supplies to each region. |
| H3: income → premium share | **Dropped** | Rejected in the pre-submission notebook (ρ ≈ 0.01 in Jan–Jun). It is reported as a negative finding, not re-tested. |
| H4: % aged 21–34 → RTD / flavoured share | **Dropped** | The age effect disappeared once county size was controlled (it was a proxy for urbanisation). |
| (new) | **H3: store entry** | The final project must support a decision. ABD licenses every store, and 2025–26 has ~20 new stores a month, so entry can be studied as a natural experiment (difference-in-differences). |

The pre-submission notebook screened 14 candidates. The course brief asks for 2–3, so the final
project tests **three**, each linked to a decision.

---

## H1 — Cross-border leakage (revises pre-submission H1)

> Counties where cheaper out-of-state spirits are close by buy fewer liters per adult from Iowa stores.

- **H1a (primary).** Counties that border Missouri buy fewer liters per adult 21+ than other counties,
  controlling for log population, median household income, median age and % bachelor's degree.
- **H1b (dose-response).** The bigger the gap between Iowa's spirits tax and the cheapest adjacent
  state's tax (Tax Foundation 2026; 0 for interior counties), the fewer liters per adult.
- **H1c (fingerprint).** The shortfall is concentrated in 1.75 L "handle" bottles, the stock-up size
  for which a trip across the border pays: Missouri-border counties have a lower handle share of liters.

**Unit:** county (n = 99). **Outcome:** log(liters per adult 21+), Jan–Oct 2026.
**Test:** OLS with HC3 robust SEs, one-sided in the stated direction, *plus* a permutation test of the
coefficient (5,000 shuffles of the border flag). **Supported if** both p < 0.05 on 2026 **and** the
sign holds with population weights and after removing influential counties (Cook's D > 4/n).

## H2 — Ethnic composition and agave share (carried from pre-submission H2)

> Counties with a larger Hispanic population buy a larger share of tequila/mezcal (agave), after
> controlling for log population and median income.

**Outcome:** agave share of liters, Jan–Oct 2026. Same test and decision rule as H1.

## H3 — New licences mostly redistribute sales (new)

> When a new Class E licensee opens in a city, the stores already there order less, so the city's
> total purchases rise by less than the newcomer's volume.

- **Design:** stacked difference-in-differences event study. *Treated:* incumbent stores in cities
  where a genuinely new store first orders between Apr 2025 and Jun 2026. *Control:* incumbents in
  cities with no store entry or exit in the window. Store fixed effects and cohort-by-month fixed
  effects; standard errors clustered by city.
- **Outcome:** incumbent store's monthly liters (log(1+L) for the % effect; liters for the volume effect).
- **Supported if** the average post-entry effect is negative with p < 0.05, the pre-entry (lead)
  coefficients are jointly insignificant (p > 0.05), and the placebo passes (see below).
- **Placebo:** a new licence at the address of a store that has just stopped ordering is an ownership
  change, not new competition, so it should show **no** effect on incumbents.
- **Derived quantity:** diversion ratio = incumbents' lost liters ÷ entrant's liters.

## Forecasting KPI (decision support)

- **Decision:** liters to plan per county × spirits family for next month (warehouse procurement
  and delivery capacity).
- **Primary metric:** WAPE (weighted absolute % error) on an untouched out-of-time test period,
  Jul–Oct 2026. **Status-quo baseline:** seasonal naive (same month last year).
- **Success:** at least a 10% relative WAPE reduction against the status-quo baseline at the
  county × family level, with absolute bias within ±3%.

All three hypotheses are reported whatever the outcome. An unsupported or inconclusive result is a
valid finding.
