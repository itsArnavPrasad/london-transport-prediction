# Modelling Commuter Mode Choice in London: Discrete Choice Models, Gradient Boosting and a Fine-Tuned Small Language Model

Project plan, from data to report. Hardware target: MacBook Pro M1, 8 GB RAM.

---

## 1. Objective and research questions

The project models which transport mode a traveller chooses for a trip, using 81,086 real trips from the London Passenger Mode Choice (LPMC) dataset. The four modes are walking, cycling, public transport (PT) and driving. The work follows the structure of the course brief on commuter mode choice: identify the factors associated with mode selection, estimate choice probabilities, compare model performance, and assess the relative importance of individual, trip-related and service attributes.

Five questions organise the study.

1. **Which factors explain mode choice?** Estimated with a Multinomial Logit (MNL) and a Nested Logit (NL), including the relative importance of person, trip and service attribute groups.
2. **How well can each model predict the mode chosen in a later year?** Compared across MNL, NL, gradient boosting, and a small language model used zero-shot and after fine-tuning.
3. **What do travellers value?** Value of travel time (£/hour) and demand elasticities, estimated from the logit models and compared with what the machine-learning models imply.
4. **How do mode shares respond to policy?** Scenarios for PT fares, congestion charging, driving time and PT service speed.
5. **Does a small language model learn travel behaviour from data?** Zero-shot against LoRA fine-tuned, with a sample-efficiency curve showing how much survey data the fine-tuning needs.

**Scope note for the course.** The brief lists walking, two-wheelers, auto-rickshaws, cars, buses and local trains, and assumes a dataset with income, education, occupation, dwelling type and parking. No public dataset with those variables exists. LPMC is the closest open substitute: real revealed-preference trips, individual-level socio-demographics, and mode-specific times and costs. The method is identical; only the city, the mode set and the variables differ. This should be confirmed with the instructor early (see section 12).

---

## 2. Data

### 2.1 Source and documentation

The data come from Hillel, Elshafie and Jin (2018), *Recreating passenger mode choice-sets for transport simulation: a case study of London, UK* (Proceedings of the ICE, Smart Infrastructure and Construction 171(1), 29-42), and must be cited. The documentation is `data/raw/CS_LPMC.pdf`, the data file is `data/raw/lpmc.dat` (tab-separated, 81,086 rows, 32 columns).

Trip diaries come from the London Travel Demand Survey (LTDS), a rolling household survey run by Transport for London, covering April 2012 to March 2015. Trips outside London and trips that did not use TfL services were removed. A trip diary records only the mode that was used, not the alternatives that were available. The authors therefore generated the missing alternatives: an online journey planner produced a route for each of the four modes, and a cost model computed individual-specific PT fares, fuel costs and congestion charges. **The attributes of the chosen mode are therefore modelled, not observed, in the same way as those of the rejected modes.** This is a limitation that belongs in the report.

### 2.2 Variables

All times are in hours, all costs in GBP, distance in metres.

| Group | Variables (as documented) |
|---|---|
| Identifiers | `trip_id`, `household_id`, `person_n`, `trip_n` |
| **Target** | `travel_mode`: 1 walk, 2 cycle, 3 public transport, 4 drive |
| Person | `age`, `female`, `driving_license`, `car_ownership` (0 no car, 1 fewer than one car per adult, 2 one or more cars per adult), `faretype` (1 full fare, 2 age 16+, 3 child, 4 disabled, 5 free), `bus_scale` (0 free, 0.5 half price, 1 full price) |
| Trip | `purpose` (1 home-based work, 2 home-based education, 3 home-based other, 4 employer's business, 5 non-home-based other), `distance` (straight line), `start_time` (0-24 decimal hours), `day_of_week`, `travel_year`, `travel_month`, `travel_date`, `survey_year` (1: 2012/13, 2: 2013/14, 3: 2014/15) |
| Walk and cycle | `dur_walking`, `dur_cycling` |
| Public transport | `dur_pt_access`, `dur_pt_rail`, `dur_pt_bus`, `dur_pt_int`, `pt_interchanges`, `cost_transit` |
| Drive | `dur_driving`, `cost_driving_fuel`, `cost_driving_ccharge`, `driving_traffic_percent`, `fueltype` |

The documentation notes that total PT time is the sum of access, rail, bus and interchange durations, and that total driving cost is fuel plus congestion charge. Expansion factors exist in the source survey but are not provided, so every result describes the sample, not the London population.

### 2.3 Audit findings (already verified on the file)

| Check | Result | Consequence |
|---|---|---|
| Missing values, duplicate rows or trip IDs | None | No imputation needed |
| Counts | 81,086 trips, 31,954 persons, 17,616 households | Matches the documentation |
| Households or persons appearing in more than one survey year | **0** | A split by survey year has no household leakage |
| Trips per person | Mean 2.5, maximum 16 | Trips of one person are correlated, so validation must be grouped |
| Mode shares overall | Walk 17.6%, cycle 3.0%, PT 35.3%, drive 44.2% | Cycling is rare, so macro-F1 and per-mode recall are needed alongside accuracy |
| Mode shares by survey year | Drive 44.0 / 45.6 / 42.8%, PT 35.2 / 34.6 / 36.1% | Mild drift; the test year is a fair but not trivial forecast |
| Drive trips by people without a licence | 8,322 of 35,808 (23%); 5,053 are by people under 17 | **"Drive" includes car passengers.** Availability cannot be defined as "licence and car". All four modes are treated as available for every trip, as in the source paper, and the mode is labelled "car (driver or passenger)" |
| Zero-valued alternatives | Driving time and total PT time are never zero | No missing alternatives |
| PT fare equal to zero | 26,120 trips (32%); 20,182 are `faretype` 5 (free) | The cost coefficient is identified mainly from fare-paying travellers; the value of time must be reported for that group |
| `fueltype` | Type 6 ("average car") occurs almost only where `car_ownership` is 0 | A proxy for car ownership used by the cost model; excluded from all models |

Two statements I made earlier in the conversation were corrected by this audit: `car_ownership` is "cars per adult" rather than a count of cars, and driving availability cannot be derived from licence and car ownership.

---

## 3. Problem formulation

Each trip is one observation of a choice among four alternatives. For alternative *j* and trip *n*, the model produces a probability P(*j* | trip *n*) that sums to one over the four modes. All models share the same trip information, so any difference in performance reflects the model and not the data it saw. The predicted mode is the one with the highest probability.

---

## 4. Train, validation and test design

This design is the foundation of the whole comparison, so the reasoning is set out in full.

**Test set: survey year 3 (April 2014 to March 2015), 26,320 trips.** The models are trained on the earlier years and evaluated on a later one, which mimics the real use of a mode choice model: estimating behaviour from past surveys and forecasting a future year. The audit shows that no household or person appears in both periods, so the test set cannot leak information through repeated observations of the same people. The test set is locked: it is not used for model selection, hyperparameters, prompt design or any decision about specification. It is evaluated once per model, at the end.

**Training pool: survey years 1 and 2, 54,766 trips (April 2012 to March 2014).**

**Validation inside the training pool.** Trips by the same person are correlated, so random row-wise validation would overstate performance. All tuning uses grouped splits by `household_id`:

- For gradient boosting: 5-fold grouped cross-validation (stratified by mode where possible).
- For logit specification and nest structure: likelihood-ratio tests and information criteria on the training pool (no tuning of parameters is needed).
- For the language model: a fixed validation set of 1,000 trips drawn from households that are excluded from the fine-tuning data. Prompt wording, learning rate and number of epochs are chosen only on this set.

**The shared evaluation subsample.** Running a language model over all 26,320 test trips three times is too slow on an 8 GB M1. Therefore:

- *Full test set (26,320 trips):* MNL, NL and gradient boosting.
- *Head-to-head test subsample (4,000 trips):* a simple random sample drawn from the test set with a fixed seed, saved to disk before any modelling. Every model is scored on it. The classical models are also scored on the full set, which shows whether the subsample is representative. With 4,000 trips the standard error of an accuracy estimate is about 0.75 percentage points.
- *Scenario subsample (1,000 trips):* the first 1,000 of the 4,000, used for the language-model policy runs.

**Uncertainty.** Every reported metric carries a 95% confidence interval from a cluster bootstrap that resamples whole households (1,000 resamples). Model comparisons use paired bootstrap differences on the same resamples.

**Other safeguards.** Scaling and encoding are fitted on training data only. All random seeds are fixed. Trip IDs of every split are written to `data/processed/splits.csv`.

---

## 5. Features and preprocessing

**Used by every model.** Times are converted to minutes.

| Group | Features |
|---|---|
| Person | age, female, driving licence, car ownership, fare type, bus scale |
| Trip | purpose, distance, start time, day of week (plus peak-period and weekend indicators for the logit models) |
| Walk | walking time |
| Cycle | cycling time |
| PT | access time, rail time, bus time, interchange time, total time, number of interchanges, fare |
| Drive | driving time, fuel cost, congestion charge, total cost, traffic variability |

**Excluded, and why.** Identifiers and `trip_n` (no behavioural meaning); `survey_year` and the travel date fields (they define the split); `fueltype` (a proxy for car ownership, see the audit).

**Engineered.** Total PT time, total driving cost, peak-period indicator (07:00-10:00 and 16:00-19:00), weekend indicator, and a free-fare indicator for PT.

**Not available, stated as limitations.** Income, education, occupation, dwelling type, parking and satisfaction, all of which appear in the brief. `faretype`, `bus_scale` and `car_ownership` are weak proxies for income, and this is a main limitation of the value-of-time analysis.

---

## 6. Models

### 6.1 Multinomial Logit

The utility of each mode is a linear function of its own attributes and of the traveller's characteristics, plus an alternative-specific constant, with walking as the reference alternative. Estimation uses Biogeme 3.3.x, with robust standard errors.

The specification is built in steps so that each addition is justified by a likelihood-ratio test:

| Model | Content |
|---|---|
| M0 | Alternative-specific constants only (reproduces the market shares; the null model) |
| M1 | + one generic travel-time and one generic cost coefficient |
| M2 | + mode-specific time coefficients; PT time split into access, in-vehicle (rail, bus), interchange, and a penalty per interchange |
| M3 | + person attributes (age, gender, licence, car ownership, fare type) as shifters on the mode constants |
| M4 | + trip attributes (purpose, distance, peak and weekend) |

The best-supported model is carried forward. Reported outputs: coefficients with standard errors and t-statistics, log-likelihood, rho-squared and adjusted rho-squared, AIC and BIC, and the likelihood-ratio tests between steps. Sign checks are part of the analysis: time and cost coefficients should be negative.

### 6.2 Nested Logit

Nested Logit relaxes the independence of irrelevant alternatives assumption of the MNL by allowing alternatives in the same nest to be closer substitutes. Two nesting structures are estimated on the best MNL specification:

- **Active versus motorised:** {walk, cycle} and {PT, drive}.
- **Car versus non-car:** {drive} alone and {walk, cycle, PT}.

In Biogeme's convention the nest parameter must be at least 1, and a value of exactly 1 collapses the nest back to MNL. It is tested against 1 with a t-test and a likelihood-ratio test. The structure that is both statistically significant and behaviourally sensible is kept. If neither nest differs from 1, the finding is that MNL is adequate, which is a legitimate result.

### 6.3 Gradient boosting

`HistGradientBoostingClassifier` from scikit-learn, because it is already part of the stack and handles this data size in minutes without extra dependencies. Inputs are the raw feature set above. Hyperparameters (learning rate, maximum leaf nodes, L2 penalty, minimum samples per leaf) are tuned by grouped cross-validation on the training pool with early stopping. Probabilities are checked for calibration on the validation folds, and a recalibration step (temperature or isotonic) is applied only if the reliability curve calls for it.

Importance is measured by permutation importance on validation data, aggregated by attribute group: person, trip, walk and cycle, PT, drive. The same grouping is used for the logit models (drop a group and measure the loss in log-likelihood), so the two views can be compared.

### 6.4 Language model: Qwen2.5-0.5B-Instruct

A 0.5-billion-parameter model is the largest that trains reliably on 8 GB of unified memory. It runs with Apple's MLX framework, quantised to 4 bits for training (QLoRA).

**From a trip to a prompt.** Each trip is rendered as a short text. The same template serves every language-model run:

> You are a traveller in London. Profile: a 34-year-old woman, holds a driving licence, lives in a household with fewer than one car per adult, pays full public transport fares. Trip: commute to work, 6.2 km, Tuesday 08:15.
> Options: walk 78 min. cycle 24 min. public transport 31 min (8 min access, 18 min in vehicle, 3 min interchange, 1 change), fare £2.40. drive 19 min, fuel £0.90, congestion charge £0.00, heavy traffic.
> Which mode do you choose? Answer with one word: walk, cycle, public or drive.
> Answer:

**From a prompt to probabilities.** Instead of asking the model to write an answer and parsing it, the model is run once on the prompt and the scores of the first token of each of the four answer words are read at the answer position and passed through a softmax. This gives a proper probability distribution, one forward pass per trip, and no parsing failures. It also avoids the position bias that letter options (A, B, C, D) cause. A check on 1,000 validation trips with the option order reversed measures any remaining order effect.

**Zero-shot run.** The instruction-tuned model, no training, the prompt above. A prior-corrected variant is reported alongside the raw one: a single additive adjustment per mode, fitted on validation trips, which shows how much of a zero-shot model's error is only a mode-share bias.

**Fine-tuned run.** LoRA adapters (rank 8) on the attention projections, trained on 8,000 randomly chosen training-pool trips with the loss applied only to the answer token. The target is the chosen mode. Learning rate is chosen from {1e-4, 2e-4} on the validation set, one epoch over 8,000 trips, gradient checkpointing, batch size 2-4 and a maximum sequence length of about 320 tokens to stay inside memory. Inference then uses the same one-pass scoring as the zero-shot run.

**Sample-efficiency curve.** Fine-tuning is repeated on nested subsets of 500, 2,000 and 8,000 trips. MNL and gradient boosting are refitted on the same subsets, plus the full training pool. The result is a plot of test log-loss against the number of training trips for each model family.

I am writing the MLX command-line options from memory. The exact flags and the model names on Hugging Face (`Qwen/Qwen2.5-0.5B-Instruct` and an MLX 4-bit community conversion) must be checked against `mlx_lm.lora --help` before the first run.

---

## 7. Evaluation

All metrics are computed per model on the 4,000-trip head-to-head subsample, and for the classical models also on the full test set.

| Metric | Why |
|---|---|
| **Multiclass log-loss** (primary) | A proper scoring rule that works for every model type, because all of them output probabilities |
| Accuracy | Familiar headline number |
| Macro-F1 and per-mode recall | Cycling is 3% of trips, so accuracy alone hides failures |
| Brier score | Second proper scoring rule |
| Expected calibration error and reliability curves | Whether a stated 60% really means 60% |
| **Mode-share error** | Mean absolute difference between the predicted share (average of the probabilities) and the observed share, for the four modes. Planning use cares about aggregates |
| Confusion matrices | Where each model's mistakes go |
| Segment breakdown | Metrics by purpose, car ownership, and distance band |

Reference predictors give the floor: always predict drive (majority class), and predict the training market shares for every trip (the MNL M0 model).

---

## 8. Behavioural analysis

### 8.1 Value of travel time

From the best logit model, the value of time for a component is the ratio of its time coefficient to the cost coefficient, converted to £/hour. It is reported for driving, PT in-vehicle time, PT access and egress, PT interchange time, walking and cycling, with delta-method confidence intervals. It is repeated by trip purpose and by car ownership, using the interactions that the specification supports. Results are compared qualitatively with published UK transport appraisal values (to be looked up in the DfT appraisal guidance, not quoted from memory). The caveat that the cost coefficient is identified mainly from travellers who pay a fare applies throughout.

For the gradient-boosting and language models there are no coefficients, so an **implied value of time** is defined on the predicted probabilities: for a mode, the ratio of the change in its choice probability per minute of extra travel time to the change per extra pound of cost, averaged over trips by finite differences. For a logit model this ratio equals the coefficient ratio exactly, so the definition is consistent across all models. Trips where the mode's probability is saturated near 0 or 1 are excluded, and medians are reported because the ratio is noisy.

### 8.2 Elasticities

Arc elasticities for every model, computed by perturbing one variable for all trips and measuring the percentage change in the aggregate predicted share of the affected mode: PT fare, driving cost, driving time, and PT in-vehicle time. Using the same perturbation procedure for all models makes the comparison fair. For the logit models, results are cross-checked against Biogeme's analytical elasticities.

### 8.3 Policy scenarios

Applied to the test trips, with the base case being each model's own predictions without a change:

| Scenario | Change |
|---|---|
| S1 | PT fares reduced by 20% |
| S2 | Driving costs (fuel and congestion charge) increased by 20% |
| S3 | Driving time increased by 15% (more congestion or reallocated road space) |
| S4 | PT in-vehicle time reduced by 15% (faster and more frequent service) |
| S5 | S1 + S2 + S4 as a combined package |

The output is the shift in mode shares in percentage points with confidence intervals, for each model. Scenarios S1 to S4 also provide the perturbations needed for the elasticities and implied values of time in 8.1 and 8.2. Machine-learning and language models can respond implausibly to a counterfactual, so every model also gets a **monotonicity check**: the share of trips where the predicted probability of a mode moves in the wrong direction when its own cost or time improves or worsens. Results are shares of the sample; without expansion factors they cannot be read as London-wide totals.

### 8.4 Importance of attribute groups

For the logit models, each group of variables (person, trip, walk and cycle, PT service, drive service) is dropped in turn and the loss in log-likelihood is tested with a likelihood-ratio test. For gradient boosting, group permutation importance is used. This answers the brief's request to assess the relative importance of individual, trip-related and service attributes, with socio-demographic coverage limited by the variables available.

---

## 9. Compute budget (M1, 8 GB)

These times are estimates. The language-model timings in particular are not measured; they will be re-measured on the first 100 trips and the plan adjusted.

| Stage | Task | Estimated time |
|---|---|---|
| 0 | Environment, data audit script, splits | 30 min |
| 1 | MNL specification ladder | 30 min |
| 2 | Nested Logit | 20 min |
| 3 | Gradient boosting: tuning and final fit | 30 min |
| 4 | Prompt design on validation, zero-shot evaluation | 40 min |
| 5 | LoRA fine-tuning on 8,000 trips, evaluation | 60 min |
| 6 | Sample-efficiency curve (500 and 2,000 trips, plus classical refits) | 30 min |
| 7 | Value of time, elasticities, scenarios (classical, plus language models on 1,000 trips) | 60 min |
| **Total compute** | | **about 4.5 hours** |
| 8 | Figures, tables and report writing | 2-3 hours, not compute |

That is inside the 5-6 hour training budget. If time runs short, the first thing to cut is the 500-trip point of the sample-efficiency curve, then the second nesting structure. Fine-tuning must run with other applications closed, since macOS needs about 3-4 GB of the 8 GB.

**Gates between stages.** After stage 1, the MNL signs must be sensible and the log-likelihood must beat M0 by a large margin; if not, the data pipeline is wrong, not the model. After stage 3, gradient boosting should beat MNL on log-loss; if it does not, features are being fed incorrectly. After stage 4, the zero-shot model should differ from the majority-class predictor; if every probability collapses to the same distribution, the prompt or token scoring is broken.

---

## 10. Repository layout and reproducibility

```
ai-ds-project/
  PLAN.md
  pyproject.toml            # Python 3.12, locked by uv
  data/
    raw/                    # lpmc.dat, CS_LPMC.pdf (never edited)
    processed/              # splits.csv, features, language-model JSONL
  src/
    common.py               # loading, splits, features, metrics, bootstrap
    s0_audit.py             # checks from section 2.3, writes the splits
    s1_mnl.py
    s2_nested.py
    s3_gbm.py
    s4_llm_zeroshot.py
    s5_llm_finetune.py
    s6_sample_curve.py
    s7_behaviour.py         # value of time, elasticities, scenarios
    s8_figures.py
  results/
    tables/  figures/  predictions/    # one probability file per model
  report/
```

Every script is numbered by stage and reads only from the files produced before it. Each model writes its predicted probabilities for the test trips to `results/predictions/`, so all comparison tables and figures can be regenerated without refitting. Seeds are fixed (42), the environment is created with `uv venv -p 3.12`, and packages are Biogeme, pandas, numpy, scikit-learn, matplotlib and mlx-lm. The earlier attempt to create a virtual environment was interrupted, and no `.venv` exists yet.

---

## 11. Risks and limitations

| Risk or limitation | Handling |
|---|---|
| Zero-shot 0.5B model barely beats the majority class | Expected for a model this small. Report it honestly; the prior-corrected variant and the fine-tuned run are the informative comparison |
| Out-of-memory during fine-tuning | Shorter sequences, smaller batch, gradient checkpointing, close other applications; fall back to fewer adapter layers |
| Nested Logit does not converge, or the nest parameter lies outside its valid range | Report as a finding; keep the MNL |
| Cycling is rare (3%) | Macro-F1, per-mode recall, and confidence intervals that make clear how uncertain cycling results are |
| Test year differs from the training years | Intended: it is the forecast setting. Report mode-share drift between years |
| Zero fares for 32% of trips | Report the value of time for fare-paying travellers; run the logit with and without a free-fare indicator as a robustness check |
| Alternatives are simulated, not observed | State clearly; it applies to all models equally |
| "Drive" includes car passengers | Label the mode "car"; passengers do not choose the mode, which weakens a behavioural reading of the car coefficients for children |
| No income variable, no expansion factors | State clearly; value of time is not income-specific and results describe the sample |
| Language-model scenario responses may be implausible | Monotonicity check, reported next to the scenario results |

---

## 12. Report outline

1. **Introduction and objectives** (the five questions, the scope adaptation from the brief).
2. **Data** (source, collection, variables, audit findings, limitations; descriptive statistics and mode shares by distance, purpose and car ownership).
3. **Methodology** (train/validation/test design, features, the four model families, evaluation metrics). The split reasoning from section 4 appears here in full prose.
4. **Discrete choice results** (the MNL ladder with tests, the final coefficient table, the nested logit comparison).
5. **Predictive comparison** (the main table and figures: all models on the shared subsample, calibration, confusion matrices, segment results, sample-efficiency curve).
6. **Behavioural insights** (value of time, elasticities, importance of attribute groups).
7. **Policy scenarios** (mode-share shifts with intervals, the monotonicity diagnostic).
8. **Discussion** (what each model family is good for: interpretation against prediction against a language model that needs no feature engineering; implications for sustainable urban transport planning; limitations).
9. **Conclusion.**
10. **References and appendix** (full coefficient tables, prompt template, hyperparameters, reproducibility notes).

Planned figures: mode share by distance band; the MNL coefficient plot; the sample-efficiency curve; reliability curves; confusion-matrix panel; value-of-time comparison across models; scenario shift bars with confidence intervals.

---

## 13. Decisions made on your behalf

These are the open choices I settled; any of them can be changed.

- Test set is the last survey year; validation is grouped by household inside the training pool.
- Language-model comparison on a 4,000-trip test subsample, scenarios on 1,000, for compute reasons.
- Answer scoring by first-token probabilities, not generated text.
- Gradient boosting uses scikit-learn's histogram implementation, no extra library.
- The few-shot variant, a larger local model, Mixed Logit and Claude models are out of scope.
- The sample-efficiency curve is included but is the first item to cut.
- All four modes are treated as available for every trip.

**To confirm before starting:** whether the instructor accepts a London dataset in place of the Indian mode set (section 1).
