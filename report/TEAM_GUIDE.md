# Team Guide: Mode Choice in London (LPMC)

A learning document for the team. It explains where the data came from, what was and was not cleaned, how every method works, exactly what we did, and what you should be able to say in the viva. It is not the course report; the report (`REPORT.md`) is the formal write-up of the results. Read this first if you want to understand the project well enough to defend it.

---

## 1. The project on one page

**Question.** Given a trip in London and the characteristics of the traveller, which of four modes will they choose: walk, cycle, public transport (PT) or car?

**Data.** The London Passenger Mode Choice dataset (LPMC): 81,086 real trips by 31,954 people in 17,616 households, April 2012 to March 2015. Each trip has the mode actually used, plus estimated travel time and cost for all four modes.

**Models.** Two classical discrete choice models (Multinomial Logit and Nested Logit), three machine-learning models (gradient boosting, random forest, a small neural network), and a small language model (Qwen2.5-0.5B) used zero-shot and after LoRA fine-tuning.

**Evaluation.** Train on the first two survey years, test on the third. Compare models by log-loss, accuracy, calibration and mode-share error, with confidence intervals. Then use the models for behavioural analysis: value of travel time, elasticities, and what happens to mode shares under policy scenarios.

**Why this design.** It follows how choice models are used in practice (estimate on past surveys, forecast a later year) and follows the validation advice in the literature (no leakage between trips of the same household).

**What the course brief asked for, and what we did instead.** The brief describes an Indian dataset with walking, two-wheelers, auto-rickshaws, cars, buses and trains, plus income, education and dwelling variables. No public dataset like that exists, so we used LPMC, the closest open one. The method is the same; only the city, the mode set and the available variables differ.

---

## 2. The data: where it came from

### 2.1 The original survey

LPMC is built on the **London Travel Demand Survey (LTDS)**, a rolling household travel survey run by Transport for London (TfL). Households fill in a trip diary: every trip a person made on a given day, with origin, destination, purpose, start time and the mode used. TfL did the sampling. Our file covers survey years 1 to 3 (April 2012 to March 2015).

The key limitation of any trip diary is that it records only **the mode that was chosen**. It does not record what the alternatives were or what they would have cost. A choice model needs those, because the whole idea is to compare the chosen mode with the ones that were not chosen.

### 2.2 How the missing alternatives were created

Hillel, Elshafie and Jin (2018) built the dataset with a data-fusion method:

1. For every trip, an **online journey planner** produced a route and travel time for each of the four modes: walking, cycling, public transport and driving.
2. A **cost model** then calculated individual-specific costs: PT fares (depending on the traveller's fare type and the zones crossed), driving fuel costs, and congestion charges.
3. The result is one row per trip containing the chosen mode and the modelled level-of-service (travel time and cost) of all four modes.

**Consequence you must be able to state:** the times and costs of the alternatives, and also of the chosen mode, are *modelled*, not observed. If someone actually walked a trip, the dataset still shows the planner's estimate of the walking time, not their real time. This is true for every model in every paper that uses this dataset.

### 2.3 What was cleaned before we received the file

According to the dataset documentation (`data/raw/CS_LPMC.pdf`), the authors removed all trips outside London and all trips that did not use TfL services. That is all that is documented. Expansion factors (survey weights that make the sample represent the population) exist in the original survey but are **not provided**, so every result in this project describes the *sample of survey respondents*, not London as a whole.

### 2.4 What we did to the data

We did not drop, impute or alter any rows. We checked, and nothing needed fixing:

| Check | Result |
|---|---|
| Missing values | none |
| Duplicate rows or trip IDs | none |
| Counts match documentation | 81,086 trips, 31,954 people, 17,616 households |
| Driving time and PT time ever zero | never |
| Households that appear in more than one survey year | none |

What we did change is the *representation*: durations were converted from hours to minutes, and totals were recomputed from their components (total PT time is access + rail + bus + interchange time; total driving cost is fuel + congestion charge), exactly as the documentation advises. For the logit models we also created dummy variables (for example car ownership 1 and 2, trip purposes), a log-distance variable, and peak-time and weekend indicators. Nothing was winsorised or removed; extreme values (for example 9-hour walking times for 40 km trips) are real modelled values for long trips, and the models can cope with them.

### 2.5 The variables

| Group | Variables |
|---|---|
| Target | `travel_mode`: 1 walk, 2 cycle, 3 public transport, 4 drive |
| Person | age, gender, driving licence, `car_ownership` (0 no car, 1 fewer than one car per adult, 2 one or more per adult), `faretype` (1 full fare, 2 age 16+, 3 child, 4 disabled, 5 free), `bus_scale` (0 free, 0.5 half price, 1 full price) |
| Trip | purpose (home-based work, home-based education, home-based other, employer's business, non-home-based other), straight-line distance, start time, day of week |
| Walk, cycle | modelled travel time |
| Public transport | access/egress time, rail in-vehicle time, bus in-vehicle time, interchange time, number of interchanges, fare |
| Drive | modelled driving time, fuel cost, congestion charge, traffic variability |

Not in the data: income, education, occupation, dwelling type, parking, satisfaction. `faretype`, `bus_scale` and `car_ownership` are weak proxies for income at best.

### 2.6 Known quirks (these come up in questions)

- **"Drive" includes car passengers.** 23% of "drive" trips are made by people without a driving licence, and about 5,000 are by people under 17. These travellers do not choose the car in the usual sense. We call the mode "car (driver or passenger)". A consequence: we could **not** define car availability as "has a licence and a household car", as I first assumed. All four modes are treated as available for every trip, as in the original paper.
- **Cycling is only 3% of trips.** Every model struggles to predict it; recall is close to zero. Accuracy alone hides this, so we also report macro-F1 and per-mode recall.
- **A third of PT fares are zero** (free passes and concessions), and PT fare is strongly tied to distance (correlation 0.74). Because of this, the fare effect is hard to identify; see section 6.
- **Households are disjoint across survey years.** We verified this, and it is what makes the survey-year split safe.
- **`fueltype`** is almost a copy of car ownership (the value "average car" is used mainly where the household has no car), so we left it out.

---

## 3. Framing the problem: discrete choice

A discrete choice problem has a decision maker, a finite set of alternatives, and attributes of each. The standard theory is **random utility maximisation**: each alternative *j* gives person *n* a utility

    U_nj = V_nj + ε_nj

where V is the part we can measure (a function of time, cost and personal characteristics) and ε is everything we cannot. The person picks the alternative with the highest utility. Because ε is random, we can only talk about **probabilities** of choosing each alternative, and what we estimate is a probability for each mode. This is why every model in the project produces four probabilities per trip, and why we judge models with probability-based scores, not just accuracy.

A convenient consequence: only **differences** in utility matter, so one alternative must be the reference (we use walking), with its constant set to zero.

---

## 4. How we split the data, and why

This is the part of the project most likely to be probed, so know it well.

**Test set: survey year 3** (April 2014 to March 2015), 26,320 trips. **Training pool: years 1 and 2**, 54,766 trips.

Reasons:

1. **It mimics real use.** Transport models are estimated on past surveys and used to forecast a future situation. Training on the past and testing on a later year is the honest version of that.
2. **There is no leakage.** We checked that no household or person appears in both periods. A random split of trips would put different trips of the same person in training and test, and since one person's trips are correlated (same home, same habits, same car), the test score would look better than it should.
3. **It matches the better papers.** The published LPMC benchmarks we compared against also use the first two years for training and a later period for testing, and Hillel's systematic review of this literature warns about trip-level leakage.

**Validation inside the training pool.** For anything that needed tuning (gradient boosting settings, language-model learning rate, prompt checks) we used five folds **grouped by household**, so no household is split across folds. The test set was never used for any decision.

**Honesty point.** Early in the project I printed test results once for a quick smoke test of the logit models. No decision was based on them (the model ladder was fixed in advance and chosen with training data only), but it should be disclosed if asked.

**Shared test subsample.** Running the language model on all 26,320 test trips three times is too slow on a MacBook with 8 GB of memory. So all models are also scored on a fixed random sample of 4,000 test trips drawn before any modelling. The classical models are additionally scored on the full test set, which confirms the subsample is representative. Standard error of an accuracy figure on 4,000 trips is about 0.7 percentage points.

---

## 5. The methods, one by one

### 5.1 Multinomial Logit (MNL)

**Idea.** Assume the random terms ε follow a particular distribution (Gumbel). Then the choice probability has a closed form, the softmax:

    P(j) = exp(V_j) / Σ_k exp(V_k)

**Specification.** V_j is a linear function of the alternative's own attributes and the person's characteristics. For example, driving utility = driving constant + b_time_drive × driving minutes + b_cost × (fuel + charge) + b_traffic × traffic variability + person and trip terms. Walking has no constant. Coefficients on time and cost should be **negative**: more time or money makes an alternative less attractive. We checked this.

**Estimation.** Maximum likelihood, using Biogeme (the standard software from EPFL). It finds coefficients that make the observed choices most probable.

**The model ladder (M0 to M4).** We added complexity step by step and tested each addition:

| Model | Adds | Parameters |
|---|---|---|
| M0 | constants only (reproduces market shares) | 3 |
| M1 | one common time coefficient and one cost coefficient | 5 |
| M2 | separate time coefficient per mode; PT time split into access, rail, bus, interchange, plus a penalty per interchange; traffic variability | 13 |
| M3 | person variables (age, age squared, gender, licence, car ownership, fare type, bus scale) | 46 |
| M4 | trip variables (purpose, log distance, peak, weekend) | 67 |

**How to compare nested models.** Because each model contains the previous one as a special case, we use a **likelihood-ratio test**: twice the gain in log-likelihood is compared with a chi-square distribution whose degrees of freedom equal the number of added parameters. We also use **BIC** (which penalises extra parameters strongly) and **rho-squared** (1 minus the ratio of the model's log-likelihood to that of the constants-only model, a pseudo R-squared). We selected the model with the lowest BIC, which was M4.

**Important weakness: IIA.** Logit assumes the ratio of probabilities of two alternatives does not depend on which other alternatives exist (independence of irrelevant alternatives). That is unrealistic when two modes are close substitutes. Nested logit relaxes it.

**Robust standard errors** are used; they protect against some misspecification. They do not account for correlation of repeated trips by the same person, which is a limitation.

### 5.2 Nested Logit

**Idea.** Group alternatives into nests whose error terms are correlated. Inside a nest, alternatives compete more closely with each other than with outsiders. A nest parameter μ (at least 1 in Biogeme's convention) measures this: μ = 1 means no correlation, so the model collapses back to MNL; larger μ means stronger within-nest similarity.

**What we tried.** (a) {walk, cycle} versus {PT, drive}, and (b) {walk, cycle, PT} versus {drive} (car versus non-car).

**What we found.** Structure (b) is supported: μ is about 1.31, significantly above 1, and the likelihood and BIC improve over MNL. Structure (a) is not: the nest parameter for the active nest sits at exactly 1 (the lower bound), so the data do not support grouping walking with cycling. Interpretation: the biggest substitution pattern is between the car and everything else. However, even though the fit improves, **prediction does not**: out-of-sample log-loss is essentially identical to MNL, a result also seen in the literature.

### 5.3 Gradient boosting

**Idea.** Build many small decision trees one after another, each one correcting the mistakes of the ones before. The sum of all trees gives a score for each mode, and a softmax turns scores into probabilities. It captures non-linearities and interactions automatically (for example, the effect of time may depend on distance), without being told about them.

**Our setup.** scikit-learn's `HistGradientBoostingClassifier`. We searched 10 random combinations of learning rate, tree size, minimum leaf size and regularisation, evaluating each on two household-grouped folds with the number of trees chosen by validation log-loss. The winning setting was a learning rate of 0.05, at most 15 leaves per tree, at least 50 samples per leaf, L2 penalty 10, and about 159 trees. Cross-validated log-loss on the training years was 0.641. We checked calibration, found it already good (ECE 0.004), and did not recalibrate.

**Why it is not simply "better".** It predicts more accurately but gives no interpretable coefficients, and its response to a changed input (like a higher fare) is not constrained to make sense. See the policy analysis.

**Feature importance by permutation.** For each group of variables (person, trip, walk/cycle, PT, drive), shuffle the group's columns jointly in held-out data and measure how much log-loss worsens. A big worsening means the model relied on that group.

### 5.4 Random forest and neural network

Added because the published LPMC benchmarks include them. **Random forest** averages many deep trees trained on random resamples of the data. The **neural network** is a small multilayer perceptron with standardised inputs and one-hot encoding for purpose and fare type, averaged over five random starts. Both were tuned lightly on household-grouped fold 0 of the training years.

### 5.5 The language model

**Model.** Qwen2.5-0.5B-Instruct, a 0.5-billion-parameter open model, run locally with Apple's MLX framework in bf16 precision. It is small compared with the 7-billion-plus models in most published LLM mode choice studies, so we expect it to be weaker; the point is a controlled comparison on a laptop.

**Turning a trip into text.** Each trip becomes a short prompt: who the traveller is (age, gender, licence, car availability, fare status), the trip (purpose, distance, day, time), and the four options with times and costs, ending with the question "Which mode does the traveller choose? Answer with one word: walk, cycle, public or drive." Times are shown to one decimal minute. We learned this the hard way: with whole minutes, a 15% change in a 3-minute driving time would show up as no change at all, and the model would look insensitive for reasons unrelated to its behaviour.

**From text to probabilities.** Instead of letting the model write an answer and parsing it, we run it once and read the model's internal scores (logits) for the first token of each of the four answer words, then apply a softmax. This gives a valid probability distribution for every trip, needs only one forward pass, and cannot fail to parse.

**Zero-shot.** No training; the model just sees the prompt. We found a strong **position bias**: reversing the order of the options changed the prediction for 74% of validation trips, and the model essentially chose walk or drive regardless of the trip. A "prior-corrected" variant adds one fixed bias per mode (fitted on the 1,000 validation trips) to show how much of the problem is just a wrong mode-share prior; it is a partial fix, not a real model.

**Fine-tuning with LoRA.** Full fine-tuning would update all 494 million parameters. LoRA instead freezes the model and trains small low-rank "adapter" matrices added to the attention layers (query, key, value and output projections of all 24 layers). We used rank 8, which means about 1.08 million trainable parameters (0.22% of the model). The training loss is applied **only to the answer word**, not to the prompt (this is "loss masking"), so the model learns to map trip descriptions to the chosen mode instead of memorising the format. Batch size 4, one pass over the training trips, sequences up to 320 tokens, gradient checkpointing to save memory.

**Learning rate.** We tried five rates on a 2,000-trip run, scoring only the validation set: 2e-4, 1e-4, 5e-5, 2e-5, 1e-5. Validation log-loss improved steadily as the rate fell (1.30, 1.04, 0.92, 0.81, 0.76), so we used 1e-5. The best value sat at the edge of the grid, so a lower rate might do better still; we stopped because of the time budget.

**Sample efficiency.** We repeated the fine-tuning on nested subsets of 500, 2,000 and 8,000 trips and compare to MNL and gradient boosting fitted on the same subsets.

### 5.6 Evaluation metrics

| Metric | What it is | Why it matters here |
|---|---|---|
| **Log-loss** (cross-entropy) | average of −log(probability given to the chosen mode) | Primary score. A proper scoring rule: rewards honest, well-calibrated probabilities. Works for all models. |
| Accuracy | share of trips where the top-probability mode is the chosen one | Familiar, but ignores confidence and hides the failure on rare modes |
| Macro-F1, per-mode recall | quality per mode averaged equally | Exposes that cycling is basically never predicted |
| Brier score | mean squared error of the probabilities | A second proper score |
| ECE | gap between stated confidence and actual hit rate, averaged over bins | Calibration: does "70%" mean 70%? |
| **Mode-share error** | mean absolute gap between the average predicted share and the actual share, over the four modes | What planners use: aggregates, not single trips |
| GMPCA | exp(−log-loss) | The score reported in some published LPMC papers, so we can compare |

**Uncertainty.** All intervals are 95% **cluster bootstrap** intervals: we resample whole households with replacement 1,000 times and recompute the metric. Resampling households (not trips) respects the correlation between trips of the same family. Model comparisons use **paired** differences on the same resamples, which is more powerful than comparing two intervals by eye.

### 5.7 Behavioural analysis

**Value of travel time (VoT).** In a logit model, the ratio of the time coefficient to the cost coefficient is the amount of money a traveller would trade for a minute of time. Multiply by 60 for pounds per hour. Standard errors come from the delta method (propagating the coefficient covariance through the ratio). We estimate it for the whole sample and for subgroups (fare-paying travellers, commuters, other purposes, with and without a household car).

**Implied VoT for models without coefficients.** For gradient boosting and the language models we compute the same trade-off from probabilities: how much the probability of a mode falls per extra minute, divided by how much it falls per extra pound, using trips where both effects have the right sign.

**Arc elasticities.** Change one variable for all trips by a fixed percentage, and measure the percentage change in the mode's aggregate share, divided by the percentage change in the variable.

**Policy scenarios.** S1 PT fares −20%; S2 driving costs +20%; S3 driving time +15%; S4 PT in-vehicle time −15%; S5 combined (S1+S2+S4). Reported as percentage-point shifts in mode share, with intervals. A **monotonicity check** counts the trips where a model moves the "wrong way" (for example, higher driving cost making driving *more* likely).

---

## 6. Pitfalls we found, and how they were handled

1. **PT fares cannot be separated from distance.** When we give PT fare and driving cost separate coefficients, the PT fare coefficient turns *positive*, which is the wrong sign. Fares are determined mostly by distance and zones, a third of them are zero, and free-pass holders are a distinct group. So a single shared cost coefficient is effectively identified from driving costs. We keep the shared coefficient (theory says money has the same value whichever mode you pay for) and state that the fare-cut scenario therefore rests on that assumption.
2. **High value of time.** Our driving VoT is about £38 per hour (£29 with separate cost coefficients). That looks high compared with UK appraisal values, but it matches earlier work on this exact dataset (about £40 per hour in Hillel's linear models, as cited in the RUMBoost paper). It is a feature of the data's modelled cost, probably amplified by weak cost variation, not a coding error.
3. **Tree models respond unreliably to counterfactuals.** Gradient boosting predicts well but, when driving costs rise, it moves the wrong way for about a third of trips. This echoes the literature (Martín-Baos et al.: flexible ML gives inconsistent value of time while MNL is robust).
4. **The home folder name breaks the model loader.** The `[02]` in the project path is read as a pattern by the file search inside `mlx_lm`, so the model weights live in `~/ai-ds-models/` instead.
5. **Prompt rounding.** Fixed by showing one decimal (section 5.5).
6. **The fine-tuned model still depends on option order.** It was trained with one fixed order (walk, cycle, public transport, car). Reversing the order on validation trips changes 33% of its predictions and drops accuracy from 73% to 56% (zero-shot: 74% of predictions changed). Randomising the order during training would likely help; we did not try it.

---

## 7. How our numbers compare with published work

Same split style (train years 1-2, test the later year) in Martín-Baos et al. (Transportation Research Part C, 2023) and Salvadé & Hillel (RUMBoost, 2024). Figures I got through readers of those papers; check them against the PDFs before quoting.

- Accuracy: published MNL about 72.5%, XGBoost about 74.7%. Ours: MNL 73.8%, gradient boosting 74.6% on the full test set.
- Log-loss: published MNL 0.709, nested logit 0.709, LightGBM 0.654, neural nets 0.667 to 0.674. Ours: MNL 0.684, nested logit 0.683, gradient boosting 0.657.
- Our MNL beats the published MNL. A training-years ablation showed that the reason is **not** the fare-type variables (removing them costs only 0.0015 log-loss) but the trip variables and nonlinear coding: removing purpose, log distance, peak and weekend costs about 0.03.
- Value of time on LPMC for linear logit is about £40 per hour for driving in earlier work; ours is £38.
- The published LLM papers are not comparable with ours: they use 7B-plus models, hard labels only, and often do not state how the data were split.

---

## 8. Limitations to state openly

- All alternatives' times and costs are modelled by a journey planner and cost model.
- No income or education; fare type and car ownership are weak proxies.
- No survey weights: results describe the sample.
- Mode "car" includes passengers.
- Cycling is rare; no model predicts it.
- The nested logit improves in-sample fit but not prediction.
- The PT fare effect is not identified from this data; fare-based scenarios depend on an assumption.
- The language model is tiny (0.5B); the learning-rate optimum may lie below the grid we tried; the language-model comparison uses a 4,000-trip test sample.
- Standard errors in the logit models do not account for repeated trips by the same person (the bootstrap intervals do).
- Test-set numbers were seen once during a smoke test (no decisions depended on them).

---

## 9. Viva questions and short answers

**Why LPMC and not an Indian dataset?** No public Indian dataset has individual-level choices with mode-specific times and costs. LPMC is open, large, and has the same structure. The methodology transfers.

**What is the dependent variable?** The chosen mode among walk, cycle, public transport, car.

**Why a temporal split?** It matches the forecasting use of choice models and, since no household spans survey years, it avoids leakage between trips of the same person.

**Why not a random 80/20 split?** Trips by the same person are correlated; a random split lets the model see the person's other trips and overstates accuracy.

**What is a utility function here?** A weighted sum of the alternative's attributes (time, cost) and the person's characteristics, plus a constant. The probability of a mode is a softmax of utilities.

**Why is walking the reference?** Only utility differences matter, so one constant must be fixed. The choice is arbitrary and does not affect predictions.

**What does the sign of a coefficient tell you?** Direction of effect: negative time and cost coefficients mean longer or costlier alternatives are less likely to be chosen.

**What is IIA and why does it matter?** The probability ratio of two modes is unaffected by a third mode. It is unrealistic when modes are close substitutes, which motivates nested logit.

**What does μ = 1.31 mean?** Walk, cycle and PT are correlated alternatives relative to the car; the nest parameter is significantly above 1.

**Why did nested logit not predict better?** The correlation structure matters for substitution patterns and elasticities, but only slightly changes the probability of the chosen alternative, so prediction scores barely move.

**How did you choose between MNL specifications?** Likelihood-ratio tests between nested specifications and BIC; the one with the lowest BIC was M4.

**What is value of time and how do you compute it?** The marginal rate of substitution between time and money: coefficient on time divided by coefficient on cost, times 60.

**Why is the value of time so high?** It matches earlier work on this dataset; the modelled cost varies little independently of distance and time, which makes the cost coefficient small and the ratio large.

**Why did you use log-loss?** Every model outputs probabilities, and log-loss rewards good probabilities, not only the top choice. It is a proper scoring rule.

**Why report accuracy at all, if it hides cycling?** It is the common headline, so we report it, with macro-F1 and per-mode recall to show the limits.

**What is calibration?** Whether predicted probabilities match observed frequencies. We measure ECE and draw reliability curves.

**What is LoRA?** Training a small low-rank correction to a frozen model: here 1.08 million parameters instead of 494 million.

**Why score the answer tokens instead of generating text?** It yields exact probabilities, is deterministic, needs one forward pass, and avoids parsing errors.

**Why does the zero-shot model perform so poorly?** A 0.5B model has no learned sense of how time and cost trade off in London, and it is dominated by prompt position effects (reversing option order changed 74% of its predictions).

**Why did lower learning rates win?** Larger rates made the small model's training unstable. The optimum may be lower than the grid we tested.

**Did the language model beat the classical models?** No. After fine-tuning on 8,000 trips it reaches log-loss 0.766 and 70.3% accuracy, against 0.686 and 73.2% for the logit and 0.661 and 73.9% for boosting on the same trips. It is also less well calibrated and its response to price and time changes is weak or sometimes in the wrong direction. The comparison is at small scale (0.5B parameters); published claims that language models beat classical models used larger models and often did not state their data splits.

**Why can the boosting model be more accurate yet less trustworthy for policy?** It optimises prediction, not behavioural consistency; nothing forces its output to fall when the cost rises. The logit model has the right structure built in.

**What would you do with more time?** Monotonic constraints on boosting (as in RUMBoost), larger language models, a lower learning-rate grid, person-level clustered standard errors, and income-like variables.

---

## 10. Glossary

- **Alternative-specific constant (ASC):** a mode's baseline attractiveness after controlling for measured attributes.
- **Cluster bootstrap:** resampling groups (households) instead of single rows.
- **Elasticity:** percentage change in demand per percentage change in a variable.
- **ECE:** expected calibration error.
- **LoRA/QLoRA:** low-rank adapter fine-tuning (QLoRA with a quantised base model; we used full-precision bf16 base weights).
- **Logit:** the model where the choice probability is a softmax of utilities.
- **LR test:** likelihood-ratio test.
- **Reference alternative:** the mode whose constant is fixed to zero (walking).
- **Revealed preference (RP):** data on actual choices, as opposed to hypothetical stated-preference surveys.
- **Zero-shot:** the model sees the task description but no training examples.

---

## 11. Where everything is

| What | Where |
|---|---|
| Raw data and documentation | `data/raw/` |
| Frozen split definitions | `data/processed/splits.csv` |
| Code, in order of execution | `src/s0_audit.py` through `src/s9_figures.py` (plus `common.py`, `choice_models.py`, `llm_common.py`) |
| Result tables | `results/tables/` |
| Test-set probabilities for each model | `results/predictions/` |
| Figures | `results/figures/` |
| Fine-tuned adapters and the base model | `~/ai-ds-models/` |
| Plan | `PLAN.md` |

---

## 12. Results at a glance

Test results, survey year 3, shared 4,000-trip sample (full-test numbers for the classical models in brackets).

| Model | Log-loss | Accuracy |
|---|---|---|
| Market shares (reference) | 1.146 | 42.3% |
| Multinomial Logit | 0.686 (0.684) | 73.2% (73.8%) |
| Nested Logit | 0.686 (0.683) | 73.3% (73.8%) |
| Gradient boosting | 0.661 (0.657) | 73.9% (74.6%) |
| Random forest | 0.679 (0.676) | 73.6% (74.2%) |
| Neural network | 0.657 (0.657) | 74.9% (74.6%) |
| Qwen 0.5B zero-shot | 2.637 | 39.4% |
| Qwen 0.5B fine-tuned, 8,000 trips | 0.766 | 70.3% |

Key messages to remember:

1. A well-specified logit is nearly as good as the best machine-learning model, and much more useful for behaviour.
2. Gradient boosting and the neural network are better by about 0.025 log-loss, but respond to price changes unreliably (wrong direction for 35% of trips when driving costs rise, for boosting).
3. Nested logit improves the fit but not the prediction.
4. Zero-shot, a 0.5B language model is worse than guessing the market shares; fine-tuning on 8,000 trips brings it to 70% accuracy, still 0.08 log-loss behind the logit, and it is less well calibrated.
5. The logit needs about 500 to 2,000 trips to get most of its accuracy; the language model needs thousands more.
6. Time effects are better supported than cost effects. Driving time +15% moves about 1.8 percentage points of trips from car (mostly to public transport); a 20% driving-cost rise moves only 0.3. The fare effect cannot be identified.
7. Value of time for driving is about £38 per hour, which looks high but matches earlier work on this dataset.

Timing on the MacBook (M1, 8 GB): logit models seconds to a minute; boosting minutes; fine-tuning 4 minutes (500 trips), 17 minutes (2,000 trips), 66 minutes (8,000 trips); scoring 4,000 trips with the language model about 10 minutes.
