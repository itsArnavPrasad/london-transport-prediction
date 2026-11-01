# Team Guide: Mode Choice in London (LPMC)

A learning and preparation document for the team (Arnav Prasad, Aditya Chaurasiya, Bhavik Katariya, Sumit Yadav, Vedant Patel, Ritesh Srivastava). It explains where the data came from, what was and was not cleaned, how every method works, exactly what we did and why, what we found, what a professor might challenge, and what you should be able to say in the viva. The formal write-up is `report/report.pdf` (source `report/report.tex`); this guide is for understanding it.

**How to use it.** Read sections 1 to 3 first (about 20 minutes). Then read the method sections that match the part of the report you will present, and finish with the viva questions in section 11. Everyone should be able to explain sections 1, 3, 5 and 7 without notes.

---

## 1. The project on one page

**Question.** Given a trip in London and the characteristics of the traveller, which of four modes will they choose: walk, cycle, public transport (PT) or car?

**Data.** The London Passenger Mode Choice dataset (LPMC): 81,086 real trips by 31,954 people in 17,616 households, April 2012 to March 2015. Each trip has the mode actually used, plus estimated travel time and cost for all four modes.

**Models.** Multinomial Logit (MNL) and Nested Logit (the classical choice models); gradient boosting, random forest and a small neural network (machine learning); and a small language model (Qwen2.5-0.5B), used zero-shot and after LoRA fine-tuning.

**Design.** Train on the first two survey years, test on the third; all tuning on folds grouped by household. Compare models by log-loss, accuracy, calibration and mode-share error, with confidence intervals from a household-level bootstrap.

**Behaviour.** Use the models to compute the value of travel time, elasticities, and what happens to mode shares under five policy scenarios.

**Why London and not the dataset in the brief.** The brief describes an Indian dataset with six modes (walk, two-wheeler, auto-rickshaw, car, bus, local train) and variables such as income and dwelling type. No such public dataset exists. LPMC is the closest open one. The methods are the same; only the city, the mode set (four, not six) and the personal variables differ. **Tell the professor this early and plainly; it is the single biggest risk to the project.**

---

## 2. Is this good enough? An honest assessment

**Overall.** It is a strong course project: sound design, results with uncertainty, a comparison with published work, and limitations stated openly. It will not be seen as rubbish. Whether it earns top marks depends on three things.

**Risk 1: the dataset swap.** The professor can fairly say we did not do the assigned problem. Mitigation: raise it early, show the mapping table in the report (introduction, Table 1) that links each requirement of the brief to a section, and be ready to say that the methods transfer unchanged.

**Risk 2: breadth versus depth in the viva.** The project covers logit, nested logit, boosting, a neural network, LoRA fine-tuning and policy scenarios. If a team member cannot explain a part, it hurts more than a smaller project would. Divide the topics (section 12) and make sure everyone can defend the core: the data, the split, the MNL and the metrics.

**Risk 3: AI assistance.** Check your course's policy on the use of AI tools and disclose it if required.

**Claims that are solid**: the data audit and description; the split design; the ranking of models with confidence intervals; the finding that boosting and the language models respond unreliably to price changes; the finding that the PT fare effect cannot be identified.

**Claims that must be stated with care** (the report already does this):
- *Time matters more than money.* True of our estimates, but the cost coefficient is weakly identified and probably biased towards zero by measurement error in the modelled costs. Say "in these estimates", never "travellers value time more than money".
- *Value of time about £38 per hour for driving.* High against UK appraisal values. It matches what earlier work found for this dataset, but do not present it as willingness to pay.
- *Our MNL is better than published MNLs because of a richer specification.* Our test shows the extra variables matter; we cannot confirm which variables the published models contained. Say "consistent with".
- *Nested logit.* The car-versus-rest structure is supported with robust standard errors that ignore repeated trips; we did not bootstrap the nested model.
- *Language-model conclusions.* They apply to a 0.5B model, one option order and a learning rate at the edge of the grid we tried. Do not generalise to language models in general.

---

## 3. The data: where it came from and what we did to it

### 3.1 The original survey
LPMC is built on the **London Travel Demand Survey (LTDS)**, a rolling household travel survey run by Transport for London (TfL). Households fill in a trip diary: every trip a person made on a given day, with origin, destination, purpose, start time and the mode used. TfL did the sampling. The key limitation of any trip diary is that it records only **the mode that was chosen**. It does not record what the alternatives were or what they would have cost, which a choice model needs.

### 3.2 How the missing alternatives were created
Hillel, Elshafie and Jin (2018) built the dataset by data fusion:
1. An **online journey planner** produced a route and travel time for each of the four modes for every trip.
2. A **cost model** then computed individual-specific costs: PT fares (depending on fare type and zones), driving fuel costs, and congestion charges.
3. Each row therefore holds the chosen mode and the modelled level of service of all four modes.

**Consequence you must be able to state:** the times and costs of every mode, including the chosen one, are *modelled*, not observed. This holds for every model in every paper that uses this dataset.

### 3.3 What was cleaned before we received the file
According to the documentation (`CS_LPMC.pdf`), the authors removed trips outside London and trips that did not use TfL services. Nothing else is documented. Expansion factors (survey weights) exist in the original survey but are **not provided**, so every result describes the *sample*, not London.

### 3.4 What we did
We removed and changed nothing. We checked, and nothing needed fixing:

| Check | Result |
|---|---|
| Missing values | none |
| Duplicate rows or trip IDs | none |
| Counts match the documentation | 81,086 trips, 31,954 people, 17,616 households |
| Driving time and PT time ever zero | never |
| Households in more than one survey year | none |

We changed the *representation*: hours to minutes, totals recomputed from components (total PT time is access + rail + bus + interchange; total driving cost is fuel + congestion charge), dummy variables for the logit (car ownership, purpose, fare type), log distance, and peak and weekend indicators. Extreme values (for example 9-hour walking times for 40 km trips) are real modelled values and were kept.

### 3.5 The variables

| Group | Variables |
|---|---|
| Target | `travel_mode`: 1 walk, 2 cycle, 3 public transport, 4 drive |
| Person | age, gender, driving licence, `car_ownership` (0 no car, 1 fewer than one car per adult, 2 one or more per adult), `faretype` (1 full fare, 2 age 16+, 3 child, 4 disabled, 5 free), `bus_scale` (0 free, 0.5 half, 1 full) |
| Trip | purpose (home-based work, home-based education, home-based other, employer's business, non-home-based other), straight-line distance, start time, day of week |
| Walk, cycle | modelled travel time |
| Public transport | access/egress time, rail time, bus time, interchange time, number of interchanges, fare |
| Car | driving time, fuel cost, congestion charge, traffic variability |

Not in the data: income, education, occupation, dwelling type, parking, satisfaction. Fare type, bus scale and car ownership are weak proxies for income at best.

### 3.6 Quirks that come up in questions
- **"Drive" includes car passengers.** 23% of car trips are by people without a licence; about 5,000 are by under-17s. So we could not define car availability as "has a licence and a household car"; all four modes are treated as available.
- **Cycling is 3% of trips.** No model predicts it (recall under 1%). Accuracy alone hides this, so we also report macro-F1 and per-mode recall.
- **A third of PT fares are zero** (free passes) and fare is strongly tied to distance (correlation 0.74). The fare effect is therefore not identified.
- **Households are disjoint across survey years**, which is what makes the survey-year split safe.
- **`fueltype`** is almost a copy of car ownership, so we left it out. Hillel et al. (2021) warn that treating the chosen mode differently from the others can leak the answer; we checked this variable for that reason.

---

## 4. Framing the problem: discrete choice

A discrete choice problem has a decision maker, a finite set of alternatives, and attributes of each. The standard theory is **random utility maximisation**: each alternative gives person *n* a utility

    U_nj = V_nj + ε_nj

where V is the part we can measure (time, cost, personal characteristics) and ε is everything we cannot. The person picks the alternative with the highest utility. Because ε is random we can only state **probabilities**, so every model in the project outputs four probabilities per trip, and we judge models with probability-based scores, not just accuracy. Only utility *differences* matter, so one alternative is the reference (walking, constant set to zero).

---

## 5. How we split the data, and why

**Test set: survey year 3** (April 2014 to March 2015), 26,320 trips. **Training pool: years 1 and 2**, 54,766 trips.

1. **It mimics real use.** Choice models are estimated on past surveys to forecast a later situation.
2. **No leakage.** We verified that no household or person appears in both periods. A random split of trips would put other trips of the same person in training, and one person's trips are correlated, so the test score would look better than it should.
3. **It matches the better papers.** Martín-Baos et al. (2023) used the last year (2014/15) as the test set and kept each individual's responses in one set only; Salvadé and Hillel (2024) trained on the first two years with household-grouped folds. Hillel et al. (2021) review 70 studies and identify trip-level leakage as a common pitfall.

**Validation inside the training pool.** Anything that needed tuning used five folds **grouped by household**. The test set was never used for any decision.

**Shared test sample.** Running the language model over all 26,320 test trips several times is too slow on an 8 GB laptop, so all models are also scored on one fixed random sample of 4,000 test trips (drawn before modelling). The classical models are also scored on the full set, which confirms the sample is representative. The standard error of an accuracy figure on 4,000 trips is about 0.7 percentage points.

**Honesty point.** Early on I printed test results once in a smoke test of the logit models. No decision was based on them (the model ladder was fixed in advance and chosen with training data), but the report discloses it, and you should too if asked.

---

## 6. The methods

### 6.1 Multinomial Logit (MNL)
**Idea.** Assume the random terms follow a Gumbel distribution. The probability then has a closed form, the softmax: P(j) = exp(V_j) / Σ exp(V_k).

**Specification.** V_j is linear in the mode's own attributes and the person's characteristics, plus a constant. Time and cost coefficients should be **negative**; we checked this.

**Estimation.** Maximum likelihood in Biogeme (EPFL).

**The ladder.** M0 constants only (3 parameters); M1 common time and cost coefficients (5); M2 mode-specific times, PT split into access, rail, bus and interchange time plus a penalty per interchange, and traffic variability (13); M3 person variables (46); M4 trip variables (67). Each step contains the previous model, so we use a **likelihood-ratio test** (twice the gain in log-likelihood, compared with a chi-square with as many degrees of freedom as added parameters), plus **BIC** (penalises extra parameters) and **pseudo R²** (1 minus the ratio of the model's log-likelihood to the constants-only model's). The lowest BIC was M4.

**Independence from irrelevant alternatives (IIA).** In logit the ratio of the probabilities of two modes does not depend on the others. That is unrealistic when two modes are close substitutes, which motivates nested logit.

**Standard errors, and why we bootstrapped them.** A single fit treats all 54,766 trips as independent, but trips of one person or household are correlated, so its standard errors and tests are overconfident. We resampled whole households with replacement **200 times**, refitted M4 each time, and used the distribution of coefficients. The median bootstrap standard error is **1.50 times** the robust one. 56 of 67 coefficients are significant by the robust standard error and 48 by the bootstrap intervals. The substantive conclusions did not change.

### 6.2 Nested Logit
Group alternatives into nests with correlated errors. The nest parameter μ ≥ 1 (Biogeme's convention) measures similarity within a nest; μ = 1 collapses to MNL. We tried (a) {walk, cycle} against {PT, car} and (b) {walk, cycle, PT} against {car}.

**Findings.** (b) is supported: μ ≈ 1.31, t = 6.8 against 1, likelihood-ratio 66 with 1 degree of freedom, lower BIC. In (a) the walk-and-cycle nest sits at the lower bound of 1, so it is not supported. The published nested logit (Salvadé and Hillel) groups PT and driving with μ = 1.35; in our estimates that grouping is only weakly supported once the car is separated. Although nested logit fits better, **it does not predict better**, as in the published results.

### 6.3 Gradient boosting
Many small decision trees built one after another, each correcting the previous ones; their summed scores go through a softmax. It finds non-linearities and interactions automatically. We used scikit-learn's histogram implementation, searched 10 random settings on two household-grouped folds, and chose the number of trees by validation log-loss. Winner: learning rate 0.05, at most 15 leaves, at least 50 samples per leaf, L2 penalty 10, 159 trees. Cross-validated log-loss on the training years 0.641; calibration already good (ECE 0.004), so no recalibration.

**Permutation importance.** Shuffle a group of variables jointly in held-out data; a big increase in log-loss means the model relied on it.

### 6.4 Random forest and neural network
Included because the published LPMC benchmarks include them. The random forest averages many deep trees on random resamples. The neural network is a small multilayer perceptron (128 and 64 units) with standardised inputs and one-hot encoding for purpose and fare type, averaged over five random starts.

### 6.5 The language model
**Model.** Qwen2.5-0.5B-Instruct, 0.5 billion parameters, run locally with Apple's MLX in bf16. It is small next to the 7B+ models in most published studies; the point is a controlled comparison on a laptop.

**Trip to text.** Each trip becomes a short prompt: who the traveller is, the trip, and the four options with times and costs, ending "Answer with one word: walk, cycle, public or drive." Times are shown to one decimal minute: with whole minutes a 15% change in a 3-minute driving time would show as no change and the model would look insensitive for no behavioural reason.

**Text to probabilities.** We run the model once and read its scores (logits) for the first token of each answer word, then apply a softmax. This gives a valid distribution, needs one forward pass, and cannot fail to parse.

**Zero-shot.** No training. We found strong **position bias**: reversing the option order changed 74% of validation predictions. A "prior-corrected" variant adds one fixed bias per mode (fitted on validation trips) to show how much of the failure is a wrong prior.

**LoRA fine-tuning.** Instead of updating all 494 million parameters, LoRA freezes the model and trains small low-rank adapter matrices on the attention layers (rank 8 on the query, key, value and output projections of all 24 layers): 1.08 million trainable parameters, 0.22%. The loss is applied **only to the answer word** ("loss masking"), so the model learns trip-to-mode and not the prompt format. Batch 4, one epoch, sequences up to 320 tokens, gradient checkpointing to save memory.

**Learning rate.** Five rates on a 2,000-trip run, scored on validation only: 2e-4, 1e-4, 5e-5, 2e-5, 1e-5 gave validation log-loss 1.30, 1.04, 0.92, 0.81, 0.76. We used 1e-5. The best value is at the **edge of the grid**, so a lower rate might do better; we stopped because of the time budget.

**Sample efficiency.** Fine-tuned on nested subsets of 500, 2,000 and 8,000 trips; the logit and boosting were fitted on the same subsets.

**Remaining weakness.** Trained with one option order only, the fine-tuned model still changes 33% of validation predictions (accuracy 73% to 56%) when the order is reversed.

### 6.6 Metrics
| Metric | Meaning | Why |
|---|---|---|
| **Log-loss** | average −log(probability given to the chosen mode) | Primary. A proper scoring rule: rewards honest probabilities. |
| Accuracy | share of trips where the top-probability mode is the chosen one | Familiar; ignores confidence; hides failure on rare modes |
| Macro-F1, per-mode recall | quality per mode, averaged equally | Shows that cycling is never predicted |
| ECE | gap between stated confidence and hit rate, over bins | Calibration |
| **Mode-share error** | mean absolute gap between average predicted and observed shares | What planners use |
| GMPCA | exp(−log-loss) | Reported in published LPMC papers, so we can compare |

**Uncertainty.** 95% percentile intervals from a **household-level bootstrap** (1,000 resamples). Model comparisons use **paired** differences on the same resamples.

### 6.7 Behavioural analysis
**Value of time.** MNL: coefficient on time divided by coefficient on cost, times 60, in pounds per hour; intervals from the delta method and from the household bootstrap. **Implied value of time** for models without coefficients: the change in a mode's probability per extra minute divided by the change per extra pound, from 15% and 20% perturbations, using trips where both effects have the right sign. **Arc elasticity:** percentage change in the aggregate mode share divided by the percentage change in the variable. **Scenarios:** S1 PT fares −20%; S2 driving costs +20%; S3 driving time +15%; S4 PT in-vehicle time −15%; S5 = S1 + S2 + S4. **Monotonicity check:** the share of trips where a model moves a mode's probability the wrong way.

---

## 7. What we found

Test results, survey year 3, shared 4,000-trip sample (full-test numbers in brackets for the classical models).

| Model | Log-loss | Accuracy |
|---|---|---|
| Market shares (reference) | 1.146 | 42.4% |
| Multinomial Logit | 0.686 (0.684) | 73.2% (73.8%) |
| Nested Logit | 0.686 (0.683) | 73.3% (73.8%) |
| Gradient boosting | 0.661 (0.657) | 73.9% (74.6%) |
| Random forest | 0.679 (0.676) | 73.6% (74.2%) |
| Neural network | 0.657 (0.657) | 74.9% (74.6%) |
| Qwen 0.5B zero-shot | 2.637 | 39.4% |
| Qwen 0.5B fine-tuned, 8,000 trips | 0.766 | 70.3% |

**Messages to remember**
1. A well-specified logit is nearly as good as the best ML model and far more useful for behaviour.
2. Boosting and the neural network are better by 0.025 to 0.029 log-loss, but respond to price changes unreliably (wrong direction for 35% of trips when driving costs rise, for boosting).
3. Nested logit improves the fit but not the prediction.
4. Zero-shot, the 0.5B language model is worse than guessing the market shares; fine-tuning on 8,000 trips brings it to 70% accuracy, still 0.08 log-loss behind the logit, less well calibrated.
5. The logit gets most of its accuracy from 500 to 2,000 trips; the language model needs thousands more, and an hour of training.
6. Car availability dominates choice; PT access time, interchanges and traffic variability are the heaviest service penalties.
7. Under the logit, driving time +15% moves about 1.8 percentage points of trips off the car; a 20% driving-cost rise only 0.3. The time-over-money comparison depends on weakly identified cost coefficients.
8. PT fare effects cannot be identified: with separate coefficients the fare coefficient is positive.
9. Driving value of time ≈ £38 per hour (household-bootstrap interval £30.5 to £48.3); high, but in line with earlier work on this dataset.

**Timing (M1 MacBook, 8 GB).** Logit seconds to a minute; boosting minutes; fine-tuning 4 minutes (500 trips), 17 (2,000), 66 (8,000); scoring 4,000 trips with the language model about 10 minutes.

---

## 8. What we learned the hard way (pitfalls)

1. **PT fares cannot be separated from distance.** Fare is determined by distance and zones, a third are zero, and the shared cost coefficient is effectively identified from driving costs. Kept the shared coefficient (theory: one marginal utility of money) and flagged it. Martín-Baos et al. report a related sign problem for this dataset.
2. **Boosting responds unreliably to counterfactuals.** It predicts well but nothing forces its output to fall when cost rises.
3. **The car mode includes passengers**, so availability could not be defined from licence and car ownership.
4. **Prompt rounding** hid small changes; fixed with one decimal.
5. **The `[02]` in the project path broke the model loader** (square brackets are read as a pattern), so the weights live in `~/ai-ds-models/`.
6. **Order dependence of the language model** (zero-shot 74%, fine-tuned 33% of predictions change under reversed options).
7. **My first explanation for why our MNL beats published MNLs was wrong.** I guessed the fare-type variables; the ablation showed they matter little (0.0015 log-loss) and the trip variables matter much more (about 0.03).
8. **The learning rate kept improving as it fell**, ending at the edge of the grid.
9. **Memory pressure** on an 8 GB machine slowed late runs; close other applications for reruns.

---

## 9. What we verified in the literature

Every published number in the report was checked by reading the original paper (not summaries).

| Source | What we cite | Where in the paper |
|---|---|---|
| Martín-Baos et al. (2023), *Transportation Research Part C* 156, 104318 | Test on 2014/15, train on 2012/13 to 2013/14, same individual in one set only | Section 4.1.1 |
| | LPMC test accuracy/GMPCA: MNL 72.54/48.85; XGBoost 74.72/51.85; neural net 74.25/51.03; deep net 74.05/51.16 | Table 8 |
| | Value of time: MNL 42.39 (PT access time) and 41.62 (car); random forest and XGBoost inconsistent, 40.57% and 42.92% invalid values on LPMC | Section 5.4, Table 11 |
| | GMPCA is the geometric mean of the probabilities of the chosen modes | Equation 26 |
| Salvadé and Hillel (2024), arXiv:2401.11954 | 62-parameter MNL; train on first two years, household-grouped five-fold CV | Section 3.1 |
| | Holdout cross-entropy: MNL 0.7085, Nested Logit 0.7091, neural net 0.6667, deep net 0.6735, LightGBM 0.6537 | Tables 4 and 6 |
| | Nested logit nest = PT and driving, μ = 1.35, no gain over MNL | Section 4.2 |
| | Linear-model value of time £8.73 (PT) and £40 (driving), attributed to Hillel (2019) thesis p. 133 | Section 3.5 (thesis not consulted) |
| Zhai et al. (2024), arXiv:2406.13558 | LLaMA3 accuracy 0.576 zero-shot, 0.768 and 0.772 with demonstrations; logit 0.690; no train/test split described; model sizes not stated | Table II |
| Alsaleh and Farooq (2025), arXiv:2507.21432 | Fine-tuned Gemma 3 12B on **Swissmetro** (not LPMC), weighted F1 0.6845, test sample of 200; untuned models on LPMC best F1 0.576 / 0.661 / 0.639 | Abstract, Sections 2 and 3 |
| Hillel et al. (2021), *J. Choice Modelling* 38, 100221 | 70 articles from 574 records; ten limitations, five likely to bias performance estimates; leakage | Abstract and Section 2 |

One earlier statement of mine was inaccurate and is corrected here: the fine-tuned model of Alsaleh and Farooq was trained and tested on Swissmetro, not LPMC.

---

## 10. Limitations to state openly
- Times and costs of all modes are modelled, not observed.
- No income, education or occupation; fare type and car ownership are weak proxies.
- No survey weights: results describe the sample.
- Mode "car" includes passengers; cycling is never predicted.
- Nested logit and the segment value-of-time estimates are not bootstrapped.
- Cost effects (and fares in particular) are weakly identified.
- The language-model comparison uses a 0.5B model, a 4,000-trip test sample, one option order, and a learning-rate grid whose optimum is at the lower edge.
- Scenario results for models other than the logit depend on the models extrapolating to 15 to 20% changes.
- Test numbers were seen once in a smoke test (no decisions depended on them).

---

## 11. Viva questions and short answers

**Why LPMC and not an Indian dataset?** No public Indian dataset has individual-level choices with mode-specific times and costs. LPMC is open, large and has the same structure; the methods transfer.

**What is the dependent variable?** The chosen mode among walk, cycle, public transport, car.

**Why a survey-year split?** It mirrors forecasting use, and since no household spans survey years it avoids leakage between trips of the same person.

**Why not a random 80/20 split?** Trips by the same person are correlated; a random split lets the model see the person's other trips and overstates accuracy.

**What is the utility function?** A weighted sum of the alternative's attributes and the person's characteristics plus a constant; the probability of a mode is the softmax of utilities.

**Why is walking the reference?** Only utility differences matter, so one constant must be fixed; the choice does not affect predictions.

**What does the sign of a coefficient tell you?** The direction of the effect. Negative time and cost coefficients mean longer or costlier alternatives are chosen less.

**What is IIA?** The probability ratio of two modes is unaffected by a third. It fails for close substitutes, which motivates nested logit.

**What does μ = 1.31 mean?** Walk, cycle and PT are correlated alternatives relative to the car; the parameter is significantly above 1.

**Why did nested logit not predict better?** Correlation structure changes substitution patterns and elasticities, but only slightly changes the probability given to the chosen mode.

**How did you choose the MNL specification?** Likelihood-ratio tests between nested specifications and BIC; lowest BIC was M4.

**Why did you bootstrap by household?** A person's trips are correlated; single-fit standard errors treat them as independent. Resampling whole households respects the correlation.

**What is value of time and how is it computed?** The marginal rate of substitution between time and money: time coefficient divided by cost coefficient, times 60.

**Why is the value of time so high?** It matches earlier work on this dataset (about £40 for driving in linear logit models). The modelled costs vary little independently of distance and time, so the cost coefficient is small and the ratio large. We use it for relative comparisons.

**Why can you not say anything about fares?** With separate coefficients the PT fare coefficient is positive; fares are determined by distance and a third are zero.

**Does time matter more than money to travellers?** In our estimates, yes, but the cost coefficient is weakly identified and probably biased towards zero, so we state it as a property of these estimates.

**Why log-loss?** Every model outputs probabilities, and log-loss rewards good probabilities, not only the top choice. It is a proper scoring rule.

**Why report accuracy if it hides cycling?** It is the common headline; we add macro-F1 and per-mode recall to show the limits.

**What is calibration?** Whether stated probabilities match observed frequencies; we use ECE and reliability curves.

**Why is boosting better but less trustworthy for policy?** It optimises prediction; nothing forces its output to fall when cost rises. The logit has the right structure built in.

**What is LoRA?** Training a small low-rank correction to a frozen model: 1.08 million parameters instead of 494 million.

**Why score answer tokens instead of generating text?** Exact probabilities, deterministic, one forward pass, no parsing errors.

**Why does zero-shot fail?** A 0.5B model has no learned sense of how time and cost trade off, and it is dominated by prompt position effects.

**Why did lower learning rates win?** Higher rates made the small model's training unstable. The optimum may be lower than our grid.

**Did the language model beat the classical models?** No: fine-tuned on 8,000 trips it gets log-loss 0.766 and 70.3% accuracy against 0.686 and 73.2% for the logit and 0.661 and 73.9% for boosting. It is also less well calibrated. This is at small scale; published claims that language models win use larger models and, in the case we checked, unreported splits.

**How do your results compare with published ones?** Same design as Martín-Baos et al. and Salvadé and Hillel. Our boosting and neural network match the best published tree and network models; our MNL is better than their MNLs, consistent with a richer specification.

**What would you do with more time?** Monotonic constraints on boosting, larger language models with randomised option order, income-like variables or observed costs, bootstrapping the nested logit, and the analysis on an Indian dataset with six modes.

---

## 12. Suggested division of explanation

| Topic | Who should be able to present it |
|---|---|
| Data origin, quality, split design | one person (sections 3 and 5) |
| MNL and nested logit | one person (6.1, 6.2) |
| Machine learning models and metrics | one person (6.3, 6.4, 6.6) |
| Language model | one person (6.5) |
| Behaviour, policy and conclusions | one person (6.7, 7) |
| Literature, limitations, the dataset swap | one person (2, 9, 10) |

Everyone: the project in one page (section 1), the split (section 5) and the main results table (section 7).

---

## 13. What to learn

**Concepts, in order**
1. Random utility and the logit model; reference alternative; likelihood, rho-squared, likelihood-ratio test, BIC.
2. IIA and nested logit.
3. Proper scoring rules: log-loss, Brier score, calibration; why accuracy is not enough.
4. Leakage, grouped validation and temporal splits.
5. Bootstrap, and why clusters matter.
6. Value of time, elasticities, scenario simulation.
7. Gradient boosting and permutation importance.
8. Language models: tokens, logits, softmax; zero-shot against fine-tuning; LoRA.

**Reading**
- Train, K. (2009), *Discrete Choice Methods with Simulation*, chapters 2 (logit) and 4 (GEV models, including nested logit); free online.
- Hillel et al. (2021), a systematic review of machine learning for mode choice (validation pitfalls).
- Martín-Baos et al. (2023) and Salvadé and Hillel (2024): the two benchmarks for our results.
- Hu et al. (2022), LoRA.
- Bierlaire (2023), a short introduction to Biogeme.

---

## 14. Where everything is

| What | Where |
|---|---|
| Report (PDF and LaTeX source) | `report/report.pdf`, `report/report.tex`, tables in `report/tables/` |
| Raw data and documentation (not in the repository) | `data/raw/` |
| Frozen split definitions | `data/processed/splits.csv` |
| Code in execution order | `src/s0_audit.py` to `src/s11_latex_tables.py`, plus `common.py`, `choice_models.py`, `llm_common.py` |
| Result tables | `results/tables/` |
| Test-set probabilities for every model | `results/predictions/` |
| Figures | `results/figures/` |
| Base model and fine-tuned adapters | `~/ai-ds-models/` (outside the project) |
| Plan | `PLAN.md` |
| Repository | https://github.com/itsArnavPrasad/london-transport-prediction |
