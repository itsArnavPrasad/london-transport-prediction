# Mode choice in London: discrete choice models, machine learning and a fine-tuned small language model

Group project on commuter mode choice using the London Passenger Mode Choice (LPMC) dataset (81,086 trips, April 2012 to March 2015). We predict whether a trip is made on foot, by bicycle, by public transport or by car, compare Multinomial Logit, Nested Logit, gradient boosting, a random forest, a neural network and a small language model (Qwen2.5-0.5B, zero-shot and LoRA fine-tuned), and use the models to estimate values of travel time, elasticities and policy scenarios.

Authors: Arnav Prasad, Aditya Chaurasiya, Bhavik Katariya, Sumit Yadav, Vedant Patel, Ritesh Srivastava.

## Design in one paragraph

Models are trained on survey years 1 and 2 (54,766 trips) and tested on survey year 3 (26,320 trips). No household appears in more than one survey year, and all tuning uses five folds grouped by household. All models are scored on the same fixed 4,000-trip test sample; the classical models are also scored on the full test set. Intervals are household-level bootstrap intervals. See `PLAN.md` for the design and `report/` for the write-up.

## Data

The raw data are not included in this repository. Download `lpmc.dat` (tab-separated) and the documentation `CS_LPMC.pdf` from the data section of <https://biogeme.epfl.ch> and place them in `data/raw/`. Please cite Hillel, Elshafie and Jin (2018).

## Layout

| Path | Content |
|---|---|
| `src/s0_audit.py` | checks the data against the documentation and writes the frozen splits |
| `src/s1_mnl.py`, `s2_nested.py` | Multinomial Logit ladder and Nested Logit (Biogeme) |
| `src/s3_gbm.py`, `s3b_rf_mlp.py`, `s3c_mnl_ablation.py` | gradient boosting, random forest and neural network, MNL ablation |
| `src/s4_llm_zeroshot.py`, `s5_llm_finetune.py`, `s5b_order_check.py` | language model: zero-shot, LoRA fine-tuning, option-order check |
| `src/s6_sample_curve.py` | sample-efficiency curve |
| `src/s7_behaviour.py` | value of time, elasticities, policy scenarios, monotonicity |
| `src/s8_eval.py`, `s9_figures.py`, `s10_export.py` | comparison tables, figures, document export |
| `src/common.py`, `choice_models.py`, `llm_common.py` | shared code |
| `results/tables`, `results/predictions`, `results/figures` | outputs, including test probabilities for every model |
| `report/` | report and team guide |

## Reproducing

Python 3.12. `uv venv -p 3.12 .venv && uv pip install biogeme scikit-learn pandas numpy matplotlib mlx-lm markdown`. Run the scripts in `src/` in numeric order from inside `src/`. The language-model stages need Apple silicon (MLX) and the weights `mlx-community/Qwen2.5-0.5B-Instruct-bf16` in `~/ai-ds-models/qwen2.5-0.5b-bf16` (outside the project, because square brackets in a path break the MLX file loader). Fine-tuning the 8,000-trip model takes about an hour on an 8 GB M1.
