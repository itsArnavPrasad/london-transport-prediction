"""Stage 1b: household-clustered bootstrap of the final MNL. Trips of the same person and household are correlated, so
model-based and robust standard errors treat them as more independent than they are. We resample whole households
(with replacement) from the TRAINING years, refit the selected specification, and read standard errors and
value-of-time intervals from the distribution of refits."""
import json, sys, time, numpy as np, pandas as pd
from biogeme.results_processing import get_pandas_estimated_parameters
from common import load_split, RES, SEED
import choice_models as cm

B = int(sys.argv[1]) if len(sys.argv) > 1 else 100
d = load_split(); tr = d[d.split == "train"].reset_index(drop=True)
mnl = json.load(open(RES / "models/mnl.json")); lvl, b0 = mnl["level"], mnl["betas"]
x = cm.prep(tr)
idx = pd.Series(np.arange(len(tr))).groupby(tr.household_id.values).apply(np.asarray).values
VOT = {"walking": "b_t_walk", "cycling": "b_t_cycle", "PT access/egress": "b_t_pt_access", "PT rail in-vehicle": "b_t_pt_rail",
       "PT bus in-vehicle": "b_t_pt_bus", "PT interchange": "b_t_pt_int", "driving": "b_t_drive"}
rng = np.random.default_rng(SEED)
rows, t0 = [], time.time()
for b in range(B):
    take = np.concatenate(idx[rng.integers(0, len(idx), len(idx))])
    r = cm.fit(cm.utilities(lvl), x.iloc[take].reset_index(drop=True), name="boot", start=b0)
    if not r.algorithm_has_converged:
        print(f"rep {b}: not converged, skipped", flush=True); continue
    rows.append(r.get_beta_values())
    if (b + 1) % 10 == 0:
        print(f"{b + 1}/{B} done, {(time.time() - t0) / 60:.1f} min", flush=True)
bt = pd.DataFrame(rows); bt.to_csv(RES / "tables/mnl_bootstrap_betas.csv", index=False)

# coefficient table: model-based robust SE (from the original fit) against the bootstrap SE
orig = pd.read_csv(RES / "tables/mnl_coefficients.csv"); orig.columns = ["name", "value", "robust_se", "t", "p"]
orig["boot_se"] = orig.name.map(bt.std(ddof=1)); orig["boot_lo"] = orig.name.map(bt.quantile(.025)); orig["boot_hi"] = orig.name.map(bt.quantile(.975))
orig["se_ratio"] = orig.boot_se / orig.robust_se
orig["sig_robust"] = orig.p < 0.05; orig["sig_boot"] = (orig.boot_lo > 0) | (orig.boot_hi < 0)
orig.to_csv(RES / "tables/mnl_bootstrap_coefficients.csv", index=False)
print(f"\n{len(bt)} converged refits. Median bootstrap/robust SE ratio {orig.se_ratio.median():.2f} (range {orig.se_ratio.min():.2f}-{orig.se_ratio.max():.2f}).")
print(f"significant at 5%: robust SE {orig.sig_robust.sum()} of {len(orig)}; bootstrap {orig.sig_boot.sum()} of {len(orig)}")

# value of time with bootstrap intervals (all training trips)
vt = []
for lab, p in VOT.items():
    est = 60 * b0[p] / b0["b_cost"]; s = 60 * bt[p] / bt["b_cost"]
    vt.append(dict(component=lab, vot_gbp_per_hour=est, boot_lo=s.quantile(.025), boot_hi=s.quantile(.975), boot_median=s.median()))
vt = pd.DataFrame(vt); vt.to_csv(RES / "tables/mnl_bootstrap_vot.csv", index=False); print(vt.round(1).to_string(index=False))
