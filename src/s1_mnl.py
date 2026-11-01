"""Stage 1: Multinomial Logit ladder M0-M4, group-importance tests, test-set probabilities.
Estimation uses TRAIN trips only (survey years 1-2). The test set is touched once, at the end, for predictions."""
import json, numpy as np, pandas as pd
from scipy.stats import chi2
from biogeme.results_processing import get_pandas_estimated_parameters
from common import load_split, RES
import choice_models as cm

d = load_split()
tr, te = d[d.split == "train"], d[d.split == "test"]
xtr, xte = cm.prep(tr), cm.prep(te)
n = len(xtr)

# ---- M0..M4 ladder
fits = {}
for lvl in range(5):
    fits[lvl] = cm.fit(cm.utilities(lvl), xtr, name=f"mnl_m{lvl}")
    assert fits[lvl].algorithm_has_converged, f"M{lvl} did not converge"
ll0 = fits[0].final_loglikelihood
rows = []
for lvl, r in fits.items():
    k, ll = len(r.get_beta_values()), r.final_loglikelihood
    row = dict(model=cm.LEVELS[lvl], k=k, loglik=ll, rho2=1 - ll / ll0, adj_rho2=1 - (ll - k) / ll0,
               AIC=2 * k - 2 * ll, BIC=k * np.log(n) - 2 * ll)
    if lvl:  # likelihood-ratio test against the previous (nested) model
        prev = fits[lvl - 1]
        stat, df = 2 * (ll - prev.final_loglikelihood), k - len(prev.get_beta_values())
        row.update(LR=stat, df=df, p=chi2.sf(stat, df))
    rows.append(row)
ladder = pd.DataFrame(rows)
ladder.to_csv(RES / "tables/mnl_ladder.csv", index=False)
print(ladder.round(3).to_string(index=False))

# ---- final model = lowest BIC
best = int(ladder.BIC.idxmin()); R = fits[best]
print(f"\nselected: M{best} (lowest BIC)")
coef = get_pandas_estimated_parameters(estimation_results=R)["Estimated parameters"]
coef.to_csv(RES / "tables/mnl_coefficients.csv", index=False)
b = R.get_beta_values()
for p in ["b_time", "b_t_walk", "b_t_cycle", "b_t_pt_access", "b_t_pt_rail", "b_t_pt_bus", "b_t_pt_int", "b_interch", "b_t_drive", "b_cost"]:
    if p in b:
        assert b[p] < 0, f"unexpected sign for {p}: {b[p]}"
print("sign check passed: all time, interchange and cost coefficients are negative")

# ---- importance of attribute groups: drop one group, likelihood-ratio test against the selected model
if best >= 3:
    g = []
    for grp in cm.GROUPS:
        if grp == "trip" and best < 4:
            continue
        r = cm.fit(cm.utilities(best, drop=(grp,)), xtr, name=f"drop_{grp}")
        stat, df = 2 * (R.final_loglikelihood - r.final_loglikelihood), len(b) - len(r.get_beta_values())
        g.append(dict(group_dropped=grp, params_removed=df, loglik_loss=R.final_loglikelihood - r.final_loglikelihood,
                      LR=stat, p=chi2.sf(stat, df)))
    g = pd.DataFrame(g).sort_values("LR", ascending=False)
    g.to_csv(RES / "tables/mnl_group_importance.csv", index=False)
    print("\n", g.round(3).to_string(index=False))

# ---- sensitivity: separate cost coefficients for PT fare and driving cost (relaxes the generic marginal utility of money)
from biogeme.expressions import Beta, Variable
U2 = cm.utilities(best)
U2[3] = U2[3] + Beta("d_cost_pt", 0, None, None, 0) * Variable("cost_transit")
U2[4] = U2[4] + Beta("d_cost_drive", 0, None, None, 0) * Variable("cost_drive")
r2 = cm.fit(U2, xtr, name="sep_cost"); b2 = r2.get_beta_values()
cpt, cdr = b2["b_cost"] + b2["d_cost_pt"], b2["b_cost"] + b2["d_cost_drive"]
sens = pd.DataFrame([dict(loglik=r2.final_loglikelihood, LR_vs_pooled=2 * (r2.final_loglikelihood - R.final_loglikelihood),
                          b_cost_pooled=b["b_cost"], b_cost_pt=cpt, b_cost_drive=cdr,
                          vot_drive_separate=60 * b2["b_t_drive"] / cdr, vot_drive_pooled=60 * b["b_t_drive"] / b["b_cost"],
                          corr_fare_distance=float(tr[tr.cost_transit > 0][["cost_transit", "distance"]].corr().iloc[0, 1]),
                          share_pt_fare_zero=float((tr.cost_transit == 0).mean()))])
sens.to_csv(RES / "tables/mnl_cost_sensitivity.csv", index=False)
print("\ncost sensitivity:", sens.round(3).to_dict("records")[0])

# ---- save parameters (for scenarios) and test-set probabilities (touched once)
(RES / "models").mkdir(exist_ok=True)
json.dump({"level": best, "betas": b}, open(RES / "models/mnl.json", "w"), indent=1)
P = cm.predict(cm.utilities(best), b, xte)
out = pd.DataFrame(P, columns=["p_walk", "p_cycle", "p_pt", "p_drive"]); out.insert(0, "trip_id", te.trip_id.values)
out.to_csv(RES / "predictions/mnl.csv", index=False)
print("saved", out.shape, "test probabilities")
