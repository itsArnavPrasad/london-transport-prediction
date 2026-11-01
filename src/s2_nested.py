"""Stage 2: Nested Logit on the selected MNL specification. Two nesting structures, tested against MNL (mu = 1)."""
import json, numpy as np, pandas as pd
from scipy.stats import chi2, norm
from biogeme.results_processing import get_pandas_estimated_parameters
from common import load_split, RES
import choice_models as cm

d = load_split()
tr, te = d[d.split == "train"], d[d.split == "test"]
xtr, xte = cm.prep(tr), cm.prep(te)
mnl = json.load(open(RES / "models/mnl.json")); lvl = mnl["level"]
ll_mnl = pd.read_csv(RES / "tables/mnl_ladder.csv").loglik[lvl]
k_mnl = len(mnl["betas"]); n = len(xtr)

rows, fits = [], {}
for kind in ["active_motor", "car_vs_rest"]:
    r = cm.fit(cm.utilities(lvl), xtr, nest=cm.nests(kind), name=f"nl_{kind}")
    fits[kind] = r
    co = get_pandas_estimated_parameters(estimation_results=r)["Estimated parameters"].set_index("Name")
    mus = [m for m in co.index if m.startswith("mu_")]
    k, ll = len(r.get_beta_values()), r.final_loglikelihood
    stat = 2 * (ll - ll_mnl)
    row = dict(structure=kind, converged=r.algorithm_has_converged, k=k, loglik=ll, BIC=k * np.log(n) - 2 * ll,
               LR_vs_MNL=stat, df=k - k_mnl, p_LR=chi2.sf(stat, k - k_mnl))
    for m in mus:  # t-test of the nest parameter against 1 (not against 0)
        t = (co.Value[m] - 1) / co["Robust std err."][m]
        row.update({m: co.Value[m], f"t_vs_1[{m}]": t})
    rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(RES / "tables/nested_comparison.csv", index=False)
print(f"MNL (M{lvl}): loglik {ll_mnl:.1f}, BIC {pd.read_csv(RES / 'tables/mnl_ladder.csv').BIC[lvl]:.1f}")
print(res.round(3).T.to_string())

# keep the converged structure with the lowest BIC for the prediction comparison
ok = res[res.converged]
best = ok.loc[ok.BIC.idxmin(), "structure"]; R = fits[best]
print(f"\nNL structure used for predictions: {best}")
CO = get_pandas_estimated_parameters(estimation_results=R)["Estimated parameters"]
CO.to_csv(RES / "tables/nested_coefficients.csv", index=False)
json.dump({"level": lvl, "structure": best, "betas": R.get_beta_values()}, open(RES / "models/nested.json", "w"), indent=1)
P = cm.predict(cm.utilities(lvl), R.get_beta_values(), xte, nest=cm.nests(best))
out = pd.DataFrame(P, columns=["p_walk", "p_cycle", "p_pt", "p_drive"]); out.insert(0, "trip_id", te.trip_id.values)
out.to_csv(RES / "predictions/nested.csv", index=False)
print("saved", out.shape)
