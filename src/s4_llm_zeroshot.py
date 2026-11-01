"""Stage 4: Qwen2.5-0.5B-Instruct zero-shot (no training). Prompt checks use the VALIDATION set only;
the 4,000-trip test subsample is scored once. Also reports a prior-corrected variant (3 bias terms fitted on validation)."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from common import load_split, metrics, RES
import llm_common as L

d = load_split()
val = d[d.llm_val].reset_index(drop=True)
te = d[d.test_rank >= 0].sort_values("test_rank").reset_index(drop=True)
model, tok = L.load()
pr = lambda df, order=(0, 1, 2, 3): [L.render(r, order) for r in df.itertuples()]

zv = L.score(model, tok, pr(val), log_every=500)
zv_rev = L.score(model, tok, pr(val, (3, 2, 1, 0)), log_every=500)  # option order reversed: position-bias check
Pv, Pv_rev = L.softmax(zv), L.softmax(zv_rev)
print("VALIDATION zero-shot:", {k: round(float(v), 4) for k, v in metrics(val.y, Pv).items()})
print("VALIDATION reversed order:", {k: round(float(v), 4) for k, v in metrics(val.y, Pv_rev).items()})
print("order effect: mean |dP| =", round(float(np.abs(Pv - Pv_rev).mean()), 4), "| prediction changes for",
      round(float((Pv.argmax(1) != Pv_rev.argmax(1)).mean()), 3), "of trips")
print("predicted mode counts on validation:", np.bincount(Pv.argmax(1), minlength=4).tolist(), "| true:", np.bincount(val.y, minlength=4).tolist())

# prior correction: additive bias per mode (walk fixed at 0), fitted on validation logits only
def nll(b):
    P = L.softmax(zv + np.r_[0, b]); return -np.log(P[np.arange(len(val)), val.y]).mean()
bias = np.r_[0, minimize(nll, np.zeros(3), method="BFGS").x]
print("fitted bias", bias.round(3), "| VALIDATION logloss", round(nll(bias[1:]), 4))
np.save(RES / "models/qwen_zeroshot_bias.npy", bias)

zt = L.score(model, tok, pr(te), log_every=500)  # the one pass over the test subsample
np.save(RES / "predictions/qwen_zeroshot_logits_test4000.npy", zt)
for name, P in [("qwen_zeroshot", L.softmax(zt)), ("qwen_zeroshot_prior", L.softmax(zt + bias))]:
    out = pd.DataFrame(P, columns=["p_walk", "p_cycle", "p_pt", "p_drive"]); out.insert(0, "trip_id", te.trip_id.values)
    out.to_csv(RES / f"predictions/{name}.csv", index=False)
print("saved zero-shot predictions for", len(te), "test trips")
