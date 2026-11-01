"""Stage 8: the comparison tables. Reads only saved probabilities, so it can be rerun at any time.
Primary table: every model on the shared 4,000-trip test subsample. Secondary: classical models on the full test set."""
import numpy as np, pandas as pd
from common import load_split, metrics, RES, MODES, PURPOSES

B = 1000
d = load_split()
te = d[d.split == "test"].set_index("trip_id")
tr_share = np.bincount(d[d.split == "train"].y, minlength=4) / (d.split == "train").sum()
sub = d[d.test_rank >= 0].sort_values("test_rank").trip_id.values

def read(name):
    p = pd.read_csv(RES / f"predictions/{name}.csv").set_index("trip_id")
    return p[["p_walk", "p_cycle", "p_pt", "p_drive"]]

NAMES = {"mnl": "Multinomial Logit", "nested": "Nested Logit", "gbm": "Gradient boosting", "rf": "Random forest", "mlp": "Neural net (MLP)",
         "qwen_zeroshot": "Qwen 0.5B zero-shot", "qwen_zeroshot_prior": "Qwen 0.5B zero-shot (prior-corrected)",
         "qwen_ft_n8000": "Qwen 0.5B LoRA fine-tuned (8,000 trips)"}
P = {k: read(k) for k in NAMES if (RES / f"predictions/{k}.csv").exists()}
base = pd.DataFrame(np.tile(tr_share, (len(te), 1)), index=te.index, columns=["p_walk", "p_cycle", "p_pt", "p_drive"])
maj = base * 0 + np.array([0.0005, 0.0005, 0.0005, 0.9985])
P = {"market_share": base, "always_drive": maj, **P}; NAMES = {"market_share": "Training market shares (reference)", "always_drive": "Always drive (reference)", **NAMES}

def run(ids, models, label):
    y = te.loc[ids, "y"].values; hh = te.loc[ids, "household_id"].values
    idx = pd.Series(np.arange(len(ids))).groupby(hh).apply(np.asarray).values
    rng = np.random.default_rng(42); draws = [np.concatenate(idx[rng.integers(0, len(idx), len(idx))]) for _ in range(B)]
    rows, boots = [], {}
    for m in models:
        p = P[m].loc[ids].values
        est = metrics(y, p)
        bs = pd.DataFrame([metrics(y[t], p[t]) for t in draws]); boots[m] = bs
        rows.append({"model": NAMES[m], **{f"{k}": est[k] for k in est}, **{f"{k}_lo": bs[k].quantile(.025) for k in est}, **{f"{k}_hi": bs[k].quantile(.975) for k in est}})
    t = pd.DataFrame(rows)
    t["gmpca"], t["gmpca_lo"], t["gmpca_hi"] = np.exp(-t.logloss), np.exp(-t.logloss_hi), np.exp(-t.logloss_lo)  # geometric-mean probability of correct assignment, as in the published LPMC papers
    t.to_csv(RES / f"tables/compare_{label}.csv", index=False)
    return t, boots

t4, boots = run(sub, list(P), "test4000")
print("=== shared 4,000-trip test subsample (95% household-bootstrap CIs) ===")
show = lambda t, k: f"{t[k]:.3f} [{t[k + '_lo']:.3f}, {t[k + '_hi']:.3f}]"
for _, r in t4.iterrows():
    print(f"{r.model:42s} logloss {show(r,'logloss')}  acc {show(r,'accuracy')}  macroF1 {r.macro_f1:.3f}  ECE {r.ece:.3f}  shareErr {r.share_err:.4f}")

classical = [m for m in ["market_share", "always_drive", "mnl", "nested", "gbm", "rf", "mlp"] if m in P]
tf, _ = run(te.index.values, classical, "test_full")
print("\n=== classical models on the full test set (26,320 trips) ===")
for _, r in tf.iterrows():
    print(f"{r.model:42s} logloss {show(r,'logloss')}  acc {show(r,'accuracy')}  macroF1 {r.macro_f1:.3f}")

# paired log-loss differences on the same bootstrap draws (negative = first model better)
pairs = [("rf", "mnl"), ("mlp", "mnl"), ("gbm", "mlp"), ("gbm", "rf"), ("gbm", "mnl"), ("nested", "mnl"), ("gbm", "nested"), ("qwen_ft_n8000", "qwen_zeroshot"), ("qwen_ft_n8000", "mnl"),
         ("qwen_ft_n8000", "gbm"), ("qwen_zeroshot_prior", "qwen_zeroshot")]
pr = []
for a, b in pairs:
    if a in boots and b in boots:
        dlt = boots[a].logloss - boots[b].logloss
        pr.append(dict(a=NAMES[a], b=NAMES[b], diff_logloss=dlt.mean(), lo=dlt.quantile(.025), hi=dlt.quantile(.975)))
pr = pd.DataFrame(pr); pr.to_csv(RES / "tables/paired_logloss.csv", index=False)
print("\n=== paired log-loss differences (a - b; negative = a better) ==="); print(pr.round(4).to_string(index=False))

# per-mode recall and confusion matrices on the shared subsample
y4 = te.loc[sub, "y"].values; conf, rec = {}, []
for m in P:
    pred = P[m].loc[sub].values.argmax(1)
    cm_ = np.zeros((4, 4), int); np.add.at(cm_, (y4, pred), 1); conf[m] = cm_
    rec.append({"model": NAMES[m], **{f"recall_{MODES[k]}": cm_[k, k] / max(cm_[k].sum(), 1) for k in range(4)}})
pd.DataFrame(rec).to_csv(RES / "tables/recall_by_mode.csv", index=False)
np.save(RES / "tables/confusion.npy", np.stack([conf[m] for m in P])); pd.Series(list(P)).to_csv(RES / "tables/confusion_models.csv", index=False)
print("\nrecall by mode:"); print(pd.DataFrame(rec).round(3).to_string(index=False))

# segments on the shared subsample
seg = te.loc[sub].assign(distband=lambda x: pd.cut(x.distance / 1000, [0, 1, 2.5, 5, 10, 100], labels=["<1km", "1-2.5", "2.5-5", "5-10", ">10km"]).astype(str))
out = []
for col, lab in [("purpose", "purpose"), ("car_ownership", "car_ownership"), ("distband", "distance")]:
    for v, g in seg.groupby(col):
        for m in [k for k in P if k not in ("market_share", "always_drive")]:
            mt = metrics(g.y.values, P[m].loc[g.index].values)
            out.append(dict(segment=lab, value=v, n=len(g), model=NAMES[m], logloss=mt["logloss"], accuracy=mt["accuracy"]))
pd.DataFrame(out).to_csv(RES / "tables/segments.csv", index=False)
print("saved tables to results/tables/")
