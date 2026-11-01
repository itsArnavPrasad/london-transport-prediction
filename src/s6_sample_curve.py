"""Stage 6: sample-efficiency curve. Test log-loss on the shared 4,000-trip subsample as a function of the number of
training trips (nested subsets of the household-disjoint pool), for MNL, gradient boosting and the fine-tuned Qwen."""
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from common import load_split, gbm_X, metrics, cluster_boot, RES, SEED
import choice_models as cm

d = load_split()
pool = d[d.llm_rank >= 0].sort_values("llm_rank")
te = d[d.test_rank >= 0].sort_values("test_rank")
gj = json.load(open(RES / "models/gbm.json"))
NS = [500, 2000, 8000, len(pool)]
rows = []

def add(model, n, P):
    ci = cluster_boot(te.y.values, P, te.household_id.values, B=300)
    rows.append(dict(model=model, n_train=n, logloss=ci.est.logloss, lo=ci.lo.logloss, hi=ci.hi.logloss, accuracy=ci.est.accuracy))

xte = cm.prep(te)
for n in NS:
    sub = pool.head(n); xs = cm.prep(sub)
    for lvl in (2, 4):  # M2 is stable with a few hundred trips; M4 has 67 parameters
        skip = tuple(c for c in cm.PERSON + cm.TRIP if xs[c].sum() < 10 or xs[c].nunique() < 2)
        try:
            r = cm.fit(cm.utilities(lvl, skip=skip), xs, name="curve"); assert r.algorithm_has_converged
            add(f"MNL M{lvl}", n, cm.predict(cm.utilities(lvl, skip=skip), r.get_beta_values(), xte))
        except Exception as e:
            print(f"MNL M{lvl} n={n} failed: {type(e).__name__}")
    g = HGB(max_iter=500, early_stopping=True, n_iter_no_change=15, validation_fraction=0.15, random_state=SEED, **gj["params"]).fit(gbm_X(sub), sub.y)
    add("Gradient boosting", n, g.predict_proba(gbm_X(te)))
    print("done", n, flush=True)

def read(name):
    p = RES / f"predictions/{name}.csv"
    return pd.read_csv(p).set_index("trip_id").loc[te.trip_id].values if p.exists() else None
if (P := read("qwen_zeroshot")) is not None: add("Qwen 0.5B zero-shot", 0, P)
for n in (500, 2000, 8000):
    if (P := read(f"qwen_ft_n{n}")) is not None: add("Qwen 0.5B fine-tuned", n, P)
out = pd.DataFrame(rows); out.to_csv(RES / "tables/sample_curve.csv", index=False)
print(out.round(4).to_string(index=False))
