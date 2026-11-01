"""Stage 3: gradient boosting. Tuning, early stopping and calibration use household-grouped folds inside the
training years only. The test set is touched once, for the final predictions."""
import json, time, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar
from common import load_split, metrics, FEATURES, GROUPS, RES, SEED

d = load_split()
tr, te = d[d.split == "train"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)


def X(df):
    x = df[FEATURES].copy()
    for c, n in [("purpose", 5), ("faretype", 5)]:  # unordered codes -> categorical
        x[c] = pd.Categorical(x[c], categories=range(1, n + 1))
    return x


MAXIT = 500
def fit_fold(params, k):
    a, v = tr[tr.fold != k], tr[tr.fold == k]
    m = HGB(max_iter=MAXIT, early_stopping=False, random_state=SEED, **params).fit(X(a), a.y)
    ll = np.array([log_loss(v.y, p, labels=[0, 1, 2, 3]) for p in m.staged_predict_proba(X(v))])
    return m, ll


# ---- random search on two folds
rng = np.random.default_rng(SEED)
space = dict(learning_rate=[0.05, 0.1], max_leaf_nodes=[15, 31, 63], min_samples_leaf=[20, 50, 100], l2_regularization=[0, 1, 10])
configs = [{k: v[rng.integers(len(v))] for k, v in space.items()} for _ in range(10)]
res = []
for i, p in enumerate(configs):
    t = time.time()
    lls = [fit_fold(p, k)[1] for k in (0, 1)]
    m = np.mean(lls, axis=0)
    res.append(dict(**p, best_iter=int(m.argmin()) + 1, cv_logloss=m.min()))
    print(i, p, "->", round(m.min(), 4), "@", int(m.argmin()) + 1, f"({time.time() - t:.0f}s)", flush=True)
search = pd.DataFrame(res).sort_values("cv_logloss"); search.to_csv(RES / "tables/gbm_search.csv", index=False)
best = search.iloc[0]
params = dict(learning_rate=float(best.learning_rate), max_leaf_nodes=int(best.max_leaf_nodes),
              min_samples_leaf=int(best.min_samples_leaf), l2_regularization=float(best.l2_regularization))

# ---- 5-fold grouped CV with the chosen config: out-of-fold probabilities, iteration count, fold-0 model
oof, iters, models = np.zeros((len(tr), 4)), [], {}
for k in range(5):
    m, ll = fit_fold(params, k)
    it = int(ll.argmin()) + 1; iters.append(it)
    v = tr.fold == k
    oof[v.values] = list(m.staged_predict_proba(X(tr[v])))[it - 1]
    models[k] = (m, it)
n_iter = int(np.mean(iters))
cv = metrics(tr.y, oof)
print("5-fold grouped CV:", {k: round(float(x), 4) for k, x in cv.items()}, "| iterations per fold", iters)

# ---- optional temperature scaling fitted on out-of-fold probabilities (applied only if it clearly helps)
lg = np.log(np.clip(oof, 1e-9, 1))
def nll(T):
    z = lg / T; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return log_loss(tr.y, p, labels=[0, 1, 2, 3])
T = minimize_scalar(nll, bounds=(0.5, 2), method="bounded").x
use_T = nll(1.0) - nll(T) > 0.001
print(f"temperature T={T:.3f}, logloss {nll(1.0):.4f} -> {nll(T):.4f}; applied: {use_T}")

# ---- permutation importance by attribute group, on fold 0 with the fold-0 model (never the test set)
m0, it0 = models[0]; v0 = tr[tr.fold == 0].reset_index(drop=True)
def lloss(df): return log_loss(df.y, list(m0.staged_predict_proba(X(df)))[it0 - 1], labels=[0, 1, 2, 3])
base, imp = lloss(v0), []
for g, cols in GROUPS.items():
    inc = []
    for r in range(5):
        w = v0.copy(); perm = np.random.default_rng(SEED + r).permutation(len(w))
        w[cols] = w[cols].iloc[perm].values  # permute the whole group jointly, rows stay intact
        inc.append(lloss(w) - base)
    imp.append(dict(group=g, logloss_increase=np.mean(inc), sd=np.std(inc)))
imp = pd.DataFrame(imp).sort_values("logloss_increase", ascending=False); imp.to_csv(RES / "tables/gbm_group_importance.csv", index=False)
print(imp.round(4).to_string(index=False))

# ---- final model on the whole training pool, one pass over the test set
final = HGB(max_iter=n_iter, early_stopping=False, random_state=SEED, **params).fit(X(tr), tr.y)
P = final.predict_proba(X(te))
if use_T:
    z = np.log(np.clip(P, 1e-9, 1)) / T; z -= z.max(1, keepdims=True); P = np.exp(z); P /= P.sum(1, keepdims=True)
json.dump(dict(params=params, n_iter=n_iter, T=float(T), use_T=bool(use_T), cv=cv), open(RES / "models/gbm.json", "w"), indent=1, default=float)
out = pd.DataFrame(P, columns=["p_walk", "p_cycle", "p_pt", "p_drive"]); out.insert(0, "trip_id", te.trip_id.values)
out.to_csv(RES / "predictions/gbm.csv", index=False)
print("saved", out.shape, "| params", params, "| n_iter", n_iter)
