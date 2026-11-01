"""Stage 3b: random forest and a small neural net (the other model families used in the published LPMC benchmarks).
Settings are chosen on household-grouped fold 0 of the TRAINING years; the test set is scored once at the end."""
import json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from common import load_split, FEATURES, RES, SEED

d = load_split()
tr, te = d[d.split == "train"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)

def design(df):  # numeric features + one-hot of the two unordered codes
    x = df[[c for c in FEATURES if c not in ("purpose", "faretype")]].copy()
    for c in ("purpose", "faretype"):
        for k in range(1, 6): x[f"{c}_{k}"] = (df[c] == k) * 1
    return x.astype(float)

Xtr, Xte = design(tr), design(te)
sc = StandardScaler().fit(Xtr)  # scaling statistics from the training years only
a, v = tr.fold != 0, tr.fold == 0
ll = lambda m, X, y: log_loss(y, np.clip(m.predict_proba(X), 1e-9, 1), labels=[0, 1, 2, 3])

res = []
rf_cfg = [dict(min_samples_leaf=5, max_features="sqrt"), dict(min_samples_leaf=20, max_features="sqrt"),
          dict(min_samples_leaf=5, max_features=0.5), dict(min_samples_leaf=20, max_features=0.5)]
for c in rf_cfg:
    m = RF(n_estimators=300, n_jobs=4, random_state=SEED, **c).fit(Xtr[a], tr.y[a]); res.append(dict(model="rf", cfg=json.dumps(c), val_logloss=ll(m, Xtr[v], tr.y[v])))
mlp_cfg = [dict(hidden_layer_sizes=(64, 64), alpha=1e-3), dict(hidden_layer_sizes=(128, 64), alpha=1e-3), dict(hidden_layer_sizes=(64, 64), alpha=1e-2)]
for c in mlp_cfg:  # early stopping uses a random 10% of the fitting data (not household-grouped): a minor optimism in the stop point only
    m = MLP(max_iter=200, early_stopping=True, n_iter_no_change=8, random_state=SEED, batch_size=256, **c).fit(sc.transform(Xtr[a]), tr.y[a])
    res.append(dict(model="mlp", cfg=json.dumps(c), val_logloss=ll(m, sc.transform(Xtr[v]), tr.y[v])))
res = pd.DataFrame(res); res.to_csv(RES / "tables/extra_ml_search.csv", index=False); print(res.round(4).to_string(index=False))

def save(name, P):
    out = pd.DataFrame(P, columns=["p_walk", "p_cycle", "p_pt", "p_drive"]); out.insert(0, "trip_id", te.trip_id.values); out.to_csv(RES / f"predictions/{name}.csv", index=False)

b = res[res.model == "rf"].sort_values("val_logloss").iloc[0]
save("rf", RF(n_estimators=500, n_jobs=4, random_state=SEED, **json.loads(b.cfg)).fit(Xtr, tr.y).predict_proba(Xte))
b = res[res.model == "mlp"].sort_values("val_logloss").iloc[0]
Ps = []  # average 5 seeds: single small nets are noisy
for s in range(5):
    m = MLP(max_iter=200, early_stopping=True, n_iter_no_change=8, random_state=SEED + s, batch_size=256, **json.loads(b.cfg)).fit(sc.transform(Xtr), tr.y)
    Ps.append(m.predict_proba(sc.transform(Xte)))
save("mlp", np.mean(Ps, axis=0))
print("saved rf and mlp test probabilities")
