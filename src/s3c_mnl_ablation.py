"""Stage 3c: why does our MNL predict better than published MNLs? Remove variable sets and score on grouped fold 0 of the TRAINING years."""
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
from common import load_split, RES
import choice_models as cm

d = load_split(); tr = d[d.split == "train"]
a, v = tr[tr.fold != 0], tr[tr.fold == 0]
xa, xv = cm.prep(a), cm.prep(v)
FARE = ["ft_child", "ft_dis", "ft_free", "bus_free", "bus_half"]
variants = {"full M4": (), "without fare-type / bus-scale (free-pass) variables": tuple(FARE),
            "without fare variables, age squared, peak, weekend": tuple(FARE + ["age10sq", "peak", "weekend"]),
            "without fare variables and trip variables": tuple(FARE + cm.TRIP)}
rows = []
for name, skip in variants.items():
    r = cm.fit(cm.utilities(4, skip=skip), xa, name="abl"); P = cm.predict(cm.utilities(4, skip=skip), r.get_beta_values(), xv)
    rows.append(dict(variant=name, params=len(r.get_beta_values()), val_logloss=log_loss(v.y, np.clip(P, 1e-9, 1), labels=[0, 1, 2, 3]), val_accuracy=(P.argmax(1) == v.y).mean()))
t = pd.DataFrame(rows); t.to_csv(RES / "tables/mnl_ablation.csv", index=False); print(t.round(4).to_string(index=False))
