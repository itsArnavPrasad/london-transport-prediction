"""Stage 5b: does the fine-tuned Qwen still depend on the ORDER of the options in the prompt? (validation trips only)"""
import json, numpy as np, pandas as pd
from common import load_split, metrics, RES
import llm_common as L

d = load_split(); val = d[d.llm_val].reset_index(drop=True)
tag = json.load(open(RES / "models/qwen_ft_best.json"))["tag"]
rows = []
for name, adapter in [("fine-tuned", str(L.ADAPTERS / tag))]:  # zero-shot order effect was measured in s4
    m, t = L.load(adapter)
    P = L.softmax(L.score(m, t, [L.render(r) for r in val.itertuples()]))
    Pr = L.softmax(L.score(m, t, [L.render(r, (3, 2, 1, 0)) for r in val.itertuples()]))
    rows.append(dict(model=name, logloss=metrics(val.y, P)["logloss"], logloss_reversed=metrics(val.y, Pr)["logloss"],
                     accuracy=metrics(val.y, P)["accuracy"], accuracy_reversed=metrics(val.y, Pr)["accuracy"],
                     mean_abs_dP=np.abs(P - Pr).mean(), pct_predictions_changed=(P.argmax(1) != Pr.argmax(1)).mean() * 100))
t = pd.DataFrame(rows); t.to_csv(RES / "tables/qwen_order_check.csv", index=False); print(t.round(3).to_string(index=False))
