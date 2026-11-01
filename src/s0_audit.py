"""Stage 0: verify the data against the documentation, then write the frozen splits."""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from common import load, PROC, SEED

d = load()
# ---- checks against CS_LPMC.pdf (any failure stops the pipeline)
assert d.shape[0] == 81086 and d.person_n.notna().all() and d.isna().sum().sum() == 0
assert d.trip_id.is_unique and d.household_id.nunique() == 17616
assert d.groupby(["household_id", "person_n"]).ngroups == 31954
assert d.groupby("household_id").survey_year.nunique().max() == 1, "household spans survey years"
assert set(d.travel_mode) == {1, 2, 3, 4} and d.survey_year.isin([1, 2, 3]).all()
assert (d.t_drive > 0).all() and (d.t_pt > 0).all()
print("audit passed: no nulls/duplicates, counts match docs, households disjoint across survey years")
print("mode share by split:"); print(pd.crosstab(d.survey_year == 3, d.travel_mode, normalize="index").round(3))

# ---- splits
s = pd.DataFrame({"trip_id": d.trip_id, "split": np.where(d.survey_year == 3, "test", "train")})
tr = d[d.survey_year < 3]
fold = np.full(len(d), -1)
for k, (_, v) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
                           .split(tr, tr.travel_mode, tr.household_id)):
    fold[tr.index.values[v]] = k
s["fold"] = fold  # grouped-by-household CV fold within the training pool; -1 = test

rng = np.random.default_rng(SEED)
# LLM validation = 1000 trips from fold 0; LLM training pool = folds 1-4 (households disjoint from fold 0)
s["llm_val"] = False
s.loc[rng.choice(s.index[s.fold == 0], 1000, replace=False), "llm_val"] = True
pool = s.index[s.fold > 0]
s["llm_rank"] = -1
s.loc[rng.permutation(pool), "llm_rank"] = np.arange(len(pool))  # first N ranks = nested fine-tune subsets
# test subsamples: rank 0..3999 = head-to-head set, rank 0..999 = scenario set
s["test_rank"] = -1
t = s.index[s.split == "test"]
s.loc[rng.choice(t, 4000, replace=False), "test_rank"] = rng.permutation(4000)

assert set(d.household_id[s.fold == 0]).isdisjoint(d.household_id[s.fold > 0])
s.to_csv(PROC / "splits.csv", index=False)
print(s.split.value_counts().to_dict(), "| folds:", np.bincount(fold[fold >= 0]).tolist(),
      "| test4000:", (s.test_rank >= 0).sum(), "| llm_val:", s.llm_val.sum(), "| llm pool:", len(pool))
