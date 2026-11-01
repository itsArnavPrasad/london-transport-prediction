"""Stage 5: LoRA fine-tuning of Qwen2.5-0.5B-Instruct on N training trips (answer-token loss only).
usage: s5_llm_finetune.py N LR [--test]   N = number of training trips (nested subsets), --test scores the test subsample.
Always scores the 1,000-trip validation set; the test subsample is scored only with --test (final runs)."""
import json, subprocess, sys, time, numpy as np, pandas as pd
from common import load_split, metrics, RES
import llm_common as L

N, LR, DO_TEST, SKIP = int(sys.argv[1]), float(sys.argv[2]), "--test" in sys.argv, "--skip-train" in sys.argv
d = load_split()
val = d[d.llm_val].reset_index(drop=True)
tr = d[(d.llm_rank >= 0) & (d.llm_rank < N)].sort_values("llm_rank").reset_index(drop=True)  # nested subsets, folds 1-4 only
te = d[d.test_rank >= 0].sort_values("test_rank").reset_index(drop=True)
assert len(tr) == N and set(tr.household_id).isdisjoint(val.household_id)

tag = f"n{N}_lr{LR:g}"
work, adapters = L.ADAPTERS / f"data_{tag}", L.ADAPTERS / tag
work.mkdir(parents=True, exist_ok=True)
model, tok = L.load()  # for the chat template and the post-training scoring base


def rows(df):
    for r in df.itertuples():
        yield {"messages": [{"role": "system", "content": L.SYSTEM}, {"role": "user", "content": L.render(r)},
                            {"role": "assistant", "content": L.WORDS[int(r.y)]}]}
for name, df in [("train", tr), ("valid", val)]:
    with open(work / f"{name}.jsonl", "w") as f:
        for row in rows(df):
            f.write(json.dumps(row) + "\n")
BATCH = 4
(work / "lora.yaml").write_text("lora_parameters:\n  rank: 8\n  scale: 20.0\n  dropout: 0.0\n"
                                '  keys: ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]\n')
del model  # free memory before training
cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", L.MODEL, "--train", "--data", str(work), "--mask-prompt",
       "--num-layers", "24", "--batch-size", str(BATCH), "--iters", str(N // BATCH), "--learning-rate", str(LR),
       "--max-seq-length", "320", "--grad-checkpoint", "--steps-per-report", "50", "--steps-per-eval", "250",
       "--val-batches", "25", "--save-every", "250", "--adapter-path", str(adapters), "-c", str(work / "lora.yaml"), "--seed", "42"]
t0 = time.time()
if not SKIP:  # --skip-train: reuse an adapter that was already trained (e.g. to score the test subsample later)
    log = (RES / f"tmp/ft_{tag}.log").open("w")
    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    print(f"{tag}: trained in {(time.time() - t0) / 60:.1f} min")
train_s = time.time() - t0

model, tok = L.load(str(adapters))
pr = lambda df: [L.render(r) for r in df.itertuples()]
zv = L.score(model, tok, pr(val))
mv = metrics(val.y, L.softmax(zv))
print(f"{tag} VALIDATION:", {k: round(float(v), 4) for k, v in mv.items()})
runs = RES / "tables/qwen_ft_runs.csv"
rec = pd.DataFrame([dict(n_train=N, lr=LR, train_minutes=train_s / 60, val_logloss=mv["logloss"], val_accuracy=mv["accuracy"])])
if not SKIP:
    rec.to_csv(runs, mode="a", header=not runs.exists(), index=False)

if DO_TEST:
    zt = L.score(model, tok, pr(te), log_every=1000)  # one pass over the test subsample
    out = pd.DataFrame(L.softmax(zt), columns=["p_walk", "p_cycle", "p_pt", "p_drive"]); out.insert(0, "trip_id", te.trip_id.values)
    out.to_csv(RES / f"predictions/qwen_ft_n{N}.csv", index=False)
    print("saved test predictions", out.shape)
