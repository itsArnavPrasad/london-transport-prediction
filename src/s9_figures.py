"""Stage 9: every figure in the report, drawn from saved tables and predictions only."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_split, RES, MODES

FIG = RES / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 130})
C = {"Multinomial Logit": "#1f77b4", "Nested Logit": "#17becf", "Gradient boosting": "#2ca02c", "Random forest": "#8c564b", "Neural net (MLP)": "#9467bd",
     "Qwen 0.5B zero-shot": "#d62728", "Qwen 0.5B fine-tuned": "#ff7f0e", "MNL M2": "#aec7e8", "MNL M4": "#1f77b4", "Qwen 0.5B zero-shot (prior-corrected)": "#ff9896", "Qwen 0.5B LoRA fine-tuned (8,000 trips)": "#ff7f0e"}
LAB = ["walk", "cycle", "public transport", "drive"]
d = load_split(); te = d[d.test_rank >= 0].sort_values("test_rank")
def tab(n): return pd.read_csv(RES / f"tables/{n}.csv")
def save(name): plt.tight_layout(); plt.savefig(FIG / f"{name}.png", bbox_inches="tight"); plt.close()
def attempt(fn):
    try: fn()
    except Exception as e: print(f"skipped {fn.__name__}: {type(e).__name__}: {e}")

def f1_mode_share_by_distance():
    b = pd.cut(d.distance / 1000, [0, .5, 1, 2, 5, 10, 50], labels=["<0.5", "0.5-1", "1-2", "2-5", "5-10", ">10"])
    s = pd.crosstab(b, d.travel_mode, normalize="index") * 100; s.columns = LAB
    s.plot(kind="bar", stacked=True, figsize=(6.5, 3.8), color=["#7fc97f", "#beaed4", "#fdc086", "#386cb0"], width=.8)
    plt.xlabel("straight-line distance (km)"); plt.ylabel("share of trips (%)"); plt.xticks(rotation=0); plt.legend(title="chosen mode", fontsize=8, bbox_to_anchor=(1, 1)); save("f1_mode_share_by_distance")

def f2_model_comparison():
    t = tab("compare_test4000"); t = t[~t.model.str.contains("reference")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, (k, ttl) in zip(ax, [("logloss", "log-loss (lower is better)"), ("accuracy", "accuracy (higher is better)")]):
        y = np.arange(len(t))[::-1]
        a.barh(y, t[k], color=[C.get(m, "gray") for m in t.model], xerr=[t[k] - t[k + "_lo"], t[k + "_hi"] - t[k]], capsize=2)
        a.set_yticks(y); a.set_yticklabels(t.model if k == "logloss" else [""] * len(t)); a.set_title(ttl)
    save("f2_model_comparison")

def f3_sample_curve():
    t = tab("sample_curve"); fig, ax = plt.subplots(figsize=(6.5, 4))
    z = t[t.n_train == 0].logloss.iloc[0]; t = t[t.n_train > 0]
    for m, g in t.groupby("model"):
        g = g.sort_values("n_train")
        ax.errorbar(g.n_train, g.logloss, yerr=[g.logloss - g.lo, g.hi - g.logloss], marker="o", capsize=2, label=m, color=C.get(m))
    ax.set_xscale("log"); ax.set_xlabel("training trips (nested subsets; last point = whole 43,812-trip pool)"); ax.set_ylabel("test log-loss"); ax.set_ylim(0.6, 1.4)
    ax.text(0.98, 0.62, f"Qwen zero-shot (no training): log-loss {z:.2f}, off the scale", transform=ax.transAxes, ha="right", fontsize=8, color="#d62728"); ax.legend(fontsize=8); save("f3_sample_efficiency")

def f4_reliability():
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for k, a in enumerate(ax):
        a.plot([0, 1], [0, 1], "k--", lw=1)
        for name, lab in [("mnl", "Multinomial Logit"), ("gbm", "Gradient boosting"), ("qwen_ft_n8000", "Qwen 0.5B LoRA fine-tuned (8,000 trips)"), ("qwen_zeroshot", "Qwen 0.5B zero-shot")]:
            p = RES / f"predictions/{name}.csv"
            if not p.exists(): continue
            P = pd.read_csv(p).set_index("trip_id").loc[te.trip_id].values[:, k]; y = (te.y.values == k)
            q = pd.qcut(P, 8, duplicates="drop"); g = pd.DataFrame({"p": P, "y": y}).groupby(q, observed=True).mean()
            a.plot(g.p, g.y, marker="o", ms=3, label=lab, color=C.get(lab))
        a.set_title(LAB[k]); a.set_xlabel("predicted probability")
    ax[0].set_ylabel("observed frequency"); ax[0].legend(fontsize=7); save("f4_reliability")

def f5_confusion():
    conf, names = np.load(RES / "tables/confusion.npy"), pd.read_csv(RES / "tables/confusion_models.csv").iloc[:, 0].tolist()
    show = [n for n in ["mnl", "gbm", "qwen_zeroshot", "qwen_ft_n8000"] if n in names]
    T = {"mnl": "Multinomial Logit", "gbm": "Gradient boosting", "qwen_zeroshot": "Qwen zero-shot", "qwen_ft_n8000": "Qwen fine-tuned"}
    fig, ax = plt.subplots(1, len(show), figsize=(3.3 * len(show), 3.3))
    for a, n in zip(np.atleast_1d(ax), show):
        c = conf[names.index(n)]; r = c / c.sum(1, keepdims=True); a.imshow(r, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4): a.text(j, i, f"{r[i, j]:.2f}", ha="center", va="center", color="white" if r[i, j] > .5 else "black", fontsize=8)
        a.set_xticks(range(4)); a.set_xticklabels(["walk", "cyc", "PT", "drive"]); a.set_yticks(range(4)); a.set_yticklabels(["walk", "cyc", "PT", "drive"]); a.set_title(T[n], fontsize=9); a.grid(False); a.set_xlabel("predicted")
    np.atleast_1d(ax)[0].set_ylabel("actual (row-normalised)"); save("f5_confusion")

def f6_vot():
    t = tab("mnl_vot"); a = t[t.segment == "all training trips"].set_index("component")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(a))[::-1]; ax[0].barh(y, a.vot_gbp_per_hour, xerr=[a.vot_gbp_per_hour - a.lo, a.hi - a.vot_gbp_per_hour], capsize=2, color="#1f77b4")
    ax[0].set_yticks(y); ax[0].set_yticklabels(a.index); ax[0].set_xlabel("value of time (GBP/hour), MNL, 95% CI")
    i = tab("implied_vot"); i = i.assign(lab=i.model + " / " + i["mode"])
    y = np.arange(len(i))[::-1]; ax[1].barh(y, i.median_gbp_per_hour, xerr=[(i.median_gbp_per_hour - i.q25).clip(lower=0), (i.q75 - i.median_gbp_per_hour).clip(lower=0)], capsize=2, color="#7f7f7f")
    ax[1].set_yticks(y); ax[1].set_yticklabels(i.lab, fontsize=7); ax[1].set_xlabel("implied value of time, median and IQR"); save("f6_value_of_time")

def f7_scenarios():
    t = tab("scenario_shifts"); t = t[(t["sample"] == "subset")]; sc = t.scenario.unique(); models = t.model.unique()
    fig, ax = plt.subplots(1, len(sc), figsize=(3 * len(sc), 3.4), sharey=True)
    for a, s in zip(np.atleast_1d(ax), sc):
        g = t[t.scenario == s]
        for j, m in enumerate(models):
            h = g[g.model == m].set_index("mode").loc[MODES]; x = np.arange(4) + (j - (len(models) - 1) / 2) * .8 / len(models)
            a.bar(x, h.shift_pp, width=.8 / len(models), yerr=[h.shift_pp - h.lo, h.hi - h.shift_pp], capsize=1, label=m, color=C.get(m, None))
        a.set_xticks(range(4)); a.set_xticklabels(["walk", "cyc", "PT", "drive"]); a.set_title(s, fontsize=8); a.axhline(0, color="k", lw=.6)
    np.atleast_1d(ax)[0].set_ylabel("mode-share shift (pp)"); np.atleast_1d(ax)[0].legend(fontsize=6); save("f7_policy_scenarios")

def f8_group_importance():
    m, g = tab("mnl_group_importance"), tab("gbm_group_importance")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].barh(m.group_dropped[::-1], m.loglik_loss[::-1], color="#1f77b4"); ax[0].set_xlabel("log-likelihood lost when the group is removed (MNL)")
    ax[1].barh(g.group[::-1], g.logloss_increase[::-1], xerr=g.sd[::-1], color="#2ca02c"); ax[1].set_xlabel("log-loss increase when the group is permuted (boosting)"); save("f8_group_importance")

def f9_mnl_coefficients():
    c = tab("mnl_coefficients"); c = c[c.Name.str.startswith(("b_t", "b_interch", "b_cost", "b_traffic"))].copy()
    c["lo"], c["hi"] = c.Value - 1.96 * c["Robust std err."], c.Value + 1.96 * c["Robust std err."]; y = np.arange(len(c))[::-1]
    plt.figure(figsize=(5.5, 3.4)); plt.errorbar(c.Value, y, xerr=1.96 * c["Robust std err."], fmt="o", capsize=2); plt.yticks(y, c.Name); plt.axvline(0, color="k", lw=.6)
    plt.xlabel("coefficient (utility per minute, per interchange, per GBP)"); save("f9_mnl_coefficients")

for f in [f1_mode_share_by_distance, f2_model_comparison, f3_sample_curve, f4_reliability, f5_confusion, f6_vot, f7_scenarios, f8_group_importance, f9_mnl_coefficients]:
    attempt(f)
print("figures:", sorted(p.name for p in FIG.glob("*.png")))
