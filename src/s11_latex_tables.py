"""Stage 11: turn the result tables into LaTeX fragments (report/tables/*.tex) so that every number in the report
comes straight from results/tables/*.csv."""
import numpy as np, pandas as pd
from common import RES, ROOT

OUT = ROOT / "report/tables"; OUT.mkdir(parents=True, exist_ok=True)
T = lambda n: pd.read_csv(RES / f"tables/{n}.csv")
esc = lambda s: str(s).replace("\\", "").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
f = lambda x, d=3: f"{x:.{d}f}".replace("-", "$-$")
ci = lambda e, lo, hi, d=3: f"{f(e, d)} [{f(lo, d)}, {f(hi, d)}]"
pct = lambda x, d=1: f"{100 * x:.{d}f}\\%"
b = lambda s: f"\\textbf{{{s}}}"

def write(name, header, rows, spec, note=None, midrules=()):
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
    for i, r in enumerate(rows):
        if i in midrules: lines.append("\\midrule")
        lines.append(" & ".join(r) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")

SHORT = {"Training market shares (reference)": "Market shares (reference)", "Qwen 0.5B zero-shot (prior-corrected)": "Qwen 0.5B zero-shot, prior-corrected",
         "Qwen 0.5B LoRA fine-tuned (8,000 trips)": "Qwen 0.5B fine-tuned (8,000 trips)"}

# 1. MNL ladder
t = T("mnl_ladder")
rows = []
for i, r in t.iterrows():
    lr = "" if i == 0 else f"{r.LR:,.0f} ({int(r.df)})"
    rows.append([esc(r.model), str(int(r.k)), f"{r.loglik:,.0f}".replace("-", "$-$"), f(r.rho2), f"{r.BIC:,.0f}", lr])
write("ladder", ["Model", "Parameters", "Log-likelihood", "Pseudo-$R^2$", "BIC", "LR test (df)"], rows, "lrrrrr")

# 2. comparison on the shared sample, and 3. full test set
def compare(name, file, drop=()):
    t = T(file); t = t[~t.model.isin(drop)]
    trained = t[~t.model.str.contains("reference")]
    bl, ba = trained.logloss.min(), trained.accuracy.max()
    rows = []
    for _, r in t.iterrows():
        ll, ac = ci(r.logloss, r.logloss_lo, r.logloss_hi), f"{100 * r.accuracy:.1f} [{100 * r.accuracy_lo:.1f}, {100 * r.accuracy_hi:.1f}]"
        if r.logloss == bl: ll = b(ll)
        if r.accuracy == ba: ac = b(ac)
        rows.append([esc(SHORT.get(r.model, r.model)), ll, ac, f(r.macro_f1), f(r.ece), f"{r.share_err:.4f}", f(r.gmpca)])
    write(name, ["Model", "Log-loss [95\\% CI]", "Accuracy (\\%) [95\\% CI]", "Macro-F1", "ECE", "Share error", "GMPCA"], rows, "lccrrrr",
          midrules=(1,))
compare("compare4000", "compare_test4000", drop=("Always drive (reference)",))
tf = T("compare_test_full"); tf = tf[tf.model != "Always drive (reference)"]
rows = [[esc(SHORT.get(r.model, r.model)), ci(r.logloss, r.logloss_lo, r.logloss_hi), f"{100 * r.accuracy:.1f} [{100 * r.accuracy_lo:.1f}, {100 * r.accuracy_hi:.1f}]", f(r.macro_f1), f(r.gmpca)] for _, r in tf.iterrows()]
write("comparefull", ["Model", "Log-loss [95\\% CI]", "Accuracy (\\%) [95\\% CI]", "Macro-F1", "GMPCA"], rows, "lccrr", midrules=(1,))

# 4. paired differences
p = T("paired_logloss"); keep = [("Gradient boosting", "Multinomial Logit"), ("Neural net (MLP)", "Multinomial Logit"), ("Random forest", "Multinomial Logit"),
    ("Nested Logit", "Multinomial Logit"), ("Gradient boosting", "Neural net (MLP)"), ("Gradient boosting", "Random forest"),
    ("Qwen 0.5B LoRA fine-tuned (8,000 trips)", "Qwen 0.5B zero-shot"), ("Qwen 0.5B LoRA fine-tuned (8,000 trips)", "Multinomial Logit"), ("Qwen 0.5B LoRA fine-tuned (8,000 trips)", "Gradient boosting")]
rows = []
for a, bb in keep:
    r = p[(p.a == a) & (p.b == bb)].iloc[0]
    rows.append([esc(SHORT.get(a, a)), esc(SHORT.get(bb, bb)), f(r.diff_logloss), f"[{f(r.lo)}, {f(r.hi)}]"])
write("paired", ["Model A", "Model B", "A $-$ B", "95\\% CI"], rows, "llrc")

# 5. recall by mode
r_ = T("recall_by_mode"); r_ = r_[~r_.model.str.contains("reference|prior")]
rows = [[esc(SHORT.get(r.model, r.model))] + [f(r[c]) for c in ["recall_walk", "recall_cycle", "recall_public", "recall_drive"]] for _, r in r_.iterrows()]
write("recall", ["Model", "Walk", "Cycle", "Public transport", "Car"], rows, "lrrrr")

# 6. sample efficiency
s = T("sample_curve"); s = s[s.model.isin(["MNL M4", "Gradient boosting", "Qwen 0.5B fine-tuned"])]
pv = s.pivot(index="n_train", columns="model", values="logloss")
rows = [[f"{n:,}"] + [f(pv.loc[n, c]) if n in pv.index and not np.isnan(pv.loc[n, c]) else "--" for c in ["MNL M4", "Gradient boosting", "Qwen 0.5B fine-tuned"]] for n in pv.index]
write("sample", ["Training trips", "MNL (M4)", "Gradient boosting", "Qwen fine-tuned"], rows, "rrrr")

# 7. group importance
m, g = T("mnl_group_importance"), T("gbm_group_importance")
LABEL = {"person": "Person characteristics", "trip": "Trip variables", "walk_cycle": "Walking and cycling times", "pt": "Public transport service", "drive": "Car service"}
rows = [[LABEL[r.group_dropped], str(int(r.params_removed)), f"{r.loglik_loss:,.0f}", f(g.set_index("group").loc[r.group_dropped, "logloss_increase"])] for _, r in m.iterrows()]
write("groups", ["Variable group", "Parameters", "Log-likelihood lost (MNL)", "Log-loss increase (boosting)"], rows, "lrrr")

# 8. nested logit
n_ = T("nested_comparison"); mnl_ll = T("mnl_ladder").loglik.iloc[-1]; mnl_bic = T("mnl_ladder").BIC.iloc[-1]; mnl_k = int(T("mnl_ladder").k.iloc[-1])
LB = {"active_motor": "\\{walk, cycle\\} and \\{PT, car\\}", "car_vs_rest": "\\{walk, cycle, PT\\} and \\{car\\}"}
rows = [["Multinomial Logit (M4)", str(mnl_k), f"{mnl_ll:,.0f}".replace("-", "$-$"), f"{mnl_bic:,.0f}", "--", "--"]]
for _, r in n_.iterrows():
    mu = ", ".join(f"{k.replace('mu_', '')} {r[k]:.2f}" for k in n_.columns if k.startswith("mu_") and not np.isnan(r[k]))
    rows.append([LB[r.structure], str(int(r.k)), f"{r.loglik:,.0f}".replace("-", "$-$"), f"{r.BIC:,.0f}", f"{r.LR_vs_MNL:.0f} ({int(r.df)})", mu])
write("nested", ["Structure", "Parameters", "Log-likelihood", "BIC", "LR vs MNL (df)", "Nest parameter $\\mu$"], rows, "lrrrrl")

# 9. language-model learning-rate sweep and ablation
q = T("qwen_ft_runs"); q2 = q[q.n_train == 2000].sort_values("lr", ascending=False)
rows = [[f"{r.lr:.0e}".replace("e-0", "e-"), f(r.val_logloss), pct(r.val_accuracy), f"{r.train_minutes:.0f}"] for _, r in q2.iterrows()]
write("lrsweep", ["Learning rate", "Validation log-loss", "Validation accuracy", "Minutes"], rows, "lrrr")
a = T("mnl_ablation")
rows = [[esc(r.variant), str(int(r.params)), f(r.val_logloss, 4), pct(r.val_accuracy)] for _, r in a.iterrows()]
write("ablation", ["MNL variant", "Parameters", "Validation log-loss", "Validation accuracy"], rows, "lrrr")

# 10. value of time (MNL, all trips): delta method and (if available) household-clustered bootstrap; implied VoT; segments
vt = T("mnl_vot"); allv = vt[vt.segment == "all training trips"].set_index("component")
order = ["driving", "PT rail in-vehicle", "PT bus in-vehicle", "PT access/egress", "PT interchange", "walking", "cycling"]
try:
    bv = T("mnl_bootstrap_vot").set_index("component")
    rows = [[esc(c), f"{allv.loc[c, 'vot_gbp_per_hour']:.1f}", f"[{allv.loc[c, 'lo']:.1f}, {allv.loc[c, 'hi']:.1f}]", f"[{bv.loc[c, 'boot_lo']:.1f}, {bv.loc[c, 'boot_hi']:.1f}]"] for c in order]
    write("vot", ["Component", "\\pounds/hour", "Delta-method 95\\% CI", "Household bootstrap 95\\% CI"], rows, "lrcc")
except FileNotFoundError:
    print("bootstrap VoT not available yet")
segs = ["all training trips", "fare-paying travellers only", "commute (purpose 1)", "other purposes (3, 5)", "car in household"]
head = ["All trips", "Fare-paying", "Commuting", "Other purposes", "Car in household"]
rows = [[esc(c)] + [f"{vt[(vt.segment == s) & (vt.component == c)].vot_gbp_per_hour.iloc[0]:.1f}" for s in segs] for c in order]
write("votseg", ["Component"] + head, rows, "lrrrrr")
iv = T("implied_vot"); iv = iv[iv.model != "Qwen 0.5B zero-shot"] if False else iv
rows = []
for mod in ["Multinomial Logit", "Nested Logit", "Gradient boosting", "Qwen 0.5B zero-shot", "Qwen 0.5B fine-tuned"]:
    row = [esc(mod)]
    for mode in ["drive", "public transport"]:
        r = iv[(iv.model == mod) & (iv["mode"] == mode)].iloc[0]
        row += ["--", "0 / 0"] if np.isnan(r.median_gbp_per_hour) else [f"{r.median_gbp_per_hour:.1f} ({r.q25:.1f}--{r.q75:.1f})", f"{int(r.trips_correct_signs)} / {int(r.trips_eligible)}"]
    rows.append(row)
write("implied", ["Model", "Car: median (IQR)", "Right signs / eligible", "PT: median (IQR)", "Right signs / eligible"], rows, "lrrrr")

# 11. elasticities, scenarios, monotonicity (scenario subsample)
MODELS = ["Multinomial Logit", "Nested Logit", "Gradient boosting", "Qwen 0.5B fine-tuned", "Qwen 0.5B zero-shot"]
e = T("elasticities"); e = e[e["sample"] == "subset"]
SC = {"S1 PT fares -20%": "S1 PT fares $-20\\%$", "S2 driving costs +20%": "S2 driving costs $+20\\%$", "S3 driving time +15%": "S3 driving time $+15\\%$",
      "S4 PT in-vehicle time -15%": "S4 PT in-vehicle time $-15\\%$", "S5 package (S1+S2+S4)": "S5 package (S1, S2, S4)"}
rows = []
for sc in list(SC)[:4]:
    rows.append([SC[sc]] + [f"{e[(e.model == m) & (e.scenario == sc)].elasticity.iloc[0]:+.2f}".replace("-", "$-$").replace("+", "$+$") for m in MODELS])
write("elast", ["Scenario"] + ["MNL", "Nested", "Boosting", "Qwen FT", "Qwen ZS"], rows, "lrrrrr")
sh = T("scenario_shifts"); sh = sh[sh["sample"] == "subset"]
cols = MODELS[:4]
rows = []
for sc in SC:
    for mode, lab in [("public", "PT"), ("drive", "Car")]:
        row = [SC[sc] if mode == "public" else "", lab]
        for m in cols:
            r = sh[(sh.model == m) & (sh.scenario == sc) & (sh["mode"] == mode)].iloc[0]
            row.append(f"{r.shift_pp:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]".replace("-", "$-$").replace("+", "+"))
        rows.append(row)
write("scen", ["Scenario", "Mode", "MNL", "Nested", "Boosting", "Qwen FT"], rows, "llrrrr", midrules=(2, 4, 6, 8))
mo = T("monotonicity")
rows = []
for s_ in ["S1", "S2", "S3", "S4"]:
    rows.append([SC[[k for k in SC if k.startswith(s_)][0]]] + [f"{100 * mo[(mo.model == m) & (mo.scenario == s_)].wrong_direction.iloc[0]:.1f}\\%" for m in MODELS])
write("mono", ["Scenario", "MNL", "Nested", "Boosting", "Qwen FT", "Qwen ZS"], rows, "lrrrrr")
cs = T("mnl_cost_sensitivity").iloc[0]
(OUT / "costsens.tex").write_text("\\begin{tabular}{lr}\n\\toprule\nQuantity & Value \\\\\n\\midrule\n"
  f"Cost coefficient, shared & ${cs.b_cost_pooled:.3f}$ \\\\\nCost coefficient, car only (separate) & ${cs.b_cost_drive:.3f}$ \\\\\nCost coefficient, PT fare (separate) & ${cs.b_cost_pt:+.3f}$ \\\\\n"
  f"Likelihood-ratio statistic, separate vs shared (1 df) & {cs.LR_vs_pooled:.0f} \\\\\nCorrelation of PT fare with distance (fare-paying trips) & {cs.corr_fare_distance:.2f} \\\\\n"
  f"Share of trips with zero PT fare & {100 * cs.share_pt_fare_zero:.1f}\\% \\\\\nValue of time, car, shared cost (\\pounds/hour) & {cs.vot_drive_pooled:.1f} \\\\\nValue of time, car, separate cost (\\pounds/hour) & {cs.vot_drive_separate:.1f} \\\\\n\\bottomrule\n\\end{{tabular}}\n")

# 12. appendix: complete MNL coefficient table with robust and bootstrap standard errors
try:
    c = T("mnl_bootstrap_coefficients")
    rows = [[f"\\texttt{{{esc(r['name'])}}}", f(r.value), f(r.robust_se), f(r.boot_se), f"{r.se_ratio:.2f}", "yes" if r.sig_boot else "no"] for _, r in c.iterrows()]
    (OUT / "coefs.tex").write_text("\\begin{longtable}{lrrrrc}\n\\toprule\nCoefficient & Estimate & Robust SE & Bootstrap SE & Ratio & Significant (bootstrap)\\\\\n\\midrule\n\\endhead\n" + "\n".join(" & ".join(r) + r" \\" for r in rows) + "\n\\bottomrule\n\\end{longtable}\n")
except FileNotFoundError:
    print("bootstrap coefficients not available yet")
print("tables written:", sorted(p.name for p in OUT.glob("*.tex")))

# 13. macros for numbers quoted in the text that come from the household bootstrap
try:
    c = T("mnl_bootstrap_coefficients"); bv = T("mnl_bootstrap_vot").set_index("component"); bt = T("mnl_bootstrap_betas")
    M = {"bootN": len(bt), "bootRatioMedian": f"{c.se_ratio.median():.2f}", "bootRatioMin": f"{c.se_ratio.min():.2f}", "bootRatioMax": f"{c.se_ratio.max():.2f}",
         "bootSigRobust": int(c.sig_robust.sum()), "bootSigBoot": int(c.sig_boot.sum()), "nCoef": len(c),
         "bootCarLo": f"{bv.loc['driving', 'boot_lo']:.1f}", "bootCarHi": f"{bv.loc['driving', 'boot_hi']:.1f}",
         "bootCarMedian": f"{bv.loc['driving', 'boot_median']:.1f}"}
    (OUT / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
    print("macros:", M)
except FileNotFoundError:
    print("bootstrap not available: numbers.tex not written")
