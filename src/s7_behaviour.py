"""Stage 7: value of time, elasticities, policy scenarios and monotonicity checks.
usage: s7_behaviour.py [--no-llm]   (language-model scoring is cached in results/predictions/scen_*.npy)
All scenarios are applied to TEST trips: the full test set for MNL/NL/GBM, the 1,000-trip scenario subset for every model."""
import json, sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from biogeme.results_processing import get_pandas_estimated_parameters
from common import load_split, derive, gbm_X, RES, SEED, MODES
import choice_models as cm

NO_LLM = "--no-llm" in sys.argv
d = load_split()
tr, te = d[d.split == "train"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
scen_set = d[(d.test_rank >= 0) & (d.test_rank < 1000)].sort_values("test_rank").reset_index(drop=True)

SCEN = {"S1 PT fares -20%": ("S1", -0.20, "pt"), "S2 driving costs +20%": ("S2", 0.20, "drive"),
        "S3 driving time +15%": ("S3", 0.15, "drive"), "S4 PT in-vehicle time -15%": ("S4", -0.15, "pt"),
        "S5 package (S1+S2+S4)": ("S5", None, None)}
def apply(df, s):
    x = df.copy()
    if s in ("S1", "S5"): x["cost_transit"] *= 0.8
    if s in ("S2", "S5"): x["cost_driving_fuel"] *= 1.2; x["cost_driving_ccharge"] *= 1.2
    if s == "S3": x["t_drive"] *= 1.15
    if s in ("S4", "S5"): x["t_pt_rail"] *= 0.85; x["t_pt_bus"] *= 0.85
    return derive(x)
SIDS = ["base", "S1", "S2", "S3", "S4", "S5"]

# ---------------------------------------------------------------- probability functions per model
mnl, nl = json.load(open(RES / "models/mnl.json")), json.load(open(RES / "models/nested.json"))
gj = json.load(open(RES / "models/gbm.json"))
gbm = HGB(max_iter=gj["n_iter"], early_stopping=False, random_state=SEED, **gj["params"]).fit(gbm_X(tr), tr.y)
FUN = {"Multinomial Logit": lambda x, s: cm.predict(cm.utilities(mnl["level"]), mnl["betas"], cm.prep(x)),
       "Nested Logit": lambda x, s: cm.predict(cm.utilities(nl["level"]), nl["betas"], cm.prep(x), nest=cm.nests(nl["structure"])),
       "Gradient boosting": lambda x, s: gbm.predict_proba(gbm_X(x))}
if not NO_LLM:
    import llm_common as L
    best = json.load(open(RES / "models/qwen_ft_best.json"))
    def llm_fun(tag, adapter):
        cache = {}
        def f(x, s):
            p = RES / f"predictions/scen_{tag}_{s}.npy"
            if p.exists(): return L.softmax(np.load(p))
            if "m" not in cache: cache["m"], cache["t"] = L.load(adapter)
            z = L.score(cache["m"], cache["t"], [L.render(r) for r in x.itertuples()], log_every=500); np.save(p, z)
            return L.softmax(z)
        return f
    FUN["Qwen 0.5B zero-shot"] = llm_fun("zeroshot", None)
    FUN["Qwen 0.5B fine-tuned"] = llm_fun("ft", str(L.ADAPTERS / best["tag"]))

# probabilities under every scenario: {model: {sid: n x 4}} for the 1000-trip set; classical also on the full test set
PS = {m: {s: f(apply(scen_set, s) if s != "base" else scen_set, s) for s in SIDS} for m, f in FUN.items()}
PF = {m: {s: FUN[m](apply(te, s) if s != "base" else te, s) for s in SIDS} for m in ["Multinomial Logit", "Nested Logit", "Gradient boosting"]}

# ---------------------------------------------------------------- cluster bootstrap over households
def draws(df, B=500):
    rng = np.random.default_rng(SEED); idx = pd.Series(np.arange(len(df))).groupby(df.household_id.values).apply(np.asarray).values
    return [np.concatenate(idx[rng.integers(0, len(idx), len(idx))]) for _ in range(B)]
DS, DF = draws(scen_set), draws(te, 300)

rows, el = [], []
for label, PP, df, dr, tag in [("scenario subset (1,000 trips)", PS, scen_set, DS, "subset"), ("full test set", PF, te, DF, "full")]:
    for m, ps in PP.items():
        base = ps["base"]
        for name, (s, delta, mode) in SCEN.items():
            D = ps[s] - base
            bm = np.array([D[t].mean(0) for t in dr]) * 100
            for k, mo in enumerate(MODES):
                rows.append(dict(sample=tag, model=m, scenario=name, mode=mo, base_share=base[:, k].mean() * 100,
                                 shift_pp=D[:, k].mean() * 100, lo=np.percentile(bm[:, k], 2.5), hi=np.percentile(bm[:, k], 97.5)))
            if delta is not None:  # arc elasticity of the affected mode's aggregate share
                k = MODES.index({"pt": "public", "drive": "drive"}[mode]) if mode in ("pt", "drive") else 0
                e = [(ps[s][t][:, k].mean() / base[t][:, k].mean() - 1) / delta for t in dr]
                el.append(dict(sample=tag, model=m, scenario=name, mode=MODES[k], elasticity=(ps[s][:, k].mean() / base[:, k].mean() - 1) / delta,
                               lo=np.percentile(e, 2.5), hi=np.percentile(e, 97.5)))
sh = pd.DataFrame(rows); sh.to_csv(RES / "tables/scenario_shifts.csv", index=False)
el = pd.DataFrame(el); el.to_csv(RES / "tables/elasticities.csv", index=False)
print("=== arc elasticities (scenario subset, 95% CI) ===")
print(el[el["sample"] == "subset"].assign(v=lambda x: x.elasticity.round(2).astype(str) + " [" + x.lo.round(2).astype(str) + ", " + x.hi.round(2).astype(str) + "]")
        .pivot(index="scenario", columns="model", values="v").to_string())
print("\n=== mode-share shift in percentage points, package S5 (scenario subset) ===")
print(sh[(sh["sample"] == "subset") & sh.scenario.str.startswith("S5")].assign(v=lambda x: x.shift_pp.round(2).astype(str) + " [" + x.lo.round(2).astype(str) + ", " + x.hi.round(2).astype(str) + "]")
        .pivot(index="mode", columns="model", values="v").to_string())

# ---------------------------------------------------------------- monotonicity: probabilities must move in the expected direction
mono = []
EXPECT = {"S1": (2, +1), "S2": (3, -1), "S3": (3, -1), "S4": (2, +1)}  # scenario -> (own mode index, expected sign of dP)
for m, ps in PS.items():
    for s, (k, sg) in EXPECT.items():
        dP = sg * (ps[s][:, k] - ps["base"][:, k]); mono.append(dict(model=m, scenario=s, wrong_direction=(dP < -1e-3).mean(), flat=(np.abs(dP) <= 1e-3).mean()))
mono = pd.DataFrame(mono); mono.to_csv(RES / "tables/monotonicity.csv", index=False)
print("\n=== share of trips moving in the WRONG direction / not moving at all (scenario subset) ===")
print(mono.pivot(index="scenario", columns="model", values="wrong_direction").round(3).to_string()); print(mono.pivot(index="scenario", columns="model", values="flat").round(3).to_string())

# ---------------------------------------------------------------- implied value of time from probability responses (GBP/hour)
def implied(ps, df):
    b = ps["base"]; out = {}
    # driving: dP_drive per minute of time over dP_drive per pound of cost
    dt = 0.15 * df.t_drive.values; dc = 0.20 * df.cost_drive.values
    a, c = (ps["S3"][:, 3] - b[:, 3]) / dt, (ps["S2"][:, 3] - b[:, 3]) / np.where(dc > 0, dc, np.nan)
    ok = (df.cost_drive.values > 0.3) & (b[:, 3] > .05) & (b[:, 3] < .95)
    good = ok & (a < 0) & (c < 0); out["drive"] = (60 * a[good] / c[good], ok.sum(), good.sum())
    # public transport: in-vehicle time (rail+bus) vs fare
    dt = -0.15 * (df.t_pt_rail + df.t_pt_bus).values; dc = -0.20 * df.cost_transit.values
    a, c = (ps["S4"][:, 2] - b[:, 2]) / np.where(dt != 0, dt, np.nan), (ps["S1"][:, 2] - b[:, 2]) / np.where(dc != 0, dc, np.nan)
    ok = (df.cost_transit.values > 0.3) & ((df.t_pt_rail + df.t_pt_bus).values > 3) & (b[:, 2] > .05) & (b[:, 2] < .95)
    good = ok & (a < 0) & (c < 0); out["public transport"] = (60 * a[good] / c[good], ok.sum(), good.sum())
    return out
iv = []
for m, ps in PS.items():
    for mode, (v, n_ok, n_good) in implied(ps, scen_set).items():
        iv.append(dict(model=m, mode=mode, median_gbp_per_hour=np.median(v) if len(v) else np.nan, q25=np.percentile(v, 25) if len(v) else np.nan,
                       q75=np.percentile(v, 75) if len(v) else np.nan, trips_eligible=n_ok, trips_correct_signs=n_good))
iv = pd.DataFrame(iv); iv.to_csv(RES / "tables/implied_vot.csv", index=False)
print("\n=== implied value of time (median, GBP/hour) from probability responses ==="); print(iv.round(2).to_string(index=False))

# ---------------------------------------------------------------- parametric value of time from the logit coefficients
def vot_table(df, label, level=mnl["level"]):
    x = cm.prep(df); skip = tuple(c for c in cm.PERSON + cm.TRIP if x[c].nunique() < 2 or x[c].sum() < 30)
    r = cm.fit(cm.utilities(level, skip=skip), x, name="vot"); b = r.get_beta_values()
    out = []
    for nm, p in [("walking", "b_t_walk"), ("cycling", "b_t_cycle"), ("PT access/egress", "b_t_pt_access"), ("PT rail in-vehicle", "b_t_pt_rail"),
                  ("PT bus in-vehicle", "b_t_pt_bus"), ("PT interchange", "b_t_pt_int"), ("driving", "b_t_drive")]:
        a, c = b[p], b["b_cost"]; S = r.get_sub_variance_covariance_matrix([p, "b_cost"]); g = np.array([60 / c, -60 * a / c ** 2])
        se = float(np.sqrt(g @ S @ g)); out.append(dict(segment=label, n=len(df), component=nm, vot_gbp_per_hour=60 * a / c, lo=60 * a / c - 1.96 * se, hi=60 * a / c + 1.96 * se))
    return out
segs = [("all training trips", tr), ("fare-paying travellers only", tr[(tr.faretype.isin([1, 2])) & (tr.bus_scale == 1)]),
        ("commute (purpose 1)", tr[tr.purpose == 1]), ("other purposes (3, 5)", tr[tr.purpose.isin([3, 5])]),
        ("no car in household", tr[tr.car_ownership == 0]), ("car in household", tr[tr.car_ownership > 0])]
vt = pd.DataFrame([r for lab, df in segs for r in vot_table(df, lab)]); vt.to_csv(RES / "tables/mnl_vot.csv", index=False)
print("\n=== MNL value of time, GBP/hour (delta-method 95% CI) ===")
print(vt.assign(v=lambda x: x.vot_gbp_per_hour.round(1).astype(str) + " [" + x.lo.round(1).astype(str) + ", " + x.hi.round(1).astype(str) + "]")
        .pivot(index="component", columns="segment", values="v").to_string())
