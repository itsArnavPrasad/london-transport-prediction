"""Shared loading, splits, features and metrics. Every stage imports from here."""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC, RES = ROOT / "data/raw/lpmc.dat", ROOT / "data/processed", ROOT / "results"
SEED = 42
MODES = ["walk", "cycle", "public", "drive"]  # travel_mode 1..4 -> index 0..3
PURPOSES = {1: "commute to work", 2: "travel to school or university", 3: "a home-based trip (shopping, leisure, errands)",
            4: "a business trip for their employer", 5: "a non-home-based trip"}
CARS = {0: "no car in the household", 1: "fewer than one car per adult in the household",
        2: "at least one car per adult in the household"}
FARES = {1: "full fare", 2: "full fare (age 16+)", 3: "child fare", 4: "disabled concession", 5: "free travel pass"}

# feature groups, used for modelling and for the importance analysis
GROUPS = {
    "person": ["age", "female", "driving_license", "car_ownership", "faretype", "bus_scale"],
    "trip": ["purpose", "distance", "start_time", "day_of_week"],
    "walk_cycle": ["t_walk", "t_cycle"],
    "pt": ["t_pt_access", "t_pt_rail", "t_pt_bus", "t_pt_int", "t_pt", "pt_interchanges", "cost_transit"],
    "drive": ["t_drive", "cost_driving_fuel", "cost_driving_ccharge", "cost_drive", "driving_traffic_percent"],
}
FEATURES = [c for g in GROUPS.values() for c in g]


def derive(d: pd.DataFrame) -> pd.DataFrame:
    """Recompute the totals from their components (call after perturbing a component in a scenario)."""
    d["t_pt"] = d[["t_pt_access", "t_pt_rail", "t_pt_bus", "t_pt_int"]].sum(axis=1)
    d["cost_drive"] = d.cost_driving_fuel + d.cost_driving_ccharge
    return d


def load() -> pd.DataFrame:
    d = pd.read_csv(RAW, sep="\t")
    for new, old in [("t_walk", "dur_walking"), ("t_cycle", "dur_cycling"), ("t_pt_access", "dur_pt_access"),
                     ("t_pt_rail", "dur_pt_rail"), ("t_pt_bus", "dur_pt_bus"), ("t_pt_int", "dur_pt_int"),
                     ("t_drive", "dur_driving")]:
        d[new] = d[old] * 60  # hours -> minutes
    d["y"] = d.travel_mode - 1
    return derive(d)


def gbm_X(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for gradient boosting (purpose and faretype are unordered codes -> categorical)."""
    x = df[FEATURES].copy()
    for c in ("purpose", "faretype"):
        x[c] = pd.Categorical(x[c], categories=range(1, 6))
    return x


def load_split() -> pd.DataFrame:
    """Raw data joined with splits.csv (written by s0_audit.py)."""
    return load().merge(pd.read_csv(PROC / "splits.csv"), on="trip_id")


# ---------------------------------------------------------------- metrics
def _ece(y, P, bins=10):
    conf, hit = P.max(1), P.argmax(1) == y
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return e


def metrics(y, P) -> dict:
    y, P = np.asarray(y), np.clip(np.asarray(P, float), 1e-9, 1)
    P = P / P.sum(1, keepdims=True)
    n, Y = len(y), np.eye(4)[y]
    pred = P.argmax(1)
    f1 = []
    for k in range(4):
        tp = ((pred == k) & (y == k)).sum()
        p, r = tp / max((pred == k).sum(), 1), tp / max((y == k).sum(), 1)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {"logloss": -np.log(P[np.arange(n), y]).mean(), "accuracy": (pred == y).mean(),
            "macro_f1": np.mean(f1), "brier": ((P - Y) ** 2).sum(1).mean(), "ece": _ece(y, P),
            "share_err": np.abs(P.mean(0) - Y.mean(0)).mean()}


def cluster_boot(y, P, groups, fn=metrics, B=1000, seed=SEED) -> pd.DataFrame:
    """Household-cluster bootstrap: point estimate and 95% CI for every metric."""
    y, P, groups = np.asarray(y), np.asarray(P), np.asarray(groups)
    rng = np.random.default_rng(seed)
    idx = pd.Series(np.arange(len(y))).groupby(groups).apply(np.asarray).values
    reps = []
    for _ in range(B):
        take = np.concatenate(idx[rng.integers(0, len(idx), len(idx))])
        reps.append(fn(y[take], P[take]))
    reps = pd.DataFrame(reps)
    out = pd.DataFrame({"est": pd.Series(fn(y, P)), "lo": reps.quantile(.025), "hi": reps.quantile(.975)})
    return out
