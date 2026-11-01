"""Biogeme model definitions shared by s1 (MNL), s2 (nested logit) and s7 (behaviour).
One definition of the utilities is used for estimation AND prediction, so they cannot drift apart."""
import os
import numpy as np, pandas as pd
import biogeme.database as db
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, Variable
from biogeme.models import loglogit, logit, lognested, nested
from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit
from common import RES, derive

_TMP = RES / "tmp"; _TMP.mkdir(exist_ok=True)
os.chdir(_TMP)  # Biogeme drops a biogeme.toml in the working directory

PERSON = ["age10", "age10sq", "female", "lic", "car1", "car2", "ft_child", "ft_dis", "ft_free", "bus_free", "bus_half"]
TRIP = ["p2", "p3", "p4", "p5", "ldist", "peak", "weekend"]
LEVELS = {0: "M0 constants", 1: "M1 generic time+cost", 2: "M2 mode-specific times", 3: "M3 +person", 4: "M4 +trip"}
GROUPS = ("person", "trip", "walk_cycle", "pt", "drive")  # for drop-one-group tests


def prep(d: pd.DataFrame) -> pd.DataFrame:
    """Raw frame -> numeric design frame for Biogeme (walk is the reference alternative)."""
    d = derive(d.copy())
    x = pd.DataFrame(index=d.index)
    x["choice"] = d.y + 1
    for c in ["t_walk", "t_cycle", "t_pt", "t_pt_access", "t_pt_rail", "t_pt_bus", "t_pt_int", "pt_interchanges",
              "t_drive", "cost_transit", "cost_drive", "driving_traffic_percent"]:
        x[c] = d[c]
    x["age10"], x["age10sq"] = d.age / 10, (d.age / 10) ** 2
    x["female"], x["lic"] = d.female, d.driving_license
    x["car1"], x["car2"] = (d.car_ownership == 1) * 1, (d.car_ownership == 2) * 1
    x["ft_child"], x["ft_dis"], x["ft_free"] = (d.faretype == 3) * 1, (d.faretype == 4) * 1, (d.faretype == 5) * 1
    x["bus_free"], x["bus_half"] = (d.bus_scale == 0) * 1, (d.bus_scale == 0.5) * 1
    for k in range(2, 6):
        x[f"p{k}"] = (d.purpose == k) * 1
    x["ldist"] = np.log(d.distance / 1000)
    st = d.start_time
    x["peak"] = (((st >= 7) & (st < 10)) | ((st >= 16) & (st < 19))) * 1
    x["weekend"] = (d.day_of_week >= 6) * 1
    return x.astype(float)


def utilities(level: int, drop=(), skip=()) -> dict:
    """Utility of each alternative (1 walk, 2 cycle, 3 public transport, 4 drive). `drop` removes a variable group."""
    B = lambda n: Beta(n, 0, None, None, 0)
    V = Variable
    t = {1: [], 2: [B("asc_cycle")], 3: [B("asc_pt")], 4: [B("asc_drive")]}
    if level == 1:
        bt, bc = B("b_time"), B("b_cost")
        t[1].append(bt * V("t_walk")); t[2].append(bt * V("t_cycle"))
        t[3] += [bt * V("t_pt"), bc * V("cost_transit")]; t[4] += [bt * V("t_drive"), bc * V("cost_drive")]
    if level >= 2:
        bc = B("b_cost")
        if "walk_cycle" not in drop:
            t[1].append(B("b_t_walk") * V("t_walk")); t[2].append(B("b_t_cycle") * V("t_cycle"))
        if "pt" not in drop:
            t[3] += [B("b_t_pt_access") * V("t_pt_access"), B("b_t_pt_rail") * V("t_pt_rail"),
                     B("b_t_pt_bus") * V("t_pt_bus"), B("b_t_pt_int") * V("t_pt_int"),
                     B("b_interch") * V("pt_interchanges"), bc * V("cost_transit")]
        if "drive" not in drop:
            t[4] += [B("b_t_drive") * V("t_drive"), bc * V("cost_drive"),
                     B("b_traffic") * V("driving_traffic_percent")]
    shifters = [v for v in (PERSON if level >= 3 and "person" not in drop else []) +
                (TRIP if level >= 4 and "trip" not in drop else []) if v not in skip]  # skip: constant columns in a subsample
    for j, m in [(2, "cycle"), (3, "pt"), (4, "drive")]:
        t[j] += [B(f"{v}_{m}") * V(v) for v in shifters]
    return {j: sum(terms[1:], terms[0]) if terms else 0 for j, terms in t.items()}


def nests(kind: str):
    """Nested-logit structures. mu >= 1 (Biogeme scale convention); mu = 1 collapses the nest to MNL."""
    mu = lambda n: Beta(n, 1, 1, None, 0)
    if kind == "active_motor":
        ns = (OneNestForNestedLogit(mu("mu_active"), [1, 2], "active"), OneNestForNestedLogit(mu("mu_motor"), [3, 4], "motor"))
    elif kind == "car_vs_rest":
        ns = (OneNestForNestedLogit(mu("mu_noncar"), [1, 2, 3], "noncar"),)
    else:
        raise ValueError(kind)
    return NestsForNestedLogit(choice_set=[1, 2, 3, 4], tuple_of_nests=ns)


def fit(U, x: pd.DataFrame, nest=None, name="model", start: dict | None = None):
    """Estimate by maximum likelihood. `start` = optional starting values (used to warm-start bootstrap refits)."""
    lp = lognested(U, None, nest, Variable("choice")) if nest else loglogit(U, None, Variable("choice"))
    m = BIOGEME(db.Database(name, x), lp, generate_html=False, generate_yaml=False, save_iterations=False)
    m.model_name = name
    if start:
        m.change_init_values(start)
    return m.estimate()


def predict(U, betas: dict, x: pd.DataFrame, nest=None) -> np.ndarray:
    """n x 4 choice probabilities at the given parameter values."""
    P = {f"P{k}": (nested(U, None, nest, k) if nest else logit(U, None, k)) for k in (1, 2, 3, 4)}
    x = x.copy(); x["choice"] = 1  # unused when simulating probabilities
    out = BIOGEME(db.Database("sim", x), P, generate_html=False, generate_yaml=False, save_iterations=False).simulate(betas)
    return out[["P1", "P2", "P3", "P4"]].to_numpy()
