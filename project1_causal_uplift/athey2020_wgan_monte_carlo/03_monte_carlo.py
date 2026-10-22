# %% [markdown]
# Athey, Imbens, Metzger & Munro (2020) — Sections 4.3-4.5: Monte Carlo comparison (Tables 6-8), small scale.
# Truth = ATT in each WGAN population built from the authors' released generators (02_wgan_population.py).
# These generators differ from the paper's (Table 2 check), so compare conclusions, not exact numbers.

# %% Cell 1: populations and true ATTs
import importlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))
import estimators as est
importlib.reload(est)
warnings.filterwarnings("ignore")                     # NN/convergence warnings are known and intended

DATA_DIR, RESULTS_DIR = HERE / "data", HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DESIGNS = {"E": "experimental", "CPS": "cps", "PSID": "psid"}
SAMPLE_SIZES = {"E": (185, 260), "CPS": (185, 15_992), "PSID": (185, 2_490)}   # (treated, controls), as in real data

pops, truth, idx = {}, {}, {}
for d, name in DESIGNS.items():
    pops[d] = pd.read_parquet(DATA_DIR / f"population_{name}.parquet").astype("float64")   # generators output float32
    t = pops[d].t.values
    idx[d] = (np.flatnonzero(t == 1), np.flatnonzero(t == 0))
    treated = pops[d][t == 1]
    truth[d] = (treated.y1 - treated.y0).mean()
    print(f"{d:5s} population={len(pops[d]):,}  true ATT={truth[d]:.3f}")

print("dtypes:", pops["E"].dtypes.value_counts().to_dict())   # should show only float64


# %% Cell 2: Monte Carlo machinery
def draw_sample(d, rng):
    """One simulated dataset: N1 treated + N0 controls drawn without replacement from the population."""
    (i1, i0), (n1, n0) = idx[d], SAMPLE_SIZES[d]
    rows = np.concatenate([rng.choice(i1, n1, replace=False), rng.choice(i0, n0, replace=False)])
    return pops[d].iloc[rows].reset_index(drop=True)


def run_mc(d, estimators, n_reps, seed=0):
    """Each estimator returns either an Estimate, or a dict of Estimates (a family fitted once)."""
    rng = np.random.default_rng(seed)
    out = []
    for rep in range(n_reps):
        sample = draw_sample(d, rng)
        for name, f in estimators.items():
            res = f(sample)
            items = res.items() if isinstance(res, dict) else [(None, res)]
            for sub, r in items:
                lo, hi = r.ci()
                out.append(dict(design=d, rep=rep, estimator=name if sub is None else f"{name}: {sub}",
                                att=r.att, se=r.se, covers=lo <= truth[d] <= hi))
    return pd.DataFrame(out)


def summarise(res):
    res = res.assign(err=res.att - res.design.map(truth))
    g = res.groupby(["design", "estimator"], sort=False)
    return pd.DataFrame({"rmse": g.err.apply(lambda e: np.sqrt(np.mean(e ** 2))), "bias": g.err.mean(),
                         "sdev": g.att.std(), "mean se": g.se.mean(), "coverage": g.covers.mean()}).round(2)


FAST = {"DIFF": est.att_naive, "BCM": est.att_bcm, "Outcome: linear": est.att_outcome_linear,
        "Propensity: logit": est.att_ipw_logit, "DR: linear": est.att_dr_linear,
        "Residual balancing": est.att_residual_balancing}

print("new run_mc" if "isinstance" in run_mc.__code__.co_names else "OLD run_mc still loaded")


# %% Cell 3: smoke test — 5 replications per design, to check it runs and time it
for d in DESIGNS:
    t0 = time.time()
    res = run_mc(d, FAST, n_reps=5)
    print(f"{d:5s} 5 reps in {time.time() - t0:5.1f}s  ->  ~{(time.time() - t0) / 5 * 100 / 60:4.1f} min for 100")
print(summarise(res))


# %% Cell 4: full Monte Carlo, fast estimators — ours vs paper Tables 6-8
N_REPS = 500                                  # paper: 2,000

PAPER_MC = {   # (rmse, bias, coverage) from paper Tables 6 (E), 7 (CPS), 8 (PSID)
    ("E", "DIFF"): (0.49, 0.06, 0.94), ("E", "BCM"): (0.58, 0.00, 0.96), ("E", "Outcome: linear"): (0.52, -0.06, 0.88),
    ("E", "Propensity: logit"): (0.52, -0.08, 0.99), ("E", "DR: linear"): (0.51, -0.08, 0.95), ("E", "Residual balancing"): (0.52, -0.09, 0.95),
    ("CPS", "DIFF"): (11.12, -11.11, 0.00), ("CPS", "BCM"): (0.73, 0.07, 0.96), ("CPS", "Outcome: linear"): (2.14, -2.08, 0.02),
    ("CPS", "Propensity: logit"): (0.51, 0.00, 0.98), ("CPS", "DR: linear"): (0.53, 0.03, 0.96), ("CPS", "Residual balancing"): (0.57, -0.22, 0.89),
    ("PSID", "DIFF"): (18.81, -18.81, 0.00), ("PSID", "BCM"): (0.98, -0.02, 0.98), ("PSID", "Outcome: linear"): (1.95, -1.82, 0.12),
    ("PSID", "Propensity: logit"): (1.11, -0.64, 0.96), ("PSID", "DR: linear"): (0.98, -0.35, 0.92), ("PSID", "Residual balancing"): (1.06, 0.33, 0.75),
}

runs = []
for d in DESIGNS:
    t0 = time.time()
    runs.append(run_mc(d, FAST, N_REPS, seed=1))
    print(f"{d:5s} {N_REPS} reps in {(time.time() - t0) / 60:4.1f} min", flush=True)
fast_res = pd.concat(runs, ignore_index=True)
fast_res.to_csv(RESULTS_DIR / "mc_fast_raw.csv", index=False)         # raw draws: re-summarise any time

# %% Cell 4b: summarise the saved run and compare with the paper
summary = summarise(fast_res)
paper = pd.DataFrame(PAPER_MC, index=["paper rmse", "paper bias", "paper cov"]).T
paper.index.names = ["design", "estimator"]                 # match the summary's index names so join can align rows
summary = summary.join(paper)
summary.to_csv(RESULTS_DIR / "mc_fast_summary.csv")
pd.set_option("display.width", 200)
print(summary)

# %% Cell 5: flexible estimators — time ONE replication per design, to plan the full run
FLEX = {   # one split per replication: each replication is already a fresh dataset
    "RF": lambda s: est.flexible_family(s, "rf", n_repeats=1),
    "NN": lambda s: est.flexible_family(s, "nn", n_repeats=1),
    "Causal forest": lambda s: est.att_causal_forest(s, n_repeats=1),
}

timing = {}
for d in DESIGNS:
    s = draw_sample(d, np.random.default_rng(99))
    for name, f in FLEX.items():
        t0 = time.time()
        f(s)
        timing[(d, name)] = time.time() - t0
        print(f"{d:5s} {name:14s} {timing[(d, name)]:6.1f}s per replication", flush=True)

per_rep = pd.Series(timing).groupby(level=0).sum()
print("\nprojected runtime for the whole flexible set (minutes):")
for R in [50, 100, 200]:
    print(f"  R={R:3d}:  " + "  ".join(f"{d}={per_rep[d] * R / 60:5.1f}" for d in DESIGNS)
          + f"   total={per_rep.sum() * R / 60:5.1f}")

# %% Cell 6: full Monte Carlo, flexible estimators — ours vs paper Tables 6-8
N_REPS_FLEX = 200

PAPER_MC_FLEX = {   # (rmse, bias, coverage) from paper Tables 6 (E), 7 (CPS), 8 (PSID)
    ("E", "RF: outcome"): (0.51, -0.07, 0.88), ("E", "NN: outcome"): (1.32, 0.04, 0.75),
    ("E", "RF: ipw"): (0.52, -0.06, 0.99), ("E", "NN: ipw"): (0.52, 0.01, 0.99),
    ("E", "RF: dr"): (0.52, -0.04, 0.95), ("E", "NN: dr"): (0.79, -0.05, 0.95), ("E", "Causal forest"): (0.50, -0.09, 0.94),
    ("CPS", "RF: outcome"): (1.00, -0.87, 0.54), ("CPS", "NN: outcome"): (0.63, 0.14, 0.88),
    ("CPS", "RF: ipw"): (1.00, -0.87, 0.73), ("CPS", "NN: ipw"): (0.65, 0.23, 0.94),
    ("CPS", "RF: dr"): (0.54, -0.05, 0.93), ("CPS", "NN: dr"): (0.62, 0.20, 0.94), ("CPS", "Causal forest"): (0.55, 0.11, 0.91),
    ("PSID", "RF: outcome"): (2.30, -2.22, 0.02), ("PSID", "NN: outcome"): (2.97, -0.93, 0.59),
    ("PSID", "RF: ipw"): (2.21, -2.05, 0.32), ("PSID", "NN: ipw"): (1.82, -1.43, 0.69),
    ("PSID", "RF: dr"): (0.98, -0.57, 0.84), ("PSID", "NN: dr"): (0.98, -0.38, 0.92), ("PSID", "Causal forest"): (1.13, -0.89, 0.73),
}

flex_runs = []
for d in DESIGNS:
    t0 = time.time()
    r = run_mc(d, FLEX, N_REPS_FLEX, seed=1)              # seed=1: the SAME samples the fast estimators saw (first 200)
    r.to_csv(RESULTS_DIR / f"mc_flex_raw_{d}.csv", index=False)   # saved per design, in case the run is interrupted
    flex_runs.append(r)
    print(f"{d:5s} {N_REPS_FLEX} reps in {(time.time() - t0) / 60:5.1f} min", flush=True)
flex_res = pd.concat(flex_runs, ignore_index=True)

flex_summary = summarise(flex_res)
paper_flex = pd.DataFrame(PAPER_MC_FLEX, index=["paper rmse", "paper bias", "paper cov"]).T
paper_flex.index.names = ["design", "estimator"]
flex_summary = flex_summary.join(paper_flex)
flex_summary.to_csv(RESULTS_DIR / "mc_flex_summary.csv")
print(flex_summary)

# %% Cell 7: combined reproduction of Tables 6-8 — all estimators on the SAME 200 simulated datasets
fast_summary_200 = summarise(fast_res[fast_res.rep < 200]).join(paper)      # first 200 = the flexible set's samples
combined = pd.concat([fast_summary_200, flex_summary]).loc[list(DESIGNS)]     # keep E, CPS, PSID order
combined.insert(0, "rank", combined.groupby(level="design").rmse.rank(method="min").astype(int))
combined.to_csv(RESULTS_DIR / "mc_combined_summary.csv")
pd.set_option("display.max_rows", None)
print(combined)

print("\nBest 3 by RMSE in each design:")
for d in DESIGNS:
    top = combined.loc[d].sort_values("rmse").head(3)
    print(f"  {d:5s} " + ", ".join(f"{name} ({r.rmse:.2f})" for name, r in top.iterrows()))

# %%
