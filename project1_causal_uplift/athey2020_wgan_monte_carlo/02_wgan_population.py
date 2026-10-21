# %% [markdown]
# Athey, Imbens, Metzger & Munro (2020) — Section 3.3: WGAN population and true ATT
# Uses the authors' pretrained generators (github.com/gsbDBI/ds-wgan/trained_models) and the variable
# setup from their Colab tutorial. Generators work in DOLLARS; we convert to $1,000s only for comparisons.

# %% Cell 1: load the authors' pretrained generators
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import wgan

DATA_DIR = Path(__file__).parent / "data" if "__file__" in globals() else Path("data")
MODEL_DIR = DATA_DIR / "trained_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPO = "https://raw.githubusercontent.com/gsbDBI/ds-wgan/master/"
DESIGNS = {"E": "experimental", "CPS": "cps", "PSID": "psid"}
MODEL_TAG = {"E": "exp", "CPS": "cps", "PSID": "psid"}

# Variable setup exactly as in the authors' Colab tutorial (order matters for the pretrained weights)
CONT_0, CAT_0, CTX_0, LB_0 = ["age", "education", "re74", "re75"], ["black", "hispanic", "married", "nodegree"], ["t"], {"re74": 0, "re75": 0}
CONT_1, CAT_1, CTX_1, LB_1 = ["re78"], [], ["t", "age", "education", "re74", "re75", "black", "hispanic", "married", "nodegree"], {"re78": 0}


def fetch(rel_path, dest):
    if not dest.exists():
        r = requests.get(REPO + rel_path, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


def load_original(design):
    """Original LDW data in DOLLARS, without u74/u75 — what the generators were trained on."""
    path = fetch(f"data/original_data/{DESIGNS[design]}.feather", DATA_DIR / f"ldw_{DESIGNS[design]}.feather")
    return pd.read_feather(path).drop(columns=["u74", "u75"])


def load_generators(design, seed=0):
    """Rebuild the authors' DataWrappers on a balanced resample (as in their tutorial) and load the weights."""
    df = load_original(design)
    p = df.t.mean()
    df_bal = df.sample(2 * len(df), weights=(1 - p) * df.t + p * (1 - df.t), replace=True, random_state=seed)
    wrappers = [wgan.DataWrapper(df_bal, CONT_0, CAT_0, CTX_0, LB_0),
                wgan.DataWrapper(df_bal, CONT_1, CAT_1, CTX_1, LB_1)]
    generators = []
    for k, dw in enumerate(wrappers):
        g = wgan.Generator(wgan.Specifications(dw, device="cpu", print_every=10**9))
        path = fetch(f"trained_models/G_{k}_{MODEL_TAG[design]}.pth", MODEL_DIR / f"G_{k}_{MODEL_TAG[design]}.pth")
        g.load_state_dict(torch.load(path, map_location="cpu"))
        generators.append(g)              # left in training mode: dropout on, as in the authors' generation
    return df, wrappers, generators


models = {d: load_generators(d) for d in DESIGNS}
print("\nloaded:", list(models))

# %% Cell 2: generate the 1M-unit population, check against paper Table 2, compute the true ATT
N_POP = 1_000_000


def generate_population(design, n=N_POP, seed=0):
    df, (dw_x, dw_y), (g_x, g_y) = models[design]
    torch.manual_seed(seed)
    with torch.no_grad():                                                  # no gradients needed: saves memory
        pop = dw_x.apply_generator(g_x, df.sample(n, replace=True, random_state=seed))   # X | t
        pop = dw_y.apply_generator(g_y, pop)                                # observed Y(t) | X, t
        flipped = pop.assign(t=1 - pop.t)
        y_cf = dw_y.apply_generator(g_y, flipped)["re78"].values            # counterfactual Y(1-t)
    pop["y1"] = np.where(pop.t == 1, pop.re78, y_cf)
    pop["y0"] = np.where(pop.t == 1, y_cf, pop.re78)
    pop[["re74", "re75", "re78", "y1", "y0"]] /= 1000                      # $1,000s, as in the paper's tables
    return pop


populations = {}
for d in DESIGNS:
    print(f"generating {d} ...", flush=True)
    populations[d] = generate_population(d)
    populations[d].to_parquet(DATA_DIR / f"population_{DESIGNS[d]}.parquet", index=False)   # reused by the Monte Carlo

# Table 2: means of the generated data (ours) vs the paper's generated data
TABLE2 = pd.DataFrame({   # black, hispanic, age, married, nodegree, education, re74, re75, re78
    "trainees (E)":   [0.92, 0.03, 25.61, 0.15, 0.69, 10.35,  3.07,  2.06,  5.31],
    "controls (E)":   [0.91, 0.03, 25.54, 0.10, 0.81, 10.02,  2.13,  1.29,  3.97],
    "controls (CPS)": [0.08, 0.07, 32.82, 0.72, 0.29, 12.03, 14.07, 13.73, 14.89],
    "controls (PSID)":[0.26, 0.03, 34.95, 0.87, 0.30, 12.32, 19.86, 20.41, 22.62],
}, index=["black", "hispanic", "age", "married", "nodegree", "education", "re74", "re75", "re78"])

ours = pd.DataFrame({
    "trainees (E)":    populations["E"].query("t == 1")[TABLE2.index].mean(),
    "controls (E)":    populations["E"].query("t == 0")[TABLE2.index].mean(),
    "controls (CPS)":  populations["CPS"].query("t == 0")[TABLE2.index].mean(),
    "controls (PSID)": populations["PSID"].query("t == 0")[TABLE2.index].mean(),
})
print("\nTable 2 check — ours / paper:")
print(ours.round(2).astype(str) + " / " + TABLE2.astype(str))

print("\nTrue ATT in each generated population ($1,000s):")
for d, pop in populations.items():
    tr = pop[pop.t == 1]
    print(f"  {d:5s} ATT = {(tr.y1 - tr.y0).mean():5.2f}   (n treated = {len(tr):,})")

# %% Diagnostic: is the Table 2 gap from dropout or the balanced-resample scaling? (E trainees, 200k draws)
def trainee_means(seed_bal, dropout_on, n=200_000):
    df, (dw_x, _), (g_x, _) = load_generators("E", seed=seed_bal)
    g_x.train(dropout_on)                                  # train(True) = dropout on; train(False) = off
    torch.manual_seed(0)
    with torch.no_grad():
        x = dw_x.apply_generator(g_x, df[df.t == 1].sample(n, replace=True, random_state=0))
    return x[["black", "hispanic", "married", "re74", "re75"]].mean().round(2).tolist()


print("paper Table 2:                  [0.92, 0.03, 0.15, 3.07, 2.06]  (re74/re75 in $1,000s)")
for seed_bal in [0, 1]:
    for drop in [True, False]:
        m = trainee_means(seed_bal, drop)
        m[3], m[4] = round(m[3] / 1000, 2), round(m[4] / 1000, 2)
        print(f"balance seed={seed_bal} dropout={'on ' if drop else 'off'}:  {m}")

# the pretrained generators released in the authors’ repo are not the ones that produced the paper’s 
# Table 2. On the experimental data they actually fit the real data better. 
# We can reproduce the paper’s method with the authors’ own released simulator, 
# but not its exact simulated world, so Tables 6–8 won’t match number for number. 
# We should compare qualitative conclusions instead: which estimator families are biased, 
# and whose confidence intervals fail. That’s still a legitimate and interesting reproduction.
