# %% [markdown]
# Athey, Imbens, Metzger & Munro (2020) — reproducing Table 5
# ATT estimators on the real Lalonde-Dehejia-Wahba data.
# Units: thousands of 1978 dollars, as in the paper.

# %% Cell 1: load the authors' data
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data" if "__file__" in globals() else Path("data")
DATA_DIR.mkdir(exist_ok=True)

URL = "https://raw.githubusercontent.com/gsbDBI/ds-wgan/master/data/original_data/{}.feather"
DESIGNS = {"E": "experimental", "CPS": "cps", "PSID": "psid"}

# The paper's eight pre-treatment variables (Section 3.2). u74/u75 are in the file but NOT used.
COVARIATES = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]


def load_ldw(design):
    """Download once, cache locally; return t, the 8 covariates and re78, earnings in $1,000s."""
    path = DATA_DIR / f"ldw_{DESIGNS[design]}.feather"
    if not path.exists():
        r = requests.get(URL.format(DESIGNS[design]), timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    df = pd.read_feather(path)
    df[["re74", "re75", "re78"]] /= 1000
    df["t"] = df["t"].astype(int)
    return df[["t", *COVARIATES, "re78"]]


data = {d: load_ldw(d) for d in DESIGNS}
for d, df in data.items():
    print(f"{d:5s} treated={(df.t == 1).sum():3d}  controls={(df.t == 0).sum():6,d}")

ESTIMATORS = {}   # name -> function(df) returning the ATT; filled in by the cells below

# %% Cell 2: covariate balance — reproduces paper Table 1 (means, s.d. in parentheses)
groups = {
    "Trainees (185)":       data["E"].query("t == 1"),
    "Exp. controls (260)":  data["E"].query("t == 0"),
    "CPS controls (15,992)": data["CPS"].query("t == 0"),
    "PSID controls (2,490)": data["PSID"].query("t == 0"),
}
variables = ["black", "hispanic", "age", "married", "nodegree", "education", "re74", "re75", "re78"]

table1 = pd.DataFrame({
    name: [f"{g[v].mean():.2f} ({g[v].std():.2f})" for v in variables]
    for name, g in groups.items()
}, index=variables)
print(table1.to_string())

# %% Cell 3: linear outcome model — tau_cm,lm (paper Section 4.1)
from sklearn.linear_model import LinearRegression


def att_outcome_linear(df):
    """Fit mu(0,x) on CONTROLS only; average (Y - mu_hat(0,X)) over the TREATED."""
    treated, controls = df[df.t == 1], df[df.t == 0]
    mu0 = LinearRegression().fit(controls[COVARIATES], controls.re78)
    return np.mean(treated.re78 - mu0.predict(treated[COVARIATES]))


def att_outcome_linear(df):
    """Fit mu(0,x) on CONTROLS only; average (Y - mu_hat(0,X)) over the TREATED."""
    treated, controls = df[df.t == 1], df[df.t == 0]
    mu0 = LinearRegression().fit(controls[COVARIATES], controls.re78)
    return np.mean(treated.re78 - mu0.predict(treated[COVARIATES]))


ESTIMATORS["Outcome: linear"] = att_outcome_linear


# %% Cell 3b: naive difference in means — DIFF baseline (paper Table 5, first row)
def att_naive(df):
    """Average earnings of treated minus average earnings of controls. No adjustment."""
    return df.loc[df.t == 1, "re78"].mean() - df.loc[df.t == 0, "re78"].mean()


ESTIMATORS["DIFF"] = att_naive


# %% Cell 4: propensity score weighting — tau_ht,lm (paper Section 4.1), trimmed at e > 0.95
from sklearn.linear_model import LogisticRegression

TRIM = 0.95


def att_ipw_logit(df):
    """Logit propensity score; reweight controls by e/(1-e) so they resemble the treated."""
    logit = LogisticRegression(C=np.inf, max_iter=10_000)          # C=inf means no regularisation (plain logit)
    e = logit.fit(df[COVARIATES], df.t).predict_proba(df[COVARIATES])[:, 1]

    keep = e <= TRIM                                               # paper: drop units with e > 0.95
    t, y, e = df.t.values[keep], df.re78.values[keep], e[keep]

    w = e / (1 - e)                                                # odds weights for controls
    treated_mean = y[t == 1].mean()
    control_mean = np.sum(w[t == 0] * y[t == 0]) / np.sum(w[t == 0])
    return treated_mean - control_mean


ESTIMATORS["Propensity: logit"] = att_ipw_logit

# Quick sanity check while developing (the full comparison comes in the final cell)
print({d: round(att_ipw_logit(df), 2) for d, df in data.items()})

   # Reproduction note: E (1.81) and CPS (1.19 vs 1.18) match Table 5. PSID gives 1.75 vs the paper's 1.26.
   # The gap persists across sklearn/statsmodels fits, trimming rules (max e = 0.935, so trimming never binds),
   # default L2 penalty in $ or $1,000s, and normalised/unnormalised weights. The authors' estimation code is not
   # public, so this is recorded as an unexplained discrepancy rather than tuned away.

# %% Cell 5: doubly robust — tau_dr,lm (paper Section 4.1): linear outcome model + logit propensity
def att_dr_linear(df):
    """Outcome-model estimate, corrected by propensity-weighted control residuals."""
    X, t, y = df[COVARIATES], df.t.values, df.re78.values

    # Outcome model: mu_hat(0,x) fit on controls (as in Cell 3)
    mu0 = LinearRegression().fit(X[t == 0], y[t == 0]).predict(X)
    resid = y - mu0                                                  # actual minus predicted no-training earnings

    # Propensity model: plain logit (as in Cell 4), with the paper's trimming rule
    e = LogisticRegression(C=np.inf, max_iter=10_000).fit(X, t).predict_proba(X)[:, 1]
    keep = e <= TRIM
    t, resid, e = t[keep], resid[keep], e[keep]

    w = e / (1 - e)
    treated_part = resid[t == 1].mean()                              # step 1: the outcome-model estimate
    control_correction = np.sum(w[t == 0] * resid[t == 0]) / np.sum(w[t == 0])   # steps 2-3
    return treated_part - control_correction                         # step 4


ESTIMATORS["DR: linear + logit"] = att_dr_linear

print({d: round(att_dr_linear(df), 2) for d, df in data.items()})   # quick check

# Reproduction note: E (1.80) and CPS (1.28 vs 1.27) match Table 5; PSID gives 2.05 vs 1.50. The gap appears
# only in propensity-based estimators (IPW +0.49, DR +0.55; outcome model 0.00), pointing to the paper's PSID
# propensity score as the source. See Cell 4 note.
