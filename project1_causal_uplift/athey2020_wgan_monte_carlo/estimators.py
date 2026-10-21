"""ATT estimators for the Athey et al. (2020) reproduction.
Every estimator takes a DataFrame (columns t, the 8 covariates, re78) and returns Estimate(att, se)."""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

COVARIATES = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
TRIM = 0.95   # paper: drop units with estimated propensity > 0.95


@dataclass
class Estimate:
    att: float
    se: float

    def ci(self, z=1.96):
        return self.att - z * self.se, self.att + z * self.se


def weighted_att(t, r, w=None):
    """Shared form of all our estimators: ATT = mean of r over treated - weighted mean of r over controls.
    SE from each unit's influence-function contribution, treating fitted models (mu0, e) as fixed.
    w=None means no control term (outcome-model-only estimators)."""
    treated = t == 1
    n1 = treated.sum()
    m1 = r[treated].mean()
    contrib = np.where(treated, (r - m1) / n1, 0.0)
    att = m1
    if w is not None:
        ctrl = ~treated
        W = w[ctrl].sum()
        m0 = np.sum(w[ctrl] * r[ctrl]) / W
        att = m1 - m0
        contrib[ctrl] = -w[ctrl] * (r[ctrl] - m0) / W
    return Estimate(float(att), float(np.sqrt(np.sum(contrib ** 2))))


# --- nuisance models (linear family) -------------------------------------------------------------
def mu0_linear(df):
    """Linear regression of re78 on covariates, fit on controls; predictions for everyone."""
    c = df.t == 0
    return LinearRegression().fit(df.loc[c, COVARIATES], df.loc[c, "re78"]).predict(df[COVARIATES])


def e_logit(df):
    """Plain (unpenalised) logit propensity score."""
    return LogisticRegression(C=np.inf, max_iter=10_000).fit(df[COVARIATES], df.t).predict_proba(df[COVARIATES])[:, 1]


# --- estimators -------------------------------------------------------------------------------
def att_naive(df):
    return weighted_att(df.t.values, df.re78.values, np.ones(len(df)))


def att_outcome_linear(df):
    return weighted_att(df.t.values, df.re78.values - mu0_linear(df))


def att_ipw_logit(df):
    e = e_logit(df)
    keep = e <= TRIM
    return weighted_att(df.t.values[keep], df.re78.values[keep], (e / (1 - e))[keep])


def att_dr_linear(df):
    e, r = e_logit(df), df.re78.values - mu0_linear(df)
    keep = e <= TRIM
    return weighted_att(df.t.values[keep], r[keep], (e / (1 - e))[keep])