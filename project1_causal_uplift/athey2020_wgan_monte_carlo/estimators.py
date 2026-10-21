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

# --- flexible family: random forests and neural nets, cross-fitted ---------------------------------
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_FOLDS = 2   # with R-style forests this reproduced the paper's RF numbers (see sensitivity cell in 01)
RF_REG = dict(n_estimators=500, min_samples_leaf=5, max_features=1 / 3)      # R randomForest regression defaults
RF_CLF = dict(n_estimators=500, min_samples_leaf=1, max_features="sqrt")     # R randomForest classification defaults
NN_REG = dict(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=2000, early_stopping=True)
NN_CLF = dict(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=500)         # no early stopping: accuracy-based, fails with rare treatment

LEARNERS = {   # name -> (outcome-model maker, propensity-model maker), each taking a seed
    "rf": (lambda s: RandomForestRegressor(**RF_REG, random_state=s, n_jobs=-1),
           lambda s: RandomForestClassifier(**RF_CLF, random_state=s, n_jobs=-1)),
    "nn": (lambda s: make_pipeline(StandardScaler(), MLPRegressor(**NN_REG, random_state=s)),
           lambda s: make_pipeline(StandardScaler(), MLPClassifier(**NN_CLF, random_state=s))),
}


def crossfit(df, learner, seed):
    """Out-of-fold mu0_hat(x) and e_hat(x) for every unit, both from the SAME fold split."""
    X, t, y = df[COVARIATES], df.t.values, df.re78.values
    make_reg, make_clf = LEARNERS[learner]
    mu0, e = np.empty(len(df)), np.empty(len(df))
    for train, test in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, t):
        ctrl = train[t[train] == 0]
        mu0[test] = make_reg(seed).fit(X.iloc[ctrl], y[ctrl]).predict(X.iloc[test])
        e[test] = make_clf(seed).fit(X.iloc[train], t[train]).predict_proba(X.iloc[test])[:, 1]
    return mu0, e


def flexible_family_once(df, learner, seed):
    """One split: outcome-model, weighting and DR estimates from the same fitted nuisance models."""
    t, y = df.t.values, df.re78.values
    mu0, e = crossfit(df, learner, seed)
    keep = e <= TRIM
    w = np.zeros_like(e)
    w[keep] = e[keep] / (1 - e[keep])        # weights only for kept units (avoids 1/0 when e = 1)
    return {"outcome": weighted_att(t, y - mu0),
            "ipw": weighted_att(t[keep], y[keep], w[keep]),
            "dr": weighted_att(t[keep], (y - mu0)[keep], w[keep])}


def combine_repeats(ests):
    """Median estimate over splits; SE adds between-split spread (DML-style, from memory — unverified)."""
    atts = np.array([e.att for e in ests])
    med = np.median(atts)
    se = np.median([np.sqrt(e.se ** 2 + (e.att - med) ** 2) for e in ests])
    return Estimate(float(med), float(se))


def flexible_family(df, learner, n_repeats=5):
    runs = [flexible_family_once(df, learner, seed) for seed in range(n_repeats)]
    return {k: combine_repeats([r[k] for r in runs]) for k in runs[0]}

# --- bias-corrected matching (Abadie & Imbens) ------------------------------------------------------
from sklearn.neighbors import NearestNeighbors


def att_bcm(df):
    """1 nearest-neighbour match per treated unit (with replacement, diagonal Mahalanobis), bias-corrected
    by regressing pair outcome gaps on pair covariate gaps. Equals weighted_att with r = Y - X b and
    w = K (number of times each control is used), so the SE accounts for control reuse."""
    X, t, y = df[COVARIATES].values, df.t.values, df.re78.values
    Xs = X / X.std(axis=0)
    ctrl = np.flatnonzero(t == 0)
    match = ctrl[NearestNeighbors(n_neighbors=1).fit(Xs[ctrl]).kneighbors(Xs[t == 1], return_distance=False)[:, 0]]
    beta = LinearRegression().fit(X[t == 1] - X[match], y[t == 1] - y[match]).coef_
    K = np.zeros(len(df))
    np.add.at(K, match, 1)                                   # how many times each control is used
    return weighted_att(t, y - X @ beta, K)


# --- residual balancing (Athey, Imbens & Wager 2018, Algorithm 1) -------------------------------------
import cvxpy as cp
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold

ZETA = 0.5
BALANCE_ON_RAW = True   # raw-unit balancing reproduced the paper's PSID estimate (see sensitivity cell in 01)


def balancing_weights(Bc, bbar_t, zeta=ZETA):
    n_c = Bc.shape[0]
    g, s = cp.Variable(n_c), cp.Variable()
    gap = bbar_t - Bc.T @ g
    cp.Problem(cp.Minimize((1 - zeta) * cp.sum_squares(g) + zeta * cp.square(s)),
               [gap <= s, -gap <= s, g >= 0, cp.sum(g) == 1, g <= n_c ** (-2 / 3)]).solve(solver=cp.CLARABEL)
    return g.value


def elastic_net_1se(Z, y, l1_ratio=0.9, n_folds=10, seed=0):
    cv = ElasticNetCV(l1_ratio=l1_ratio, cv=KFold(n_folds, shuffle=True, random_state=seed)).fit(Z, y)
    mean = cv.mse_path_.mean(axis=1)
    se = cv.mse_path_.std(axis=1, ddof=1) / np.sqrt(n_folds)
    lam = cv.alphas_[mean <= mean.min() + se[mean.argmin()]].max()
    return ElasticNet(alpha=lam, l1_ratio=l1_ratio).fit(Z, y)


def att_residual_balancing(df):
    X, t, y = df[COVARIATES].values, df.t.values, df.re78.values
    Z = StandardScaler().fit_transform(X)
    B = X if BALANCE_ON_RAW else Z
    w = np.zeros(len(df))
    w[t == 0] = balancing_weights(B[t == 0], B[t == 1].mean(axis=0))
    enet = elastic_net_1se(Z[t == 0], y[t == 0])
    return weighted_att(t, y - enet.predict(Z), w)

# --- causal forest (Wager & Athey 2018) -----------------------------------------------------------------
from econml.dml import CausalForestDML


def att_causal_forest_once(df, seed):
    """Causal forest with RF nuisance models (R-style settings, cross-fitted). ATT = plain average of the
    forest's CATEs over the treated, with econml's SE for that average.
    (econml's DR-corrected att_ was tried and rejected: it reweights by untrimmed RF propensities, which
    reach ~1, and the estimate explodes — E 1.15 (se 4.5), CPS -8.26 (se 16.9), PSID 8.88 (se 17.8).)"""
    X, t, y = df[COVARIATES].values, df.t.values, df.re78.values
    cf = CausalForestDML(model_y=RandomForestRegressor(**RF_REG, random_state=seed, n_jobs=-1),
                         model_t=RandomForestClassifier(**RF_CLF, random_state=seed, n_jobs=-1),
                         discrete_treatment=True, cv=N_FOLDS, n_estimators=1000, random_state=seed)
    cf.fit(y, t, X=X)
    inf = cf.ate_inference(X=X[t == 1])           # average of CATEs over the treated units
    return Estimate(float(np.ravel(inf.mean_point)[0]), float(np.ravel(inf.stderr_mean)[0]))


def att_causal_forest(df, n_repeats=5):
    return combine_repeats([att_causal_forest_once(df, seed) for seed in range(n_repeats)])