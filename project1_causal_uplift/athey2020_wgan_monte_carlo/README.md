# Reproducing Athey, Imbens, Metzger & Munro (2020)

**"Using Wasserstein Generative Adversarial Networks for the Design of Monte Carlo Simulations"**
([arXiv:1909.02210](https://arxiv.org/abs/1909.02210)) — data and pretrained generators from
[gsbDBI/ds-wgan](https://github.com/gsbDBI/ds-wgan).

The paper compares 13 estimators of the average treatment effect on the treated (ATT) on the
Lalonde–Dehejia–Wahba (LDW) job-training data, then uses WGANs to build realistic simulations with a
known true effect and ranks the estimators by bias, RMSE and confidence-interval coverage.

## Summary

| Part | Status |
|---|---|
| Table 1 — sample summary statistics | Reproduced exactly (all 36 values) |
| Table 5 — 13 estimators on real data | Mostly reproduced; gaps traced to unreported settings or documented |
| Table 2 — WGAN-generated data | **Not reproduced:** the released generators are not the models behind the paper's Table 2 |
| Tables 6–8 — Monte Carlo | Reproduced in method; compared on conclusions, not numbers (different simulated world) |

**Headline findings**
1. **No single estimator wins everywhere** (the paper's main result) — reproduced.
2. **Doubly robust estimators with flexible models are the safe default** (the paper's recommendation) — holds in our
   simulated world too: never the worst, never far from the best.
3. **The simulated world depends on the trained generator.** With the authors' released generators, CPS is the hard
   design and PSID the easy one — the reverse of the paper. GAN training randomness is itself a design choice.
4. **An erratum:** the paper's Table 5 standard errors for the difference in means swap the group sizes.

## The three designs

All share the same 185 NSW trainees; only the comparison group changes.

| Design | Controls | Represents |
|---|---|---|
| E | 260 randomised NSW controls | Experiment: no confounding |
| CPS | 15,992 Current Population Survey men | Observational, heavy confounding, large pool |
| PSID | 2,490 Panel Study of Income Dynamics men | Observational, worst overlap |

Experimental benchmark (E, difference in means): **$1,794** (1978 dollars). Units below are $1,000s.

## Part 1 — Real data (`01_ate_estimators_real_data.py`, paper Table 5)

| Estimator | E ours / paper | CPS ours / paper | PSID ours / paper |
|---|---|---|---|
| DIFF | 1.79 / 1.79 | −8.50 / −8.50 | −15.20 / −15.20 |
| Bias-corrected matching | 1.77 / 2.12 | 2.20 / 2.15 | −0.07 / 0.57 |
| Outcome: linear | 1.79 / 1.79 | 0.69 / 0.69 | 0.79 / 0.79 |
| Outcome: random forest | 1.71 / 1.69 | 1.10 / 0.85 | −0.16 / −0.20 |
| Outcome: neural net | 2.60 / 1.49 | 1.53 / 1.70 | 0.77 / 1.47 |
| Propensity: logit | 1.81 / 1.81 | 1.19 / 1.18 | 1.75 / 1.26 |
| Propensity: random forest | 2.27 / 1.90 | 1.66 / 0.73 | 0.32 / 0.24 |
| Propensity: neural net | 1.82 / 1.69 | 1.82 / 1.38 | 0.16 / 0.42 |
| DR: linear + logit | 1.80 / 1.80 | 1.28 / 1.27 | 2.05 / 1.50 |
| DR: random forest | 2.23 / 1.93 | 1.48 / 1.63 | 1.35 / 0.98 |
| DR: neural net | 2.71 / 1.90 | 1.54 / 1.63 | 0.20 / 1.56 |
| Causal forest | 1.70 / 1.72 | 1.80 / 1.58 | 0.45 / 0.59 |
| Residual balancing | 1.80 / 1.73 | 1.16 / 0.93 | 0.77 / 0.72 |

Flexible estimators use cross-fitting and report the median over 5 random splits; the paper's single-split values
generally fall within our split-to-split ranges.

**What matched, what didn't, and why**
- **Exact:** DIFF and linear outcome model (all designs); logit weighting and DR-linear on E and CPS.
- **PSID logit-based gap (unexplained):** our logit weighting gives 1.75 vs 1.26, DR-linear 2.05 vs 1.50. The gap
  appears only in estimators that use the propensity score and survives every check (two fitting libraries, three
  trimming rules, default regularisation in two unit scales, normalised vs unnormalised weights). The authors'
  estimation code is not public.
- **Random forests depend on unreported settings.** On PSID the same RF outcome model ranges from **−0.20 to 1.26**
  across forest settings; R `randomForest` defaults with 2-fold splitting reproduce the paper's −0.20 exactly.
- **Residual balancing depends on covariate scaling.** Balancing on raw (unstandardised) covariates reproduces the
  paper's PSID 0.72 (ours 0.77); standardised gives 1.36. Across choices, PSID ranges 0.77–1.81.
- **Neural nets:** architecture not reported. scikit-learn's `MLPClassifier` early stopping monitors *accuracy*, which
  fails with rare treatment (1.1% treated in CPS): training stops almost immediately and the propensity model stays
  near-constant. Training without early stopping fixes CPS/PSID but overfits on E.
- **Matching:** likely tie handling (R's `Matching` averages ties; we pick one).

**Erratum — DIFF standard errors.** The paper's Table 5 DIFF SEs (0.63 / 0.71 / 1.15) equal
√(s₁²/n₀ + s₀²/n₁), i.e. each group's variance divided by the *other* group's size. The correct
√(s₁²/n₁ + s₀²/n₀) gives 0.67 / 0.58 / 0.66. Estimates are unaffected.

## Part 2 — WGAN population (`02_wgan_population.py`, paper Section 3.3)

Generators from the authors' repository; variable setup (ordering, bounds, balanced training resample) from their
Colab tutorial. A 1,000,000-unit population per design; both potential outcomes generated for every unit.

| Design | True ATT in simulated population |
|---|---|
| E | 1.89 |
| CPS | 1.10 |
| PSID | 0.97 |

Each simulated world has its own truth: the generator embeds the outcome model it learned from that design's data,
with unconfoundedness built in. It is not the experimental $1,794.

**Table 2 does not match.** Generated E trainees are 85% black vs the paper's 92% (real data: 84%); re74 2.11 vs 3.07
(real: 2.10). Neither dropout on/off nor the balanced-resample seed moves these. The released generators fit the
real E data *better* than the paper's, but they are evidently different trained models.

## Part 3 — Monte Carlo (`03_monte_carlo.py`, paper Tables 6–8)

Each replication draws a sample the size of the real data (E 185+260, CPS 185+15,992, PSID 185+2,490) from the
population. 200 replications per design for all 19 estimators (fast estimators also run to 500; one cross-fitting
split per replication). Influence-function standard errors; the paper does not state how its SEs were computed.

Best three by RMSE (ties within ~±0.03 Monte Carlo error):

| Design | Best | 
|---|---|
| E | Outcome: linear 0.56, Residual balancing 0.56, Propensity: logit 0.57 |
| CPS | Causal forest 0.49, NN: outcome 0.54, NN: DR 0.54 |
| PSID | Outcome: linear 0.57, Causal forest 0.62, Residual balancing 0.63 |

Full results: `results/mc_combined_summary.csv` (rmse, bias, sdev, mean SE, coverage, paper values).

**Lessons**
1. **DR is not protection when both models are wrong.** On CPS every *linear* model-based estimator shares a bias of
   about −0.5 (DR-linear −0.45, coverage 0.83). Flexible DR removes it (DR-RF −0.04, DR-NN 0.00, coverage ~0.93).
   Cf. Kang & Schafer (2007).
2. **Flexible outcome models alone can be worse than linear ones under poor overlap.** PSID: RF outcome −1.10
   (coverage 0.35), NN outcome +0.88 (0.46), linear outcome +0.05. Forests cannot extrapolate to trainees' very low
   prior earnings, where few PSID controls exist. Weighting and DR versions of the same models are unbiased.
3. **Diagnose coverage with mean SE vs sdev.** Outcome-model SEs are too small (they ignore uncertainty in the fitted
   model: E coverage 0.88, as in the paper); matching SEs are too large (coverage 0.99); causal-forest SEs from econml
   are 2–3× too large (coverage 1.00 everywhere). IPW, DR and residual-balancing SEs are well calibrated, so their
   coverage failures come from bias.
4. **Extreme propensity weights break estimators.** Random-forest propensities reach 1 (pure leaves). econml's
   doubly robust ATT for the causal forest reweights by untrimmed propensities and explodes (CPS −8.26, SE 16.9);
   the plain average of CATEs is stable.
5. **The simulated "experiment" isn't quite randomised.** DIFF is biased by −0.28 on E: the generator learned the real
   sample's chance imbalances as systematic differences.

## Reproduced tables (ours vs paper)

### Table 1 — real data: mean (s.d.), ours / paper

| variable | Trainees (185) | Exp. controls (260) | CPS controls (15,992) | PSID controls (2,490) |
|---|---|---|---|---|
| black | 0.84 (0.36) / 0.84 (0.36) | 0.83 (0.38) / 0.83 (0.38) | 0.07 (0.26) / 0.07 (0.26) | 0.25 (0.43) / 0.25 (0.43) |
| hispanic | 0.06 (0.24) / 0.06 (0.24) | 0.11 (0.31) / 0.11 (0.31) | 0.07 (0.26) / 0.07 (0.26) | 0.03 (0.18) / 0.03 (0.18) |
| age | 25.82 (7.16) / 25.82 (7.16) | 25.05 (7.06) / 25.05 (7.06) | 33.23 (11.05) / 33.23 (11.05) | 34.85 (10.44) / 34.85 (10.44) |
| married | 0.19 (0.39) / 0.19 (0.39) | 0.15 (0.36) / 0.15 (0.36) | 0.71 (0.45) / 0.71 (0.45) | 0.87 (0.34) / 0.87 (0.34) |
| nodegree | 0.71 (0.46) / 0.71 (0.46) | 0.83 (0.37) / 0.83 (0.37) | 0.30 (0.46) / 0.30 (0.46) | 0.31 (0.46) / 0.31 (0.46) |
| education | 10.35 (2.01) / 10.35 (2.01) | 10.09 (1.61) / 10.09 (1.61) | 12.03 (2.87) / 12.03 (2.87) | 12.12 (3.08) / 12.12 (3.08) |
| re74 | 2.10 (4.89) / 2.10 (4.89) | 2.11 (5.69) / 2.11 (5.69) | 14.02 (9.57) / 14.02 (9.57) | 19.43 (13.41) / 19.43 (13.41) |
| re75 | 1.53 (3.22) / 1.53 (3.22) | 1.27 (3.10) / 1.27 (3.10) | 13.65 (9.27) / 13.65 (9.27) | 19.06 (13.60) / 19.06 (13.60) |
| re78 | 6.35 (7.87) / 6.35 (7.87) | 4.55 (5.48) / 4.55 (5.48) | 14.85 (9.65) / 14.85 (9.65) | 21.55 (15.56) / 21.55 (15.56) |

### Table 2 — WGAN-generated data: means, ours / paper

| variable | Trainees (E) | Exp. controls (E) | CPS controls | PSID controls |
|---|---|---|---|---|
| black | 0.85 / 0.92 | 0.83 / 0.91 | 0.09 / 0.08 | 0.29 / 0.26 |
| hispanic | 0.07 / 0.03 | 0.10 / 0.03 | 0.08 / 0.07 | 0.03 / 0.03 |
| age | 26.32 / 25.61 | 25.89 / 25.54 | 33.19 / 32.82 | 36.50 / 34.95 |
| married | 0.19 / 0.15 | 0.18 / 0.10 | 0.72 / 0.72 | 0.85 / 0.87 |
| nodegree | 0.73 / 0.69 | 0.84 / 0.81 | 0.29 / 0.29 | 0.33 / 0.30 |
| education | 9.98 / 10.35 | 9.79 / 10.02 | 11.89 / 12.03 | 11.44 / 12.32 |
| re74 | 2.11 / 3.07 | 2.07 / 2.13 | 14.63 / 14.07 | 18.04 / 19.86 |
| re75 | 1.42 / 2.06 | 1.32 / 1.29 | 14.35 / 13.73 | 16.74 / 20.41 |
| re78 | 5.90 / 5.31 | 4.29 / 3.97 | 15.51 / 14.89 | 19.29 / 22.62 |

### Table 6 — Monte Carlo, E design (true ATT 1.89; 200 replications; paper 2,000)

| estimator | RMSE ours | RMSE paper | bias ours | bias paper | coverage ours | coverage paper | sdev ours | mean SE ours |
|---|---|---|---|---|---|---|---|---|
| DIFF | 0.59 | 0.49 | -0.25 | 0.06 | 0.92 | 0.94 | 0.54 | 0.53 |
| BCM | 0.66 | 0.58 | 0.16 | 0.00 | 0.98 | 0.96 | 0.64 | 0.85 |
| Outcome: linear | 0.56 | 0.52 | -0.01 | -0.06 | 0.89 | 0.88 | 0.56 | 0.46 |
| RF: outcome | 0.57 | 0.51 | -0.15 | -0.07 | 0.90 | 0.88 | 0.55 | 0.46 |
| NN: outcome | 0.74 | 1.32 | 0.23 | 0.04 | 0.79 | 0.75 | 0.71 | 0.46 |
| Propensity: logit | 0.57 | 0.52 | 0.02 | -0.08 | 0.92 | 0.99 | 0.57 | 0.54 |
| RF: ipw | 0.64 | 0.52 | -0.04 | -0.06 | 0.94 | 0.99 | 0.64 | 0.61 |
| NN: ipw | 0.89 | 0.52 | -0.33 | 0.01 | 0.91 | 0.99 | 0.83 | 0.74 |
| DR: linear | 0.57 | 0.51 | 0.02 | -0.08 | 0.94 | 0.95 | 0.57 | 0.54 |
| RF: dr | 0.67 | 0.52 | -0.08 | -0.04 | 0.92 | 0.95 | 0.66 | 0.60 |
| NN: dr | 0.84 | 0.79 | -0.05 | -0.05 | 0.92 | 0.95 | 0.84 | 0.74 |
| Causal forest | 0.61 | 0.50 | -0.28 | -0.09 | 1.00 | 0.94 | 0.54 | 1.14 |
| Residual balancing | 0.56 | 0.52 | -0.03 | -0.09 | 0.92 | 0.95 | 0.56 | 0.54 |

### Table 7 — Monte Carlo, CPS design (true ATT 1.10; 200 replications; paper 2,000)

| estimator | RMSE ours | RMSE paper | bias ours | bias paper | coverage ours | coverage paper | sdev ours | mean SE ours |
|---|---|---|---|---|---|---|---|---|
| DIFF | 10.55 | 11.12 | -10.54 | -11.11 | 0.00 | 0.00 | 0.41 | 0.40 |
| BCM | 0.61 | 0.73 | 0.02 | 0.07 | 0.98 | 0.96 | 0.61 | 0.80 |
| Outcome: linear | 0.78 | 2.14 | -0.64 | -2.08 | 0.66 | 0.02 | 0.46 | 0.42 |
| RF: outcome | 0.79 | 1.00 | -0.64 | -0.87 | 0.62 | 0.54 | 0.46 | 0.41 |
| NN: outcome | 0.54 | 0.63 | -0.21 | 0.14 | 0.85 | 0.88 | 0.50 | 0.41 |
| Propensity: logit | 0.67 | 0.51 | -0.48 | 0.00 | 0.78 | 0.98 | 0.47 | 0.44 |
| RF: ipw | 0.58 | 1.00 | -0.12 | -0.87 | 0.92 | 0.73 | 0.56 | 0.54 |
| NN: ipw | 0.57 | 0.65 | 0.18 | 0.23 | 0.92 | 0.94 | 0.54 | 0.53 |
| DR: linear | 0.65 | 0.53 | -0.46 | 0.03 | 0.84 | 0.96 | 0.47 | 0.46 |
| RF: dr | 0.57 | 0.54 | -0.04 | -0.05 | 0.94 | 0.93 | 0.57 | 0.54 |
| NN: dr | 0.54 | 0.62 | 0.00 | 0.20 | 0.92 | 0.94 | 0.54 | 0.54 |
| Causal forest | 0.49 | 0.55 | -0.10 | 0.11 | 1.00 | 0.91 | 0.48 | 1.49 |
| Residual balancing | 0.79 | 0.57 | -0.62 | -0.22 | 0.71 | 0.89 | 0.49 | 0.46 |

### Table 8 — Monte Carlo, PSID design (true ATT 0.97; 200 replications; paper 2,000)

| estimator | RMSE ours | RMSE paper | bias ours | bias paper | coverage ours | coverage paper | sdev ours | mean SE ours |
|---|---|---|---|---|---|---|---|---|
| DIFF | 15.17 | 18.81 | -15.17 | -18.81 | 0.00 | 0.00 | 0.47 | 0.47 |
| BCM | 0.83 | 0.98 | 0.18 | -0.02 | 0.94 | 0.98 | 0.81 | 0.84 |
| Outcome: linear | 0.57 | 1.95 | 0.05 | -1.82 | 0.84 | 0.12 | 0.56 | 0.44 |
| RF: outcome | 1.21 | 2.30 | -1.10 | -2.22 | 0.35 | 0.02 | 0.51 | 0.41 |
| NN: outcome | 1.09 | 2.97 | 0.88 | -0.93 | 0.46 | 0.59 | 0.66 | 0.44 |
| Propensity: logit | 0.65 | 1.11 | -0.03 | -0.64 | 0.92 | 0.96 | 0.66 | 0.63 |
| RF: ipw | 0.75 | 2.21 | -0.07 | -2.05 | 0.96 | 0.32 | 0.75 | 0.74 |
| NN: ipw | 0.81 | 1.82 | 0.14 | -1.43 | 0.94 | 0.69 | 0.80 | 0.80 |
| DR: linear | 0.72 | 0.98 | 0.21 | -0.35 | 0.93 | 0.92 | 0.69 | 0.67 |
| RF: dr | 0.83 | 0.98 | 0.21 | -0.57 | 0.88 | 0.84 | 0.80 | 0.73 |
| NN: dr | 0.85 | 0.98 | 0.09 | -0.38 | 0.94 | 0.92 | 0.85 | 0.82 |
| Causal forest | 0.62 | 1.13 | 0.04 | -0.89 | 1.00 | 0.73 | 0.62 | 1.32 |
| Residual balancing | 0.63 | 1.06 | -0.13 | 0.33 | 0.92 | 0.75 | 0.62 | 0.57 |


## Critique

- **Researcher discretion persists.** The paper argues WGANs reduce discretion in simulation design. But (i) several
  of its own real-data results depend on settings it does not report, and (ii) a different training run of the same
  generator produces a simulated world in which estimator rankings change.
- **The simulation tests estimators under unconfoundedness; it cannot test unconfoundedness.** The generator builds the
  assumption in, so a good simulated ranking says nothing about whether the 8 covariates suffice in the real data
  (cf. Advani, Kitagawa & Słoczyński 2019).
- **Robustness checks cover only CPS** (paper Section 5).

## How to run

```bash
conda activate causal          # numpy, pandas, scikit-learn, statsmodels, econml, cvxpy, pytorch
pip install git+https://github.com/gsbDBI/ds-wgan
```
Run in order: `01_ate_estimators_real_data.py` → `02_wgan_population.py` → `03_monte_carlo.py`.
Estimators live in `estimators.py`; every one returns an estimate and an SE. Data downloads on first run into
`data/` (git-ignored); results go to `results/`.