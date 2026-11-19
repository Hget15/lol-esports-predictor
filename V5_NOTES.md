# V5 — the corrected model library

`main` holds the V1–V4 feature pipeline and the original model library. This branch adds a
rewrite of the model library that fixes the limitations documented in the original's
comments, with tests and a like-for-like benchmark.

> **Correction, November 2026.** An earlier version of these notes benchmarked the two
> libraries on V4 features that contained look-ahead (see the main README), and reported
> that the "patch adaptation" finding replicated. It did not: that finding was an artifact
> of the same bug. The benchmark below was rerun on the corrected features and the full
> dataset.

## What was wrong in the V4 library, and what V5 does instead

| # | V4 limitation | V5 |
|---|---|---|
| 1 | GBDT re-used the Gini **classification** tree on continuous pseudo-residuals — Gini of a residual mean is not a real split criterion | One `GradientTree` that optimises the exact second-order objective. Split gain `½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)]`, leaf `w* = −G/(H+λ)`. With `g=−y, h=1, λ=0` it is a variance-reduction regression tree (≡ Gini for a 0/1 target), so Random Forest and GBDT share one correct implementation |
| 2 | Comment said "Newton-Raphson leaf values"; code did first-order mean-residual leaves | True Newton leaves with `g = p−y`, `h = p(1−p)` and L2 leaf regularisation |
| 3 | Split search: Python loop over 20 percentile thresholds, each re-masking the whole node; per-row recursive prediction | Quantile-binned **histogram** splits: one `np.bincount` + `cumsum` per feature per node, binning computed once per dataset and shared by all trees; prediction routes index arrays through the tree |
| 4 | LR mini-batch sampling and dropout masks were **unseeded** → not reproducible | One seeded `RandomState` drives every random choice; same seed ⇒ bit-identical predictions (tested) |
| 5 | Stacking re-fit the caller's model objects in place; meta-learner combined raw probabilities under strong L2 (good AUC, squashed calibration) | Deep-copies base models per fold (caller's objects untouched, genuinely out-of-fold), meta-learner works in **logit space** — Brier on the smoke test went 0.184 → 0.100 |
| 6 | Feature importance = split count | Gain-weighted importance (split-count still available) |
| 7 | NaN went **left** at fit time (bin 0) but **right** at predict time (`x <= thr` is False for NaN) | Consistent "missing goes left" at both |
| 8 | AUC tie handling in a Python `while` loop | Vectorised average ranks via `np.unique` — identical result, tested against a brute-force pairwise AUC |
| 9 | `warnings.filterwarnings('ignore')` at module level | Nothing suppressed globally; the one legitimately noisy spot (masked 0/0 candidates in split search) uses a scoped `np.errstate` with a comment saying why |

Also new: validation-based early stopping for GBDT and NN (keeps the best iteration/epoch),
random-forest out-of-bag AUC, column subsampling for GBDT, input validation, docstrings that
explain the *why* on every component.

## How it was verified

- **`tests/test_ml_v5.py` — 24 unit tests, ~2 s, pass under `python -W error`** (any warning
  fails the run). They check AUC against a brute-force pairwise definition including ties,
  the tree's leaf/split/binning contracts, that every model learns a known non-linear
  synthetic problem, that early stopping keeps the best iteration, OOB scoring, bit-for-bit
  reproducibility of every model (including dropout), that different seeds differ, and that
  stacking never mutates its templates.
- **CI:** `.github/workflows/tests.yml` runs the suite, the feature look-ahead tests and both
  libraries' smoke tests on Python 3.10 and 3.12 with warnings-as-errors on every push / PR.
- **Head-to-head benchmark** against the V4 library on the same data and split (below).

## Benchmark: V4 library vs V5 library

`python src/benchmark_v5.py` on the corrected `data/engineered_features_v4.csv`: 53,175
games with at least five prior games per team, 130 features, time-ordered 85/15 split →
**45,198 train / 7,977 test**, matched hyper-parameters.

| Model | Lib | AUC | Brier | LogLoss | Acc | Fit (s) |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | V4 | 0.7252 | 0.2114 | 0.6104 | 0.666 | 2.2 |
| Logistic Regression | **V5** | 0.7258 | 0.2110 | 0.6094 | 0.666 | 0.8 |
| GBDT | V4 | 0.7177 | 0.2141 | 0.6169 | 0.654 | 394.3 |
| GBDT | **V5** | **0.7212** | **0.2123** | **0.6118** | **0.660** | **24.4** |
| Random Forest | V4 | 0.7138 | 0.2148 | 0.6179 | 0.655 | 102.8 |
| Random Forest | **V5** | 0.7140 | 0.2148 | 0.6181 | 0.658 | **12.5** |
| Neural Network | V4 | 0.7095 | 0.2185 | 0.6278 | 0.650 | 55.3 |
| Neural Network | **V5** | 0.7086 | 0.2189 | 0.6302 | 0.650 | 46.1 |

**V5 GBDT with validation early stopping** (which V4 does not have): stopped at 118 of a
500-tree cap → AUC 0.7208, Brier 0.2122, 29 s.

Top V5 GBDT features by gain: composite team strength 18.8%, Elo gap 11.9%, 20-game gold
per minute 10.0%, roster experience 7.4%. No patch feature is in the top ten.

### How to read this

- **Speed is the clear win.** GBDT trains about **16× faster** (394 s → 24 s) and the random
  forest about 8×, from the vectorised histogram splits.
- **GBDT is slightly better on every metric** (AUC +0.0035, Brier and log-loss lower). With
  7,977 test games the standard error of an AUC near 0.72 is about 0.006, so this is a
  consistent small edge, not a decisive one.
- **Logistic regression, random forest and the neural network are ties**, as they should be:
  V5 changes how they are implemented, not what they compute.
- **Logistic regression is still the best model on this data** in either library.
- These numbers use an 85/15 split, so they sit a little above the main README's table,
  which trains on the first 75% of games.

## Running it

```bash
python -m unittest tests.test_ml_v5 -v        # unit tests
python src/ml_from_scratch_v5.py               # smoke test on synthetic data
python src/benchmark_v5.py [features.csv]      # V4-vs-V5 head-to-head on your local data
```

The benchmark needs an engineered-feature CSV; see `data/README.md` for building one from
the Oracle's Elixir download.

## What I'd do next

1. Swap the V5 library into `src/evaluate_versions.py` so the headline table runs in a
   couple of minutes instead of twenty.
2. Tune V5's new knobs (`reg_lambda`, `colsample`, `n_bins`, early-stopping patience) on the
   validation slice — everything above uses V4's hyper-parameters unchanged.
3. Use gain-based importance to prune the 130 features; many are `_a` / `_b` / `_diff`
   triples of the same quantity.
