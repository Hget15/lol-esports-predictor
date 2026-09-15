# V5 — the corrected model library

`main` keeps the V1–V4 code exactly as it produced the published results (AUC 0.797 on
51,088 games). This branch is where the limitations documented in that code's comments
are actually fixed, and where I try to push the library further. Nothing here changes
the numbers in the main README.

## What was wrong in V4, and what V5 does instead

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
- **CI:** `.github/workflows/tests.yml` runs the suite plus both libraries' smoke tests on
  Python 3.10 and 3.12 with warnings-as-errors on every push / PR.
- **Head-to-head benchmark** against the V4 library on the same data and split (below).

## Local benchmark: V4 library vs V5 library

`python src/benchmark_v5.py` on `data/engineered_features_v4.csv` built from the **2026
season only** (the sole raw file available locally): 1,355 games after the ≥5-games filter,
130 V4 features, temporal 85/15 split → **1,151 train / 204 test**, matched hyper-parameters.

| Model | Lib | AUC | Brier | LogLoss | Acc | Fit (s) |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | V4 | 0.8000 | 0.1835 | 0.5787 | 0.740 | 0.2 |
| Logistic Regression | **V5** | **0.8228** | **0.1728** | **0.5207** | 0.735 | 0.0 |
| GBDT | V4 | **0.8401** | 0.1718 | 0.5230 | 0.755 | 61.9 |
| GBDT | **V5** | 0.8317 | **0.1670** | **0.4997** | 0.755 | **5.5** |
| Random Forest | V4 | **0.8217** | **0.1806** | **0.5419** | 0.760 | 10.9 |
| Random Forest | **V5** | 0.8124 | 0.1845 | 0.5515 | 0.745 | **2.0** |
| Neural Network | V4 | 0.7403 | 0.2411 | 1.0671 | 0.706 | 1.5 |
| Neural Network | **V5** | 0.7345 | 0.2473 | 1.1329 | 0.696 | 1.0 |

**V5 GBDT with validation early stopping** (a feature V4 does not have): stopped at 121 of a
500-tree cap → **AUC 0.8397, Brier 0.1643** (best Brier in the table), 6.6 s.

`patch_wr_delta_diff` is again the top feature (19.0% gain-weighted importance) — the
project's headline finding reproduces on a different season with a different tree
implementation.

### How to read this honestly

- **Calibration improved.** V5 is better on Brier and log-loss for LR and GBDT, i.e. its
  probabilities are more trustworthy, which is what Newton leaves + regularisation are for.
- **Speed improved a lot.** GBDT trains **~11× faster**, RF ~5×, with no change in accuracy —
  the vectorised histogram splits doing exactly what they were meant to.
- **AUC is a statistical tie.** With only 204 test games the standard error of an AUC around
  0.83 is roughly ±0.03. V4's nominal +0.01 on GBDT/RF and V5's +0.02 on LR are both inside
  that; neither library "wins" on ranking here. With early stopping V5 lands on V4's best AUC
  anyway, at the best calibration of all.
- **Neither NN is good on this much data** (1,151 rows, 130 features → both overfit). That
  matches the original project's finding that tree models dominate on this problem.
- **This is not the headline number.** The README's 0.797 AUC came from 51,088 games across
  2021–2026; a single season is smaller, noisier and — as the higher absolute AUCs show — an
  easier slice. The value of this benchmark is the *relative* comparison and the timing.

## Running it

```bash
python -m unittest tests.test_ml_v5 -v        # unit tests
python src/ml_from_scratch_v5.py               # smoke test on synthetic data
python src/benchmark_v5.py [features.csv]      # V4-vs-V5 head-to-head on your local data
```

The benchmark needs an engineered-feature CSV; see `data/README.md` for building one from
the Oracle's Elixir download.

## What I'd do next with the full dataset

1. Re-run the V4 training pipeline with the V5 library swapped in, to get a like-for-like
   51k-game number and a proper walk-forward CV spread rather than one split.
2. Tune V5's new knobs (`reg_lambda`, `colsample`, `n_bins`, early-stopping patience) on the
   validation slice — everything above uses V4's hyper-parameters unchanged.
3. Use gain-based importance to prune the 130 features; the V4 notes suspected redundancy
   among the `_a` / `_b` / `_diff` triples.
