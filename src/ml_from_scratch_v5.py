"""
ml_from_scratch_v5 — corrected and improved from-scratch ML library (NumPy only).

V5 is the "fix everything" rewrite of ml_from_scratch.py. The V1–V4 library on
`main` is kept exactly as it was so the published results stay reproducible;
this module is where the limitations documented there are actually fixed.

What changed and why
--------------------
1. ONE principled tree. A histogram-binned *gradient tree* replaces the
   Gini-on-residuals hack. It optimises the exact second-order objective
   (the XGBoost formulation): for a leaf collecting gradients G = Σg_i and
   hessians H = Σh_i, the optimal weight is  w* = -G / (H + λ)  and a split's
   gain is  ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ].
   With g = −y, h = 1, λ = 0 the very same tree is an ordinary MSE /
   variance-reduction regression tree whose leaves are class means — which
   for a 0/1 target is equivalent to Gini — so Random Forest and GBDT share a
   single correct implementation instead of a classification tree being
   re-purposed for residuals.
2. TRUE Newton boosting. The V4 comment claimed "Newton-Raphson leaf values";
   the code did first-order mean-residual leaves. V5 uses g = p − y,
   h = p(1 − p) with L2 leaf regularisation, column subsampling, and optional
   validation-based early stopping that keeps the best iteration.
3. Speed. Split finding is histogram-based and fully vectorised (np.bincount +
   cumsum over ≤ n_bins quantile bins per feature) instead of a Python loop
   over 20 percentile thresholds that re-masked the whole node each time.
   Binning is done once per dataset and shared by every tree. Prediction
   routes *index arrays* through the tree instead of recursing per row.
4. Reproducibility. Every source of randomness — LR mini-batches, dropout
   masks, bootstraps, column subsampling — is driven by one seeded
   RandomState. In V4, LR batch sampling and dropout were unseeded.
5. Stacking deep-copies base models per fold, so the caller's objects are
   never mutated and out-of-fold meta-features are genuinely out-of-fold.
6. Feature importance is gain-weighted (total objective reduction per
   feature); split-count is still available for comparison.
7. Missing values are routed consistently to the LEFT child at both fit and
   predict time ("missing goes left"). V4 compared `x <= thr`, which is False
   for NaN, so NaN silently went right at predict time only.
8. AUC tie handling is vectorised (average ranks via np.unique) — identical
   result to the V4 Python loop, far faster on tree ensembles that emit many
   repeated probabilities.

The public surface deliberately mirrors V4 — fit / predict_proba / predict —
so V4 and V5 models can be benchmarked side by side (see benchmark_v5.py).
"""

from __future__ import annotations

import copy
import numpy as np

__all__ = [
    "sigmoid", "logloss", "brier_score", "roc_auc_score", "accuracy_score",
    "calibration_bins", "train_test_split_temporal", "standardize",
    "GradientTree", "LogisticRegression", "GradientBoostedTrees",
    "RandomForest", "NeuralNetwork", "StackingEnsemble", "walk_forward_cv",
]


# ============================================================
# UTILITIES / METRICS
# ============================================================

def sigmoid(z):
    # Clip before exp(): exp(-z) overflows for z < ~-709 and would poison every
    # downstream metric with NaN. ±500 is safely inside float64 range while the
    # output is indistinguishable from 0 or 1 at the extremes.
    z = np.clip(np.asarray(z, dtype=float), -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def logloss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    # log(0) = -inf: a single confidently-wrong 0.0/1.0 would make the mean
    # infinite, so clip into the open interval (0, 1).
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-15, 1 - 1e-15)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def brier_score(y_true, y_pred):
    # Mean squared error of the probabilities. Tracked next to AUC because AUC
    # only measures ranking; Brier also punishes miscalibration.
    return float(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2))


def _average_ranks(x):
    """1-based ranks with ties averaged — vectorised, O(n log n).

    Sort once, find tie groups with np.unique on the sorted array, and give
    every member of a group the mean of the ranks it spans. Same result as a
    Python while-loop over ties, without the per-element interpreter cost.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    _, inv, counts = np.unique(sx, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts + 1           # first rank of each group
    avg = starts + (counts - 1) / 2.0                 # mean rank of each group
    ranks = np.empty(x.shape[0], dtype=float)
    ranks[order] = avg[inv]
    return ranks


def roc_auc_score(y_true, y_pred):
    """ROC AUC via the rank-based (Mann-Whitney U) identity.

    AUC is exactly P(score(random positive) > score(random negative)), which
    the U statistic gives in closed form from ranks — exact (no trapezoid
    approximation) and one sort. Ties are average-ranked, otherwise AUC is
    biased whenever a model emits repeated probabilities (tree ensembles do).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("roc_auc_score: y_true and y_pred must have the same shape")
    if np.isnan(y_pred).any():
        raise ValueError("roc_auc_score: y_pred contains NaN")
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(y_pred)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def accuracy_score(y_true, y_pred_proba, threshold=0.5):
    y_pred = (np.asarray(y_pred_proba, float) >= threshold).astype(int)
    return float(np.mean(np.asarray(y_true).astype(int) == y_pred))


def calibration_bins(y_true, y_pred, n_bins=10):
    """Return bin midpoints, observed positive fraction, and bin counts."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    edges = np.linspace(0, 1, n_bins + 1)
    mids, fracs, counts = [], [], []
    for i in range(n_bins):
        hi_inclusive = (i == n_bins - 1)  # last bin must include p == 1.0
        mask = (y_pred >= edges[i]) & ((y_pred <= edges[i + 1]) if hi_inclusive else (y_pred < edges[i + 1]))
        c = int(mask.sum())
        if c > 0:
            mids.append((edges[i] + edges[i + 1]) / 2)
            fracs.append(float(y_true[mask].mean()))
            counts.append(c)
    return np.array(mids), np.array(fracs), np.array(counts)


def train_test_split_temporal(X, y, dates, test_frac=0.2):
    """Most recent `test_frac` of games become the test set.

    A shuffled split would let the model see the future (and the rolling
    features already encode the past), inflating every metric. Ordering by
    date mirrors real use: predict games that haven't happened from games
    that have.
    """
    order = np.argsort(dates, kind="mergesort")
    X, y, dates = X[order], y[order], dates[order]
    split = int(len(y) * (1 - test_frac))
    return X[:split], X[split:], y[:split], y[split:], dates[:split], dates[split:]


def standardize(X_train, X_test=None):
    """Z-score using TRAINING statistics only (fitting on test data leaks)."""
    mu = np.nanmean(X_train, axis=0)
    sigma = np.nanstd(X_train, axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)  # constant columns -> harmless 0, not NaN
    X_train_s = (X_train - mu) / sigma
    if X_test is not None:
        return X_train_s, (X_test - mu) / sigma, mu, sigma
    return X_train_s, mu, sigma


def _seed_from(rng):
    """Draw a child seed from a RandomState (keeps every component reproducible)."""
    return int(rng.randint(0, 2**31 - 1))


# ============================================================
# GRADIENT TREE (histogram-binned, second-order objective)
# ============================================================

class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "gain", "n_samples")

    def __init__(self):
        self.feature = None      # None => leaf
        self.threshold = None
        self.left = None
        self.right = None
        self.value = 0.0
        self.gain = 0.0
        self.n_samples = 0


class GradientTree:
    """Histogram-binned regression tree on (gradient, hessian) statistics.

    `fit(X, g, h)` minimises  Σ_i [ g_i·w + ½ h_i·w² ] + ½ λ w²  in each leaf —
    the second-order Taylor expansion of any twice-differentiable loss — so
    the leaf weight is  w* = -G / (H + λ).

    How callers use it:
      * GBDT on log-loss:   g = p − y,  h = p(1−p)        -> Newton boosting
      * Random forest:      g = −y,     h = 1,  λ = 0      -> leaf = mean(y);
        split gain = variance reduction (≡ Gini on a 0/1 target)

    Why histograms: each feature is bucketed into ≤ n_bins quantile bins ONCE
    (per dataset, shared by all trees). At a node, the per-bin sums of g and
    h come from a single np.bincount, and every candidate split's gain falls
    out of one cumulative sum — O(n) per feature per node with no Python
    inner loop, versus V4's 20 thresholds × full-array masking each.

    Missing values: NaN is placed in bin 0 at fit time and sent left at
    predict time, so training and inference agree ("missing goes left").
    """

    def __init__(self, max_depth=6, min_samples_leaf=5, max_features=None,
                 n_bins=32, reg_lambda=1.0, min_gain=0.0, random_state=None):
        if n_bins < 2 or n_bins > 65535:
            raise ValueError("n_bins must be in [2, 65535]")
        self.max_depth = max_depth
        self.min_samples_leaf = max(1, int(min_samples_leaf))
        self.max_features = max_features
        self.n_bins = n_bins
        self.reg_lambda = float(reg_lambda)
        self.min_gain = float(min_gain)
        self.rng = np.random.RandomState(random_state)
        self.root = None
        self.edges_ = None
        self.n_features_ = None
        self._gain_importance = None
        self._split_counts = None

    # ---------- binning (shared by every tree in an ensemble) ----------
    @staticmethod
    def bin_features(X, n_bins=32):
        """Quantile-bin every column once. Returns (edges, bins).

        edges[f] holds the interior bin edges of feature f (≤ n_bins − 1, fewer
        when values repeat); bins is a uint16 matrix of bin indices with
        bins[i, f] = #(edges[f] < X[i, f]), so `bin <= k` ⇔ `x <= edges[f][k]`.
        NaN is forced to bin 0.
        """
        X = np.asarray(X, dtype=float)
        n, m = X.shape
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = []
        bins = np.zeros((n, m), dtype=np.uint16)
        for f in range(m):
            col = X[:, f]
            nan_mask = np.isnan(col)
            finite = col[~nan_mask]
            if finite.size == 0:
                edges.append(np.empty(0))
                continue
            e = np.unique(np.quantile(finite, qs))
            edges.append(e)
            if e.size:
                bins[:, f] = np.searchsorted(e, col, side="left")
                bins[nan_mask, f] = 0   # searchsorted puts NaN last; we want "missing goes left"
        return edges, bins

    # ---------- fitting ----------
    def fit(self, X, g, h=None):
        X = np.asarray(X, dtype=float)
        edges, bins = self.bin_features(X, self.n_bins)
        return self.fit_binned(bins, edges, g, h)

    def fit_binned(self, bins, edges, g, h=None):
        """Fit on a pre-binned matrix (what GBDT / RandomForest call per tree)."""
        g = np.asarray(g, dtype=float)
        h = np.ones_like(g) if h is None else np.asarray(h, dtype=float)
        if bins.shape[0] != g.shape[0]:
            raise ValueError("bins and g must have the same number of rows")
        self.edges_ = edges
        self.n_features_ = bins.shape[1]
        self._gain_importance = np.zeros(self.n_features_)
        self._split_counts = np.zeros(self.n_features_)
        self.root = self._build(bins, g, h, np.arange(g.shape[0]), 0)
        return self

    def _build(self, B, g, h, rows, depth):
        node = _Node()
        n = rows.size
        G = float(g[rows].sum())
        H = float(h[rows].sum())
        node.n_samples = n
        node.value = -G / (H + self.reg_lambda)   # Newton-optimal leaf weight
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return node
        best = self._best_split(B, g, h, rows, G, H)
        if best is None:
            return node
        f, k, gain = best
        node.feature = f
        node.threshold = float(self.edges_[f][k])
        node.gain = gain
        self._gain_importance[f] += gain
        self._split_counts[f] += 1
        col = B[rows, f]
        left_rows = rows[col <= k]
        right_rows = rows[col > k]
        node.left = self._build(B, g, h, left_rows, depth + 1)
        node.right = self._build(B, g, h, right_rows, depth + 1)
        return node

    def _best_split(self, B, g, h, rows, G, H):
        m = B.shape[1]
        lam = self.reg_lambda
        min_leaf = self.min_samples_leaf
        if self.max_features is None or self.max_features >= m:
            feats = range(m)
        else:
            # Random feature subset per split — what decorrelates forest trees.
            feats = self.rng.choice(m, self.max_features, replace=False)
        Bn = B[rows]
        gn = g[rows]
        hn = h[rows]
        n = rows.size
        parent = G * G / (H + lam)
        best_gain = self.min_gain
        best = None
        for f in feats:
            nb = self.edges_[f].size + 1
            if nb < 2:
                continue                       # constant feature: nothing to split
            b = Bn[:, f].astype(np.intp)
            Gb = np.bincount(b, weights=gn, minlength=nb)
            Hb = np.bincount(b, weights=hn, minlength=nb)
            Cb = np.bincount(b, minlength=nb)
            GL = np.cumsum(Gb)[:-1]
            HL = np.cumsum(Hb)[:-1]
            CL = np.cumsum(Cb)[:-1]
            GR, HR, CR = G - GL, H - HL, n - CL
            valid = (CL >= min_leaf) & (CR >= min_leaf)
            if not valid.any():
                continue
            # Invalid candidates can have H_L + λ == 0 (empty side with λ = 0);
            # they are masked out below, so the 0/0 they'd produce is noise, not
            # signal. Suppress exactly that, locally, rather than globally.
            with np.errstate(divide="ignore", invalid="ignore"):
                gain = 0.5 * (GL * GL / (HL + lam) + GR * GR / (HR + lam) - parent)
            gain = np.where(valid, gain, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best_gain:
                best_gain = float(gain[k])
                best = (int(f), k, best_gain)
        return best

    # ---------- prediction (vectorised routing) ----------
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.root is None:
            raise RuntimeError("GradientTree is not fitted")
        out = np.empty(X.shape[0], dtype=float)
        self._route(self.root, X, np.arange(X.shape[0]), out)
        return out

    def _route(self, node, X, idx, out):
        # Instead of walking the tree once per row, push the whole index array
        # down and split it at each node: one vectorised comparison per node.
        if node.feature is None:
            out[idx] = node.value
            return
        x = X[idx, node.feature]
        go_left = (x <= node.threshold) | np.isnan(x)      # missing goes left
        if go_left.any():
            self._route(node.left, X, idx[go_left], out)
        if (~go_left).any():
            self._route(node.right, X, idx[~go_left], out)

    def feature_importance(self, kind="gain"):
        v = self._gain_importance if kind == "gain" else self._split_counts
        s = v.sum()
        return v / s if s > 0 else v.copy()

    def depth(self):
        def d(node):
            return 0 if node is None or node.feature is None else 1 + max(d(node.left), d(node.right))
        return d(self.root)


def _resolve_max_features(spec, m):
    if spec is None:
        return None
    if spec == "sqrt":
        return max(1, int(np.sqrt(m)))
    if spec == "log2":
        return max(1, int(np.log2(m)))
    if isinstance(spec, float):
        return max(1, int(spec * m)) if spec < 1.0 else None
    return int(spec)


# ============================================================
# LOGISTIC REGRESSION (mini-batch SGD, L2, seeded, epoch-based)
# ============================================================

class LogisticRegression:
    """Binary logistic regression trained with epoch-based mini-batch SGD + L2.

    V5 changes: batches come from a seeded per-epoch permutation (each sample
    is visited exactly once per epoch) instead of unseeded sampling with
    replacement, and training stops early when the full-data loss plateaus.
    """

    def __init__(self, lr=0.05, n_epochs=30, lambda_reg=0.01, batch_size=256,
                 decay=1e-3, tol=1e-6, verbose=False, random_state=42):
        self.lr = lr
        self.n_epochs = n_epochs
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.decay = decay
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0
        self.losses = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, m = X.shape
        rng = np.random.RandomState(self.random_state)
        self.weights = np.zeros(m)
        self.bias = 0.0
        self.losses = []
        step = 0
        prev = np.inf
        for epoch in range(self.n_epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                p = sigmoid(X[idx] @ self.weights + self.bias)
                err = p - y[idx]
                dw = X[idx].T @ err / idx.size + self.lambda_reg * self.weights
                db = float(err.mean())
                cur_lr = self.lr / (1.0 + self.decay * step)   # Robbins-Monro decay
                self.weights -= cur_lr * dw
                self.bias -= cur_lr * db
                step += 1
            loss = logloss(y, self.predict_proba(X))
            self.losses.append(loss)
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"  LR epoch {epoch + 1}: loss={loss:.4f}")
            if abs(prev - loss) < self.tol:
                break
            prev = loss
        return self

    def predict_proba(self, X):
        return sigmoid(np.asarray(X, float) @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# GRADIENT BOOSTED TREES (Newton boosting on log-loss)
# ============================================================

class GradientBoostedTrees:
    """Second-order (Newton) gradient boosting for binary classification.

    Each round fits a GradientTree to g = p − y, h = p(1 − p); its leaves are
    the Newton steps −G/(H+λ), shrunk by the learning rate. Compared to V4's
    first-order mean-residual leaves this converges in fewer trees and the
    L2 term keeps tiny-hessian leaves (near-certain predictions) from
    producing huge weights. Row and column subsampling add the usual
    variance reduction; with a validation set, boosting stops when val
    log-loss hasn't improved for `early_stopping_rounds` and keeps the best
    iteration.
    """

    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=4,
                 min_samples_leaf=5, subsample=0.8, colsample=1.0, reg_lambda=1.0,
                 n_bins=32, early_stopping_rounds=None, verbose=False, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample = colsample
        self.reg_lambda = reg_lambda
        self.n_bins = n_bins
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.random_state = random_state
        self.trees = []
        self.initial_pred = 0.0
        self.train_losses = []
        self.val_losses = []
        self.best_iteration_ = None
        self.n_features_ = None

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, m = X.shape
        self.n_features_ = m
        rng = np.random.RandomState(self.random_state)
        edges, bins = GradientTree.bin_features(X, self.n_bins)   # once, for all trees

        # Start at the base-rate log-odds so tree #1 only explains deviations.
        p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        self.initial_pred = float(np.log(p0 / (1 - p0)))
        F = np.full(n, self.initial_pred)
        Fv = None
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)
            Fv = np.full(X_val.shape[0], self.initial_pred)

        mf = None if self.colsample >= 1.0 else max(1, int(self.colsample * m))
        n_sub = n if self.subsample >= 1.0 else max(1, int(n * self.subsample))
        self.trees, self.train_losses, self.val_losses = [], [], []
        best_val, best_iter = np.inf, 0

        for i in range(self.n_estimators):
            p = sigmoid(F)
            g = p - y                 # ∂logloss/∂F
            h = p * (1.0 - p)         # ∂²logloss/∂F²
            idx = rng.choice(n, n_sub, replace=False) if n_sub < n else np.arange(n)
            tree = GradientTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                max_features=mf, n_bins=self.n_bins, reg_lambda=self.reg_lambda,
                                random_state=_seed_from(rng))
            tree.fit_binned(bins[idx], edges, g[idx], h[idx])
            F += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            self.train_losses.append(logloss(y, sigmoid(F)))
            if Fv is not None:
                Fv += self.learning_rate * tree.predict(X_val)
                vl = logloss(y_val, sigmoid(Fv))
                self.val_losses.append(vl)
                if vl < best_val - 1e-9:
                    best_val, best_iter = vl, i + 1
                elif self.early_stopping_rounds and (i + 1 - best_iter) >= self.early_stopping_rounds:
                    if self.verbose:
                        print(f"  GBDT early stop at {i + 1}, best iteration {best_iter}")
                    self.trees = self.trees[:best_iter]
                    break
            if self.verbose and (i + 1) % 50 == 0:
                msg = f"  GBDT iter {i + 1}/{self.n_estimators}: train loss={self.train_losses[-1]:.4f}"
                if self.val_losses:
                    msg += f", val loss={self.val_losses[-1]:.4f}"
                print(msg)
        self.best_iteration_ = len(self.trees)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return sigmoid(F)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self, n_features=None, kind="gain"):
        """Gain-weighted (default) or split-count importance, normalised to 1.

        `n_features` is accepted for V4 API compatibility and validated only.
        """
        if n_features is not None and n_features != self.n_features_:
            raise ValueError("n_features does not match the fitted model")
        total = np.zeros(self.n_features_)
        for tree in self.trees:
            total += tree._gain_importance if kind == "gain" else tree._split_counts
        s = total.sum()
        return total / s if s > 0 else total


# ============================================================
# RANDOM FOREST (bagged gradient trees in regression mode)
# ============================================================

class RandomForest:
    """Bootstrap-aggregated trees with per-split feature subsampling.

    Each tree is a GradientTree in regression mode (g = −y, h = 1, λ = 0), so
    its leaves are class means and splits maximise variance reduction — the
    same criterion as Gini for a binary target, now via one shared, correct
    implementation. With `oob_score=True` every tree is also scored on the
    rows its bootstrap left out, giving a free validation AUC (`oob_auc_`).
    """

    def __init__(self, n_estimators=100, max_depth=8, min_samples_leaf=5,
                 max_features="sqrt", bootstrap=True, n_bins=32, oob_score=False,
                 verbose=False, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_bins = n_bins
        self.oob_score = oob_score
        self.verbose = verbose
        self.random_state = random_state
        self.trees = []
        self.oob_auc_ = None
        self.oob_prediction_ = None
        self.n_features_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, m = X.shape
        self.n_features_ = m
        rng = np.random.RandomState(self.random_state)
        edges, bins = GradientTree.bin_features(X, self.n_bins)
        mf = _resolve_max_features(self.max_features, m)
        g = -y
        h = np.ones(n)
        oob_sum = np.zeros(n)
        oob_cnt = np.zeros(n)
        self.trees = []
        for i in range(self.n_estimators):
            idx = rng.choice(n, n, replace=True) if self.bootstrap else np.arange(n)
            tree = GradientTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                max_features=mf, n_bins=self.n_bins, reg_lambda=0.0,
                                random_state=_seed_from(rng))
            tree.fit_binned(bins[idx], edges, g[idx], h[idx])
            self.trees.append(tree)
            if self.oob_score and self.bootstrap:
                in_bag = np.zeros(n, dtype=bool)
                in_bag[idx] = True
                oob = ~in_bag
                if oob.any():
                    oob_sum[oob] += tree.predict(X[oob])
                    oob_cnt[oob] += 1
            if self.verbose and (i + 1) % 25 == 0:
                print(f"  RF tree {i + 1}/{self.n_estimators} built")
        if self.oob_score and self.bootstrap:
            has = oob_cnt > 0
            self.oob_prediction_ = np.where(has, oob_sum / np.maximum(oob_cnt, 1), np.nan)
            if has.sum() > 1:
                self.oob_auc_ = roc_auc_score(y[has], self.oob_prediction_[has])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.zeros(X.shape[0])
        for tree in self.trees:
            preds += tree.predict(X)
        return np.clip(preds / len(self.trees), 0.0, 1.0)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self, n_features=None, kind="gain"):
        if n_features is not None and n_features != self.n_features_:
            raise ValueError("n_features does not match the fitted model")
        total = np.zeros(self.n_features_)
        for tree in self.trees:
            total += tree._gain_importance if kind == "gain" else tree._split_counts
        s = total.sum()
        return total / s if s > 0 else total


# ============================================================
# NEURAL NETWORK (MLP: He init, Adam, inverted dropout, early stopping)
# ============================================================

class NeuralNetwork:
    """Fully-reproducible 2-hidden-layer MLP for binary classification.

    Same recipe as V4 (He init, Adam, inverted dropout, early stopping with
    best-weight restore) with two fixes: dropout masks now come from the
    seeded RNG (V4 created an unseeded RandomState per forward pass), and
    when a validation set is supplied early stopping watches *validation*
    loss instead of training loss, which is what actually guards against
    memorisation.
    """

    def __init__(self, hidden_sizes=(64, 32), lr=0.001, n_epochs=200, batch_size=256,
                 lambda_reg=0.001, dropout_rate=0.3, patience=20, verbose=False, random_state=42):
        self.hidden_sizes = tuple(hidden_sizes)
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.dropout_rate = dropout_rate
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state
        self.weights = {}
        self.losses = []
        self.val_losses = []
        self.best_epoch_ = None

    def _init_weights(self, n_features, rng):
        sizes = [n_features] + list(self.hidden_sizes) + [1]
        self.weights = {}
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i])                       # He init for ReLU
            self.weights[f"W{i}"] = rng.randn(sizes[i], sizes[i + 1]) * scale
            self.weights[f"b{i}"] = np.zeros(sizes[i + 1])

    @staticmethod
    def _relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_deriv(z):
        return (z > 0).astype(float)

    def _forward(self, X, rng=None):
        """Forward pass. Dropout is applied only when an RNG is supplied (training)."""
        n_layers = len(self.hidden_sizes) + 1
        cache = {"A0": X}
        A = X
        for i in range(n_layers):
            Z = A @ self.weights[f"W{i}"] + self.weights[f"b{i}"]
            cache[f"Z{i + 1}"] = Z
            if i < n_layers - 1:
                A = self._relu(Z)
                if rng is not None and self.dropout_rate > 0:
                    # Inverted dropout: scale survivors by 1/(1-p) so inference
                    # (no dropout) needs no special-casing.
                    mask = (rng.rand(*A.shape) > self.dropout_rate).astype(float)
                    A = A * mask / (1.0 - self.dropout_rate)
                    cache[f"mask{i + 1}"] = mask
            else:
                A = sigmoid(Z)
            cache[f"A{i + 1}"] = A
        return A, cache

    def _backward(self, y, cache, n_layers):
        m = y.shape[0]
        grads = {}
        dA = cache[f"A{n_layers}"] - y.reshape(-1, 1)      # d(BCE∘sigmoid)/dZ
        for i in range(n_layers - 1, -1, -1):
            if i == n_layers - 1:
                dZ = dA
            else:
                dZ = dA * self._relu_deriv(cache[f"Z{i + 1}"])
                if f"mask{i + 1}" in cache:
                    dZ = dZ * cache[f"mask{i + 1}"] / (1.0 - self.dropout_rate)
            grads[f"dW{i}"] = cache[f"A{i}"].T @ dZ / m + self.lambda_reg * self.weights[f"W{i}"]
            grads[f"db{i}"] = dZ.sum(axis=0) / m
            if i > 0:
                dA = dZ @ self.weights[f"W{i}"].T
        return grads

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, m = X.shape
        rng = np.random.RandomState(self.random_state)
        self._init_weights(m, rng)
        n_layers = len(self.hidden_sizes) + 1
        m_state = {k: np.zeros_like(v) for k, v in self.weights.items()}
        v_state = {k: np.zeros_like(v) for k, v in self.weights.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        monitor_val = X_val is not None
        if monitor_val:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)
        self.losses, self.val_losses = [], []
        best_loss, best_weights, wait, t = np.inf, None, 0, 0
        for epoch in range(self.n_epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                _, cache = self._forward(X[idx], rng=rng)
                grads = self._backward(y[idx], cache, n_layers)
                t += 1
                for key in self.weights:
                    gk = grads[f"d{key}"]
                    m_state[key] = beta1 * m_state[key] + (1 - beta1) * gk
                    v_state[key] = beta2 * v_state[key] + (1 - beta2) * gk * gk
                    m_hat = m_state[key] / (1 - beta1 ** t)
                    v_hat = v_state[key] / (1 - beta2 ** t)
                    self.weights[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
            train_loss = logloss(y, self.predict_proba(X))
            self.losses.append(train_loss)
            watched = train_loss
            if monitor_val:
                val_loss = logloss(y_val, self.predict_proba(X_val))
                self.val_losses.append(val_loss)
                watched = val_loss
            if watched < best_loss - 1e-6:
                best_loss, wait = watched, 0
                best_weights = {k: v.copy() for k, v in self.weights.items()}
                self.best_epoch_ = epoch + 1
            else:
                wait += 1
                if wait >= self.patience:
                    if self.verbose:
                        print(f"  NN early stop at epoch {epoch + 1} (best {self.best_epoch_})")
                    break
            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"  NN epoch {epoch + 1}: loss={train_loss:.4f}")
        if best_weights is not None:
            self.weights = best_weights
        return self

    def predict_proba(self, X):
        out, _ = self._forward(np.asarray(X, dtype=float), rng=None)
        return out.ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# STACKING ENSEMBLE (clone-per-fold, out-of-fold meta-features)
# ============================================================

class StackingEnsemble:
    """Meta-learner over base-model probabilities, done without side effects.

    Base models are passed as unfitted templates. For each fold a deep copy
    is trained on the other folds and scored on the held-out one, so the
    meta-features are genuinely out-of-fold; the caller's objects are never
    touched. Folds are contiguous (not shuffled) so the scheme stays sane on
    time-ordered data. Final base models are fresh copies refit on all rows.

    The meta-learner sees base predictions in LOGIT space (log-odds), not raw
    probabilities. A logistic meta-model is then a linear blend of log-odds —
    the natural way to combine probabilistic classifiers — and it can scale
    the blend freely, so the stack is at least as well calibrated as its
    inputs. (Combining raw probabilities with a strongly-regularised LR, as
    V4 did, keeps the ranking but compresses outputs toward 0.5 — good AUC,
    poor Brier.)
    """

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p))

    def __init__(self, base_models, meta_model=None, n_folds=3, random_state=42):
        self.base_models = list(base_models)              # list of (name, template)
        self.meta_model = meta_model or LogisticRegression(lr=0.1, n_epochs=100, lambda_reg=1e-4,
                                                           random_state=random_state)
        self.n_folds = n_folds
        self.random_state = random_state
        self.fitted_base_ = []
        self.is_fitted = False

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        k = len(self.base_models)
        if X_val is not None:
            fitted = [(name, copy.deepcopy(tmpl).fit(X, y)) for name, tmpl in self.base_models]
            meta = np.column_stack([self._logit(mdl.predict_proba(X_val)) for _, mdl in fitted])
            self.meta_model.fit(meta, np.asarray(y_val, float))
            self.fitted_base_ = fitted
        else:
            meta = np.zeros((n, k))
            bounds = np.linspace(0, n, self.n_folds + 1).astype(int)
            for f in range(self.n_folds):
                lo, hi = bounds[f], bounds[f + 1]
                val_idx = np.arange(lo, hi)
                train_idx = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
                for j, (name, tmpl) in enumerate(self.base_models):
                    mdl = copy.deepcopy(tmpl).fit(X[train_idx], y[train_idx])
                    meta[val_idx, j] = self._logit(mdl.predict_proba(X[val_idx]))
            self.fitted_base_ = [(name, copy.deepcopy(tmpl).fit(X, y)) for name, tmpl in self.base_models]
            self.meta_model.fit(meta, y)
        self.is_fitted = True
        return self

    def _meta_features(self, X):
        return np.column_stack([self._logit(mdl.predict_proba(X)) for _, mdl in self.fitted_base_])

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble is not fitted")
        return self.meta_model.predict_proba(self._meta_features(np.asarray(X, float)))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_base_predictions(self, X):
        return {name: mdl.predict_proba(np.asarray(X, float)) for name, mdl in self.fitted_base_}


# ============================================================
# WALK-FORWARD CROSS-VALIDATION
# ============================================================

def walk_forward_cv(X, y, dates, model_factory, n_splits=5, min_train_size=1000):
    """Time-series CV: train on everything before a slice, test on the slice.

    Shuffled k-fold would train on later games to score earlier ones — the
    same leakage `train_test_split_temporal` avoids. Standardisation is refit
    inside each fold on that fold's training rows only, for the same reason.
    Returns {metric: (mean, std)} across folds.
    """
    order = np.argsort(dates, kind="mergesort")
    X, y, dates = X[order], y[order], dates[order]
    n = len(y)
    results = {"auc": [], "brier": [], "logloss": [], "accuracy": []}
    fold = (n - min_train_size) // n_splits
    for i in range(n_splits):
        tr_end = min_train_size + i * fold
        te_end = min(tr_end + fold, n)
        if tr_end >= n or te_end <= tr_end:
            break
        Xtr, ytr, Xte, yte = X[:tr_end], y[:tr_end], X[tr_end:te_end], y[tr_end:te_end]
        if len(np.unique(yte)) < 2:
            continue
        Xtr_s, Xte_s, _, _ = standardize(Xtr, Xte)
        Xtr_s, Xte_s = np.nan_to_num(Xtr_s, nan=0.0), np.nan_to_num(Xte_s, nan=0.0)
        model = model_factory().fit(Xtr_s, ytr)
        p = model.predict_proba(Xte_s)
        results["auc"].append(roc_auc_score(yte, p))
        results["brier"].append(brier_score(yte, p))
        results["logloss"].append(logloss(yte, p))
        results["accuracy"].append(accuracy_score(yte, p))
    return {k: (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan) for k, v in results.items()}


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":
    rng = np.random.RandomState(42)
    X = rng.randn(600, 6)
    y = ((X[:, 0] + 0.5 * X[:, 1] + 0.8 * (X[:, 2] * X[:, 3] > 0) + rng.randn(600) * 0.5) > 0).astype(float)
    Xtr, Xte, ytr, yte = X[:450], X[450:], y[:450], y[450:]

    def report(name, model):
        p = model.fit(Xtr, ytr).predict_proba(Xte)
        print(f"  {name:<22} AUC={roc_auc_score(yte, p):.3f}  Brier={brier_score(yte, p):.3f}")

    print("V5 smoke test")
    report("LogisticRegression", LogisticRegression(n_epochs=40))
    report("GradientBoostedTrees", GradientBoostedTrees(n_estimators=80, max_depth=3))
    report("RandomForest", RandomForest(n_estimators=40, max_depth=6))
    report("NeuralNetwork", NeuralNetwork(hidden_sizes=(32, 16), n_epochs=80, lr=0.01))
    report("StackingEnsemble", StackingEnsemble([("lr", LogisticRegression(n_epochs=30)),
                                                  ("gbdt", GradientBoostedTrees(n_estimators=50, max_depth=3))]))
    a = GradientBoostedTrees(n_estimators=30, random_state=7).fit(Xtr, ytr).predict_proba(Xte)
    b = GradientBoostedTrees(n_estimators=30, random_state=7).fit(Xtr, ytr).predict_proba(Xte)
    print(f"  reproducible (same seed -> identical preds): {np.array_equal(a, b)}")
    print("All V5 models working!")
