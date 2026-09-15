"""
Machine-learning algorithms implemented from scratch with only NumPy.
Core library for the LoL Esports Match Prediction Model.

Why from scratch? The training environment had no package internet access
(a proxy blocked `pip install`), so scikit-learn/XGBoost weren't available.
Rather than treat that as a blocker, I implemented every model by hand — which
also forced me to actually understand the mechanics rather than call `.fit()`
on a black box.

Design principle followed throughout: every estimator exposes the same small,
scikit-learn-like surface — `fit(X, y)`, `predict_proba(X)`, `predict(X)` — so
the training/evaluation scripts can treat all models interchangeably (this is
what makes the model-comparison loop and the StackingEnsemble trivial to write).

Includes:
- Logistic Regression with L2 regularization (mini-batch SGD, decaying LR)
- Decision Tree (CART, Gini impurity)
- Random Forest (bagging + feature subsampling)
- Gradient-Boosted Decision Trees (log-loss gradient boosting)
- Neural Network (2-hidden-layer MLP: He init, Adam, dropout, early stopping)
- Stacking Ensemble (out-of-fold meta-features)
- Evaluation metrics (AUC, Brier, Log Loss, calibration) + temporal CV helpers
"""

import numpy as np

# NOTE: we deliberately do NOT blanket-suppress warnings here. The numerical
# hot spots below (sigmoid, log-loss, standardization) each guard against their
# own overflow/underflow/divide-by-zero explicitly, so warnings that do surface
# are real signal worth seeing rather than noise to hide.


# ============================================================
# UTILITIES
# ============================================================

def sigmoid(z):
    # Clip before exp(): np.exp(-z) overflows to inf for z < ~-709, which would
    # produce NaN probabilities and silently poison every downstream metric.
    # Clipping at +/-500 keeps us safely inside float64 range while leaving the
    # output indistinguishable from 0 or 1 at the extremes.
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def logloss(y_true, y_pred):
    # log(0) = -inf. A single prediction of exactly 0.0 or 1.0 that's wrong
    # would make the whole loss infinite, so clip into the open interval (0, 1).
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def brier_score(y_true, y_pred):
    # Brier = mean squared error of the probabilities. I track it alongside AUC
    # because AUC only measures *ranking*; Brier also punishes miscalibration
    # (e.g. a model that's confidently wrong). Lower is better.
    return np.mean((y_true - y_pred) ** 2)

def roc_auc_score(y_true, y_pred):
    """Compute ROC AUC using the rank-based (Mann-Whitney U) method.

    Why rank-based instead of sweeping thresholds and integrating the curve?
    AUC is exactly the probability that a random positive is ranked above a
    random negative, which the Mann-Whitney U statistic gives in closed form.
    It's exact (no trapezoid approximation error) and O(n log n) from one sort.
    Ties must be handled with average ranks or AUC is biased when the model
    outputs repeated probabilities (common with tree ensembles).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Rank all predictions (ascending). Average ties.
    order = np.argsort(y_pred)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_pred) + 1, dtype=float)
    # Handle ties by averaging ranks
    sorted_pred = y_pred[order]
    i = 0
    while i < len(sorted_pred):
        j = i
        while j < len(sorted_pred) and sorted_pred[j] == sorted_pred[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    # AUC = (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    rank_sum = np.sum(ranks[y_true == 1])
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc

def accuracy_score(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return np.mean(y_true == y_pred)

def calibration_bins(y_true, y_pred, n_bins=10):
    """Return bin midpoints, fraction of positives, and bin counts."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_mids = []
    bin_true_fracs = []
    bin_counts = []
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
        if i == n_bins - 1:
            mask = (y_pred >= bins[i]) & (y_pred <= bins[i+1])
        count = np.sum(mask)
        if count > 0:
            bin_mids.append((bins[i] + bins[i+1]) / 2)
            bin_true_fracs.append(np.mean(y_true[mask]))
            bin_counts.append(count)
    return np.array(bin_mids), np.array(bin_true_fracs), np.array(bin_counts)

def train_test_split_temporal(X, y, dates, test_frac=0.2):
    """Split data temporally — most recent games as test.

    This is the single most important methodological choice in the project.
    A random shuffle-split would let the model "see the future": a game from
    June could inform a prediction about a game from March, and rolling-window
    features (Elo, recent win rate) for a March game are computed from data that
    a random split might also place in the test set. That leaks and inflates
    every metric. Ordering by date and testing only on the most recent slice
    mirrors how the model would actually be used — predicting games that
    haven't happened yet from games that have.
    """
    order = np.argsort(dates)
    X, y, dates = X[order], y[order], dates[order]
    split_idx = int(len(y) * (1 - test_frac))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:], dates[:split_idx], dates[split_idx:]

def standardize(X_train, X_test=None):
    """Z-score standardization.

    mu/sigma are computed from the TRAINING set only and then applied to the
    test set — fitting the scaler on test data would be another leakage path.
    The returned mu/sigma are also what the React app needs to reproduce the
    model's input space, which is why they get saved to model_params.npz.
    """
    mu = np.nanmean(X_train, axis=0)
    sigma = np.nanstd(X_train, axis=0)
    # A constant feature has sigma == 0, which would divide by zero and turn the
    # whole column into NaN/inf. Setting sigma to 1 leaves such columns as
    # (x - mu) = 0, i.e. harmlessly uninformative, instead of poisoning the row.
    sigma[sigma < 1e-8] = 1.0
    X_train_s = (X_train - mu) / sigma
    if X_test is not None:
        X_test_s = (X_test - mu) / sigma
        return X_train_s, X_test_s, mu, sigma
    return X_train_s, mu, sigma


# ============================================================
# LOGISTIC REGRESSION (with L2 regularization, mini-batch SGD)
# ============================================================

class LogisticRegression:
    """Binary logistic regression trained with mini-batch SGD + L2 penalty.

    Thought process: this is the baseline every other model has to beat, so it
    should be simple and hard to get wrong. Mini-batches (rather than full-batch
    gradient descent) keep each step cheap on 50k rows and add a little noise
    that helps escape flat regions; the L2 term keeps weights small so that the
    130 correlated features (many are `_a` / `_b` / `_diff` triples of the same
    quantity) don't blow up into huge offsetting coefficients.
    """

    def __init__(self, lr=0.01, n_iters=1000, lambda_reg=0.01, batch_size=256, verbose=False):
        self.lr = lr
        self.n_iters = n_iters
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.losses = []

        for i in range(self.n_iters):
            # Mini-batch
            idx = np.random.choice(n_samples, min(self.batch_size, n_samples), replace=False)
            X_batch = X[idx]
            y_batch = y[idx]

            z = X_batch @ self.weights + self.bias
            y_hat = sigmoid(z)

            # Gradients
            error = y_hat - y_batch
            dw = (1 / len(y_batch)) * (X_batch.T @ error) + self.lambda_reg * self.weights
            db = (1 / len(y_batch)) * np.sum(error)

            # Learning-rate decay: start aggressive to make fast early progress,
            # then shrink so the noisy mini-batch gradients settle near the
            # optimum instead of bouncing around it. 1/(1+k*i) is the classic
            # Robbins-Monro schedule — simple and it converges in practice here.
            current_lr = self.lr / (1 + 0.001 * i)
            self.weights -= current_lr * dw
            self.bias -= current_lr * db

            if i % 100 == 0:
                loss = logloss(y, sigmoid(X @ self.weights + self.bias))
                self.losses.append(loss)
                if self.verbose and i % 500 == 0:
                    print(f"  LR iter {i}: loss={loss:.4f}")

        return self

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# DECISION TREE (CART for binary classification)
# ============================================================

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # predicted probability if leaf

class DecisionTree:
    def __init__(self, max_depth=6, min_samples_split=10, min_samples_leaf=5,
                 max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.rng = np.random.RandomState(random_state)

    def _gini(self, y):
        # Gini impurity for a binary target: 2p(1-p). It's 0 for a pure node
        # and peaks at 0.5 for a 50/50 split. Chosen over entropy because it's
        # cheaper (no log) and gives near-identical trees in practice.
        #
        # Design note: this same tree is reused as the base learner inside
        # GradientBoostedTrees, where `y` is a vector of pseudo-residuals rather
        # than 0/1 labels. Gini then acts as a heuristic split score on the
        # residual mean instead of a true variance-reduction criterion. It works
        # well empirically (it's the criterion behind the reported V4 results),
        # but a purpose-built regression tree using variance reduction is the
        # more principled choice — that's addressed in the V5 rewrite.
        if len(y) == 0:
            return 0
        p = np.mean(y)
        return 2 * p * (1 - p)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None

        parent_gini = self._gini(y)

        # Feature subsetting: when max_features is set (Random Forest), each
        # split only considers a random subset of columns. This decorrelates the
        # trees — without it every tree would grab the same few dominant
        # features first and the ensemble would average nearly identical models.
        if self.max_features is not None:
            feature_indices = self.rng.choice(n_features, min(self.max_features, n_features), replace=False)
        else:
            feature_indices = np.arange(n_features)

        for feat_idx in feature_indices:
            col = X[:, feat_idx]
            # Candidate thresholds: trying every unique value is O(n) per feature
            # per node, which is far too slow for 50k rows x 130 features in pure
            # Python. For continuous features I instead try 20 evenly spaced
            # percentiles (5th..95th) — that captures the useful split points
            # while making the whole tree build ~100x faster. Low-cardinality
            # (<=20 unique) features still get exact thresholds.
            unique_vals = np.unique(col)
            if len(unique_vals) <= 20:
                thresholds = unique_vals
            else:
                thresholds = np.percentile(col, np.linspace(5, 95, 20))
                thresholds = np.unique(thresholds)

            for threshold in thresholds:
                left_mask = col <= threshold
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                weighted_gini = (n_left * gini_left + n_right * gini_right) / n_samples
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)

        # Stopping conditions
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return TreeNode(value=np.mean(y) if len(y) > 0 else 0.5)

        feature, threshold, gain = self._best_split(X, y)
        if feature is None or gain <= 0:
            return TreeNode(value=np.mean(y))

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature=feature, threshold=threshold, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        return self

    def _predict_single(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict_proba(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# GRADIENT BOOSTED DECISION TREES
# ============================================================

class GradientBoostedTrees:
    """Gradient boosting for binary classification (log-loss objective).

    Thought process: boosting builds an additive model where each new tree is
    fit to the *errors* of everything before it, so it can carve out the
    interactions that a linear model can't (e.g. "new patch AND low patch-reps
    AND traveling"). It ended up as the best model in V3 and V4. Shallow trees
    (max_depth=4) + a small learning rate + row subsampling are the three knobs
    that keep it from overfitting 50k rows.
    """

    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=4,
                 min_samples_split=10, min_samples_leaf=5, subsample=0.8,
                 max_features=None, verbose=False, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.verbose = verbose
        self.random_state = random_state
        self.trees = []
        self.initial_pred = None
        self.train_losses = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        rng = np.random.RandomState(self.random_state)

        # Initialize every sample at the base-rate log-odds. Starting from the
        # prior (rather than 0) means tree #1 only has to explain deviations
        # from the class balance, which converges faster and more stably.
        p = np.mean(y)
        self.initial_pred = np.log(p / (1 - p + 1e-15))
        F = np.full(n_samples, self.initial_pred)

        for i in range(self.n_estimators):
            # Pseudo-residuals = negative gradient of log-loss w.r.t. F, which
            # for the logistic link works out to the beautifully simple (y - p).
            # Each tree is fit to these, i.e. to "what the model still gets
            # wrong, and in which direction".
            p_hat = sigmoid(F)
            residuals = y - p_hat

            # Subsample
            if self.subsample < 1.0:
                idx = rng.choice(n_samples, int(n_samples * self.subsample), replace=False)
            else:
                idx = np.arange(n_samples)

            # Fit regression tree to residuals
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=rng.randint(0, 100000)
            )
            tree.fit(X[idx], residuals[idx])

            # Add the tree's contribution, shrunk by the learning rate.
            # Each leaf predicts the mean residual of the samples that land in
            # it, so this is standard first-order gradient boosting (Friedman,
            # 2001). Note: it is NOT a Newton step — a true Newton/XGBoost-style
            # leaf would divide by the summed Hessian p(1-p) to get the optimal
            # leaf weight. The learning rate ("shrinkage") is what makes boosting
            # robust: many small corrections generalize far better than a few
            # large ones. Implementing proper Newton leaves is part of V5.
            predictions = tree.predict_proba(X)
            F += self.learning_rate * predictions
            self.trees.append(tree)

            if self.verbose and (i + 1) % 50 == 0:
                loss = logloss(y, sigmoid(F))
                self.train_losses.append(loss)
                print(f"  GBDT iter {i+1}/{self.n_estimators}: loss={loss:.4f}")

        return self

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            F += self.learning_rate * tree.predict_proba(X)
        return sigmoid(F)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self, n_features):
        """Compute simple feature importance based on split frequency.

        This counts how often each feature is chosen for a split across all
        trees and normalizes to sum to 1. It's the simplest defensible
        importance measure and is what produced results/feature_importance.csv.
        Its limitation: a feature split on many times with tiny gains ranks the
        same as one split on rarely with huge gains. A gain-weighted version
        (sum of impurity reduction per split) is more informative and is what
        V5 implements.
        """
        importance = np.zeros(n_features)

        def _traverse(node):
            if node is None or node.value is not None:
                return
            importance[node.feature] += 1
            _traverse(node.left)
            _traverse(node.right)

        for tree in self.trees:
            _traverse(tree.root)

        if importance.sum() > 0:
            importance = importance / importance.sum()
        return importance


# ============================================================
# RANDOM FOREST
# ============================================================

class RandomForest:
    """Bagged decision trees with per-split feature subsampling.

    Thought process: a single deep tree overfits badly; averaging many trees
    that each saw a different bootstrap sample AND a different random feature
    subset at every split cancels out their individual quirks (variance
    reduction). It's the "cheap and robust" ensemble — deeper trees than GBDT
    (max_depth=8) because bagging tolerates overfit base learners in a way
    boosting does not.
    """

    def __init__(self, n_estimators=100, max_depth=8, min_samples_split=10,
                 min_samples_leaf=5, max_features='sqrt', bootstrap=True,
                 verbose=False, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.verbose = verbose
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        if self.max_features == 'sqrt':
            mf = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            mf = int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            mf = int(self.max_features * n_features)
        else:
            mf = self.max_features or n_features

        for i in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.choice(n_samples, n_samples, replace=True)
            else:
                idx = np.arange(n_samples)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mf,
                random_state=rng.randint(0, 100000)
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

            if self.verbose and (i + 1) % 25 == 0:
                print(f"  RF tree {i+1}/{self.n_estimators} built")

        return self

    def predict_proba(self, X):
        preds = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(preds, axis=0)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# NEURAL NETWORK (MLP with 2 hidden layers)
# ============================================================

class NeuralNetwork:
    """2-hidden-layer MLP for binary classification, written from scratch.

    Thought process: the NN was included to test whether a flexible non-linear
    learner could beat the tree ensembles on this tabular data. It didn't (trees
    won every version), which matches the usual finding that GBDTs dominate on
    small/medium tabular problems. The engineering choices — He init, Adam,
    inverted dropout, early stopping on the best epoch — are the standard
    recipe for making a small MLP train reliably without a framework.
    """

    def __init__(self, hidden_sizes=(64, 32), lr=0.001, n_epochs=200,
                 batch_size=256, lambda_reg=0.001, dropout_rate=0.3,
                 verbose=False, random_state=42):
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.dropout_rate = dropout_rate
        self.verbose = verbose
        self.random_state = random_state
        self.weights = {}
        self.losses = []

    def _init_weights(self, n_features):
        rng = np.random.RandomState(self.random_state)
        layer_sizes = [n_features] + list(self.hidden_sizes) + [1]
        self.weights = {}
        for i in range(len(layer_sizes) - 1):
            # He initialization
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.weights[f'W{i}'] = rng.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            self.weights[f'b{i}'] = np.zeros(layer_sizes[i+1])

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_deriv(self, z):
        return (z > 0).astype(float)

    def _forward(self, X, training=False):
        # Dropout uses an UNSEEDED RandomState here, so dropout masks differ
        # run-to-run even when random_state is set — the NN is therefore not
        # bit-for-bit reproducible. Weight init and batch order ARE seeded.
        # Left as-is on this branch to keep behaviour identical to the reported
        # runs (the NN was never a headline model); V5 threads the seeded RNG
        # through so the whole network is reproducible.
        rng = np.random.RandomState()
        cache = {'A0': X}
        A = X
        n_layers = len(self.hidden_sizes) + 1

        for i in range(n_layers):
            Z = A @ self.weights[f'W{i}'] + self.weights[f'b{i}']
            cache[f'Z{i+1}'] = Z

            if i < n_layers - 1:
                # Hidden layer: ReLU + dropout
                A = self._relu(Z)
                if training and self.dropout_rate > 0:
                    # "Inverted" dropout: scale the surviving activations up by
                    # 1/(1-p) at train time so their expected magnitude matches
                    # inference, where dropout is off. That way predict_proba
                    # needs no special-casing — a classic source of subtle bugs.
                    mask = (rng.rand(*A.shape) > self.dropout_rate).astype(float)
                    A = A * mask / (1 - self.dropout_rate)
                    cache[f'mask{i+1}'] = mask
            else:
                # Output layer: sigmoid
                A = sigmoid(Z)

            cache[f'A{i+1}'] = A

        return A, cache

    def _backward(self, y, cache, n_layers):
        m = len(y)
        grads = {}

        # Output layer
        dA = cache[f'A{n_layers}'] - y.reshape(-1, 1)  # derivative of BCE + sigmoid

        for i in range(n_layers - 1, -1, -1):
            if i == n_layers - 1:
                dZ = dA
            else:
                dZ = dA * self._relu_deriv(cache[f'Z{i+1}'])
                if f'mask{i+1}' in cache:
                    dZ = dZ * cache[f'mask{i+1}'] / (1 - self.dropout_rate)

            grads[f'dW{i}'] = (1/m) * (cache[f'A{i}'].T @ dZ) + self.lambda_reg * self.weights[f'W{i}']
            grads[f'db{i}'] = (1/m) * np.sum(dZ, axis=0)

            if i > 0:
                dA = dZ @ self.weights[f'W{i}'].T

        return grads

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._init_weights(n_features)
        n_layers = len(self.hidden_sizes) + 1
        rng = np.random.RandomState(self.random_state)

        # Adam optimizer state. Plain SGD was fragile here — 130 standardized
        # features with very different gradient scales meant one global learning
        # rate was either too slow for some weights or divergent for others.
        # Adam keeps a per-parameter running mean (m) and variance (v) of the
        # gradient and normalizes each update by them, which made training
        # stable without hand-tuning the LR per layer.
        m_state = {k: np.zeros_like(v) for k, v in self.weights.items()}
        v_state = {k: np.zeros_like(v) for k, v in self.weights.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        self.losses = []
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.n_epochs):
            indices = rng.permutation(n_samples)
            epoch_loss = 0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                idx = indices[start:end]
                X_batch = X[idx]
                y_batch = y[idx]

                # Forward
                y_hat, cache = self._forward(X_batch, training=True)

                # Backward
                grads = self._backward(y_batch, cache, n_layers)

                # Adam update
                t = epoch * (n_samples // self.batch_size) + n_batches + 1
                for key in self.weights:
                    g = grads[f'd{key}']
                    m_state[key] = beta1 * m_state[key] + (1 - beta1) * g
                    v_state[key] = beta2 * v_state[key] + (1 - beta2) * g**2
                    m_hat = m_state[key] / (1 - beta1**t)
                    v_hat = v_state[key] / (1 - beta2**t)
                    self.weights[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

                n_batches += 1

            # Compute epoch loss
            y_pred_full = self.predict_proba(X)
            loss = logloss(y, y_pred_full)
            self.losses.append(loss)

            # Early stopping with best-weight restore. Training loss keeps
            # falling long after the network starts memorizing, so we snapshot
            # the weights at the best epoch and roll back to them at the end
            # rather than keeping whatever the final (overfit) epoch produced.
            # Patience of 20 epochs avoids stopping on a single noisy plateau.
            if loss < best_loss - 1e-5:
                best_loss = loss
                patience_counter = 0
                self._best_weights = {k: v.copy() for k, v in self.weights.items()}
            else:
                patience_counter += 1

            if patience_counter >= 20:
                if self.verbose:
                    print(f"  NN early stop at epoch {epoch+1}")
                break

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"  NN epoch {epoch+1}/{self.n_epochs}: loss={loss:.4f}")

        # Restore best weights
        if hasattr(self, '_best_weights'):
            self.weights = self._best_weights

        return self

    def predict_proba(self, X):
        y_hat, _ = self._forward(X, training=False)
        return y_hat.flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# STACKING ENSEMBLE
# ============================================================

class StackingEnsemble:
    """Stacked generalization: a meta-learner over the base models' outputs.

    Thought process: different models make different mistakes (LR is well
    calibrated but linear; GBDT captures interactions but can be overconfident).
    Feeding their probabilities into a small logistic regression lets the data
    decide how much to trust each one. The crucial detail is that the
    meta-features must be OUT-OF-FOLD predictions — if a base model is scored
    on the same rows it trained on, the meta-learner just learns "trust the
    most overfit model", which is exactly wrong.

    Implementation note (known limitation): the k-fold path re-fits the SAME
    model instances across folds and then again on the full data, rather than
    cloning a fresh instance per fold. It works because every model's fit()
    fully re-initializes its state, but it mutates the caller's objects and
    relies on that re-init behaviour. V5 clones per fold.
    """

    def __init__(self, base_models, meta_model=None):
        """
        base_models: list of (name, model) tuples
        meta_model: model for combining predictions (default: LogisticRegression)
        """
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(lr=0.05, n_iters=500, lambda_reg=0.1)
        self.is_fitted = False

    def fit(self, X, y, X_val=None, y_val=None):
        """
        If X_val is provided, fit base models on X and generate meta-features from X_val.
        Otherwise, use 3-fold CV to generate meta-features.
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models)

        if X_val is not None:
            # Fit all base models on full training data
            for name, model in self.base_models:
                print(f"  Fitting {name}...")
                model.fit(X, y)

            # Generate meta-features from validation set
            meta_features = np.zeros((X_val.shape[0], n_models))
            for i, (name, model) in enumerate(self.base_models):
                meta_features[:, i] = model.predict_proba(X_val)

            # Fit meta-model
            print("  Fitting meta-learner...")
            self.meta_model.fit(meta_features, y_val)
        else:
            # K-fold cross-validation for meta-features
            k = 3
            fold_size = n_samples // k
            meta_features = np.zeros((n_samples, n_models))
            indices = np.arange(n_samples)

            for fold in range(k):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < k - 1 else n_samples
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

                for i, (name, model) in enumerate(self.base_models):
                    # Create a fresh copy-like approach by re-fitting
                    model.fit(X[train_idx], y[train_idx])
                    meta_features[val_idx, i] = model.predict_proba(X[val_idx])

            # Refit all base models on full data
            for name, model in self.base_models:
                model.fit(X, y)

            # Fit meta-model
            self.meta_model.fit(meta_features, y)

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            meta_features[:, i] = model.predict_proba(X)
        return self.meta_model.predict_proba(meta_features)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_base_predictions(self, X):
        """Get individual base model predictions."""
        result = {}
        for name, model in self.base_models:
            result[name] = model.predict_proba(X)
        return result


# ============================================================
# WALK-FORWARD CROSS-VALIDATION
# ============================================================

def walk_forward_cv(X, y, dates, model_factory, n_splits=5, min_train_size=1000):
    """
    Time-series cross-validation where training always precedes test.
    model_factory: callable that returns a fresh model instance

    Why not ordinary k-fold? Shuffled folds would train on 2025 games to score
    2022 games — the same future-leakage problem train_test_split_temporal
    avoids. Walk-forward instead grows the training window forward in time and
    evaluates on the slice immediately after it, for several such slices. The
    mean +/- std across folds tells us whether the headline test-set number is
    stable across eras (patch metas, roster shuffles) or a lucky split.
    Standardization is re-fit inside each fold, on that fold's training rows
    only, for the same reason.
    """
    order = np.argsort(dates)
    X, y, dates = X[order], y[order], dates[order]
    n_samples = len(y)

    results = {'auc': [], 'brier': [], 'logloss': [], 'accuracy': []}

    fold_size = (n_samples - min_train_size) // n_splits

    for i in range(n_splits):
        train_end = min_train_size + i * fold_size
        test_end = min(train_end + fold_size, n_samples)

        if train_end >= n_samples or test_end <= train_end:
            break

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]

        if len(np.unique(y_test)) < 2:
            continue

        model = model_factory()

        # Standardize
        X_tr_s, X_te_s, _, _ = standardize(X_train, X_test)
        X_tr_s = np.nan_to_num(X_tr_s, 0)
        X_te_s = np.nan_to_num(X_te_s, 0)

        model.fit(X_tr_s, y_train)
        y_pred = model.predict_proba(X_te_s)

        results['auc'].append(roc_auc_score(y_test, y_pred))
        results['brier'].append(brier_score(y_test, y_pred))
        results['logloss'].append(logloss(y_test, y_pred))
        results['accuracy'].append(accuracy_score(y_test, y_pred))

    return {k: (np.mean(v), np.std(v)) for k, v in results.items()}


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(500) * 0.5 > 0).astype(float)

    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    print("Testing Logistic Regression...")
    lr = LogisticRegression(lr=0.1, n_iters=500)
    lr.fit(X_train, y_train)
    p = lr.predict_proba(X_test)
    print(f"  AUC: {roc_auc_score(y_test, p):.3f}, Brier: {brier_score(y_test, p):.3f}")

    print("Testing GBDT...")
    gb = GradientBoostedTrees(n_estimators=50, max_depth=3, learning_rate=0.1)
    gb.fit(X_train, y_train)
    p = gb.predict_proba(X_test)
    print(f"  AUC: {roc_auc_score(y_test, p):.3f}, Brier: {brier_score(y_test, p):.3f}")

    print("Testing Random Forest...")
    rf = RandomForest(n_estimators=30, max_depth=5)
    rf.fit(X_train, y_train)
    p = rf.predict_proba(X_test)
    print(f"  AUC: {roc_auc_score(y_test, p):.3f}, Brier: {brier_score(y_test, p):.3f}")

    print("Testing Neural Network...")
    nn = NeuralNetwork(hidden_sizes=(32, 16), n_epochs=100, lr=0.01)
    nn.fit(X_train, y_train)
    p = nn.predict_proba(X_test)
    print(f"  AUC: {roc_auc_score(y_test, p):.3f}, Brier: {brier_score(y_test, p):.3f}")

    print("\nAll models working!")
