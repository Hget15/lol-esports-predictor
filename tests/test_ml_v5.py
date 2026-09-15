"""
Unit tests for ml_from_scratch_v5 (stdlib unittest — no extra dependencies).

Run from the repo root:
    python -m unittest tests.test_ml_v5 -v
or directly:
    python tests/test_ml_v5.py

What is covered and why:
  * metrics      — AUC checked against a brute-force pairwise definition
                   (with ties), edge cases, NaN rejection, finite extremes
  * tree         — leaf = mean in regression mode, a perfect 1-D split is
                   found, binning contract (NaN -> bin 0, bin = #edges < x)
  * learning     — every model clears an AUC bar on synthetic data with a
                   known non-linear interaction; early stopping keeps the
                   best iteration; RF OOB scoring works
  * safety       — every model is bit-for-bit reproducible under a seed
                   (including dropout), different seeds differ, and stacking
                   never mutates the caller's template models
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ml_from_scratch_v5 as v5  # noqa: E402


def make_data(n=800, seed=0):
    """Linear signal on x0, x1 plus an interaction on x2*x3 — trees must earn their keep."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6)
    logit = 1.5 * X[:, 0] + 0.8 * X[:, 1] + 1.2 * (X[:, 2] * X[:, 3] > 0) + rng.randn(n) * 0.4
    return X, (logit > 0).astype(float)


class TestMetrics(unittest.TestCase):
    def test_auc_matches_bruteforce_with_ties(self):
        rng = np.random.RandomState(1)
        y = rng.randint(0, 2, 200)
        p = np.round(rng.rand(200), 1)                     # heavy ties on purpose
        pos, neg = p[y == 1], p[y == 0]
        brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
        self.assertAlmostEqual(v5.roc_auc_score(y, p), brute, places=12)

    def test_auc_perfect_inverted_and_degenerate(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(v5.roc_auc_score(y, [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(v5.roc_auc_score(y, [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertEqual(v5.roc_auc_score(np.ones(4), [0.1, 0.2, 0.3, 0.4]), 0.5)

    def test_auc_rejects_nan(self):
        with self.assertRaises(ValueError):
            v5.roc_auc_score([0, 1], [0.1, np.nan])

    def test_sigmoid_and_logloss_are_finite_at_extremes(self):
        self.assertTrue(np.all(np.isfinite(v5.sigmoid(np.array([-1e6, 0.0, 1e6])))))
        self.assertTrue(np.isfinite(v5.logloss([1, 0], [0.0, 1.0])))

    def test_temporal_split_orders_time(self):
        X = np.arange(20, dtype=float).reshape(10, 2)
        y = np.arange(10) % 2
        dates = np.arange(10)[::-1]                        # deliberately reversed input
        _, _, _, _, dtr, dte = v5.train_test_split_temporal(X, y, dates, test_frac=0.3)
        self.assertLessEqual(dtr.max(), dte.min())
        self.assertEqual(len(dte), 3)

    def test_standardize_handles_constant_column(self):
        X = np.column_stack([np.ones(5), np.arange(5.0)])
        Xs, _, _ = v5.standardize(X)
        self.assertTrue(np.all(np.isfinite(Xs)))
        self.assertTrue(np.allclose(Xs[:, 0], 0.0))


class TestGradientTree(unittest.TestCase):
    def test_regression_mode_root_leaf_is_mean(self):
        X = np.random.RandomState(0).randn(50, 3)
        y = np.random.RandomState(1).rand(50)
        t = v5.GradientTree(max_depth=0, reg_lambda=0.0).fit(X, -y, np.ones(50))
        self.assertTrue(np.allclose(t.predict(X), y.mean()))

    def test_finds_perfect_one_dimensional_split(self):
        X = np.linspace(-1, 1, 200).reshape(-1, 1)
        y = (X[:, 0] > 0).astype(float)
        t = v5.GradientTree(max_depth=1, min_samples_leaf=5, reg_lambda=0.0).fit(X, -y, np.ones(200))
        p = t.predict(X)
        self.assertIsNotNone(t.root.feature)
        self.assertTrue(np.allclose(p[X[:, 0] <= t.root.threshold], 0.0))
        self.assertTrue(np.allclose(p[X[:, 0] > t.root.threshold], 1.0))

    def test_binning_contract(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0], [np.nan]])
        edges, B = v5.GradientTree.bin_features(X, n_bins=4)
        self.assertEqual(B[4, 0], 0)                       # NaN -> bin 0 (goes left)
        for i in range(4):
            self.assertEqual(B[i, 0], np.sum(edges[0] < X[i, 0]))

    def test_nan_inputs_yield_finite_predictions(self):
        X = np.random.RandomState(0).randn(300, 4)
        X[::7, 1] = np.nan
        y = (X[:, 0] > 0).astype(float)
        m = v5.GradientBoostedTrees(n_estimators=20, max_depth=3).fit(X, y)
        self.assertTrue(np.all(np.isfinite(m.predict_proba(X))))

    def test_importance_is_a_distribution_over_informative_features(self):
        X, y = make_data()
        m = v5.GradientBoostedTrees(n_estimators=30, max_depth=3).fit(X, y)
        for kind in ("gain", "split"):
            imp = m.feature_importance(kind=kind)
            self.assertAlmostEqual(imp.sum(), 1.0, places=9)
            self.assertTrue(np.all(imp >= 0))
        self.assertIn(int(np.argmax(m.feature_importance())), (0, 1, 2, 3))
        self.assertLess(m.feature_importance()[4] + m.feature_importance()[5], 0.15)  # noise features


class TestModelsLearn(unittest.TestCase):
    def setUp(self):
        X, y = make_data()
        self.Xtr, self.Xte, self.ytr, self.yte = X[:600], X[600:], y[:600], y[600:]

    def _auc(self, model):
        return v5.roc_auc_score(self.yte, model.fit(self.Xtr, self.ytr).predict_proba(self.Xte))

    def test_logistic_regression(self):
        self.assertGreater(self._auc(v5.LogisticRegression(n_epochs=40)), 0.85)

    def test_gbdt(self):
        self.assertGreater(self._auc(v5.GradientBoostedTrees(n_estimators=100, max_depth=3)), 0.93)

    def test_random_forest(self):
        self.assertGreater(self._auc(v5.RandomForest(n_estimators=50, max_depth=6)), 0.90)

    def test_neural_network(self):
        self.assertGreater(self._auc(v5.NeuralNetwork(hidden_sizes=(32, 16), n_epochs=80, lr=0.01)), 0.88)

    def test_stacking(self):
        st = v5.StackingEnsemble([("lr", v5.LogisticRegression(n_epochs=30)),
                                  ("gb", v5.GradientBoostedTrees(n_estimators=50, max_depth=3))])
        self.assertGreater(self._auc(st), 0.90)

    def test_gbdt_early_stopping_keeps_best_iteration(self):
        m = v5.GradientBoostedTrees(n_estimators=300, learning_rate=0.3, max_depth=4, early_stopping_rounds=10)
        m.fit(self.Xtr, self.ytr, self.Xte, self.yte)
        self.assertLess(len(m.trees), 300)
        self.assertEqual(m.best_iteration_, len(m.trees))
        self.assertEqual(int(np.argmin(m.val_losses)) + 1, len(m.trees))

    def test_random_forest_oob_score(self):
        m = v5.RandomForest(n_estimators=40, max_depth=6, oob_score=True).fit(self.Xtr, self.ytr)
        self.assertGreater(m.oob_auc_, 0.85)


class TestReproducibilityAndSafety(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_data(n=400)

    def _assert_same(self, factory):
        a = factory().fit(self.X, self.y).predict_proba(self.X)
        b = factory().fit(self.X, self.y).predict_proba(self.X)
        self.assertTrue(np.array_equal(a, b))

    def test_lr_reproducible(self):
        self._assert_same(lambda: v5.LogisticRegression(n_epochs=10, random_state=3))

    def test_gbdt_reproducible_with_row_and_column_subsampling(self):
        self._assert_same(lambda: v5.GradientBoostedTrees(n_estimators=20, subsample=0.7, colsample=0.5, random_state=3))

    def test_rf_reproducible(self):
        self._assert_same(lambda: v5.RandomForest(n_estimators=15, random_state=3))

    def test_nn_reproducible_with_dropout(self):
        self._assert_same(lambda: v5.NeuralNetwork(hidden_sizes=(16, 8), n_epochs=15, dropout_rate=0.5, random_state=3))

    def test_different_seeds_differ(self):
        a = v5.GradientBoostedTrees(n_estimators=20, subsample=0.6, random_state=1).fit(self.X, self.y).predict_proba(self.X)
        b = v5.GradientBoostedTrees(n_estimators=20, subsample=0.6, random_state=2).fit(self.X, self.y).predict_proba(self.X)
        self.assertFalse(np.array_equal(a, b))

    def test_stacking_does_not_mutate_templates(self):
        tmpl = v5.GradientBoostedTrees(n_estimators=10, max_depth=2)
        st = v5.StackingEnsemble([("gb", tmpl), ("lr", v5.LogisticRegression(n_epochs=5))]).fit(self.X, self.y)
        self.assertEqual(tmpl.trees, [])                   # template never fitted
        self.assertIsNot(st.fitted_base_[0][1], tmpl)


if __name__ == "__main__":
    unittest.main(verbosity=2)
