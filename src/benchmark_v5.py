"""
benchmark_v5.py — the V4 library vs. the V5 library on the SAME data and split.

Usage (from the repo root):
    python src/benchmark_v5.py [path/to/engineered_features_v4.csv]

What this is: an honest, like-for-like comparison of the two model libraries
with matched hyper-parameters on whatever engineered-feature CSV you have
locally (see data/README.md to build one). It reports AUC / Brier / log-loss /
accuracy / fit time per model for both libraries, the V5 GBDT's gain-based
feature importances, an early-stopping demo, and a reproducibility check.

What this is NOT: a re-run of the headline 51,088-game result in README.md.
That needed the full 2021–2026 Oracle's Elixir data; if you only have one
year locally the absolute numbers here will be lower and noisier. The point
is the *relative* comparison and the timing.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ml_from_scratch as v4          # noqa: E402  (the published V1–V4 library)
import ml_from_scratch_v5 as v5       # noqa: E402  (the corrected library)
from v4_enhancements import prepare_model_data  # noqa: E402


def evaluate(name, lib, model, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    p = model.predict_proba(Xte)
    row = dict(model=name, lib=lib,
               auc=v5.roc_auc_score(yte, p), brier=v5.brier_score(yte, p),
               logloss=v5.logloss(yte, p), acc=v5.accuracy_score(yte, p), fit_s=fit_s)
    print(f"  {name:<20} {lib}  AUC={row['auc']:.4f}  Brier={row['brier']:.4f}  "
          f"LogLoss={row['logloss']:.4f}  Acc={row['acc']:.3f}  fit={fit_s:6.1f}s", flush=True)
    return row


def main(path):
    if not os.path.exists(path):
        print(f"Feature file not found: {path}\n"
              "Build one with the feature-engineering scripts (see data/README.md), "
              "or pass a path as the first argument.")
        sys.exit(1)

    df = pd.read_csv(path)
    X, y, dates, names, _ = prepare_model_data(df, min_games=5)
    Xtr, Xte, ytr, yte, _, _ = v5.train_test_split_temporal(X, y, dates, test_frac=0.15)
    Xtr, Xte, _, _ = v5.standardize(Xtr, Xte)
    Xtr, Xte = np.nan_to_num(Xtr, nan=0.0), np.nan_to_num(Xte, nan=0.0)
    print(f"\nData: {path}")
    print(f"rows: train={len(ytr)}  test={len(yte)}  features={X.shape[1]}  "
          f"train base rate={ytr.mean():.3f}\n")

    # Matched hyper-parameters. V4's LR counts iterations (random batches);
    # V5's counts epochs (every row once per epoch) — 30 epochs x 512-row
    # batches on this data is roughly the same number of gradient steps.
    pairs = [
        ("Logistic Regression",
         v4.LogisticRegression(lr=0.05, n_iters=2000, lambda_reg=0.01, batch_size=512),
         v5.LogisticRegression(lr=0.05, n_epochs=30, lambda_reg=0.01, batch_size=512)),
        ("GBDT",
         v4.GradientBoostedTrees(n_estimators=100, learning_rate=0.1, max_depth=4,
                                 min_samples_split=10, min_samples_leaf=5, subsample=0.8),
         v5.GradientBoostedTrees(n_estimators=100, learning_rate=0.1, max_depth=4,
                                 min_samples_leaf=5, subsample=0.8, reg_lambda=1.0)),
        ("Random Forest",
         v4.RandomForest(n_estimators=60, max_depth=8, min_samples_split=30, min_samples_leaf=5),
         v5.RandomForest(n_estimators=60, max_depth=8, min_samples_leaf=5)),
        ("Neural Network",
         v4.NeuralNetwork(hidden_sizes=(64, 32), lr=0.001, n_epochs=150),
         v5.NeuralNetwork(hidden_sizes=(64, 32), lr=0.001, n_epochs=150)),
    ]

    print("Head-to-head (same split, matched hyper-parameters):")
    rows = []
    v5_gbdt = None
    for name, m4, m5 in pairs:
        rows.append(evaluate(name, "V4", m4, Xtr, ytr, Xte, yte))
        rows.append(evaluate(name, "V5", m5, Xtr, ytr, Xte, yte))
        if name == "GBDT":
            v5_gbdt = m5

    print("\n| Model | Lib | AUC | Brier | LogLoss | Acc | Fit (s) |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['model']} | {r['lib']} | {r['auc']:.4f} | {r['brier']:.4f} | "
              f"{r['logloss']:.4f} | {r['acc']:.3f} | {r['fit_s']:.1f} |")

    # --- V5 extras -------------------------------------------------------
    imp = v5_gbdt.feature_importance(kind="gain")
    top = np.argsort(imp)[::-1][:10]
    print("\nTop-10 V5 GBDT features (gain-weighted importance):")
    for i in top:
        print(f"  {names[i]:<34} {imp[i] * 100:5.1f}%")

    # Early stopping demo: carve a temporal validation slice out of train.
    cut = int(len(ytr) * 0.85)
    es = v5.GradientBoostedTrees(n_estimators=500, learning_rate=0.05, max_depth=4,
                                 min_samples_leaf=5, subsample=0.8, colsample=0.8,
                                 early_stopping_rounds=30, random_state=42)
    t0 = time.perf_counter()
    es.fit(Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:])
    p = es.predict_proba(Xte)
    print(f"\nV5 GBDT with validation early stopping: stopped at {es.best_iteration_} trees "
          f"(cap 500), test AUC={v5.roc_auc_score(yte, p):.4f}, "
          f"Brier={v5.brier_score(yte, p):.4f}, fit={time.perf_counter() - t0:.1f}s")

    a = v5.GradientBoostedTrees(n_estimators=30, subsample=0.7, random_state=7).fit(Xtr, ytr).predict_proba(Xte)
    b = v5.GradientBoostedTrees(n_estimators=30, subsample=0.7, random_state=7).fit(Xtr, ytr).predict_proba(Xte)
    print(f"Reproducibility (same seed -> identical predictions): {np.array_equal(a, b)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "engineered_features_v4.csv"))
