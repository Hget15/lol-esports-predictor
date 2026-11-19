"""
evaluate_versions.py — one command that produces every headline number.

Trains Logistic Regression and Gradient-Boosted Trees on each feature
generation (V1–V4) using the SAME games and the SAME time-ordered split, and
reports test-set AUC / Brier / log-loss / accuracy next to an Elo-only
baseline. Writes results/model_comparison.csv and results/feature_importance.csv.

Usage (from the repo root, after the feature scripts have been run):
    python src/evaluate_versions.py                       # last 15% of games as test
    python src/evaluate_versions.py --holdout WLDs 2026   # train on everything before
                                                          # that event, test on the event

Why this script exists: the numbers in the README used to come from separate
per-version training runs, and the V4 run was inflated by look-ahead in its
features. Putting every version through one loop on one split makes the
comparison checkable and repeatable.
"""
import argparse
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ml_from_scratch import (  # noqa: E402
    LogisticRegression, GradientBoostedTrees, standardize,
    roc_auc_score, brier_score, logloss, accuracy_score,
)

VERSIONS = [
    ("V1", "engineered_features.csv", "feature_engineering", "FEATURE_COLUMNS"),
    ("V2", "engineered_features_v2.csv", "feature_engineering_v2", "FEATURE_COLUMNS_V2"),
    ("V3", "engineered_features_v3.csv", "v3_enhancements", "FEATURE_COLUMNS_V3"),
    ("V4", "engineered_features_v4.csv", "v4_enhancements", "FEATURE_COLUMNS_V4"),
]
MIN_GAMES = 5  # both teams need this much history, as in the per-version scripts


def common_game_ids(data_dir, versions):
    """Games usable in every selected version.

    The versions do not agree exactly on which games qualify: V2 onward drop a
    few games with missing team names, which also shifts some teams' history
    counts. Intersecting the post-filter sets gives every version the same rows.
    """
    common = None
    for _, csv_name, _, _ in versions:
        d = pd.read_csv(os.path.join(data_dir, csv_name), usecols=["gameid", "n_games_a", "n_games_b"], low_memory=False)
        d = d[(d["n_games_a"] >= MIN_GAMES) & (d["n_games_b"] >= MIN_GAMES)]
        ids = set(d["gameid"].astype(str))
        common = ids if common is None else common & ids
    return common


def load_version(data_dir, csv_name, module, columns_attr, keep_ids):
    df = pd.read_csv(os.path.join(data_dir, csv_name), low_memory=False)
    df = df[df["gameid"].astype(str).isin(keep_ids)]
    # Stable sort by time; gameid breaks ties so every version gets the same order.
    df = df.sort_values(["date_numeric", "gameid"], kind="mergesort").reset_index(drop=True)
    cols = [c for c in getattr(importlib.import_module(module), columns_attr) if c in df.columns]
    X = np.clip(np.nan_to_num(df[cols].values.astype(float), nan=0.0), -1e6, 1e6)
    return df, X, cols


def split_indices(df, holdout):
    """Return (train_idx, test_idx, description). Train always precedes test in time."""
    n = len(df)
    if holdout:
        league, year = holdout
        is_event = (df["league"] == league) & (pd.to_datetime(df["date"]).dt.year == int(year))
        if not is_event.any():
            raise SystemExit(f"No games found for league={league} year={year}")
        first = int(np.flatnonzero(is_event.values)[0])
        return np.arange(first), np.flatnonzero(is_event.values), f"holdout {league} {year}"
    # Same proportions as the original training scripts: 75% train / 10% validation / 15% test.
    # Validation is left unused here so the training set matches those runs.
    return np.arange(int(n * 0.75)), np.arange(int(n * 0.85), n), "last 15% of games"


def metrics(y, p):
    return dict(auc=roc_auc_score(y, p), brier=brier_score(y, p),
                logloss=logloss(y, p), acc=accuracy_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--holdout", nargs=2, metavar=("LEAGUE", "YEAR"),
                    help="test on one event (e.g. WLDs 2026), train on all earlier games")
    ap.add_argument("--versions", nargs="*", default=[v[0] for v in VERSIONS])
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--model-dir", help="also save the last version's logistic regression "
                                        "(weights, bias, scaler, feature names) as .npz files here")
    args = ap.parse_args()

    rows, importance, predictions = [], None, None
    reference_ids = None
    selected = [v for v in VERSIONS if v[0] in args.versions]
    keep_ids = common_game_ids(args.data_dir, selected)

    for name, csv_name, module, attr in selected:
        df, X, cols = load_version(args.data_dir, csv_name, module, attr, keep_ids)
        ids = df["gameid"].astype(str).tolist()
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise SystemExit(f"{name}: games differ from the first version — split would not be identical")

        tr, te, desc = split_indices(df, args.holdout)
        y = df["result"].values.astype(float)
        Xtr, Xte, mu, sigma = standardize(X[tr], X[te])
        Xtr, Xte = np.nan_to_num(Xtr, nan=0.0), np.nan_to_num(Xte, nan=0.0)
        ytr, yte = y[tr], y[te]

        if not rows:
            d = pd.to_datetime(df["date"])
            print(f"Split: {desc}")
            print(f"  train {len(tr):>6} games  {d.iloc[tr[0]].date()} -> {d.iloc[tr[-1]].date()}")
            print(f"  test  {len(te):>6} games  {d.iloc[te[0]].date()} -> {d.iloc[te[-1]].date()}\n")
            # Baseline: no model at all, just the pre-game Elo expectation.
            m = metrics(yte, df["elo_expected"].values.astype(float)[te])
            rows.append(dict(version="-", model="Elo only (no model)", features=1, test_games=len(te), **m))
            print(f"  {'-':<3} {'Elo only (no model)':<22} AUC={m['auc']:.4f}  Brier={m['brier']:.4f}  Acc={m['acc']:.4f}")

        np.random.seed(42)  # the library's LR draws its mini-batches from the global RNG
        models = [
            ("Logistic Regression", LogisticRegression(lr=0.05, n_iters=2000, lambda_reg=0.01, batch_size=512)),
            ("GBDT", GradientBoostedTrees(n_estimators=args.trees, learning_rate=0.1, max_depth=4,
                                          min_samples_split=30, min_samples_leaf=15, subsample=0.8,
                                          random_state=42)),
        ]
        for model_name, model in models:
            t0 = time.time()
            p = model.fit(Xtr, ytr).predict_proba(Xte)
            m = metrics(yte, p)
            rows.append(dict(version=name, model=model_name, features=len(cols), test_games=len(te), **m))
            print(f"  {name:<3} {model_name:<22} AUC={m['auc']:.4f}  Brier={m['brier']:.4f}  "
                  f"Acc={m['acc']:.4f}  ({len(cols)} features, {time.time() - t0:.0f}s)", flush=True)
            if name == selected[-1][0]:
                if predictions is None:
                    predictions = df.iloc[te][["gameid", "date", "league", "team_a", "team_b", "result"]].copy()
                predictions["p_team_a_lr" if model_name == "Logistic Regression" else "p_team_a_gbdt"] = p
                if model_name == "Logistic Regression" and args.model_dir:
                    os.makedirs(args.model_dir, exist_ok=True)
                    np.savez(os.path.join(args.model_dir, "logistic_regression.npz"),
                             weights=model.weights, bias=model.bias)
                    np.savez(os.path.join(args.model_dir, "model_params.npz"),
                             mu=mu, sigma=sigma, feature_names=np.array(cols))
                if model_name == "GBDT":
                    imp = model.feature_importance(len(cols))
                    order = np.argsort(imp)[::-1][:30]
                    importance = pd.DataFrame({"feature": [cols[i] for i in order], "importance": imp[order]})

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"_{args.holdout[0]}_{args.holdout[1]}" if args.holdout else ""
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out_dir, f"model_comparison{suffix}.csv"), index=False)
    if importance is not None and not args.holdout:
        importance.to_csv(os.path.join(args.out_dir, f"feature_importance{suffix}.csv"), index=False)
    if args.holdout and predictions is not None:
        predictions.to_csv(os.path.join(args.out_dir, f"predictions{suffix}.csv"), index=False)
    print(f"\nWrote {args.out_dir}/model_comparison{suffix}.csv")


if __name__ == "__main__":
    main()
