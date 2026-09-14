"""
First Stand 2026 Tournament Simulation
Uses V3 model to predict all matches and compare with actual results.
Also runs edge case tests.
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ml_from_scratch import (
    LogisticRegression, GradientBoostedTrees, RandomForest, NeuralNetwork,
    roc_auc_score, accuracy_score, standardize, sigmoid
)
from v3_enhancements import FEATURE_COLUMNS_V3, prepare_model_data

# ============================================================
# FST 2026 Match Data (from Leaguepedia)
# ============================================================
FST_MATCHES = [
    # Group A
    {"stage": "Group A R1", "team_a": "Bilibili Gaming", "team_b": "BNK FEARX", "score_a": 3, "score_b": 2, "date": "2026-03-16"},
    {"stage": "Group A R1", "team_a": "G2 Esports", "team_b": "Team Secret Whales", "score_a": 3, "score_b": 0, "date": "2026-03-16"},
    {"stage": "Group A LB", "team_a": "BNK FEARX", "team_b": "Team Secret Whales", "score_a": 3, "score_b": 0, "date": "2026-03-18"},
    {"stage": "Group A QM", "team_a": "Bilibili Gaming", "team_b": "G2 Esports", "score_a": 3, "score_b": 0, "date": "2026-03-18"},
    {"stage": "Group A QM2", "team_a": "G2 Esports", "team_b": "BNK FEARX", "score_a": 3, "score_b": 0, "date": "2026-03-20"},
    # Group B
    {"stage": "Group B R1", "team_a": "Gen.G", "team_b": "JD Gaming", "score_a": 3, "score_b": 0, "date": "2026-03-17"},
    {"stage": "Group B R1", "team_a": "LYON", "team_b": "LOUD", "score_a": 3, "score_b": 2, "date": "2026-03-17"},
    {"stage": "Group B LB", "team_a": "JD Gaming", "team_b": "LOUD", "score_a": 3, "score_b": 0, "date": "2026-03-19"},
    {"stage": "Group B QM", "team_a": "Gen.G", "team_b": "LYON", "score_a": 3, "score_b": 0, "date": "2026-03-19"},
    {"stage": "Group B QM2", "team_a": "LYON", "team_b": "JD Gaming", "score_a": 1, "score_b": 3, "date": "2026-03-20"},
    # Knockout Stage
    {"stage": "Semifinal 1", "team_a": "Gen.G", "team_b": "G2 Esports", "score_a": 0, "score_b": 3, "date": "2026-03-21"},
    {"stage": "Semifinal 2", "team_a": "Bilibili Gaming", "team_b": "JD Gaming", "score_a": 3, "score_b": 0, "date": "2026-03-21"},
    {"stage": "Finals", "team_a": "G2 Esports", "team_b": "Bilibili Gaming", "score_a": 1, "score_b": 3, "date": "2026-03-22"},
]


def load_model_and_data():
    """Load V3 features and train model up to pre-FST cutoff."""
    print("Loading V3 features...")
    df = pd.read_csv('data/engineered_features_v3.csv', low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Use all data before FST 2026 for training (cutoff: March 15, 2026)
    cutoff = pd.Timestamp('2026-03-15')
    train_df = df[df['date'] < cutoff].copy()
    fst_df = df[df['date'] >= cutoff].copy()

    print(f"  Training data: {len(train_df)} games (before {cutoff.date()})")
    print(f"  FST 2026 data: {len(fst_df)} games")

    return train_df, fst_df, df


def get_team_features(df, team_a, team_b):
    """Find the most recent game row where these teams appear and extract features."""
    # Find latest game involving team_a
    mask_a = (df['team_a'] == team_a) | (df['team_b'] == team_a)
    mask_b = (df['team_a'] == team_b) | (df['team_b'] == team_b)

    recent_a = df[mask_a].iloc[-1] if mask_a.any() else None
    recent_b = df[mask_b].iloc[-1] if mask_b.any() else None

    return recent_a, recent_b


def find_fst_game(fst_df, team_a, team_b, date_str):
    """Find the actual FST game row in the dataset."""
    date = pd.Timestamp(date_str)
    # Look for game with these teams on this date (±1 day)
    mask = (
        ((fst_df['team_a'] == team_a) & (fst_df['team_b'] == team_b)) |
        ((fst_df['team_a'] == team_b) & (fst_df['team_b'] == team_a))
    ) & (abs((fst_df['date'] - date).dt.days) <= 2)

    if mask.any():
        return fst_df[mask].iloc[0]
    return None


def predict_match(model, mu, sigma, features, feature_names):
    """Predict using trained model."""
    x = np.array([features.get(f, 0) for f in feature_names], dtype=float)
    x = np.nan_to_num(x, 0)
    x_std = (x - mu) / np.where(sigma > 1e-10, sigma, 1)
    x_std = np.nan_to_num(x_std, 0)
    return model.predict_proba(x_std.reshape(1, -1))[0]


def run_simulation():
    print("=" * 80)
    print("  FIRST STAND 2026 TOURNAMENT SIMULATION")
    print("=" * 80)

    train_df, fst_df, full_df = load_model_and_data()

    # Prepare training data
    X, y, dates, feature_names, meta_df = prepare_model_data(train_df, min_games=3)
    print(f"\n  Training on {X.shape[0]} games with {X.shape[1]} features")

    # Standardize
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma[sigma < 1e-10] = 1
    X_std = (X - mu) / sigma
    X_std = np.nan_to_num(X_std, 0)

    # Train GBDT (best model)
    print("\n  Training GBDT...")
    gbdt = GradientBoostedTrees(
        n_estimators=100, learning_rate=0.1, max_depth=4,
        min_samples_split=30, min_samples_leaf=15,
        subsample=0.8, verbose=False, random_state=42
    )
    gbdt.fit(X, y)  # GBDT doesn't need standardization

    # Also train LR for comparison
    print("  Training LR...")
    lr = LogisticRegression(lr=0.05, n_iters=2000, lambda_reg=0.01, batch_size=512, verbose=False)
    lr.fit(X_std, y)

    # ============================================================
    # SIMULATE FST MATCHES
    # ============================================================
    print("\n" + "=" * 80)
    print("  MATCH-BY-MATCH PREDICTIONS vs ACTUAL RESULTS")
    print("=" * 80)

    correct_gbdt = 0
    correct_lr = 0
    total = 0

    for match in FST_MATCHES:
        ta, tb = match['team_a'], match['team_b']
        actual_winner = ta if match['score_a'] > match['score_b'] else tb

        # Try to find actual game in the FST data
        game_row = find_fst_game(fst_df, ta, tb, match['date'])

        if game_row is not None:
            # Use actual features from the dataset
            feats = {f: game_row.get(f, 0) for f in feature_names}
            # Check if team_a in dataset matches our team_a
            if game_row['team_a'] != ta:
                # Flip: negate diff features
                for f in feature_names:
                    if 'diff' in f or f.startswith('elo_') or f.startswith('wr_') or f.startswith('h2h'):
                        feats[f] = -feats.get(f, 0)

            x = np.array([feats.get(f, 0) for f in feature_names], dtype=float)
            x = np.nan_to_num(x, 0)

            prob_gbdt = gbdt.predict_proba(x.reshape(1, -1))[0]

            x_std = (x - mu) / sigma
            x_std = np.nan_to_num(x_std, 0)
            prob_lr = lr.predict_proba(x_std.reshape(1, -1))[0]
        else:
            prob_gbdt = 0.5
            prob_lr = 0.5

        pred_winner_gbdt = ta if prob_gbdt >= 0.5 else tb
        pred_winner_lr = ta if prob_lr >= 0.5 else tb

        gbdt_correct = "✓" if pred_winner_gbdt == actual_winner else "✗"
        lr_correct = "✓" if pred_winner_lr == actual_winner else "✗"

        if pred_winner_gbdt == actual_winner:
            correct_gbdt += 1
        if pred_winner_lr == actual_winner:
            correct_lr += 1
        total += 1

        found = "📊" if game_row is not None else "⚠️"
        score_str = f"{match['score_a']}-{match['score_b']}"

        print(f"\n  {found} {match['stage']:20s} | {ta:25s} vs {tb:25s}")
        print(f"     Actual: {actual_winner} ({score_str})")
        print(f"     GBDT:  {ta} {prob_gbdt:.1%} | {tb} {1-prob_gbdt:.1%}  {gbdt_correct}")
        print(f"     LR:    {ta} {prob_lr:.1%} | {tb} {1-prob_lr:.1%}  {lr_correct}")

    print("\n" + "=" * 80)
    print(f"  FST 2026 PREDICTION ACCURACY")
    print(f"  GBDT: {correct_gbdt}/{total} ({correct_gbdt/total:.1%})")
    print(f"  LR:   {correct_lr}/{total} ({correct_lr/total:.1%})")
    print("=" * 80)

    # ============================================================
    # EDGE CASE TESTS
    # ============================================================
    print("\n" + "=" * 80)
    print("  EDGE CASE TESTS")
    print("=" * 80)

    # Test 1: Same team vs itself (should be ~50%)
    print("\n  Test 1: Same team vs itself")
    for team in ["Gen.G", "T1", "Bilibili Gaming"]:
        mask = full_df['team_a'] == team
        if mask.any():
            row = full_df[mask].iloc[-1]
            x = np.array([row.get(f, 0) for f in feature_names], dtype=float)
            # Zero out all diff features (same team)
            for i, f in enumerate(feature_names):
                if 'diff' in f:
                    x[i] = 0
            x = np.nan_to_num(x, 0)
            p = gbdt.predict_proba(x.reshape(1, -1))[0]
            status = "✓" if 0.45 <= p <= 0.55 else "⚠️"
            print(f"    {status} {team} vs {team}: {p:.1%} (expected ~50%)")

    # Test 2: Massive Elo difference (top vs bottom)
    print("\n  Test 2: Extreme Elo difference")
    x_base = np.zeros(len(feature_names))
    for i, f in enumerate(feature_names):
        if f == 'elo_diff':
            x_base[i] = 300  # Huge elo advantage
        elif f == 'elo_expected':
            x_base[i] = 0.85  # Expected to win
        elif f == 'wr_diff_10':
            x_base[i] = 0.6  # Much better recent record
    p = gbdt.predict_proba(x_base.reshape(1, -1))[0]
    status = "✓" if p > 0.7 else "⚠️"
    print(f"    {status} +300 Elo, +0.6 WR diff: {p:.1%} (expected >70%)")

    x_base2 = x_base.copy()
    for i, f in enumerate(feature_names):
        if f == 'elo_diff':
            x_base2[i] = -300
        elif f == 'elo_expected':
            x_base2[i] = 0.15
        elif f == 'wr_diff_10':
            x_base2[i] = -0.6
    p2 = gbdt.predict_proba(x_base2.reshape(1, -1))[0]
    status = "✓" if p2 < 0.3 else "⚠️"
    print(f"    {status} -300 Elo, -0.6 WR diff: {p2:.1%} (expected <30%)")

    # Test 3: Symmetry (swapping teams should flip probability)
    print("\n  Test 3: Symmetry check")
    for team in ["Gen.G", "Bilibili Gaming"]:
        mask = full_df['team_a'] == team
        if mask.any():
            row = full_df[mask].iloc[-1]
            x = np.array([row.get(f, 0) for f in feature_names], dtype=float)
            x = np.nan_to_num(x, 0)
            p_normal = gbdt.predict_proba(x.reshape(1, -1))[0]

            # Flip all diff features
            x_flip = x.copy()
            for i, f in enumerate(feature_names):
                if 'diff' in f or f == 'h2h_a' or f == 'series_score_diff' or f == 'series_momentum_a':
                    x_flip[i] = -x_flip[i]
            p_flip = gbdt.predict_proba(x_flip.reshape(1, -1))[0]

            sym_error = abs((p_normal + p_flip) - 1.0)
            status = "✓" if sym_error < 0.15 else "⚠️"
            print(f"    {status} {team}: p={p_normal:.3f}, p_flip={p_flip:.3f}, sum={p_normal+p_flip:.3f} (expected ~1.0, err={sym_error:.3f})")

    # Test 4: Bo1 vs Bo5 impact
    print("\n  Test 4: Bo format impact")
    x_bo1 = np.zeros(len(feature_names))
    x_bo5 = np.zeros(len(feature_names))
    for i, f in enumerate(feature_names):
        if f == 'elo_diff':
            x_bo1[i] = x_bo5[i] = 50
        elif f == 'bo_format':
            x_bo1[i] = 1
            x_bo5[i] = 5
        elif f == 'is_bo1':
            x_bo1[i] = 1
            x_bo5[i] = 0
    p_bo1 = gbdt.predict_proba(x_bo1.reshape(1, -1))[0]
    p_bo5 = gbdt.predict_proba(x_bo5.reshape(1, -1))[0]
    print(f"    Bo1: {p_bo1:.1%}, Bo5: {p_bo5:.1%} (longer series may amplify skill diff)")

    # Test 5: International travel impact
    print("\n  Test 5: Travel impact")
    x_home = np.zeros(len(feature_names))
    x_travel = np.zeros(len(feature_names))
    for i, f in enumerate(feature_names):
        if f == 'elo_diff':
            x_home[i] = x_travel[i] = 50
        elif f == 'is_traveling_a':
            x_home[i] = 0
            x_travel[i] = 1
        elif f == 'is_home_a':
            x_home[i] = 1
            x_travel[i] = 0
    p_home = gbdt.predict_proba(x_home.reshape(1, -1))[0]
    p_travel = gbdt.predict_proba(x_travel.reshape(1, -1))[0]
    print(f"    Home: {p_home:.1%}, Traveling: {p_travel:.1%} (home advantage expected)")

    # V1 vs V2 vs V3 comparison summary
    print("\n" + "=" * 80)
    print("  MODEL VERSION COMPARISON (Test AUC)")
    print("=" * 80)
    v1 = {'LR': 0.6881, 'GBDT': 0.6834, 'RF': 0.6794, 'NN': 0.6798, 'Ensemble': 0.6874}
    v2 = {'LR': 0.6843, 'GBDT': 0.6975, 'RF': 0.6923, 'NN': 0.6870, 'Ensemble': 0.6976}
    v3 = {'LR': 0.6840, 'GBDT': 0.6983, 'RF': 0.6924, 'NN': 0.6821, 'Ensemble': 0.6978}

    print(f"  {'Model':15s} {'V1':>8s} {'V2':>8s} {'V3':>8s} {'V1→V3':>8s}")
    print(f"  {'-'*47}")
    for k in v1:
        diff = v3[k] - v1[k]
        arrow = '↑' if diff > 0 else '↓'
        print(f"  {k:15s} {v1[k]:8.4f} {v2[k]:8.4f} {v3[k]:8.4f} {arrow}{abs(diff):7.4f}")

    print(f"\n  Features: V1=63 → V2=79 → V3=106")
    print(f"  Best: V1 0.6881 (LR) → V2 0.6976 (Ens) → V3 0.6983 (GBDT)")


if __name__ == "__main__":
    run_simulation()
