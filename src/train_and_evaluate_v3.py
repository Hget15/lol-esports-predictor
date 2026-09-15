"""
Model Training and Evaluation Pipeline for LoL Esports Prediction.

Trains multiple models, evaluates with temporal split, produces:
- Model comparison metrics (AUC, Brier, Log Loss, Accuracy)
- Calibration curves
- Feature importance charts
- Learning curves
- Saves best model weights
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import pickle

# Warnings are deliberately left visible (no blanket filterwarnings('ignore')):
# during training, a RuntimeWarning is usually the first sign of a NaN/inf
# leaking into the feature matrix or a degenerate metric, which we want to see.

# Import our custom modules
sys.path.insert(0, os.path.dirname(__file__))
from ml_from_scratch import (
    LogisticRegression, GradientBoostedTrees, RandomForest, NeuralNetwork,
    StackingEnsemble, sigmoid, logloss, brier_score, roc_auc_score,
    accuracy_score, calibration_bins, standardize
)
from v3_enhancements import FEATURE_COLUMNS_V3 as FEATURE_COLUMNS, prepare_model_data


# ============================================================
# LOAD DATA
# ============================================================

def load_features():
    print("Loading engineered features...")
    df = pd.read_csv('data/engineered_features_v3.csv', low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print(f"  {len(df)} games loaded")
    return df


# ============================================================
# TEMPORAL TRAIN/TEST SPLIT
# ============================================================

def temporal_split(X, y, dates, meta_df, test_frac=0.15, val_frac=0.10):
    """Split into train/val/test by date (no leakage)."""
    order = np.argsort(dates)
    X, y, dates = X[order], y[order], dates[order]
    meta_df = meta_df.iloc[order].reset_index(drop=True)

    n = len(y)
    train_end = int(n * (1 - test_frac - val_frac))
    val_end = int(n * (1 - test_frac))

    splits = {
        'X_train': X[:train_end], 'y_train': y[:train_end], 'd_train': dates[:train_end],
        'X_val': X[train_end:val_end], 'y_val': y[train_end:val_end], 'd_val': dates[train_end:val_end],
        'X_test': X[val_end:], 'y_test': y[val_end:], 'd_test': dates[val_end:],
        'meta_train': meta_df.iloc[:train_end],
        'meta_val': meta_df.iloc[train_end:val_end],
        'meta_test': meta_df.iloc[val_end:],
    }

    print(f"  Train: {train_end} games (up to {pd.Timestamp(dates[train_end-1], unit='s')})")
    print(f"  Val:   {val_end - train_end} games")
    print(f"  Test:  {n - val_end} games (from {pd.Timestamp(dates[val_end], unit='s')})")
    return splits


# ============================================================
# TRAINING
# ============================================================

def train_all_models(splits, feature_names):
    """Train all models and return predictions."""
    X_tr, y_tr = splits['X_train'], splits['y_train']
    X_val, y_val = splits['X_val'], splits['y_val']
    X_te, y_te = splits['X_test'], splits['y_test']

    # Standardize
    X_tr_s, X_val_s, mu, sigma = standardize(X_tr, X_val)
    _, X_te_s, _, _ = standardize(X_tr, X_te)
    X_tr_s = np.nan_to_num(X_tr_s, 0)
    X_val_s = np.nan_to_num(X_val_s, 0)
    X_te_s = np.nan_to_num(X_te_s, 0)

    results = {}

    # 1. Logistic Regression
    print("\n--- Logistic Regression ---")
    t0 = time.time()
    lr = LogisticRegression(lr=0.05, n_iters=2000, lambda_reg=0.01, batch_size=512, verbose=True)
    lr.fit(X_tr_s, y_tr)
    p_lr_val = lr.predict_proba(X_val_s)
    p_lr_test = lr.predict_proba(X_te_s)
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  Val AUC: {roc_auc_score(y_val, p_lr_val):.4f}, Brier: {brier_score(y_val, p_lr_val):.4f}")
    print(f"  Test AUC: {roc_auc_score(y_te, p_lr_test):.4f}, Brier: {brier_score(y_te, p_lr_test):.4f}")
    results['Logistic Regression'] = {
        'model': lr, 'val_pred': p_lr_val, 'test_pred': p_lr_test,
        'train_losses': lr.losses, 'needs_standardize': True
    }

    # 2. Gradient Boosted Trees
    print("\n--- Gradient Boosted Decision Trees ---")
    t0 = time.time()
    gbdt = GradientBoostedTrees(
        n_estimators=100, learning_rate=0.1, max_depth=4,
        min_samples_split=30, min_samples_leaf=15,
        subsample=0.8, max_features=None, verbose=True, random_state=42
    )
    gbdt.fit(X_tr, y_tr)  # GBDT doesn't need standardization
    p_gb_val = gbdt.predict_proba(X_val)
    p_gb_test = gbdt.predict_proba(X_te)
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  Val AUC: {roc_auc_score(y_val, p_gb_val):.4f}, Brier: {brier_score(y_val, p_gb_val):.4f}")
    print(f"  Test AUC: {roc_auc_score(y_te, p_gb_test):.4f}, Brier: {brier_score(y_te, p_gb_test):.4f}")
    results['GBDT'] = {
        'model': gbdt, 'val_pred': p_gb_val, 'test_pred': p_gb_test,
        'train_losses': gbdt.train_losses, 'needs_standardize': False
    }

    # 3. Random Forest
    print("\n--- Random Forest ---")
    t0 = time.time()
    rf = RandomForest(
        n_estimators=60, max_depth=8, min_samples_split=30,
        min_samples_leaf=15, max_features='sqrt', verbose=True, random_state=42
    )
    rf.fit(X_tr, y_tr)
    p_rf_val = rf.predict_proba(X_val)
    p_rf_test = rf.predict_proba(X_te)
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  Val AUC: {roc_auc_score(y_val, p_rf_val):.4f}, Brier: {brier_score(y_val, p_rf_val):.4f}")
    print(f"  Test AUC: {roc_auc_score(y_te, p_rf_test):.4f}, Brier: {brier_score(y_te, p_rf_test):.4f}")
    results['Random Forest'] = {
        'model': rf, 'val_pred': p_rf_val, 'test_pred': p_rf_test,
        'needs_standardize': False
    }

    # 4. Neural Network
    print("\n--- Neural Network ---")
    t0 = time.time()
    nn = NeuralNetwork(
        hidden_sizes=(64, 32), lr=0.001, n_epochs=150,
        batch_size=512, lambda_reg=0.001, dropout_rate=0.3,
        verbose=True, random_state=42
    )
    nn.fit(X_tr_s, y_tr)
    p_nn_val = nn.predict_proba(X_val_s)
    p_nn_test = nn.predict_proba(X_te_s)
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  Val AUC: {roc_auc_score(y_val, p_nn_val):.4f}, Brier: {brier_score(y_val, p_nn_val):.4f}")
    print(f"  Test AUC: {roc_auc_score(y_te, p_nn_test):.4f}, Brier: {brier_score(y_te, p_nn_test):.4f}")
    results['Neural Network'] = {
        'model': nn, 'val_pred': p_nn_val, 'test_pred': p_nn_test,
        'train_losses': nn.losses, 'needs_standardize': True
    }

    # 5. Weighted Ensemble (simple average weighted by val AUC)
    print("\n--- Weighted Ensemble ---")
    val_aucs = {
        'lr': roc_auc_score(y_val, p_lr_val),
        'gb': roc_auc_score(y_val, p_gb_val),
        'rf': roc_auc_score(y_val, p_rf_val),
        'nn': roc_auc_score(y_val, p_nn_val),
    }
    total = sum(val_aucs.values())
    w = {k: v/total for k, v in val_aucs.items()}
    print(f"  Weights: LR={w['lr']:.3f}, GBDT={w['gb']:.3f}, RF={w['rf']:.3f}, NN={w['nn']:.3f}")

    p_ens_val = w['lr']*p_lr_val + w['gb']*p_gb_val + w['rf']*p_rf_val + w['nn']*p_nn_val
    p_ens_test = w['lr']*p_lr_test + w['gb']*p_gb_test + w['rf']*p_rf_test + w['nn']*p_nn_test

    print(f"  Val AUC: {roc_auc_score(y_val, p_ens_val):.4f}, Brier: {brier_score(y_val, p_ens_val):.4f}")
    print(f"  Test AUC: {roc_auc_score(y_te, p_ens_test):.4f}, Brier: {brier_score(y_te, p_ens_test):.4f}")
    results['Weighted Ensemble'] = {
        'val_pred': p_ens_val, 'test_pred': p_ens_test,
        'weights': w, 'needs_standardize': False
    }

    # 6. Stacking Ensemble (meta-learner trained on val predictions)
    print("\n--- Stacking Ensemble ---")
    # Stack val predictions as meta-features
    meta_val = np.column_stack([p_lr_val, p_gb_val, p_rf_val, p_nn_val])
    meta_test = np.column_stack([p_lr_test, p_gb_test, p_rf_test, p_nn_test])

    meta_lr = LogisticRegression(lr=0.1, n_iters=500, lambda_reg=0.1)
    meta_lr.fit(meta_val, y_val)
    p_stack_test = meta_lr.predict_proba(meta_test)

    print(f"  Test AUC: {roc_auc_score(y_te, p_stack_test):.4f}, Brier: {brier_score(y_te, p_stack_test):.4f}")
    results['Stacking Ensemble'] = {
        'model': meta_lr, 'test_pred': p_stack_test,
        'val_pred': meta_lr.predict_proba(meta_val),
        'needs_standardize': False
    }

    return results, mu, sigma, w


# ============================================================
# EVALUATION & VISUALIZATION
# ============================================================

def evaluate_and_plot(results, splits, feature_names, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    y_val = splits['y_val']
    y_test = splits['y_test']

    # ---- 1. Model Comparison Table ----
    print("\n" + "="*80)
    print("MODEL COMPARISON (Test Set)")
    print("="*80)
    rows = []
    for name, res in results.items():
        p = res['test_pred']
        row = {
            'Model': name,
            'AUC': roc_auc_score(y_test, p),
            'Brier': brier_score(y_test, p),
            'Log Loss': logloss(y_test, p),
            'Accuracy': accuracy_score(y_test, p),
        }
        rows.append(row)
        print(f"  {name:25s} | AUC: {row['AUC']:.4f} | Brier: {row['Brier']:.4f} | "
              f"LogLoss: {row['Log Loss']:.4f} | Acc: {row['Accuracy']:.4f}")
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

    # ---- 2. Calibration Curves ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4']
    for i, (name, res) in enumerate(results.items()):
        p = res['test_pred']
        mids, fracs, counts = calibration_bins(y_test, p, n_bins=10)
        ax.plot(mids, fracs, 'o-', color=colors[i % len(colors)], label=name, markersize=6)
    ax.set_xlabel('Predicted Probability', fontsize=13)
    ax.set_ylabel('Observed Win Rate', fontsize=13)
    ax.set_title('Calibration Curves - LoL Esports Prediction Models', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_curves.png'), dpi=150)
    plt.close()
    print(f"\n  Saved calibration_curves.png")

    # ---- 3. Feature Importance (GBDT) ----
    if 'GBDT' in results:
        gbdt = results['GBDT']['model']
        importance = gbdt.feature_importance(len(feature_names))
        imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
        imp_df = imp_df.sort_values('importance', ascending=False).head(25)

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.barh(range(len(imp_df)), imp_df['importance'].values, color='#2196F3')
        ax.set_yticks(range(len(imp_df)))
        ax.set_yticklabels(imp_df['feature'].values, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance (Split Frequency)', fontsize=12)
        ax.set_title('Top 25 Features - Gradient Boosted Trees', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
        plt.close()
        imp_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        print(f"  Saved feature_importance.png")

    # ---- 4. ROC Curves ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for i, (name, res) in enumerate(results.items()):
        p = res['test_pred']
        auc_val = roc_auc_score(y_test, p)
        # Compute ROC curve points
        thresholds = np.linspace(0, 1, 200)
        tprs, fprs = [], []
        for t in thresholds:
            pred_pos = p >= t
            tp = np.sum((pred_pos) & (y_test == 1))
            fp = np.sum((pred_pos) & (y_test == 0))
            fn = np.sum((~pred_pos) & (y_test == 1))
            tn = np.sum((~pred_pos) & (y_test == 0))
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            tprs.append(tpr)
            fprs.append(fpr)
        ax.plot(fprs, tprs, color=colors[i % len(colors)], label=f'{name} (AUC={auc_val:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves - LoL Esports Prediction Models', fontsize=15)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150)
    plt.close()
    print(f"  Saved roc_curves.png")

    # ---- 5. Prediction Distribution ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for i, (name, res) in enumerate(results.items()):
        if i >= 6:
            break
        ax = axes[i // 3][i % 3]
        p = res['test_pred']
        ax.hist(p[y_test == 1], bins=30, alpha=0.6, color='green', label='Wins', density=True)
        ax.hist(p[y_test == 0], bins=30, alpha=0.6, color='red', label='Losses', density=True)
        ax.set_title(name, fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlabel('Predicted Win Probability')
    plt.suptitle('Prediction Distributions by Outcome', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distributions.png'), dpi=150)
    plt.close()
    print(f"  Saved prediction_distributions.png")

    # ---- 6. Accuracy by Confidence Bucket ----
    best_name = max(results.keys(), key=lambda n: roc_auc_score(y_test, results[n]['test_pred']))
    best_pred = results[best_name]['test_pred']

    confidence_buckets = [(0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7),
                          (0.7, 0.75), (0.75, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 1.0)]
    # Map predictions to "confidence" (distance from 0.5)
    confidence = np.abs(best_pred - 0.5) + 0.5  # remap to [0.5, 1.0]
    predicted_winner = (best_pred >= 0.5).astype(int)
    correct = (predicted_winner == y_test)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bucket_accs = []
    bucket_labels = []
    bucket_counts = []
    for lo, hi in confidence_buckets:
        mask = (confidence >= lo) & (confidence < hi)
        if np.sum(mask) > 10:
            acc = np.mean(correct[mask])
            bucket_accs.append(acc)
            bucket_labels.append(f'{lo:.0%}-{hi:.0%}')
            bucket_counts.append(np.sum(mask))

    bars = ax.bar(range(len(bucket_accs)), bucket_accs, color='#2196F3', alpha=0.8)
    ax.set_xticks(range(len(bucket_accs)))
    ax.set_xticklabels(bucket_labels, rotation=45)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Model Confidence', fontsize=12)
    ax.set_title(f'Accuracy by Confidence Level ({best_name})', fontsize=14)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Coin flip')
    # Add count labels
    for bar, count in zip(bars, bucket_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=9)
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_confidence.png'), dpi=150)
    plt.close()
    print(f"  Saved accuracy_by_confidence.png")

    # ---- 7. Performance by League Tier ----
    meta_test = splits['meta_test']
    if 'league_tier' in meta_test.columns:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for tier_val, tier_name in [(3, 'Major (LCK/LPL/LEC/LCS)'), (2, 'Regional'), (1, 'Minor')]:
            mask = meta_test['league_tier'].values == tier_val
            if np.sum(mask) > 50:
                auc = roc_auc_score(y_test[mask], best_pred[mask])
                brier = brier_score(y_test[mask], best_pred[mask])
                ax.bar(tier_name, auc, alpha=0.8, color=colors[tier_val])
                ax.text(tier_val - 1, auc + 0.005, f'AUC={auc:.3f}\nn={np.sum(mask)}',
                        ha='center', fontsize=10)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title(f'Model Performance by League Tier ({best_name})', fontsize=14)
        ax.set_ylim(0.5, 0.85)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_by_tier.png'), dpi=150)
        plt.close()
        print(f"  Saved performance_by_tier.png")

    return comparison_df


# ============================================================
# SAVE MODELS
# ============================================================

def save_models(results, mu, sigma, feature_names, weights, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)

    model_data = {
        'feature_names': feature_names,
        'mu': mu,
        'sigma': sigma,
        'ensemble_weights': weights,
        'models': {}
    }

    for name, res in results.items():
        if 'model' in res:
            model = res['model']
            if hasattr(model, 'weights') and isinstance(model.weights, dict):
                # Neural network
                model_data['models'][name] = {
                    'type': type(model).__name__,
                    'weights': model.weights,
                    'hidden_sizes': getattr(model, 'hidden_sizes', None),
                    'needs_standardize': res.get('needs_standardize', False),
                }
            elif hasattr(model, 'weights') and isinstance(model.weights, np.ndarray):
                # Logistic regression
                model_data['models'][name] = {
                    'type': 'LogisticRegression',
                    'weights': model.weights,
                    'bias': model.bias,
                    'needs_standardize': res.get('needs_standardize', False),
                }
    # Note: Tree-based models are harder to serialize without pickle
    # Save a simplified version

    # Save with numpy
    np.savez(os.path.join(output_dir, 'model_params.npz'),
             mu=mu, sigma=sigma, feature_names=feature_names,
             ensemble_weights_lr=weights.get('lr', 0.25),
             ensemble_weights_gb=weights.get('gb', 0.25),
             ensemble_weights_rf=weights.get('rf', 0.25),
             ensemble_weights_nn=weights.get('nn', 0.25))

    # Save LR weights separately (easy to use)
    if 'Logistic Regression' in results:
        lr_model = results['Logistic Regression']['model']
        np.savez(os.path.join(output_dir, 'logistic_regression.npz'),
                 weights=lr_model.weights, bias=lr_model.bias)

    print(f"\nModels saved to {output_dir}/")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*80)
    print("  LoL ESPORTS MATCH PREDICTION MODEL V3")
    print("  Coach + Travel + Regional Playstyle Features")
    print("="*80)

    # Load features
    features_df = load_features()

    # Prepare model data
    X, y, dates, feature_names, meta_df = prepare_model_data(features_df, min_games=5)
    print(f"\nDataset: {X.shape[0]} games, {X.shape[1]} features")
    print(f"Class balance: {y.mean():.3f} (team_a wins)")

    # Temporal split
    print("\nTemporal Split:")
    splits = temporal_split(X, y, dates, meta_df, test_frac=0.15, val_frac=0.10)

    # Train models
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    results, mu, sigma, weights = train_all_models(splits, feature_names)

    # Evaluate and plot
    print("\n" + "="*80)
    print("EVALUATION & VISUALIZATION")
    print("="*80)
    comparison_df = evaluate_and_plot(results, splits, feature_names, output_dir='output_v3')

    # Save models
    save_models(results, mu, sigma, feature_names, weights, output_dir='models_v3')

    print("\n" + "="*80)
    print("  TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest model: {comparison_df.loc[comparison_df['AUC'].idxmax(), 'Model']} "
          f"(AUC={comparison_df['AUC'].max():.4f})")
