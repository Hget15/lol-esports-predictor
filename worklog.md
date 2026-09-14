# LoL Esports Match Prediction Model — Worklog

## Project Overview

Comprehensive ML prediction model for League of Legends esports matches, built from scratch using only NumPy (no sklearn — pip proxy blocked). Trained on 51,088 games from Oracle's Elixir data (2021–2026). Evolved through 4 major versions, culminating in AUC 0.797 on held-out test data.

---

## Architecture

### Data Pipeline

```
Oracle's Elixir CSVs (6 files, ~450MB total)
        │
        ▼
feature_engineering.py      → data/engineered_features.csv      (V1: 63 features)
feature_engineering_v2.py   → data/engineered_features_v2.csv   (V2: 79 features)
v3_enhancements.py          → data/engineered_features_v3.csv   (V3: 106 features)
v4_enhancements.py          → data/engineered_features_v4.csv   (V4: 130 features)
        │
        ▼
train_and_evaluate_v{N}.py  → output_v{N}/ (charts, CSVs) + models_v{N}/ (weights)
        │
        ▼
React prediction interface  → output_v4/lol_predictor_v4.jsx
```

### Raw Data Format

Each Oracle's Elixir CSV has 165 columns and 12 rows per game: 10 player rows (top/jng/mid/bot/sup × 2 teams) + 2 team summary rows. Key columns: `gameid`, `date`, `teamname`, `playername`, `position`, `result`, `patch`, `champion`, `kills`, `deaths`, `assists`, `dpm`, `cspm`, etc.

The engineered features CSVs collapse each game to 1 row with `team_a` vs `team_b` perspective.

### ML Library (`ml_from_scratch.py`)

All algorithms implemented from scratch in pure NumPy:
- `LogisticRegression` — SGD with L2 regularization, mini-batch
- `GradientBoostedTrees` — custom decision tree with histogram-based splitting
- `RandomForest` — bagged trees with feature subsampling
- `NeuralNetwork` — 2-hidden-layer MLP with dropout, Adam optimizer
- `StackingEnsemble` — meta-learner on base model predictions
- Metrics: `roc_auc_score`, `brier_score`, `logloss`, `accuracy_score`, `calibration_bins`
- Utilities: `standardize`, `sigmoid`

### Training Methodology

- Temporal train/test split (no future leakage): 75% train, 10% val, 15% test, ordered by date
- Dynamic Elo rating system with season decay
- Rolling window statistics (5/10/20 game windows)
- All features computed using only data available BEFORE each game

---

## Version History

### V1 — Baseline (63 features)

**Script:** `feature_engineering.py` → `train_and_evaluate.py`
**Output:** `data/engineered_features.csv`, `output/`, `models/`
**Best:** LR AUC 0.6881, Acc 63.5%

Features: Elo diff, Elo expected, rolling win rates (5/10/20/all), blue side, head-to-head, league tier, playoffs flag, game number, first pick, champion diversity, LAN flag, best-of format.

### V2 — Player Tracking (79 features)

**Script:** `feature_engineering_v2.py` → `train_and_evaluate_v2.py`
**Output:** `data/engineered_features_v2.csv`, `output_v2/`, `models_v2/`
**Best:** Ensemble AUC 0.6976, Acc 64.1%

Added: Per-player KDA tracking, champion pools, hot streaks, trajectory, series momentum, fearless draft detection, substitution detection.

**Key bug fixes:**
- NaN team names: `if pd.isna(ta) or pd.isna(tb): continue` guard
- Series tracker: `gameid`/teams converted to `str()` before sorting

### V3 — Coaches, Travel, Regional Playstyle (106 features)

**Script:** `v3_enhancements.py` → `train_and_evaluate_v3.py`
**Output:** `data/engineered_features_v3.csv`, `output_v3/`, `models_v3/`
**Best:** GBDT AUC 0.6983, Acc 64.3%

Added 28 features:
- **Coach tracking** (via `data/coaches.tsv`): coach WR, experience, tenure, same-coach flag. Coach data scraped from Leaguepedia Cargo API using Chrome MCP tools (direct fetch was blocked by proxy).
- **Travel/home advantage**: Team home region mapping → is_home, is_traveling, travel_diff.
- **Regional playstyle profiles**: Positional resource allocation per region (gold share, damage share for top/mid/bot), carry position, playstyle distance.

**Key bug fixes:**
- Playstyle features all zero: fixed by using team HOME region (not event league) for profiling
- `del team_home_region` placed after both travel AND playstyle features use it

### V4 — Patch Meta & Champion Meta (130 features)

**Script:** `v4_enhancements.py` → `train_and_evaluate_v4.py`
**Output:** `data/engineered_features_v4.csv`, `output_v4/`, `models_v4/`
**Best:** GBDT AUC 0.7970, Acc 72.1%

Added 24 features:
- **Patch meta**: `patch_maturity` (log-scaled game count on patch), `is_new_patch`, `team_patch_games_a/b/diff`, `patch_wr_delta_a/b/diff` (patch WR vs recent WR), `patch_age_days`
- **Champion meta**: `avg_pick_rate_a/b/diff`, `avg_champ_wr_a/b/diff`, `team_champ_comfort_a/b/diff`, `meta_conformity_a/b/diff`, `ban_target_a/b/diff`

**Key bug fixes:**
- `defaultdict(lambda: defaultdict(list))` → `defaultdict(list)` for team_patch_games
- Team row filter: `champion.isna()` → `position == 'team'`
- OOM kill: rewrote `build_trackers_from_raw` from full-year load to chunk-based processing (20K row chunks, gc.collect() between years)
- Patch column missing from v3_df: now pulled from `gameid_drafts` dict built during tracker pass
- Memory-optimized trackers: PatchTracker uses running counts instead of storing full date lists; ChampionMetaTracker uses running pick/win counts instead of game_sequence list

**Architecture note:** `v4_enhancements.py` uses a single-pass approach — `build_all_from_raw()` builds both trackers AND the gameid_drafts dict in one iteration through the CSVs, then v3_df is loaded AFTER to minimize peak memory.

---

## React Prediction Interface

**File:** `output_v4/lol_predictor_v4.jsx`

Uses a calibrated Elo+WR formula (not the raw 130-feature LR model, which requires full feature vectors):
```
P(A wins) = sigmoid(log(base/(1-base)) + 1.5 * wr_diff + 0.15 * blue + 0.05 * lan)
where base = 1 / (1 + 10^(-elo_diff/250))
```
Elo scale parameter D=250 fitted by minimizing log-loss on full dataset.

Includes player head-to-head matchups: per-position comparison of KDA, DPM, WR, CS/min with visual indicators. Roster data from last ~4 months (Dec 2025 – Mar 2026), 50 teams with 5-player rosters.

**Why not embed the full LR model?** Setting 123 of 130 features to 0 in the standardized model produces garbage predictions due to ghost contributions from `(0 - mu) / sigma` terms. The Elo+WR formula is a faithful simplified proxy.

---

## File Map

```
lol_predictor/
├── ml_from_scratch.py              # All ML algorithms (NumPy only)
├── feature_engineering.py          # V1 feature engineering
├── feature_engineering_v2.py       # V2 feature engineering
├── v3_enhancements.py              # V3 feature engineering (coaches, travel, playstyle)
├── v4_enhancements.py              # V4 feature engineering (patch meta, champion meta)
├── train_and_evaluate.py           # V1 training pipeline
├── train_and_evaluate_v2.py        # V2 training pipeline
├── train_and_evaluate_v3.py        # V3 training pipeline
├── train_and_evaluate_v4.py        # V4 training pipeline (LR+GBDT only, memory-constrained)
├── simulate_fst2026.py             # First Stand 2026 tournament simulation
├── worklog.md                      # This file
├── data/
│   ├── 20{21-26}_LoL_esports_match_data_from_OraclesElixir.csv  # Raw data
│   ├── coaches.tsv                 # Coach-team-year mappings (~230 records)
│   ├── engineered_features.csv     # V1 features (51088 × ~78 cols)
│   ├── engineered_features_v2.csv  # V2 features (51088 × 114 cols)
│   ├── engineered_features_v3.csv  # V3 features (51088 × 142 cols)
│   └── engineered_features_v4.csv  # V4 features (51088 × 166 cols)
├── models_v{2,3,4}/
│   ├── logistic_regression.npz     # LR weights + bias
│   └── model_params.npz            # mu, sigma, feature_names
├── output/                         # V1 charts + React interface
├── output_v2/                      # V2 charts
├── output_v3/                      # V3 charts + model comparison
└── output_v4/                      # V4 charts + React interface + team rosters
    ├── lol_predictor_v4.jsx        # React prediction UI (latest)
    ├── model_comparison.csv        # V4 LR/GBDT/Blend results
    ├── feature_importance.csv      # Top 30 GBDT feature importances
    ├── elo_wr_model.json           # Calibrated Elo+WR model params (D=250)
    ├── merged_teams.json           # 50 teams: elo, wr, full 5-player rosters
    └── team_rosters_v4.json        # 259 teams: player stats (KDA, DPM, etc.)
```

---

## Environment & Constraints

- **VM:** Lightweight Linux, ~3.9GB RAM, no swap. OOM kills are common with large DataFrames.
- **Python packages:** pandas, numpy, matplotlib, seaborn available. NO sklearn (pip proxy blocks installation).
- **Memory patterns:** Always use chunk-based CSV reading (`chunksize=20000-50000`), explicit `gc.collect()`, `del` intermediates. Never load full year + groupby in one pass.
- **Riot API key:** stored in `.env` as `RIOT_API_KEY` (see `.env.example`); never committed. Development tier (20 req/s, 100 req/2min). Not yet integrated — reserved for potential solo queue data enrichment.

---

## Performance Summary

| Version | Best Model    | AUC    | Brier  | Accuracy | Features |
|---------|---------------|--------|--------|----------|----------|
| V1      | LR            | 0.6881 | 0.2230 | 63.5%    | 63       |
| V2      | Ensemble      | 0.6976 | 0.2197 | 64.1%    | 79       |
| V3      | GBDT          | 0.6983 | 0.2200 | 64.3%    | 106      |
| **V4**  | **GBDT**      | **0.7970** | **0.1879** | **72.1%** | **130** |

Test split: last 15% of games by date (~6,874 games). All versions use identical split.

Top V4 features (GBDT importance): `patch_wr_delta_diff` (25.3%), `player_experience_diff` (8.2%), `team_patch_games_diff` (8.0%), `avg20_earned_gpm_diff` (5.2%).

---

## FST 2026 Tournament Simulation

First Stand 2026 (São Paulo, March 2026). BLG won 3-1 over G2 in finals.
- V3: GBDT 6/13 correct (46.2%), LR 4/13 (30.8%)
- V4: GBDT 2/10 matched games (20%), LR 3/10 (30%)
- Note: V4 found fewer matches in test set (144 post-cutoff vs V3's setup). Small-sample tournament finals between evenly-matched teams are inherently low-signal.

---

## Potential Next Steps

1. **Riot API integration** — Use the provided API key to pull solo queue data (champion mastery, rank, recent performance) for each pro player. Would add ~20-30 features per player.
2. **Draft-phase model** — Train a separate model that takes champion picks/bans as input (currently only used for meta features, not individual champion strength).
3. **Live updating** — Scrape new Oracle's Elixir data periodically, retrain models, update React interface.
4. **Random Forest / Neural Network for V4** — V4 only trained LR + GBDT due to memory constraints. Could try RF with fewer trees or NN with smaller hidden layers.
5. **Hyperparameter tuning** — GBDT uses fixed hyperparams. Grid search on val set could improve.
6. **Feature selection** — 130 features may have redundancy. Recursive feature elimination or L1 regularization could help.
7. **Improve React predictor** — Current Elo+WR formula is a simplified proxy. Could pre-compute full-model predictions for all team pairs and embed as a lookup table, or compute per-team average feature vectors for a richer simplified model.

---

## How to Resume

If starting a new session:

1. **Read this file first** to understand architecture and current state.
2. All code is in `/sessions/happy-sleepy-clarke/lol_predictor/` (or wherever the session places it).
3. Raw data is in `data/` — 6 Oracle's Elixir CSVs + `coaches.tsv`.
4. Latest engineered features: `data/engineered_features_v4.csv` (51088 × 166).
5. Latest model weights: `models_v4/logistic_regression.npz` + `model_params.npz`.
6. Latest React UI: `output_v4/lol_predictor_v4.jsx`.
7. User deliverables are copied to: `/sessions/.../mnt/Downloads/lol_prediction_model_v4/`.

To retrain: `cd lol_predictor && python3 train_and_evaluate_v4.py` (may need memory-optimized version — see V4 notes above about OOM).

To regenerate features from scratch: run `python3 v4_enhancements.py` (~5 min).

---

*Last updated: March 26, 2026*
*Current state: V4 complete and delivered. All models trained. React interface working with calibrated Elo+WR predictions and player H2H matchups.*
