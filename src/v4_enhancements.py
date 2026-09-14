"""
V4 Enhancements for LoL Esports Match Prediction.

Enriches V3 features with:
1. Patch Meta Features: patch maturity, team patch performance, patch adaptation
2. Champion Meta Features: global pick/win rates, team comfort, ban targeting

Memory-optimized: single-pass tracker building, no full-history storage.
"""

import pandas as pd
import numpy as np
import os
import gc
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')


# ============================================================
# PATCH TRACKER (memory-optimized: counts not lists)
# ============================================================

class PatchTracker:
    """Tracks patch-level statistics and team performance on patches."""

    def __init__(self):
        self.patch_games_count = defaultdict(int)        # patch -> total games before current
        self.patch_first_seen = {}                        # patch -> first_date
        self.team_patch_wins = defaultdict(int)           # (team,patch) -> wins
        self.team_patch_count = defaultdict(int)          # (team,patch) -> count
        self.team_recent = defaultdict(list)              # team -> deque of last 10 results

    def record_game(self, patch, date, team_a, result_a, team_b, result_b):
        if pd.isna(patch):
            return
        p = str(patch)
        self.patch_games_count[p] += 1
        if p not in self.patch_first_seen:
            self.patch_first_seen[p] = date

        self.team_patch_count[(team_a, p)] += 1
        self.team_patch_count[(team_b, p)] += 1
        self.team_patch_wins[(team_a, p)] += result_a
        self.team_patch_wins[(team_b, p)] += result_b

        # Keep only last 10 for recent WR
        for team, res in [(team_a, result_a), (team_b, result_b)]:
            lst = self.team_recent[team]
            lst.append(res)
            if len(lst) > 15:
                self.team_recent[team] = lst[-10:]

    def get_patch_maturity(self, patch):
        if pd.isna(patch):
            return np.nan
        return np.log1p(self.patch_games_count.get(str(patch), 0))

    def get_is_new_patch(self, patch):
        if pd.isna(patch):
            return 0
        return 1 if self.patch_games_count.get(str(patch), 0) < 50 else 0

    def get_team_patch_games(self, team, patch):
        if pd.isna(patch):
            return 0
        return self.team_patch_count.get((team, str(patch)), 0)

    def get_patch_wr(self, team, patch):
        if pd.isna(patch):
            return np.nan
        key = (team, str(patch))
        count = self.team_patch_count.get(key, 0)
        if count == 0:
            return np.nan
        return self.team_patch_wins.get(key, 0) / count

    def get_recent_wr(self, team):
        lst = self.team_recent.get(team, [])
        if not lst:
            return np.nan
        return sum(lst[-10:]) / len(lst[-10:])

    def get_patch_wr_delta(self, team, patch):
        pwr = self.get_patch_wr(team, patch)
        rwr = self.get_recent_wr(team)
        if pd.isna(pwr) or pd.isna(rwr):
            return np.nan
        return pwr - rwr

    def get_patch_age_days(self, patch, date):
        if pd.isna(patch):
            return np.nan
        first = self.patch_first_seen.get(str(patch))
        if first is None:
            return np.nan
        return max(0, (date - first).days)


# ============================================================
# CHAMPION META TRACKER (memory-optimized: running counts)
# ============================================================

class ChampionMetaTracker:
    """Tracks champion pick rates, win rates, and team comfort using running counts."""

    def __init__(self):
        self.champ_picks = defaultdict(int)         # champ -> total picks
        self.champ_wins = defaultdict(int)           # champ -> total wins
        self.total_champ_entries = 0                 # total champion entries
        self.team_champ_recent = defaultdict(list)   # team -> last 30 game champ lists

    def record_game(self, champs_a, champs_b, result_a, result_b):
        for champ in champs_a:
            if pd.notna(champ) and champ != '':
                self.champ_picks[champ] += 1
                self.champ_wins[champ] += result_a
                self.total_champ_entries += 1
        for champ in champs_b:
            if pd.notna(champ) and champ != '':
                self.champ_picks[champ] += 1
                self.champ_wins[champ] += result_b
                self.total_champ_entries += 1

    def record_team_game(self, team_a, champs_a, team_b, champs_b):
        for team, champs in [(team_a, champs_a), (team_b, champs_b)]:
            lst = self.team_champ_recent[team]
            lst.append(champs)
            if len(lst) > 35:
                self.team_champ_recent[team] = lst[-30:]

    def get_global_pick_rate(self, champ):
        if self.total_champ_entries == 0:
            return np.nan
        return self.champ_picks.get(champ, 0) / self.total_champ_entries

    def get_global_win_rate(self, champ):
        picks = self.champ_picks.get(champ, 0)
        if picks == 0:
            return np.nan
        return self.champ_wins.get(champ, 0) / picks

    def get_team_champ_comfort(self, team, champs):
        games = self.team_champ_recent.get(team, [])
        window = games[-30:]
        if not window:
            return np.nan
        comforts = []
        for champ in champs:
            if pd.notna(champ) and champ != '':
                count = sum(1 for cg in window if champ in cg)
                comforts.append(count)
        return np.mean(comforts) if comforts else np.nan

    def get_meta_conformity(self, champs):
        if self.total_champ_entries == 0:
            return np.nan
        # Top 20 most-picked champs
        sorted_champs = sorted(self.champ_picks.keys(), key=lambda c: self.champ_picks[c], reverse=True)
        top20 = set(sorted_champs[:20])
        valid = [c for c in champs if pd.notna(c) and c != '']
        if not valid:
            return np.nan
        return sum(1 for c in valid if c in top20) / len(valid)

    def get_ban_target_count(self, team, bans):
        games = self.team_champ_recent.get(team, [])
        window = games[-30:]
        if not window:
            return 0
        team_champs = defaultdict(int)
        for cg in window:
            for c in cg:
                if pd.notna(c) and c != '':
                    team_champs[c] += 1
        valid_bans = [b for b in bans if pd.notna(b) and b != '']
        return sum(1 for b in valid_bans if team_champs.get(b, 0) > 0)


# ============================================================
# SINGLE-PASS: BUILD TRACKERS + DRAFT MAP
# ============================================================

def build_all_from_raw(data_dir):
    """Build trackers and gameid_drafts in a single pass through raw data."""
    print("\nBuilding trackers + draft map from raw data...")

    patch_tracker = PatchTracker()
    champ_tracker = ChampionMetaTracker()
    gameid_drafts = {}
    games_processed = 0
    seen_gameids = set()

    usecols = ['gameid', 'date', 'patch', 'teamname', 'champion', 'position',
               'ban1', 'ban2', 'ban3', 'ban4', 'ban5',
               'pick1', 'pick2', 'pick3', 'pick4', 'pick5', 'result']

    for year in range(2021, 2027):
        path = os.path.join(data_dir, f'{year}_LoL_esports_match_data_from_OraclesElixir.csv')
        if not os.path.exists(path):
            continue
        print(f"  Reading {year}...", end='', flush=True)
        try:
            for chunk in pd.read_csv(path, usecols=usecols, chunksize=20000, low_memory=False):
                chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
                team_chunk = chunk[chunk['position'] == 'team']
                player_chunk = chunk[chunk['position'].isin(['top', 'jng', 'mid', 'bot', 'sup'])]

                for gameid, game_teams in team_chunk.groupby('gameid'):
                    if gameid in seen_gameids:
                        continue
                    if len(game_teams) != 2:
                        continue
                    seen_gameids.add(gameid)

                    ra, rb = game_teams.iloc[0], game_teams.iloc[1]
                    date = ra['date']
                    patch = ra['patch']
                    ta = str(ra['teamname'])
                    tb = str(rb['teamname'])
                    res_a, res_b = ra['result'], rb['result']

                    if pd.isna(res_a) or pd.isna(res_b) or pd.isna(date):
                        continue
                    res_a, res_b = int(res_a), int(res_b)

                    # Patch tracking
                    patch_tracker.record_game(patch, date, ta, res_a, tb, res_b)

                    # Get champion picks
                    gp = player_chunk[player_chunk['gameid'] == gameid]
                    champs_a = gp[gp['teamname'] == ra['teamname']]['champion'].dropna().tolist()
                    champs_b = gp[gp['teamname'] == rb['teamname']]['champion'].dropna().tolist()

                    picks_a = [ra.get(f'pick{i}') for i in range(1, 6)]
                    picks_b = [rb.get(f'pick{i}') for i in range(1, 6)]
                    picks_a = [p for p in picks_a if pd.notna(p)]
                    picks_b = [p for p in picks_b if pd.notna(p)]

                    if not champs_a:
                        champs_a = picks_a
                    if not champs_b:
                        champs_b = picks_b

                    # Champion tracking
                    champ_tracker.record_game(champs_a, champs_b, res_a, res_b)
                    champ_tracker.record_team_game(ta, champs_a, tb, champs_b)

                    # Draft map (for later feature computation)
                    bans_a = [ra.get(f'ban{i}') for i in range(1, 6)]
                    bans_b = [rb.get(f'ban{i}') for i in range(1, 6)]
                    gameid_drafts[gameid] = {
                        'picks_a': picks_a if picks_a else champs_a,
                        'picks_b': picks_b if picks_b else champs_b,
                        'bans_a': bans_a,
                        'bans_b': bans_b,
                        'patch': patch,
                    }

                    games_processed += 1

                del team_chunk, player_chunk
            gc.collect()
            print(f" done ({games_processed} games)")
        except Exception as e:
            print(f" ERROR: {e}")
            import traceback
            traceback.print_exc()

    del seen_gameids
    gc.collect()
    print(f"  Trackers built with {games_processed} games, {len(gameid_drafts)} draft records")
    return patch_tracker, champ_tracker, gameid_drafts


# ============================================================
# FEATURE BUILDERS
# ============================================================

def add_patch_meta_features(v3_df, patch_tracker, gameid_drafts):
    """Add patch meta features to V3 dataframe."""
    print("\n" + "="*60)
    print("COMPUTING PATCH META FEATURES")
    print("="*60)

    n = len(v3_df)
    pf = {
        'patch_maturity': np.full(n, np.nan),
        'is_new_patch': np.zeros(n, dtype=int),
        'team_patch_games_a': np.zeros(n, dtype=int),
        'team_patch_games_b': np.zeros(n, dtype=int),
        'team_patch_games_diff': np.zeros(n, dtype=np.float32),
        'patch_wr_delta_a': np.full(n, np.nan),
        'patch_wr_delta_b': np.full(n, np.nan),
        'patch_wr_delta_diff': np.full(n, np.nan),
        'patch_age_days': np.full(n, np.nan),
    }

    for idx, row in v3_df.iterrows():
        ta, tb, date = row['team_a'], row['team_b'], row['date']
        draft = gameid_drafts.get(row['gameid'], {})
        patch = draft.get('patch', np.nan)

        pf['patch_maturity'][idx] = patch_tracker.get_patch_maturity(patch)
        pf['is_new_patch'][idx] = patch_tracker.get_is_new_patch(patch)

        tpg_a = patch_tracker.get_team_patch_games(ta, patch)
        tpg_b = patch_tracker.get_team_patch_games(tb, patch)
        pf['team_patch_games_a'][idx] = tpg_a
        pf['team_patch_games_b'][idx] = tpg_b
        pf['team_patch_games_diff'][idx] = tpg_a - tpg_b

        pwd_a = patch_tracker.get_patch_wr_delta(ta, patch)
        pwd_b = patch_tracker.get_patch_wr_delta(tb, patch)
        pf['patch_wr_delta_a'][idx] = pwd_a
        pf['patch_wr_delta_b'][idx] = pwd_b
        if not (pd.isna(pwd_a) or pd.isna(pwd_b)):
            pf['patch_wr_delta_diff'][idx] = pwd_a - pwd_b

        pf['patch_age_days'][idx] = patch_tracker.get_patch_age_days(patch, date)

        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1} games...")

    for col, vals in pf.items():
        v3_df[col] = vals
    print(f"Added {len(pf)} patch meta features")
    return v3_df


def add_champion_meta_features(v3_df, champ_tracker, gameid_drafts):
    """Add champion meta features using pre-built draft map."""
    print("\n" + "="*60)
    print("COMPUTING CHAMPION META FEATURES")
    print("="*60)

    n = len(v3_df)
    cf = {
        'avg_pick_rate_a': np.full(n, np.nan),
        'avg_pick_rate_b': np.full(n, np.nan),
        'avg_pick_rate_diff': np.full(n, np.nan),
        'avg_champ_wr_a': np.full(n, np.nan),
        'avg_champ_wr_b': np.full(n, np.nan),
        'avg_champ_wr_diff': np.full(n, np.nan),
        'team_champ_comfort_a': np.full(n, np.nan),
        'team_champ_comfort_b': np.full(n, np.nan),
        'team_champ_comfort_diff': np.full(n, np.nan),
        'meta_conformity_a': np.full(n, np.nan),
        'meta_conformity_b': np.full(n, np.nan),
        'meta_conformity_diff': np.full(n, np.nan),
        'ban_target_a': np.zeros(n, dtype=int),
        'ban_target_b': np.zeros(n, dtype=int),
        'ban_target_diff': np.zeros(n, dtype=np.float32),
    }

    matched = 0
    for idx, row in v3_df.iterrows():
        gameid = row['gameid']
        ta, tb = row['team_a'], row['team_b']

        draft = gameid_drafts.get(gameid)
        if not draft:
            continue
        matched += 1

        pa = draft['picks_a']
        pb = draft['picks_b']
        ba = draft['bans_a']
        bb = draft['bans_b']

        # Pick rates
        pra = [champ_tracker.get_global_pick_rate(c) for c in pa]
        prb = [champ_tracker.get_global_pick_rate(c) for c in pb]
        va = [r for r in pra if not pd.isna(r)]
        vb = [r for r in prb if not pd.isna(r)]
        if va:
            cf['avg_pick_rate_a'][idx] = np.mean(va)
        if vb:
            cf['avg_pick_rate_b'][idx] = np.mean(vb)
        if va and vb:
            cf['avg_pick_rate_diff'][idx] = np.mean(va) - np.mean(vb)

        # Win rates
        wra = [champ_tracker.get_global_win_rate(c) for c in pa]
        wrb = [champ_tracker.get_global_win_rate(c) for c in pb]
        vwa = [w for w in wra if not pd.isna(w)]
        vwb = [w for w in wrb if not pd.isna(w)]
        if vwa:
            cf['avg_champ_wr_a'][idx] = np.mean(vwa)
        if vwb:
            cf['avg_champ_wr_b'][idx] = np.mean(vwb)
        if vwa and vwb:
            cf['avg_champ_wr_diff'][idx] = np.mean(vwa) - np.mean(vwb)

        # Comfort
        ca = champ_tracker.get_team_champ_comfort(ta, pa)
        cb = champ_tracker.get_team_champ_comfort(tb, pb)
        cf['team_champ_comfort_a'][idx] = ca
        cf['team_champ_comfort_b'][idx] = cb
        if not (pd.isna(ca) or pd.isna(cb)):
            cf['team_champ_comfort_diff'][idx] = ca - cb

        # Meta conformity
        ma = champ_tracker.get_meta_conformity(pa)
        mb = champ_tracker.get_meta_conformity(pb)
        cf['meta_conformity_a'][idx] = ma
        cf['meta_conformity_b'][idx] = mb
        if not (pd.isna(ma) or pd.isna(mb)):
            cf['meta_conformity_diff'][idx] = ma - mb

        # Ban targeting
        bta = champ_tracker.get_ban_target_count(ta, bb)
        btb = champ_tracker.get_ban_target_count(tb, ba)
        cf['ban_target_a'][idx] = bta
        cf['ban_target_b'][idx] = btb
        cf['ban_target_diff'][idx] = bta - btb

        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1} games...")

    for col, vals in cf.items():
        v3_df[col] = vals
    print(f"Added {len(cf)} champion meta features (matched {matched}/{len(v3_df)} drafts)")
    return v3_df


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS_V3 = [
    'elo_diff', 'elo_expected', 'n_games_diff',
    'wr_diff_5', 'wr_diff_10', 'wr_diff_20', 'wr_diff_all',
    'wr_a_10', 'wr_b_10', 'wwr_diff_10',
    'is_blue_a', 'side_wr_diff', 'h2h_a',
    'league_tier', 'is_playoffs', 'game_number', 'first_pick_a',
    'champ_div_diff', 'is_lan', 'bo_format',
    'has_sub_a', 'has_sub_b', 'sub_diff',
    'series_games_played', 'series_score_diff', 'is_elimination', 'series_momentum_a',
    'is_fearless',
    'avg_player_kda_diff', 'avg_player_streak_diff', 'avg_player_trajectory_diff',
    'avg_champ_comfort_diff', 'avg_champ_games_diff', 'avg_player_pool_diff',
    'star_player_kda_diff', 'weakest_player_kda_diff', 'player_experience_diff',
    'avg10_teamkills_diff', 'avg10_teamdeaths_diff',
    'avg10_team_kpm_diff', 'avg10_ckpm_diff',
    'avg10_firstblood_diff', 'avg10_firstdragon_diff',
    'avg10_dragons_diff', 'avg10_opp_dragons_diff',
    'avg10_firstherald_diff', 'avg10_heralds_diff',
    'avg10_firstbaron_diff', 'avg10_barons_diff',
    'avg10_firsttower_diff', 'avg10_towers_diff',
    'avg10_dpm_diff', 'avg10_wpm_diff', 'avg10_vspm_diff',
    'avg10_earned_gpm_diff', 'avg10_cspm_diff',
    'avg10_golddiffat10_diff', 'avg10_xpdiffat10_diff', 'avg10_csdiffat10_diff',
    'avg10_golddiffat15_diff', 'avg10_xpdiffat15_diff', 'avg10_csdiffat15_diff',
    'avg10_turretplates_diff', 'avg10_gamelength_diff',
    'avg20_teamkills_diff', 'avg20_teamdeaths_diff',
    'avg20_dragons_diff', 'avg20_barons_diff', 'avg20_towers_diff',
    'avg20_golddiffat15_diff', 'avg20_earned_gpm_diff', 'avg20_dpm_diff',
    'kda_ratio_diff', 'dragon_control_diff', 'tower_control_diff',
    'baron_control_diff', 'vision_diff', 'composite_strength_diff',
    'coach_wr_a', 'coach_wr_b', 'coach_wr_diff',
    'coach_experience_a', 'coach_experience_b', 'coach_experience_diff',
    'same_coach_a', 'same_coach_b',
    'coach_tenure_a', 'coach_tenure_b', 'coach_tenure_diff',
    'team_a_home', 'team_b_home',
    'is_home_a', 'is_home_b',
    'is_traveling_a', 'is_traveling_b', 'travel_diff',
    'region_top_gold_diff', 'region_top_dmg_diff',
    'region_bot_gold_diff', 'region_bot_dmg_diff',
    'region_mid_gold_diff', 'region_mid_dmg_diff',
    'region_carry_position_a', 'region_carry_position_b',
    'carry_position_match', 'region_playstyle_distance',
]

FEATURE_COLUMNS_V4 = FEATURE_COLUMNS_V3 + [
    'patch_maturity', 'is_new_patch',
    'team_patch_games_a', 'team_patch_games_b', 'team_patch_games_diff',
    'patch_wr_delta_a', 'patch_wr_delta_b', 'patch_wr_delta_diff',
    'patch_age_days',
    'avg_pick_rate_a', 'avg_pick_rate_b', 'avg_pick_rate_diff',
    'avg_champ_wr_a', 'avg_champ_wr_b', 'avg_champ_wr_diff',
    'team_champ_comfort_a', 'team_champ_comfort_b', 'team_champ_comfort_diff',
    'meta_conformity_a', 'meta_conformity_b', 'meta_conformity_diff',
    'ban_target_a', 'ban_target_b', 'ban_target_diff',
]


def prepare_model_data(features_df, min_games=5):
    """Prepare data for modeling."""
    mask = (features_df['n_games_a'] >= min_games) & (features_df['n_games_b'] >= min_games)
    df = features_df[mask].copy()
    print(f"After filtering (>={min_games} games): {len(df)} (dropped {len(features_df)-len(df)})")

    avail = [c for c in FEATURE_COLUMNS_V4 if c in df.columns]
    print(f"Using {len(avail)} features")

    X = df[avail].values.astype(float)
    y = df['result'].values.astype(float)
    dates = df['date_numeric'].values.astype(float)

    X = np.nan_to_num(X, nan=0.0)
    X = np.clip(X, -1e6, 1e6)
    return X, y, dates, avail, df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    data_dir = 'data'

    print("\n" + "="*60)
    print("V4 FEATURE ENGINEERING (Patch & Champion Meta)")
    print("="*60)

    # Step 1: Build trackers + draft map (BEFORE loading v3_df)
    patch_tracker, champ_tracker, gameid_drafts = build_all_from_raw(data_dir)
    gc.collect()

    # Step 2: Load V3 features
    path = os.path.join(data_dir, 'engineered_features_v3.csv')
    print(f"\nLoading V3 features from {path}...")
    v3_df = pd.read_csv(path)
    v3_df['date'] = pd.to_datetime(v3_df['date'])
    print(f"  Loaded {len(v3_df)} games")

    # Step 3: Add patch features
    v3_df = add_patch_meta_features(v3_df, patch_tracker, gameid_drafts)
    del patch_tracker
    gc.collect()

    # Step 4: Add champion features
    v3_df = add_champion_meta_features(v3_df, champ_tracker, gameid_drafts)
    del champ_tracker, gameid_drafts
    gc.collect()

    # Step 5: Save
    print("\n" + "="*60)
    print("SAVING V4 FEATURES")
    print("="*60)
    output_path = os.path.join(data_dir, 'engineered_features_v4.csv')
    v3_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({v3_df.shape})")
    print(f"Date range: {v3_df['date'].min()} to {v3_df['date'].max()}")
    print(f"Total columns: {len(v3_df.columns)}")
    print(f"\nV4 features: {len(FEATURE_COLUMNS_V4)}")

    # Summary
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    v4_new = [c for c in FEATURE_COLUMNS_V4 if c not in FEATURE_COLUMNS_V3]
    print(f"New V4-specific features ({len(v4_new)}):")
    for col in v4_new:
        if col in v3_df.columns:
            non_null = v3_df[col].notna().sum()
            mean_val = v3_df[col].mean()
            std_val = v3_df[col].std()
            print(f"  {col:40s}: mean={mean_val:8.4f}, std={std_val:8.4f}, non-null={non_null}")

    print("\nDone!")
