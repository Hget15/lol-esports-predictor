"""
V4 Enhancements for LoL Esports Match Prediction.

Enriches V3 features with:
1. Patch Meta Features: patch maturity, team patch performance, patch adaptation
2. Champion Meta Features: global pick/win rates, team comfort, ban targeting

Every feature for a game is built only from games that finished before it:
games are collected, sorted by date, and each one is read from the trackers
before it is recorded (see build_pre_game_features). The first version of this
file did not do that and its results were inflated by look-ahead.
"""

import pandas as pd
import numpy as np
import os
import gc
from collections import defaultdict

# Warnings are deliberately left visible. A module-level
# warnings.filterwarnings('ignore') would hide real problems (silent NaN
# propagation, dtype coercion, deprecated pandas behaviour) behind a clean
# console. Known-benign cases are handled at the call site instead.


# ============================================================
# PATCH TRACKER (memory-optimized: counts not lists)
# ============================================================

class PatchTracker:
    """Tracks patch-level statistics and team performance on patches.

    Why this exists (the V4 idea): a League patch can shift the meta enough
    that last month's form is a weak guide to this week's game. So instead
    of only asking "how good is this team?", V4 also asks "how good is this
    team ON THIS PATCH, relative to its own recent form?" — see
    get_patch_wr_delta.

    The tracker itself has no notion of time: it answers from whatever has
    been recorded so far. It is only leak-free when the caller queries it
    BEFORE recording the game in question, in date order.

    Memory design: an earlier version stored a full per-(team, patch) list
    of dates/results and was OOM-killed on the 3.9 GB VM. This version keeps
    only running counts (wins, games) plus a short recent-results list per
    team, which is all the features actually need.
    """

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
        """Team's win rate on the current patch minus its recent (last-10) win rate.

        Positive = the team is playing above its own baseline since the patch
        dropped; negative = the new patch knocked them off form. The idea is
        to separate adaptation from raw strength. NaN until the team has a
        recorded game on the patch.

        Computed from recorded games only, so it is a pre-game quantity only
        if the current game has not been recorded yet. When it was queried
        after recording every game, it contained the result being predicted
        and looked like the strongest feature in the model; it is not.
        """
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
    """Tracks champion pick rates, win rates, and team comfort using running counts.

    Why: the draft is the one part of a pro game decided before minute 0,
    and it is rich with signal. Three angles are captured:
      * global meta   — how often a champion is picked and how often it
                        wins (a proxy for "is this pick currently strong?");
      * team comfort  — how many of a team's picks are champions they've
                        played recently (repetition = practised comps);
      * conformity /  — how much a team plays the consensus top-20 meta, and
        ban pressure    how many of the opponent's bans target this team's
                        recent picks (a "respect" signal: you ban what you
                        fear).
    Everything is a running count, so the whole tracker fits in a few MB.
    """

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
# COLLECT GAMES, THEN BUILD FEATURES IN DATE ORDER (NO LOOK-AHEAD)
# ============================================================

ROLES = ('top', 'jng', 'mid', 'bot', 'sup')
RAW_COLS = ['gameid', 'date', 'patch', 'teamname', 'champion', 'position',
            'ban1', 'ban2', 'ban3', 'ban4', 'ban5',
            'pick1', 'pick2', 'pick3', 'pick4', 'pick5', 'result']


def collect_games_from_raw(data_dir):
    """Read every game's draft and result from the raw CSVs, sorted by date.

    Returns one compact record per game: gameid, date, patch and, for each of
    the two teams, its name, result, champions played, draft picks and bans.
    Nothing is computed here. Only the 17 needed columns are loaded, so a
    whole year fits in memory comfortably and no game is split across a chunk
    boundary (the earlier chunked reader silently dropped games whose two
    team rows landed in different chunks).
    """
    print("\nCollecting games from raw data...")
    games = []
    for year in range(2021, 2027):
        path = os.path.join(data_dir, f'{year}_LoL_esports_match_data_from_OraclesElixir.csv')
        if not os.path.exists(path):
            continue
        print(f"  Reading {year}...", end='', flush=True)
        df = pd.read_csv(path, usecols=RAW_COLS, low_memory=False)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        n_before = len(games)
        for gameid, g in df.groupby('gameid', sort=False):
            teams = g[g['position'] == 'team']
            if len(teams) != 2:
                continue
            ra, rb = teams.iloc[0], teams.iloc[1]
            if pd.isna(ra['result']) or pd.isna(rb['result']) or pd.isna(ra['date']):
                continue
            players = g[g['position'].isin(ROLES)]
            sides = []
            for r in (ra, rb):
                champs = players[players['teamname'] == r['teamname']]['champion'].dropna().tolist()
                picks = [r[f'pick{i}'] for i in range(1, 6) if pd.notna(r[f'pick{i}'])]
                bans = [r[f'ban{i}'] for i in range(1, 6) if pd.notna(r[f'ban{i}'])]
                sides.append({'team': str(r['teamname']), 'result': int(r['result']),
                              'champs': champs or picks, 'picks': picks or champs, 'bans': bans})
            games.append({'gameid': str(gameid), 'date': ra['date'], 'patch': ra['patch'], 'sides': sides})
        print(f" done ({len(games) - n_before} games)")
        del df
        gc.collect()

    # Date order is what makes "before this game" well defined. The raw files
    # are only roughly chronological, so sort explicitly.
    games.sort(key=lambda g: (g['date'], g['gameid']))
    print(f"  Collected {len(games)} games")
    return games


def _mean_or_nan(values):
    vals = [v for v in values if not pd.isna(v)]
    return float(np.mean(vals)) if vals else np.nan


def build_pre_game_features(games):
    """Walk the games in date order: read the trackers, THEN record the game.

    This ordering is the whole point. An earlier version of this file filled
    both trackers with every game in the dataset first and queried them
    afterwards, so "win rate on the current patch" included the result of the
    game being predicted (and of later games), and "recent form" was the
    team's last ten games of the entire dataset. That look-ahead inflated the
    reported V4 test AUC to 0.797 (it is 0.72 without it) and made
    patch_wr_delta_diff look like the dominant feature. Querying before
    recording removes it: each game's features are built only from games that
    finished earlier.

    Features are stored per team name, so they attach to the right side of
    the feature table whichever order the raw file listed the two teams in.
    """
    patch_tracker = PatchTracker()
    champ_tracker = ChampionMetaTracker()
    feats = {}

    for g in games:
        patch, date = g['patch'], g['date']
        a, b = g['sides']

        entry = {
            'patch_maturity': patch_tracker.get_patch_maturity(patch),
            'is_new_patch': patch_tracker.get_is_new_patch(patch),
            'patch_age_days': patch_tracker.get_patch_age_days(patch, date),
            'teams': {},
        }
        for me, opp in ((a, b), (b, a)):
            entry['teams'][me['team']] = {
                'team_patch_games': patch_tracker.get_team_patch_games(me['team'], patch),
                'patch_wr_delta': patch_tracker.get_patch_wr_delta(me['team'], patch),
                'avg_pick_rate': _mean_or_nan(champ_tracker.get_global_pick_rate(c) for c in me['picks']),
                'avg_champ_wr': _mean_or_nan(champ_tracker.get_global_win_rate(c) for c in me['picks']),
                'team_champ_comfort': champ_tracker.get_team_champ_comfort(me['team'], me['picks']),
                'meta_conformity': champ_tracker.get_meta_conformity(me['picks']),
                # "Respect" signal: how many of the OPPONENT's bans hit this team's recent picks.
                'ban_target': champ_tracker.get_ban_target_count(me['team'], opp['bans']),
            }
        feats[g['gameid']] = entry

        # Only now does this game become part of history.
        patch_tracker.record_game(patch, date, a['team'], a['result'], b['team'], b['result'])
        champ_tracker.record_game(a['champs'], b['champs'], a['result'], b['result'])
        champ_tracker.record_team_game(a['team'], a['champs'], b['team'], b['champs'])

    print(f"  Built pre-game features for {len(feats)} games")
    return feats


def _lookup(game_feats, row):
    """Pre-game features for this row's game and its two teams (None if unmatched)."""
    entry = game_feats.get(str(row['gameid']))
    if not entry:
        return None, None, None
    return entry, entry['teams'].get(str(row['team_a'])), entry['teams'].get(str(row['team_b']))


# ============================================================
# FEATURE BUILDERS
# ============================================================

def add_patch_meta_features(v3_df, game_feats):
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

    matched = 0
    for idx, row in v3_df.iterrows():
        entry, fa, fb = _lookup(game_feats, row)
        if fa is None or fb is None:
            continue
        matched += 1

        pf['patch_maturity'][idx] = entry['patch_maturity']
        pf['is_new_patch'][idx] = entry['is_new_patch']
        pf['patch_age_days'][idx] = entry['patch_age_days']

        pf['team_patch_games_a'][idx] = fa['team_patch_games']
        pf['team_patch_games_b'][idx] = fb['team_patch_games']
        pf['team_patch_games_diff'][idx] = fa['team_patch_games'] - fb['team_patch_games']

        pwd_a, pwd_b = fa['patch_wr_delta'], fb['patch_wr_delta']
        pf['patch_wr_delta_a'][idx] = pwd_a
        pf['patch_wr_delta_b'][idx] = pwd_b
        if not (pd.isna(pwd_a) or pd.isna(pwd_b)):
            pf['patch_wr_delta_diff'][idx] = pwd_a - pwd_b

        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1} games...")

    # Attach all new columns with ONE concat rather than inserting them in a
    # loop. Each single-column insert appends a new internal block, so the
    # frame becomes progressively more fragmented (pandas raises a
    # PerformanceWarning) and every later insert gets slower. Building the
    # block once is faster and yields identical values and column order.
    v3_df = pd.concat([v3_df, pd.DataFrame(pf, index=v3_df.index)], axis=1)
    print(f"Added {len(pf)} patch meta features (matched {matched}/{n} games)")
    return v3_df


def add_champion_meta_features(v3_df, game_feats):
    """Add champion meta features to V3 dataframe."""
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
        _, fa, fb = _lookup(game_feats, row)
        if fa is None or fb is None:
            continue
        matched += 1

        for name in ('avg_pick_rate', 'avg_champ_wr', 'team_champ_comfort', 'meta_conformity'):
            va, vb = fa[name], fb[name]
            cf[f'{name}_a'][idx] = va
            cf[f'{name}_b'][idx] = vb
            if not (pd.isna(va) or pd.isna(vb)):
                cf[f'{name}_diff'][idx] = va - vb

        cf['ban_target_a'][idx] = fa['ban_target']
        cf['ban_target_b'][idx] = fb['ban_target']
        cf['ban_target_diff'][idx] = fa['ban_target'] - fb['ban_target']

        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1} games...")

    # Single concat instead of per-column inserts (see add_patch_meta_features):
    # avoids DataFrame fragmentation; values and column order are identical.
    v3_df = pd.concat([v3_df, pd.DataFrame(cf, index=v3_df.index)], axis=1)
    print(f"Added {len(cf)} champion meta features (matched {matched}/{n} games)")
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

    # Step 1: Collect games and build every game's features from earlier games only
    games = collect_games_from_raw(data_dir)
    game_feats = build_pre_game_features(games)
    del games
    gc.collect()

    # Step 2: Load V3 features
    path = os.path.join(data_dir, 'engineered_features_v3.csv')
    print(f"\nLoading V3 features from {path}...")
    v3_df = pd.read_csv(path)
    v3_df['date'] = pd.to_datetime(v3_df['date'])
    print(f"  Loaded {len(v3_df)} games")

    # Step 3: Add patch features
    v3_df = add_patch_meta_features(v3_df, game_feats)

    # Step 4: Add champion features
    v3_df = add_champion_meta_features(v3_df, game_feats)
    del game_feats
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
