"""
V3 Enhancements for LoL Esports Match Prediction (Memory-Optimized).

Enriches V2 features with:
1. Coach tracking: win rates, experience, tenure, stability
2. Travel/LAN: regional mapping, home/away indicators
3. Regional playstyle: position-based resource allocation profiles
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
# CONSTANTS
# ============================================================

LEAGUE_TO_REGION = {
    'LCK': 'KR', 'LPL': 'CN', 'LEC': 'EU', 'LCS': 'NA', 'LTA North': 'NA',
    'LTA South': 'LATAM', 'CBLOL': 'BR', 'LCP': 'APAC', 'PCS': 'APAC',
    'VCS': 'VN', 'LJL': 'JP', 'KeSPA': 'KR',
    'First Stand': 'INT', 'Worlds': 'INT', 'MSI': 'INT', 'WCS': 'INT',
}

REGION_TO_INT = {
    'KR': 0, 'CN': 1, 'EU': 2, 'NA': 3, 'LATAM': 4, 'BR': 5,
    'APAC': 6, 'VN': 7, 'JP': 8, 'INT': 9,
}


# ============================================================
# COACH TRACKER
# ============================================================

class CoachTracker:
    """Compact coach performance tracker.

    Why coaches: rosters change, but a coaching staff's drafting philosophy
    and preparation can persist across them, and a long-tenured coach/team
    pairing is itself a stability signal. Coach-team-year mappings were
    scraped from Leaguepedia (data/coaches.tsv). The tracker exposes a
    coach's win rate with a given team, experience (games coached) and
    tenure (seasons together), each computed only from games before the
    current date.
    """

    def __init__(self, window=30):
        self.window = window
        self.coach_games = defaultdict(list)
        self.coach_tenure = defaultdict(lambda: defaultdict(int))

    def record_game(self, coach, team, result, date):
        if pd.isna(coach) or coach == '':
            return
        key = (coach, team)
        self.coach_games[key].append((date, result))

    def update_tenure(self, coach, team, year):
        if pd.isna(coach) or coach == '':
            return
        self.coach_tenure[coach][team] += 1

    def get_wr(self, coach, team, date):
        if pd.isna(coach) or coach == '':
            return np.nan
        key = (coach, team)
        games = self.coach_games[key]
        if len(games) == 0:
            return np.nan
        past_games = [g for g in games if g[0] < date]
        if len(past_games) == 0:
            return np.nan
        last_games = past_games[-self.window:]
        if len(last_games) == 0:
            return np.nan
        wins = sum(1 for _, result in last_games if result == 1)
        return wins / len(last_games)

    def get_experience(self, coach, team, date):
        if pd.isna(coach) or coach == '':
            return 0
        key = (coach, team)
        games = self.coach_games[key]
        if len(games) == 0:
            return 0
        past_games = [g for g in games if g[0] < date]
        return len(past_games)

    def get_tenure(self, coach, team):
        if pd.isna(coach) or coach == '':
            return 0
        return self.coach_tenure[coach][team]


# ============================================================
# REGIONAL PLAYSTYLE TRACKER
# ============================================================

class RegionalPlaystyleTracker:
    """Lightweight regional playstyle tracker - uses simple averages.

    Why: regions play the game differently — how gold and damage are split
    between top/mid/bot says whether a region funnels resources into one
    carry lane or spreads them. Comparing two teams' regional profiles gives
    a "style clash" distance feature.

    Lesson baked in from a V3 bug: profiles must be keyed by a team's HOME
    region, not the league of the event being played. At international
    events every team shares the same event "league", so keying on it gave
    both sides identical profiles and every playstyle feature came out zero.
    Running sums/counts only — no per-game history — to stay light on memory.
    """

    def __init__(self, window=200):
        self.window = window
        # Just store running sums and counts per region-position (no history)
        self.region_position_dmg = defaultdict(lambda: [0, 0])  # region,position -> [sum, count]
        self.region_position_gold = defaultdict(lambda: [0, 0])

    def record_game(self, region, position, damageshare, earnedgoldshare, date):
        if pd.isna(region) or pd.isna(position):
            return
        if pd.isna(damageshare):
            damageshare = 0
        if pd.isna(earnedgoldshare):
            earnedgoldshare = 0
        key = (region, position)
        self.region_position_dmg[key][0] += damageshare
        self.region_position_dmg[key][1] += 1
        self.region_position_gold[key][0] += earnedgoldshare
        self.region_position_gold[key][1] += 1

    def get_position_stats(self, region, position, date):
        key = (region, position)
        dmg_data = self.region_position_dmg.get(key, [0, 0])
        gold_data = self.region_position_gold.get(key, [0, 0])

        if dmg_data[1] == 0 or gold_data[1] == 0:
            return np.nan, np.nan

        dmg_avg = dmg_data[0] / dmg_data[1] if dmg_data[1] > 0 else 0
        gold_avg = gold_data[0] / gold_data[1] if gold_data[1] > 0 else 0
        return dmg_avg, gold_avg

    def get_carry_position(self, region, date):
        positions = ['top', 'jng', 'mid', 'bot', 'sup']
        pos_map = {'top': 1, 'jng': 2, 'mid': 3, 'bot': 4, 'sup': 5}
        golds = {}

        for pos in positions:
            key = (region, pos)
            gold_data = self.region_position_gold.get(key, [0, 0])
            if gold_data[1] == 0:
                continue
            golds[pos] = gold_data[0] / gold_data[1]

        if not golds:
            return np.nan
        carry_pos = max(golds, key=golds.get)
        return pos_map.get(carry_pos, np.nan)


# ============================================================
# DATA LOADING
# ============================================================

def load_v2_features(data_dir):
    """Load V2 features."""
    path = os.path.join(data_dir, 'engineered_features_v2.csv')
    print(f"Loading V2 features from {path}...")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} games")
    return df


def load_coaches(data_dir):
    """Load coach data."""
    path = os.path.join(data_dir, 'coaches.tsv')
    print(f"Loading coaches from {path}...")
    coaches_df = pd.read_csv(path, sep='\t')
    print(f"  Loaded {len(coaches_df)} coach-team-year records")
    return coaches_df


def build_team_home_region_from_raw(data_dir):
    """Build team->home_region mapping by reading raw data in chunks."""
    print("\nBuilding team home region mapping from raw data...")
    team_leagues = defaultdict(lambda: defaultdict(int))

    for year in range(2021, 2027):
        path = os.path.join(data_dir, f'{year}_LoL_esports_match_data_from_OraclesElixir.csv')
        if os.path.exists(path):
            print(f"  Reading {year}...", end='', flush=True)
            try:
                for chunk in pd.read_csv(path, usecols=['teamname', 'league'], chunksize=50000):
                    for _, row in chunk.iterrows():
                        team_leagues[row['teamname']][row['league']] += 1
                print(f" done")
            except Exception as e:
                print(f" ERROR: {e}")
                continue

    team_home_region = {}
    for team, leagues in team_leagues.items():
        if leagues:
            primary_league = max(leagues, key=leagues.get)
            region = LEAGUE_TO_REGION.get(primary_league, 'INT')
            team_home_region[team] = region

    print(f"  Mapped {len(team_home_region)} teams to home regions")
    return team_home_region


def build_coach_tracker_from_raw(data_dir, coaches_df):
    """Build coach tracker from raw data."""
    print("\nBuilding coach tracker from raw data...")

    # Build coach map
    coach_map = {}
    for _, row in coaches_df.iterrows():
        key = (row['team'], int(row['year']))
        coach_map[key] = row['coach']

    coach_tracker = CoachTracker(window=30)

    for year in range(2021, 2027):
        path = os.path.join(data_dir, f'{year}_LoL_esports_match_data_from_OraclesElixir.csv')
        if os.path.exists(path):
            print(f"  Reading {year}...", end='', flush=True)
            try:
                # Load only needed columns
                for chunk in pd.read_csv(path, usecols=['teamname', 'year', 'date', 'result'], chunksize=50000):
                    chunk['date'] = pd.to_datetime(chunk['date'])
                    chunk = chunk.sort_values('date')
                    for _, row in chunk.iterrows():
                        team = row['teamname']
                        y = int(row['year'])
                        date = row['date']
                        result = row['result']
                        if pd.isna(result):
                            continue
                        key = (team, y)
                        coach = coach_map.get(key)
                        if coach:
                            coach_tracker.record_game(coach, team, result, date)
                            prev_key = (team, y - 1)
                            prev_coach = coach_map.get(prev_key)
                            if prev_coach == coach:
                                coach_tracker.update_tenure(coach, team, y)
                print(f" done")
            except Exception as e:
                print(f" ERROR: {e}")
                continue

    print(f"  Coach tracker built")
    return coach_tracker


def build_playstyle_tracker_from_raw(data_dir):
    """Build playstyle tracker from raw data - memory optimized."""
    print("\nBuilding playstyle tracker from raw data...")

    playstyle_tracker = RegionalPlaystyleTracker(window=200)
    games_processed = 0

    for year in range(2021, 2027):
        path = os.path.join(data_dir, f'{year}_LoL_esports_match_data_from_OraclesElixir.csv')
        if os.path.exists(path):
            print(f"  Reading {year}...", end='', flush=True)
            try:
                for chunk in pd.read_csv(path, usecols=['league', 'position', 'damageshare', 'earnedgoldshare', 'date'], chunksize=100000):
                    chunk['date'] = pd.to_datetime(chunk['date'])
                    chunk = chunk[chunk['position'].isin(['top', 'jng', 'mid', 'bot', 'sup'])].copy()
                    chunk['league'] = chunk['league'].fillna('INT')
                    for _, row in chunk.iterrows():
                        league = row['league']
                        region = LEAGUE_TO_REGION.get(league, 'INT')
                        position = row['position']
                        dmg = row.get('damageshare', 0) if not pd.isna(row.get('damageshare')) else 0
                        gold = row.get('earnedgoldshare', 0) if not pd.isna(row.get('earnedgoldshare')) else 0
                        date = row['date']
                        playstyle_tracker.record_game(region, position, dmg, gold, date)
                        games_processed += 1
                print(f" done ({games_processed} games)")
            except Exception as e:
                print(f" ERROR: {e}")
                continue

    print(f"  Playstyle tracker built with {games_processed} games")
    return playstyle_tracker


# ============================================================
# FEATURE BUILDERS
# ============================================================

def add_coach_features(v2_df, coaches_df, coach_tracker):
    """Add coach features to V2 dataframe."""
    print("\n" + "="*60)
    print("COMPUTING COACH FEATURES")
    print("="*60)

    # Build coach map
    coach_map = {}
    for _, row in coaches_df.iterrows():
        key = (row['team'], int(row['year']))
        coach_map[key] = row['coach']

    v2_df['date'] = pd.to_datetime(v2_df['date'])

    coach_features = {
        'coach_wr_a': np.full(len(v2_df), np.nan),
        'coach_wr_b': np.full(len(v2_df), np.nan),
        'coach_wr_diff': np.full(len(v2_df), np.nan),
        'coach_experience_a': np.zeros(len(v2_df)),
        'coach_experience_b': np.zeros(len(v2_df)),
        'coach_experience_diff': np.zeros(len(v2_df)),
        'same_coach_a': np.zeros(len(v2_df), dtype=int),
        'same_coach_b': np.zeros(len(v2_df), dtype=int),
        'coach_tenure_a': np.zeros(len(v2_df)),
        'coach_tenure_b': np.zeros(len(v2_df)),
        'coach_tenure_diff': np.zeros(len(v2_df)),
    }

    for idx, row in v2_df.iterrows():
        team_a = row['team_a']
        team_b = row['team_b']
        year = int(row['date'].year)
        date = row['date']

        key_a = (team_a, year)
        key_b = (team_b, year)
        coach_a = coach_map.get(key_a)
        coach_b = coach_map.get(key_b)

        wr_a = coach_tracker.get_wr(coach_a, team_a, date)
        wr_b = coach_tracker.get_wr(coach_b, team_b, date)
        coach_features['coach_wr_a'][idx] = wr_a
        coach_features['coach_wr_b'][idx] = wr_b
        if not (pd.isna(wr_a) or pd.isna(wr_b)):
            coach_features['coach_wr_diff'][idx] = wr_a - wr_b

        exp_a = coach_tracker.get_experience(coach_a, team_a, date)
        exp_b = coach_tracker.get_experience(coach_b, team_b, date)
        coach_features['coach_experience_a'][idx] = exp_a
        coach_features['coach_experience_b'][idx] = exp_b
        coach_features['coach_experience_diff'][idx] = exp_a - exp_b

        prev_key_a = (team_a, year - 1)
        prev_key_b = (team_b, year - 1)
        prev_coach_a = coach_map.get(prev_key_a)
        prev_coach_b = coach_map.get(prev_key_b)
        coach_features['same_coach_a'][idx] = 1 if coach_a == prev_coach_a and coach_a else 0
        coach_features['same_coach_b'][idx] = 1 if coach_b == prev_coach_b and coach_b else 0

        tenure_a = coach_tracker.get_tenure(coach_a, team_a) if coach_a else 0
        tenure_b = coach_tracker.get_tenure(coach_b, team_b) if coach_b else 0
        coach_features['coach_tenure_a'][idx] = tenure_a
        coach_features['coach_tenure_b'][idx] = tenure_b
        coach_features['coach_tenure_diff'][idx] = tenure_a - tenure_b

        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1} games...")

    # Attach all new columns with ONE concat rather than inserting them in a
    # loop. Each single-column insert appends a new internal block, so the
    # frame becomes progressively more fragmented (pandas raises a
    # PerformanceWarning) and every later insert gets slower. Building the
    # block once is faster and yields identical values and column order.
    v2_df = pd.concat([v2_df, pd.DataFrame(coach_features, index=v2_df.index)], axis=1)

    print(f"Added {len(coach_features)} coach features")
    return v2_df


def add_travel_features(v2_df, team_home_region):
    """Add travel/regional features to V2 dataframe.

    Why: at international events one side may be playing at (or near) home
    while the other has crossed several time zones — a real, if modest,
    disadvantage in a reaction-time game. is_home / is_traveling /
    travel_diff encode that from each team's home region vs. the event's.
    """
    print("\n" + "="*60)
    print("COMPUTING TRAVEL/REGIONAL FEATURES")
    print("="*60)

    travel_features = {
        'team_a_home': np.zeros(len(v2_df), dtype=int),
        'team_b_home': np.zeros(len(v2_df), dtype=int),
        'is_home_a': np.zeros(len(v2_df), dtype=int),
        'is_home_b': np.zeros(len(v2_df), dtype=int),
        'is_traveling_a': np.zeros(len(v2_df), dtype=int),
        'is_traveling_b': np.zeros(len(v2_df), dtype=int),
        'travel_diff': np.zeros(len(v2_df), dtype=int),
    }

    for idx, row in v2_df.iterrows():
        team_a = row['team_a']
        team_b = row['team_b']
        league = row['league']

        home_a = team_home_region.get(team_a, 'INT')
        home_b = team_home_region.get(team_b, 'INT')
        travel_features['team_a_home'][idx] = REGION_TO_INT.get(home_a, 9)
        travel_features['team_b_home'][idx] = REGION_TO_INT.get(home_b, 9)

        league_region = LEAGUE_TO_REGION.get(league, 'INT')

        is_home_a = 1 if home_a == league_region else 0
        is_home_b = 1 if home_b == league_region else 0
        travel_features['is_home_a'][idx] = is_home_a
        travel_features['is_home_b'][idx] = is_home_b

        is_travel_a = 1 if league_region == 'INT' and home_a != 'INT' else 0
        is_travel_b = 1 if league_region == 'INT' and home_b != 'INT' else 0
        travel_features['is_traveling_a'][idx] = is_travel_a
        travel_features['is_traveling_b'][idx] = is_travel_b

        travel_features['travel_diff'][idx] = is_travel_a - is_travel_b

        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1} games...")

    # Single concat instead of per-column inserts (see add_coach_features):
    # avoids DataFrame fragmentation; values and column order are identical.
    v2_df = pd.concat([v2_df, pd.DataFrame(travel_features, index=v2_df.index)], axis=1)

    print(f"Added {len(travel_features)} travel/regional features")
    return v2_df


def add_playstyle_features(v2_df, playstyle_tracker, team_home_region):
    """Add playstyle features to V2 dataframe using team HOME regions."""
    print("\n" + "="*60)
    print("COMPUTING PLAYSTYLE FEATURES")
    print("="*60)

    playstyle_features = {
        'region_top_gold_diff': np.full(len(v2_df), np.nan),
        'region_top_dmg_diff': np.full(len(v2_df), np.nan),
        'region_bot_gold_diff': np.full(len(v2_df), np.nan),
        'region_bot_dmg_diff': np.full(len(v2_df), np.nan),
        'region_mid_gold_diff': np.full(len(v2_df), np.nan),
        'region_mid_dmg_diff': np.full(len(v2_df), np.nan),
        'region_carry_position_a': np.full(len(v2_df), np.nan),
        'region_carry_position_b': np.full(len(v2_df), np.nan),
        'carry_position_match': np.zeros(len(v2_df), dtype=int),
        'region_playstyle_distance': np.full(len(v2_df), np.nan),
    }

    v2_df['date'] = pd.to_datetime(v2_df['date'])

    for idx, row in v2_df.iterrows():
        ta = str(row.get('team_a', ''))
        tb = str(row.get('team_b', ''))
        # Use team HOME region (not current league) for playstyle profiling
        region_a = team_home_region.get(ta, LEAGUE_TO_REGION.get(row.get('league', 'INT'), 'INT'))
        region_b = team_home_region.get(tb, LEAGUE_TO_REGION.get(row.get('league', 'INT'), 'INT'))
        date = row['date']

        top_dmg_a, top_gold_a = playstyle_tracker.get_position_stats(region_a, 'top', date)
        top_dmg_b, top_gold_b = playstyle_tracker.get_position_stats(region_b, 'top', date)

        bot_dmg_a, bot_gold_a = playstyle_tracker.get_position_stats(region_a, 'bot', date)
        bot_dmg_b, bot_gold_b = playstyle_tracker.get_position_stats(region_b, 'bot', date)

        mid_dmg_a, mid_gold_a = playstyle_tracker.get_position_stats(region_a, 'mid', date)
        mid_dmg_b, mid_gold_b = playstyle_tracker.get_position_stats(region_b, 'mid', date)

        if not (pd.isna(top_gold_a) or pd.isna(top_gold_b)):
            playstyle_features['region_top_gold_diff'][idx] = top_gold_a - top_gold_b
        if not (pd.isna(top_dmg_a) or pd.isna(top_dmg_b)):
            playstyle_features['region_top_dmg_diff'][idx] = top_dmg_a - top_dmg_b
        if not (pd.isna(bot_gold_a) or pd.isna(bot_gold_b)):
            playstyle_features['region_bot_gold_diff'][idx] = bot_gold_a - bot_gold_b
        if not (pd.isna(bot_dmg_a) or pd.isna(bot_dmg_b)):
            playstyle_features['region_bot_dmg_diff'][idx] = bot_dmg_a - bot_dmg_b
        if not (pd.isna(mid_gold_a) or pd.isna(mid_gold_b)):
            playstyle_features['region_mid_gold_diff'][idx] = mid_gold_a - mid_gold_b
        if not (pd.isna(mid_dmg_a) or pd.isna(mid_dmg_b)):
            playstyle_features['region_mid_dmg_diff'][idx] = mid_dmg_a - mid_dmg_b

        carry_a = playstyle_tracker.get_carry_position(region_a, date)
        carry_b = playstyle_tracker.get_carry_position(region_b, date)
        playstyle_features['region_carry_position_a'][idx] = carry_a
        playstyle_features['region_carry_position_b'][idx] = carry_b

        playstyle_features['carry_position_match'][idx] = 1 if (carry_a == carry_b and not pd.isna(carry_a)) else 0

        profile_a = np.array([top_gold_a, top_dmg_a, bot_gold_a, bot_dmg_a, mid_gold_a, mid_dmg_a])
        profile_b = np.array([top_gold_b, top_dmg_b, bot_gold_b, bot_dmg_b, mid_gold_b, mid_dmg_b])

        if not (np.isnan(profile_a).any() or np.isnan(profile_b).any()):
            playstyle_features['region_playstyle_distance'][idx] = np.linalg.norm(profile_a - profile_b)

        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1} games...")

    # Single concat instead of per-column inserts (see add_coach_features):
    # avoids DataFrame fragmentation; values and column order are identical.
    v2_df = pd.concat([v2_df, pd.DataFrame(playstyle_features, index=v2_df.index)], axis=1)

    print(f"Added {len(playstyle_features)} playstyle features")
    return v2_df


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS_V2 = [
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
]

FEATURE_COLUMNS_V3 = FEATURE_COLUMNS_V2 + [
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


def prepare_model_data(features_df, min_games=5):
    """Prepare data for modeling."""
    mask = (features_df['n_games_a'] >= min_games) & (features_df['n_games_b'] >= min_games)
    df = features_df[mask].copy()
    print(f"After filtering (>={min_games} games): {len(df)} (dropped {len(features_df)-len(df)})")

    avail = [c for c in FEATURE_COLUMNS_V3 if c in df.columns]
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
    print("V3 FEATURE ENGINEERING (Memory-Optimized)")
    print("="*60)

    # Load
    v2_df = load_v2_features(data_dir)
    coaches_df = load_coaches(data_dir)

    # Build from raw data (chunk-based to save memory)
    team_home_region = build_team_home_region_from_raw(data_dir)
    coach_tracker = build_coach_tracker_from_raw(data_dir, coaches_df)
    playstyle_tracker = build_playstyle_tracker_from_raw(data_dir)

    del coaches_df
    gc.collect()

    # Add features
    v2_df = add_coach_features(v2_df, pd.read_csv(os.path.join(data_dir, 'coaches.tsv'), sep='\t'), coach_tracker)
    del coach_tracker
    gc.collect()

    v2_df = add_travel_features(v2_df, team_home_region)

    v2_df = add_playstyle_features(v2_df, playstyle_tracker, team_home_region)
    del team_home_region, playstyle_tracker
    gc.collect()

    # Save
    print("\n" + "="*60)
    print("SAVING V3 FEATURES")
    print("="*60)
    output_path = os.path.join(data_dir, 'engineered_features_v3.csv')
    v2_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({v2_df.shape})")
    print(f"Date range: {v2_df['date'].min()} to {v2_df['date'].max()}")
    print(f"Total columns: {len(v2_df.columns)}")
    print(f"\nV3 features: {len(FEATURE_COLUMNS_V3)}")

    # Summary
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    v3_new = [c for c in FEATURE_COLUMNS_V3 if c not in FEATURE_COLUMNS_V2]
    print(f"New V3-specific features ({len(v3_new)}):")
    for col in v3_new:
        if col in v2_df.columns:
            non_null = v2_df[col].notna().sum()
            mean_val = v2_df[col].mean()
            std_val = v2_df[col].std()
            print(f"  {col:40s}: mean={mean_val:8.4f}, std={std_val:8.4f}, non-null={non_null}")

    print("\nDone!")
