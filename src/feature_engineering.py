"""
Feature Engineering Pipeline for LoL Esports Match Prediction.
OPTIMIZED VERSION - uses vectorized pandas operations, minimal memory.
"""

import pandas as pd
import numpy as np
import os
import gc
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# DATA LOADING (memory optimized)
# ============================================================

# Only load columns we actually need
TEAM_COLS = [
    'gameid', 'datacompleteness', 'league', 'year', 'split', 'playoffs',
    'date', 'game', 'patch', 'side', 'position', 'teamname', 'teamid',
    'firstPick', 'result', 'gamelength',
    'teamkills', 'teamdeaths', 'team kpm', 'ckpm',
    'firstblood', 'firstdragon', 'dragons', 'opp_dragons',
    'elementaldrakes', 'firstherald', 'heralds', 'opp_heralds',
    'void_grubs', 'opp_void_grubs', 'firstbaron', 'barons', 'opp_barons',
    'firsttower', 'towers', 'opp_towers',
    'inhibitors', 'opp_inhibitors',
    'dpm', 'wpm', 'wcpm', 'vspm', 'earned gpm', 'cspm',
    'golddiffat10', 'xpdiffat10', 'csdiffat10',
    'golddiffat15', 'xpdiffat15', 'csdiffat15',
    'golddiffat20', 'xpdiffat20', 'csdiffat20',
    'turretplates', 'opp_turretplates',
    'wardsplaced', 'wardskilled', 'visionscore',
    'earnedgold', 'totalgold'
]

PLAYER_COLS = ['gameid', 'position', 'playername', 'teamname', 'champion', 'datacompleteness']


def load_all_data(data_dir='data'):
    """Load only needed columns from all years."""
    team_frames = []
    player_frames = []

    for year in range(2021, 2027):
        path = os.path.join(data_dir, f'{year}_LoL_esports_match_data_from_OraclesElixir.csv')
        if not os.path.exists(path):
            continue

        print(f"Loading {year}...", end=' ')
        # Read header to check available columns
        header = pd.read_csv(path, nrows=0).columns.tolist()
        team_cols_avail = [c for c in TEAM_COLS if c in header]
        player_cols_avail = [c for c in PLAYER_COLS if c in header]

        df = pd.read_csv(path, usecols=list(set(team_cols_avail + player_cols_avail)), low_memory=False)
        df = df[df['datacompleteness'] == 'complete']

        team_df = df[df['position'] == 'team'][team_cols_avail].copy()
        player_df = df[df['position'] != 'team'][player_cols_avail].copy()

        print(f"team: {len(team_df)}, player: {len(player_df)}")
        team_frames.append(team_df)
        player_frames.append(player_df)
        del df
        gc.collect()

    team_all = pd.concat(team_frames, ignore_index=True)
    player_all = pd.concat(player_frames, ignore_index=True)

    # Parse dates
    team_all['date'] = pd.to_datetime(team_all['date'], errors='coerce')
    team_all = team_all.sort_values('date').reset_index(drop=True)
    team_all['result'] = team_all['result'].astype(int)
    team_all['date_numeric'] = team_all['date'].astype(np.int64) // 10**9

    print(f"\nTotal: {len(team_all)} team rows ({len(team_all)//2} games), {len(player_all)} player rows")
    return team_all, player_all


# ============================================================
# BUILD GAME-LEVEL ROSTER/CHAMPION DATA (vectorized)
# ============================================================

def build_game_roster_data(player_df):
    """Build dicts of players and champions per game/team."""
    print("Building roster data...")
    game_players = {}
    game_champions = {}

    grouped = player_df.groupby(['gameid', 'teamname'])
    for (gid, team), group in grouped:
        key = (gid, team)
        game_players[key] = group['playername'].dropna().tolist()
        game_champions[key] = group['champion'].dropna().tolist()

    print(f"  {len(game_players)} game-team entries")
    return game_players, game_champions


# ============================================================
# LEAGUE TIERS
# ============================================================

MAJOR_LEAGUES = {'LCK', 'LPL', 'LEC', 'LCS', 'LTA', 'LTA North', 'LTA South', 'LCP'}
INTL_EVENTS = {'WLDs', 'MSI', 'Worlds'}

def get_league_tier(league):
    if not isinstance(league, str):
        return 1
    if league in MAJOR_LEAGUES or any(x in league for x in ['Worlds', 'MSI', 'WLDs']):
        return 3
    return 2 if league in {'PCS', 'VCS', 'CBLOL', 'LJL', 'LLA', 'LCO', 'TCL',
                           'LCK CL', 'LDL', 'NACL', 'LFL', 'PRM', 'NLC'} else 1


# ============================================================
# ELO + STATS TRACKER (optimized with numpy arrays)
# ============================================================

class EloSystem:
    def __init__(self, initial=1500, k=32):
        self.initial = initial
        self.k = k
        self.ratings = {}

    def get(self, team):
        return self.ratings.get(team, self.initial)

    def update(self, team_a, team_b, result_a, tier=1):
        ra, rb = self.get(team_a), self.get(team_b)
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        k = self.k * (0.5 + 0.5 * tier / 3)
        self.ratings[team_a] = ra + k * (result_a - ea)
        self.ratings[team_b] = rb + k * ((1 - result_a) - (1 - ea))
        return ea

    def decay(self, factor=0.75):
        for t in self.ratings:
            self.ratings[t] = self.initial + factor * (self.ratings[t] - self.initial)


class FastTracker:
    """Memory-efficient stats tracker using fixed-size rolling buffers."""

    STAT_KEYS = [
        'result', 'gamelength', 'teamkills', 'teamdeaths', 'team kpm', 'ckpm',
        'firstblood', 'firstdragon', 'dragons', 'opp_dragons',
        'elementaldrakes', 'firstherald', 'heralds', 'opp_heralds',
        'void_grubs', 'opp_void_grubs', 'firstbaron', 'barons', 'opp_barons',
        'firsttower', 'towers', 'opp_towers', 'inhibitors', 'opp_inhibitors',
        'dpm', 'wpm', 'wcpm', 'vspm', 'earned gpm', 'cspm',
        'golddiffat10', 'xpdiffat10', 'csdiffat10',
        'golddiffat15', 'xpdiffat15', 'csdiffat15',
        'golddiffat20', 'xpdiffat20', 'csdiffat20',
        'turretplates', 'opp_turretplates'
    ]
    STAT_IDX = {k: i for i, k in enumerate(STAT_KEYS)}
    N_STATS = len(STAT_KEYS)
    BUFFER_SIZE = 30  # Keep last 30 games

    def __init__(self):
        self.buffers = {}  # team -> (np.array of shape [BUFFER_SIZE, N_STATS], write_pos, count)
        self.sides = {}    # team -> list of (side, result) tuples (last 30)
        self.h2h = {}      # (team_a, team_b) -> list of results
        self.rosters = {}  # team -> set of player names
        self.champs = {}   # team -> list of recent champ sets

    def _get_buf(self, team):
        if team not in self.buffers:
            self.buffers[team] = (np.full((self.BUFFER_SIZE, self.N_STATS), np.nan), 0, 0)
        return self.buffers[team]

    def add_game(self, team, opponent, row, champions):
        buf, pos, count = self._get_buf(team)

        for key, idx in self.STAT_IDX.items():
            val = row.get(key, np.nan)
            try:
                buf[pos % self.BUFFER_SIZE, idx] = float(val) if pd.notna(val) else np.nan
            except:
                buf[pos % self.BUFFER_SIZE, idx] = np.nan

        self.buffers[team] = (buf, pos + 1, count + 1)

        # Side tracking
        if team not in self.sides:
            self.sides[team] = []
        side = row.get('side', '')
        result = float(row.get('result', 0))
        self.sides[team].append((side, result))
        if len(self.sides[team]) > 60:
            self.sides[team] = self.sides[team][-60:]

        # H2H
        key = (team, opponent)
        if key not in self.h2h:
            self.h2h[key] = []
        self.h2h[key].append(result)
        if len(self.h2h[key]) > 20:
            self.h2h[key] = self.h2h[key][-20:]

        # Champions
        if team not in self.champs:
            self.champs[team] = []
        self.champs[team].append(set(champions))
        if len(self.champs[team]) > 30:
            self.champs[team] = self.champs[team][-30:]

    def rolling_mean(self, team, n, stat_key):
        buf, pos, count = self._get_buf(team)
        if count == 0:
            return np.nan
        idx = self.STAT_IDX.get(stat_key)
        if idx is None:
            return np.nan
        actual_n = min(n, count, self.BUFFER_SIZE)
        indices = [(pos - 1 - i) % self.BUFFER_SIZE for i in range(actual_n)]
        vals = buf[indices, idx]
        valid = vals[~np.isnan(vals)]
        return np.mean(valid) if len(valid) > 0 else np.nan

    def win_rate(self, team, n=None):
        buf, pos, count = self._get_buf(team)
        if count == 0:
            return 0.5
        idx = self.STAT_IDX['result']
        actual_n = min(n or count, count, self.BUFFER_SIZE)
        indices = [(pos - 1 - i) % self.BUFFER_SIZE for i in range(actual_n)]
        vals = buf[indices, idx]
        valid = vals[~np.isnan(vals)]
        return np.mean(valid) if len(valid) > 0 else 0.5

    def side_wr(self, team, side, n=30):
        entries = self.sides.get(team, [])
        matches = [(s, r) for s, r in entries[-n*2:] if s == side]
        if not matches:
            return 0.5
        return np.mean([r for _, r in matches[-n:]])

    def h2h_wr(self, team, opp, n=10):
        results = self.h2h.get((team, opp), [])
        if not results:
            return 0.5
        return np.mean(results[-n:])

    def game_count(self, team):
        _, _, count = self._get_buf(team)
        return count

    def champ_diversity(self, team, n=20):
        recent = self.champs.get(team, [])[-n:]
        all_champs = set()
        for s in recent:
            all_champs.update(s)
        return len(all_champs)

    def roster_overlap(self, team, current_players):
        prev = self.rosters.get(team, set())
        if not prev or not current_players:
            return 0.5
        return len(prev.intersection(set(current_players))) / max(len(current_players), 1)

    def update_roster(self, team, players):
        self.rosters[team] = set(players)


# ============================================================
# MAIN FEATURE BUILDER
# ============================================================

def build_features(team_df, game_players, game_champions):
    print("\n=== Building Features ===")
    elo = EloSystem()
    tracker = FastTracker()

    feature_rows = []
    seen = set()
    current_year = None

    n_total = len(team_df)
    for idx in range(0, n_total - 1, 2):
        row_a = team_df.iloc[idx]
        row_b = team_df.iloc[idx + 1]

        if row_a['gameid'] != row_b['gameid']:
            continue

        gid = row_a['gameid']
        if gid in seen:
            continue
        seen.add(gid)

        # Year boundary -> Elo decay
        yr = row_a.get('year')
        if yr and current_year and yr != current_year:
            elo.decay(0.75)
        current_year = yr

        ta, tb = row_a['teamname'], row_b['teamname']
        sa, sb = row_a['side'], row_b['side']
        ra = int(row_a['result'])
        league = row_a.get('league', '')
        tier = get_league_tier(league)

        # === PRE-MATCH FEATURES ===
        ea, eb = elo.get(ta), elo.get(tb)
        elo_exp = 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))

        na, nb = tracker.game_count(ta), tracker.game_count(tb)

        f = {
            'gameid': gid, 'date': row_a['date'],
            'date_numeric': row_a.get('date_numeric', 0),
            'team_a': ta, 'team_b': tb, 'side_a': sa,
            'league': league, 'result': ra,

            'elo_diff': ea - eb, 'elo_expected': elo_exp,
            'n_games_a': na, 'n_games_b': nb, 'n_games_diff': na - nb,

            'wr_diff_5': tracker.win_rate(ta, 5) - tracker.win_rate(tb, 5),
            'wr_diff_10': tracker.win_rate(ta, 10) - tracker.win_rate(tb, 10),
            'wr_diff_20': tracker.win_rate(ta, 20) - tracker.win_rate(tb, 20),
            'wr_diff_all': tracker.win_rate(ta) - tracker.win_rate(tb),
            'wr_a_10': tracker.win_rate(ta, 10),
            'wr_b_10': tracker.win_rate(tb, 10),

            'is_blue_a': 1 if sa == 'Blue' else 0,
            'side_wr_diff': tracker.side_wr(ta, sa) - tracker.side_wr(tb, sb),
            'h2h_a': tracker.h2h_wr(ta, tb),

            'league_tier': tier,
            'is_playoffs': int(row_a.get('playoffs', 0)),
            'game_number': int(row_a['game']) if pd.notna(row_a.get('game')) else 1,
            'first_pick_a': int(row_a['firstPick']) if pd.notna(row_a.get('firstPick')) else 0,

            'champ_div_diff': tracker.champ_diversity(ta) - tracker.champ_diversity(tb),
        }

        # Roster overlap
        pl_a = game_players.get((gid, ta), [])
        pl_b = game_players.get((gid, tb), [])
        f['roster_overlap_diff'] = tracker.roster_overlap(ta, pl_a) - tracker.roster_overlap(tb, pl_b)

        # Rolling stat diffs (10-game)
        diff_stats = [
            'teamkills', 'teamdeaths', 'team kpm', 'ckpm',
            'firstblood', 'firstdragon', 'dragons', 'opp_dragons',
            'firstherald', 'heralds', 'opp_heralds',
            'firstbaron', 'barons', 'opp_barons',
            'firsttower', 'towers', 'opp_towers',
            'dpm', 'wpm', 'vspm', 'earned gpm', 'cspm',
            'golddiffat10', 'xpdiffat10', 'csdiffat10',
            'golddiffat15', 'xpdiffat15', 'csdiffat15',
            'turretplates', 'opp_turretplates', 'gamelength'
        ]
        for sk in diff_stats:
            va = tracker.rolling_mean(ta, 10, sk)
            vb = tracker.rolling_mean(tb, 10, sk)
            safe = sk.replace(' ', '_')
            f[f'avg10_{safe}_diff'] = (va - vb) if (pd.notna(va) and pd.notna(vb)) else np.nan

        # 20-game key stats
        for sk in ['teamkills', 'teamdeaths', 'dragons', 'barons', 'towers', 'golddiffat15', 'earned gpm', 'dpm']:
            va = tracker.rolling_mean(ta, 20, sk)
            vb = tracker.rolling_mean(tb, 20, sk)
            safe = sk.replace(' ', '_')
            f[f'avg20_{safe}_diff'] = (va - vb) if (pd.notna(va) and pd.notna(vb)) else np.nan

        # Derived metrics
        ka = tracker.rolling_mean(ta, 10, 'teamkills') or 0
        da_t = tracker.rolling_mean(ta, 10, 'teamdeaths') or 1
        kb = tracker.rolling_mean(tb, 10, 'teamkills') or 0
        db_t = tracker.rolling_mean(tb, 10, 'teamdeaths') or 1
        f['kda_ratio_diff'] = (ka / max(da_t, 0.5)) - (kb / max(db_t, 0.5))

        # Dragon control
        dr_a = tracker.rolling_mean(ta, 10, 'dragons') or 0
        odr_a = tracker.rolling_mean(ta, 10, 'opp_dragons') or 0
        dr_b = tracker.rolling_mean(tb, 10, 'dragons') or 0
        odr_b = tracker.rolling_mean(tb, 10, 'opp_dragons') or 0
        dc_a = dr_a / max(dr_a + odr_a, 0.5)
        dc_b = dr_b / max(dr_b + odr_b, 0.5)
        f['dragon_control_diff'] = dc_a - dc_b

        # Tower control
        tw_a = tracker.rolling_mean(ta, 10, 'towers') or 0
        otw_a = tracker.rolling_mean(ta, 10, 'opp_towers') or 0
        tw_b = tracker.rolling_mean(tb, 10, 'towers') or 0
        otw_b = tracker.rolling_mean(tb, 10, 'opp_towers') or 0
        f['tower_control_diff'] = tw_a / max(tw_a + otw_a, 0.5) - tw_b / max(tw_b + otw_b, 0.5)

        # Baron control
        ba_a = tracker.rolling_mean(ta, 10, 'barons') or 0
        oba_a = tracker.rolling_mean(ta, 10, 'opp_barons') or 0
        ba_b = tracker.rolling_mean(tb, 10, 'barons') or 0
        oba_b = tracker.rolling_mean(tb, 10, 'opp_barons') or 0
        f['baron_control_diff'] = ba_a / max(ba_a + oba_a, 0.01) - ba_b / max(ba_b + oba_b, 0.01)

        # Vision diff
        vs_a = tracker.rolling_mean(ta, 10, 'vspm')
        vs_b = tracker.rolling_mean(tb, 10, 'vspm')
        f['vision_diff'] = ((vs_a or 0) - (vs_b or 0))

        # Composite strength
        gd15_a = tracker.rolling_mean(ta, 10, 'golddiffat15') or 0
        gd15_b = tracker.rolling_mean(tb, 10, 'golddiffat15') or 0
        f['composite_strength_diff'] = (
            0.3 * (ea - eb) / 200 +
            0.25 * (tracker.win_rate(ta, 10) - tracker.win_rate(tb, 10)) * 2 +
            0.2 * f['kda_ratio_diff'] +
            0.15 * f['dragon_control_diff'] * 2 +
            0.1 * (gd15_a - gd15_b) / 1000
        )

        feature_rows.append(f)

        # === UPDATE TRACKERS ===
        ch_a = game_champions.get((gid, ta), [])
        ch_b = game_champions.get((gid, tb), [])
        tracker.add_game(ta, tb, row_a, ch_a)
        tracker.add_game(tb, ta, row_b, ch_b)
        tracker.update_roster(ta, pl_a)
        tracker.update_roster(tb, pl_b)
        elo.update(ta, tb, ra, tier)

        if len(feature_rows) % 5000 == 0:
            print(f"  {len(feature_rows)} games processed...")

    print(f"\nTotal: {len(feature_rows)} games with features")
    return pd.DataFrame(feature_rows), elo, tracker


# ============================================================
# FEATURE COLUMNS FOR MODELING
# ============================================================

FEATURE_COLUMNS = [
    'elo_diff', 'elo_expected',
    'n_games_diff',
    'wr_diff_5', 'wr_diff_10', 'wr_diff_20', 'wr_diff_all',
    'wr_a_10', 'wr_b_10',
    'is_blue_a', 'side_wr_diff', 'h2h_a',
    'league_tier', 'is_playoffs', 'game_number', 'first_pick_a',
    'roster_overlap_diff', 'champ_div_diff',
    # 10-game rolling diffs
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
    # 20-game rolling diffs
    'avg20_teamkills_diff', 'avg20_teamdeaths_diff',
    'avg20_dragons_diff', 'avg20_barons_diff', 'avg20_towers_diff',
    'avg20_golddiffat15_diff', 'avg20_earned_gpm_diff', 'avg20_dpm_diff',
    # Derived
    'kda_ratio_diff', 'dragon_control_diff', 'tower_control_diff',
    'baron_control_diff', 'vision_diff', 'composite_strength_diff',
]


def prepare_model_data(features_df, min_games=5):
    mask = (features_df['n_games_a'] >= min_games) & (features_df['n_games_b'] >= min_games)
    df = features_df[mask].copy()
    print(f"After filtering (>={min_games} games per team): {len(df)} games (dropped {len(features_df)-len(df)})")

    avail = [c for c in FEATURE_COLUMNS if c in df.columns]
    # Also add any n_games features
    for c in df.columns:
        if c.startswith('avg') and c.endswith('_diff') and c not in avail:
            avail.append(c)

    print(f"Using {len(avail)} features")

    X = df[avail].values.astype(float)
    y = df['result'].values.astype(float)
    dates = df['date_numeric'].values.astype(float)

    X = np.nan_to_num(X, nan=0.0)
    X = np.clip(X, -1e6, 1e6)

    return X, y, dates, avail, df


if __name__ == "__main__":
    team_df, player_df = load_all_data('data')

    game_players, game_champions = build_game_roster_data(player_df)
    del player_df
    gc.collect()

    features_df, elo_sys, tracker = build_features(team_df, game_players, game_champions)
    del team_df, game_players, game_champions
    gc.collect()

    features_df.to_csv('data/engineered_features.csv', index=False)
    print(f"\nSaved: data/engineered_features.csv ({features_df.shape})")
    print(f"Date range: {features_df['date'].min()} to {features_df['date'].max()}")
    print(f"Win rate: {features_df['result'].mean():.3f}")
    print(f"Leagues: {features_df['league'].nunique()}")
