"""
ENHANCED Feature Engineering V2 for LoL Esports Match Prediction.

New features over V1:
- Player-level performance tracking (per-position rolling stats)
- Individual player hot streaks and slumps
- Player trajectory (improving vs declining)
- Champion pool depth and comfort picks per player
- Substitution detection and roster instability
- Series format (Bo1/Bo3/Bo5) with momentum and psychological features
- Fearless draft detection
- Online vs LAN context
- Series-weighted game history (Bo3/Bo5 games weighted higher)
"""

import pandas as pd
import numpy as np
import os
import gc
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# DATA LOADING (same as V1 but with more player columns)
# ============================================================

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

PLAYER_COLS = [
    'gameid', 'datacompleteness', 'position', 'playername', 'playerid',
    'teamname', 'champion', 'side', 'result', 'date',
    'kills', 'deaths', 'assists', 'damagetochampions', 'dpm',
    'damageshare', 'earnedgold', 'earned gpm', 'earnedgoldshare',
    'cspm', 'visionscore', 'vspm', 'golddiffat10', 'golddiffat15',
    'xpdiffat10', 'xpdiffat15', 'csdiffat10', 'csdiffat15',
    'firstbloodkill', 'firstbloodvictim',
    'doublekills', 'triplekills', 'quadrakills', 'pentakills',
    'game', 'league', 'gamelength'
]


def load_all_data(data_dir='data'):
    team_frames = []
    player_frames = []

    for year in range(2021, 2027):
        path = os.path.join(data_dir, f'{year}_LoL_esports_match_data_from_OraclesElixir.csv')
        if not os.path.exists(path):
            continue
        print(f"Loading {year}...", end=' ')
        header = pd.read_csv(path, nrows=0).columns.tolist()
        tcols = [c for c in TEAM_COLS if c in header]
        pcols = [c for c in PLAYER_COLS if c in header]
        all_cols = list(set(tcols + pcols))
        df = pd.read_csv(path, usecols=all_cols, low_memory=False)
        df = df[df['datacompleteness'] == 'complete']

        team_df = df[df['position'] == 'team'].copy()
        player_df = df[df['position'] != 'team'].copy()
        print(f"team: {len(team_df)}, player: {len(player_df)}")
        team_frames.append(team_df)
        player_frames.append(player_df)
        del df
        gc.collect()

    team_all = pd.concat(team_frames, ignore_index=True)
    player_all = pd.concat(player_frames, ignore_index=True)
    team_all['date'] = pd.to_datetime(team_all['date'], errors='coerce')
    player_all['date'] = pd.to_datetime(player_all['date'], errors='coerce')
    team_all = team_all.sort_values('date').reset_index(drop=True)
    player_all = player_all.sort_values('date').reset_index(drop=True)
    team_all['result'] = team_all['result'].astype(int)
    team_all['date_numeric'] = team_all['date'].astype(np.int64) // 10**9
    print(f"\nTotal: {len(team_all)} team rows, {len(player_all)} player rows")
    return team_all, player_all


# ============================================================
# LEAGUE / CONTEXT CLASSIFICATION
# ============================================================

MAJOR_LEAGUES = {'LCK', 'LPL', 'LEC', 'LCS', 'LTA', 'LTA North', 'LTA South', 'LCP'}
LAN_EVENTS = {'WLDs', 'MSI', 'Worlds', 'LCK', 'LPL', 'LEC', 'LCS'}  # Major leagues play on LAN

def get_league_tier(league):
    if not isinstance(league, str): return 1
    if league in MAJOR_LEAGUES or any(x in league for x in ['Worlds', 'MSI', 'WLDs']): return 3
    if league in {'PCS', 'VCS', 'CBLOL', 'LJL', 'LLA', 'LCO', 'TCL', 'LCK CL', 'LDL', 'NACL', 'LFL', 'PRM', 'NLC'}: return 2
    return 1

def is_lan_event(league):
    if not isinstance(league, str): return 0
    if any(x in league for x in ['Worlds', 'MSI', 'WLDs']): return 1
    if league in MAJOR_LEAGUES: return 1
    return 0


# ============================================================
# PLAYER PERFORMANCE TRACKER
# ============================================================

class PlayerTracker:
    """Tracks individual player performance, champion pools, hot streaks."""
    BUF_SIZE = 30
    STATS = ['kills', 'deaths', 'assists', 'dpm', 'cspm', 'vspm',
             'earned gpm', 'golddiffat15', 'damageshare', 'earnedgoldshare']
    STAT_IDX = {k: i for i, k in enumerate(STATS)}
    N = len(STATS)

    def __init__(self):
        self.buffers = {}      # playerid -> (np.array [BUF_SIZE, N], pos, count)
        self.champ_stats = {}  # playerid -> {champ: [wins, games]}
        self.positions = {}    # playerid -> most recent position
        self.names = {}        # playerid -> name
        self.results = {}      # playerid -> list of recent results (for streaks)

    def _buf(self, pid):
        if pid not in self.buffers:
            self.buffers[pid] = (np.full((self.BUF_SIZE, self.N), np.nan), 0, 0)
        return self.buffers[pid]

    def add_game(self, pid, pname, position, champion, result, stats_dict):
        buf, pos, count = self._buf(pid)
        for key, idx in self.STAT_IDX.items():
            val = stats_dict.get(key, np.nan)
            try: buf[pos % self.BUF_SIZE, idx] = float(val) if pd.notna(val) else np.nan
            except: buf[pos % self.BUF_SIZE, idx] = np.nan
        self.buffers[pid] = (buf, pos + 1, count + 1)
        self.positions[pid] = position
        self.names[pid] = pname

        # Champion stats
        if pid not in self.champ_stats:
            self.champ_stats[pid] = {}
        if champion and str(champion) != 'nan':
            if champion not in self.champ_stats[pid]:
                self.champ_stats[pid][champion] = [0, 0]
            self.champ_stats[pid][champion][1] += 1
            if result: self.champ_stats[pid][champion][0] += 1

        # Result history for streaks
        if pid not in self.results:
            self.results[pid] = []
        self.results[pid].append(float(result))
        if len(self.results[pid]) > 30:
            self.results[pid] = self.results[pid][-30:]

    def rolling_mean(self, pid, n, stat):
        buf, pos, count = self._buf(pid)
        if count == 0: return np.nan
        idx = self.STAT_IDX.get(stat)
        if idx is None: return np.nan
        actual_n = min(n, count, self.BUF_SIZE)
        indices = [(pos - 1 - i) % self.BUF_SIZE for i in range(actual_n)]
        vals = buf[indices, idx]
        valid = vals[~np.isnan(vals)]
        return np.mean(valid) if len(valid) > 0 else np.nan

    def game_count(self, pid):
        _, _, count = self._buf(pid)
        return count

    def win_rate(self, pid, n=None):
        res = self.results.get(pid, [])
        if not res: return 0.5
        if n: res = res[-n:]
        return np.mean(res) if res else 0.5

    def hot_streak(self, pid, short=3, long=15):
        """Difference between recent short-term WR and longer-term WR. Positive = hot."""
        res = self.results.get(pid, [])
        if len(res) < short: return 0.0
        short_wr = np.mean(res[-short:])
        long_wr = np.mean(res[-min(long, len(res)):])
        return short_wr - long_wr

    def trajectory(self, pid, stat='dpm', recent=5, historical=20):
        """Compare recent performance to historical. Positive = improving."""
        buf, pos, count = self._buf(pid)
        if count < recent + 2: return 0.0
        idx = self.STAT_IDX.get(stat)
        if idx is None: return 0.0

        # Recent
        r_n = min(recent, count, self.BUF_SIZE)
        r_idx = [(pos - 1 - i) % self.BUF_SIZE for i in range(r_n)]
        recent_vals = buf[r_idx, idx]
        recent_vals = recent_vals[~np.isnan(recent_vals)]

        # Historical (the n games before recent)
        h_n = min(historical, count - recent, self.BUF_SIZE - recent)
        if h_n <= 0: return 0.0
        h_idx = [(pos - 1 - recent - i) % self.BUF_SIZE for i in range(h_n)]
        hist_vals = buf[h_idx, idx]
        hist_vals = hist_vals[~np.isnan(hist_vals)]

        if len(recent_vals) == 0 or len(hist_vals) == 0: return 0.0
        hist_mean = np.mean(hist_vals)
        if abs(hist_mean) < 1e-6: return 0.0
        return (np.mean(recent_vals) - hist_mean) / (abs(hist_mean) + 1e-6)

    def champ_pool_size(self, pid, n_recent=20):
        """Number of unique champions played recently."""
        cs = self.champ_stats.get(pid, {})
        return len(cs)

    def champ_comfort(self, pid, champion):
        """Win rate on this specific champion. Returns (wr, n_games)."""
        cs = self.champ_stats.get(pid, {})
        if champion not in cs: return 0.5, 0
        wins, games = cs[champion]
        return wins / max(games, 1), games

    def kda(self, pid, n=10):
        k = self.rolling_mean(pid, n, 'kills') or 0
        d = self.rolling_mean(pid, n, 'deaths') or 1
        a = self.rolling_mean(pid, n, 'assists') or 0
        return (k + a) / max(d, 0.5)


# ============================================================
# SERIES TRACKER
# ============================================================

class SeriesTracker:
    """Track series context: Bo format, game momentum within series."""

    def __init__(self):
        # Key: (date_approx, team_a, team_b) -> list of game results
        self.active_series = {}

    def get_series_key(self, gameid, team_a, team_b, date):
        """Extract series identifier from gameid or construct from teams + date."""
        gameid = str(gameid) if pd.notna(gameid) else ''
        team_a, team_b = str(team_a), str(team_b)
        parts = gameid.rsplit('_', 1) if '_' in gameid else (gameid, '')
        base = parts[0] if parts else gameid
        teams = tuple(sorted([team_a, team_b]))
        return (base, teams)

    def record_game(self, gameid, team_a, team_b, result_a, game_num, date):
        key = self.get_series_key(gameid, team_a, team_b, date)
        if key not in self.active_series:
            self.active_series[key] = []
        self.active_series[key].append({'result_a': result_a, 'game': game_num})
        # Cleanup old series (keep last 1000)
        if len(self.active_series) > 5000:
            keys = list(self.active_series.keys())
            for k in keys[:2000]:
                del self.active_series[k]

    def get_series_state(self, gameid, team_a, team_b, game_num, date):
        """Get series state before this game."""
        key = self.get_series_key(gameid, team_a, team_b, date)
        games = self.active_series.get(key, [])
        prior = [g for g in games if g['game'] < game_num]
        if not prior:
            return {
                'games_played': 0, 'series_score_a': 0, 'series_score_b': 0,
                'is_elimination': 0, 'momentum_a': 0,
            }
        score_a = sum(1 for g in prior if g['result_a'] == 1)
        score_b = len(prior) - score_a
        # Momentum: did team_a win the most recent game?
        last_result = prior[-1]['result_a']
        momentum_a = 1 if last_result == 1 else -1
        # Is this an elimination game? (e.g., 1-1 in Bo3, 2-2 in Bo5)
        is_elimination = 1 if score_a == score_b else 0
        return {
            'games_played': len(prior),
            'series_score_a': score_a, 'series_score_b': score_b,
            'is_elimination': is_elimination,
            'momentum_a': momentum_a,
        }

    def infer_bo_format(self, game_col_max):
        """Infer series format from max game number."""
        if game_col_max <= 1: return 1
        if game_col_max <= 3: return 3
        return 5


# ============================================================
# FEARLESS DRAFT DETECTOR
# ============================================================

class FearlessDraftDetector:
    """Detect if a match/tournament uses fearless draft (no champion reuse in series)."""

    def __init__(self):
        self.series_picks = {}  # series_key -> set of picked champions

    def record_picks(self, series_key, champions):
        if series_key not in self.series_picks:
            self.series_picks[series_key] = set()
        self.series_picks[series_key].update([c for c in champions if c and str(c) != 'nan'])

    def check_fearless(self, series_key, current_champions):
        """Check if any current champion was already picked in this series."""
        prev_picks = self.series_picks.get(series_key, set())
        if not prev_picks:
            return 0, 0  # no data yet
        current = set(c for c in current_champions if c and str(c) != 'nan')
        overlap = len(prev_picks.intersection(current))
        return 1 if overlap == 0 else 0, len(prev_picks)  # is_fearless, n_prev_picks


# ============================================================
# TEAM TRACKER (enhanced from V1)
# ============================================================

class TeamTracker:
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
    N = len(STAT_KEYS)
    BUF = 30

    def __init__(self):
        self.buffers = {}
        self.sides = {}
        self.h2h = {}
        self.rosters = {}
        self.champs = {}
        self.game_weights = {}  # team -> list of weights (Bo3/5 weighted higher)
        self.sub_history = {}   # team -> list of (date, was_sub_detected)

    def _buf(self, team):
        if team not in self.buffers:
            self.buffers[team] = (np.full((self.BUF, self.N), np.nan), 0, 0)
        return self.buffers[team]

    def add_game(self, team, opponent, row, champions, weight=1.0):
        buf, pos, count = self._buf(team)
        for key, idx in self.STAT_IDX.items():
            val = row.get(key, np.nan)
            try: buf[pos % self.BUF, idx] = float(val) if pd.notna(val) else np.nan
            except: buf[pos % self.BUF, idx] = np.nan
        self.buffers[team] = (buf, pos + 1, count + 1)

        # Side
        if team not in self.sides: self.sides[team] = []
        self.sides[team].append((row.get('side', ''), float(row.get('result', 0))))
        if len(self.sides[team]) > 60: self.sides[team] = self.sides[team][-60:]

        # H2H
        key = (team, opponent)
        if key not in self.h2h: self.h2h[key] = []
        self.h2h[key].append(float(row.get('result', 0)))
        if len(self.h2h[key]) > 20: self.h2h[key] = self.h2h[key][-20:]

        # Champions
        if team not in self.champs: self.champs[team] = []
        self.champs[team].append(set(c for c in champions if c and str(c) != 'nan'))
        if len(self.champs[team]) > 30: self.champs[team] = self.champs[team][-30:]

        # Game weight
        if team not in self.game_weights: self.game_weights[team] = []
        self.game_weights[team].append(weight)
        if len(self.game_weights[team]) > 30: self.game_weights[team] = self.game_weights[team][-30:]

    def rolling_mean(self, team, n, stat):
        buf, pos, count = self._buf(team)
        if count == 0: return np.nan
        idx = self.STAT_IDX.get(stat)
        if idx is None: return np.nan
        actual_n = min(n, count, self.BUF)
        indices = [(pos - 1 - i) % self.BUF for i in range(actual_n)]
        vals = buf[indices, idx]
        valid = vals[~np.isnan(vals)]
        return np.mean(valid) if len(valid) > 0 else np.nan

    def weighted_win_rate(self, team, n=10):
        """Win rate weighted by game importance (Bo3/5 > Bo1)."""
        buf, pos, count = self._buf(team)
        if count == 0: return 0.5
        idx = self.STAT_IDX['result']
        actual_n = min(n, count, self.BUF)
        indices = [(pos - 1 - i) % self.BUF for i in range(actual_n)]
        vals = buf[indices, idx]

        weights = self.game_weights.get(team, [1.0] * actual_n)[-actual_n:]
        while len(weights) < actual_n:
            weights.insert(0, 1.0)

        valid_mask = ~np.isnan(vals)
        vals = vals[valid_mask]
        w = np.array(weights[-len(vals):])
        if len(vals) == 0: return 0.5
        return np.average(vals, weights=w) if np.sum(w) > 0 else np.mean(vals)

    def win_rate(self, team, n=None):
        buf, pos, count = self._buf(team)
        if count == 0: return 0.5
        idx = self.STAT_IDX['result']
        actual_n = min(n or count, count, self.BUF)
        indices = [(pos - 1 - i) % self.BUF for i in range(actual_n)]
        vals = buf[indices, idx]
        valid = vals[~np.isnan(vals)]
        return np.mean(valid) if len(valid) > 0 else 0.5

    def side_wr(self, team, side, n=30):
        entries = self.sides.get(team, [])
        matches = [(s, r) for s, r in entries[-n*2:] if s == side]
        if not matches: return 0.5
        return np.mean([r for _, r in matches[-n:]])

    def h2h_wr(self, team, opp, n=10):
        results = self.h2h.get((team, opp), [])
        if not results: return 0.5
        return np.mean(results[-n:])

    def game_count(self, team):
        _, _, count = self._buf(team)
        return count

    def champ_diversity(self, team, n=20):
        recent = self.champs.get(team, [])[-n:]
        all_c = set()
        for s in recent: all_c.update(s)
        return len(all_c)

    def roster_overlap(self, team, current_players):
        prev = self.rosters.get(team, set())
        if not prev or not current_players: return 0.5
        return len(prev.intersection(set(current_players))) / max(len(current_players), 1)

    def detect_sub(self, team, current_players):
        """Detect if a substitute was brought in (roster change from previous game)."""
        prev = self.rosters.get(team, set())
        if not prev or not current_players: return 0, 0
        curr = set(current_players)
        new_players = curr - prev
        dropped_players = prev - curr
        return len(new_players), len(dropped_players)

    def update_roster(self, team, players):
        self.rosters[team] = set(players)


# ============================================================
# ELO SYSTEM (same as V1)
# ============================================================

class EloSystem:
    def __init__(self, initial=1500, k=32):
        self.initial = initial
        self.k = k
        self.ratings = {}

    def get(self, team): return self.ratings.get(team, self.initial)

    def update(self, team_a, team_b, result_a, tier=1):
        ra, rb = self.get(team_a), self.get(team_b)
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        k = self.k * (0.5 + 0.5 * tier / 3)
        self.ratings[team_a] = ra + k * (result_a - ea)
        self.ratings[team_b] = rb + k * ((1 - result_a) - (1 - ea))

    def decay(self, factor=0.75):
        for t in self.ratings:
            self.ratings[t] = self.initial + factor * (self.ratings[t] - self.initial)


# ============================================================
# MAIN FEATURE BUILDER (V2)
# ============================================================

def build_game_data(player_df):
    """Build per-game player data structures."""
    print("Building game data...")
    game_players = {}     # (gameid, team) -> [playernames]
    game_champions = {}   # (gameid, team) -> [champions]
    game_player_stats = {}  # (gameid, team) -> [{stats_dict per player}]
    game_player_ids = {}   # (gameid, team) -> [playerids]

    for _, row in player_df.iterrows():
        gid = row['gameid']
        team = row['teamname']
        key = (gid, team)
        if key not in game_players:
            game_players[key] = []
            game_champions[key] = []
            game_player_stats[key] = []
            game_player_ids[key] = []

        if pd.notna(row.get('playername')):
            game_players[key].append(row['playername'])
        if pd.notna(row.get('champion')):
            game_champions[key].append(row['champion'])
        if pd.notna(row.get('playerid')):
            game_player_ids[key].append(row['playerid'])

        stats = {}
        for col in PlayerTracker.STATS:
            if col in row.index:
                stats[col] = row[col]
        stats['position'] = row.get('position', '')
        stats['champion'] = row.get('champion', '')
        stats['playerid'] = row.get('playerid', '')
        stats['playername'] = row.get('playername', '')
        stats['result'] = row.get('result', 0)
        game_player_stats[key].append(stats)

    print(f"  {len(game_players)} game-team entries")
    return game_players, game_champions, game_player_stats, game_player_ids


def build_features(team_df, game_players, game_champions, game_player_stats, game_player_ids):
    print("\n=== Building Enhanced Features (V2) ===")
    elo = EloSystem()
    team_tracker = TeamTracker()
    player_tracker = PlayerTracker()
    series_tracker = SeriesTracker()
    fearless_detector = FearlessDraftDetector()

    feature_rows = []
    seen = set()
    current_year = None

    for idx in range(0, len(team_df) - 1, 2):
        row_a = team_df.iloc[idx]
        row_b = team_df.iloc[idx + 1]
        if row_a['gameid'] != row_b['gameid']: continue
        gid = row_a['gameid']
        if gid in seen: continue
        seen.add(gid)

        yr = row_a.get('year')
        if yr and current_year and yr != current_year:
            elo.decay(0.75)
        current_year = yr

        ta, tb = row_a['teamname'], row_b['teamname']
        if pd.isna(ta) or pd.isna(tb):
            continue
        ta, tb = str(ta), str(tb)
        sa, sb = row_a['side'], row_b['side']
        ra = int(row_a['result'])
        league = row_a.get('league', '')
        tier = get_league_tier(league)
        date = row_a['date']
        game_num = int(row_a['game']) if pd.notna(row_a.get('game')) else 1

        # Determine series format weight
        # Bo3/Bo5 games are more predictive than Bo1
        bo_weight = 1.0
        if game_num >= 3: bo_weight = 1.5   # Bo5 game
        elif game_num == 2: bo_weight = 1.3  # At least Bo3
        # Bo1 stays at 1.0

        # === PRE-MATCH FEATURES ===
        ea, eb = elo.get(ta), elo.get(tb)
        na, nb = team_tracker.game_count(ta), team_tracker.game_count(tb)

        f = {
            'gameid': gid, 'date': date,
            'date_numeric': row_a.get('date_numeric', 0),
            'team_a': ta, 'team_b': tb, 'side_a': sa,
            'league': league, 'result': ra,

            # V1 features (team-level)
            'elo_diff': ea - eb,
            'elo_expected': 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0)),
            'n_games_a': na, 'n_games_b': nb, 'n_games_diff': na - nb,

            'wr_diff_5': team_tracker.win_rate(ta, 5) - team_tracker.win_rate(tb, 5),
            'wr_diff_10': team_tracker.win_rate(ta, 10) - team_tracker.win_rate(tb, 10),
            'wr_diff_20': team_tracker.win_rate(ta, 20) - team_tracker.win_rate(tb, 20),
            'wr_diff_all': team_tracker.win_rate(ta) - team_tracker.win_rate(tb),
            'wr_a_10': team_tracker.win_rate(ta, 10),
            'wr_b_10': team_tracker.win_rate(tb, 10),

            # NEW: Weighted win rate (Bo3/5 weighted higher)
            'wwr_diff_10': team_tracker.weighted_win_rate(ta, 10) - team_tracker.weighted_win_rate(tb, 10),

            'is_blue_a': 1 if sa == 'Blue' else 0,
            'side_wr_diff': team_tracker.side_wr(ta, sa) - team_tracker.side_wr(tb, sb),
            'h2h_a': team_tracker.h2h_wr(ta, tb),

            'league_tier': tier,
            'is_playoffs': int(row_a.get('playoffs', 0)),
            'game_number': game_num,
            'first_pick_a': int(row_a['firstPick']) if pd.notna(row_a.get('firstPick')) else 0,
            'champ_div_diff': team_tracker.champ_diversity(ta) - team_tracker.champ_diversity(tb),

            # NEW: Context features
            'is_lan': is_lan_event(league),
            'bo_format': 1 if game_num <= 1 else (3 if game_num <= 3 else 5),
            'is_bo1': 1 if game_num == 1 else 0,  # Could be Bo1 or first game of series
        }

        # === ROSTER / SUBSTITUTION FEATURES ===
        pl_a = game_players.get((gid, ta), [])
        pl_b = game_players.get((gid, tb), [])
        pids_a = game_player_ids.get((gid, ta), [])
        pids_b = game_player_ids.get((gid, tb), [])

        f['roster_overlap_diff'] = team_tracker.roster_overlap(ta, pl_a) - team_tracker.roster_overlap(tb, pl_b)

        # Substitution detection
        subs_in_a, subs_out_a = team_tracker.detect_sub(ta, pl_a)
        subs_in_b, subs_out_b = team_tracker.detect_sub(tb, pl_b)
        f['subs_in_a'] = subs_in_a
        f['subs_in_b'] = subs_in_b
        f['has_sub_a'] = 1 if subs_in_a > 0 else 0
        f['has_sub_b'] = 1 if subs_in_b > 0 else 0
        f['sub_diff'] = subs_in_a - subs_in_b

        # === SERIES FEATURES ===
        series_state = series_tracker.get_series_state(gid, ta, tb, game_num, date)
        f['series_games_played'] = series_state['games_played']
        f['series_score_diff'] = series_state['series_score_a'] - series_state['series_score_b']
        f['is_elimination'] = series_state['is_elimination']
        f['series_momentum_a'] = series_state['momentum_a']

        # === FEARLESS DRAFT DETECTION ===
        series_key = series_tracker.get_series_key(gid, ta, tb, date)
        all_champs = game_champions.get((gid, ta), []) + game_champions.get((gid, tb), [])
        is_fearless, n_prev = fearless_detector.check_fearless(series_key, all_champs)
        f['is_fearless'] = is_fearless if n_prev > 5 else 0  # Only flag if enough prior picks

        # === PLAYER-LEVEL FEATURES ===
        pstats_a = game_player_stats.get((gid, ta), [])
        pstats_b = game_player_stats.get((gid, tb), [])

        # Aggregate player features for each team
        for side_label, pstats, pids in [('a', pstats_a, pids_a), ('b', pstats_b, pids_b)]:
            if not pstats or not pids:
                # No player data, fill with defaults
                f[f'avg_player_kda_{side_label}'] = np.nan
                f[f'avg_player_streak_{side_label}'] = 0
                f[f'avg_player_trajectory_{side_label}'] = 0
                f[f'avg_champ_comfort_{side_label}'] = 0.5
                f[f'avg_champ_games_{side_label}'] = 0
                f[f'avg_player_pool_{side_label}'] = 0
                f[f'star_player_kda_{side_label}'] = np.nan
                f[f'weakest_player_kda_{side_label}'] = np.nan
                f[f'player_experience_{side_label}'] = 0
                continue

            kdas = []
            streaks = []
            trajectories = []
            champ_comforts = []
            champ_games_on = []
            pool_sizes = []
            experiences = []

            for i, ps in enumerate(pstats):
                pid = ps.get('playerid', '')
                champ = ps.get('champion', '')
                if not pid: continue

                # Player KDA (historical)
                kda = player_tracker.kda(pid, 10)
                kdas.append(kda)

                # Hot streak / slump
                streak = player_tracker.hot_streak(pid, short=3, long=15)
                streaks.append(streak)

                # Trajectory (improving or declining DPM)
                traj = player_tracker.trajectory(pid, stat='dpm', recent=5, historical=15)
                trajectories.append(traj)

                # Champion comfort
                if champ and str(champ) != 'nan':
                    wr, ng = player_tracker.champ_comfort(pid, champ)
                    champ_comforts.append(wr)
                    champ_games_on.append(ng)

                # Champion pool size
                pool_sizes.append(player_tracker.champ_pool_size(pid))

                # Experience
                experiences.append(player_tracker.game_count(pid))

            # Aggregate
            f[f'avg_player_kda_{side_label}'] = np.nanmean(kdas) if kdas else np.nan
            f[f'avg_player_streak_{side_label}'] = np.mean(streaks) if streaks else 0
            f[f'avg_player_trajectory_{side_label}'] = np.mean(trajectories) if trajectories else 0
            f[f'avg_champ_comfort_{side_label}'] = np.mean(champ_comforts) if champ_comforts else 0.5
            f[f'avg_champ_games_{side_label}'] = np.mean(champ_games_on) if champ_games_on else 0
            f[f'avg_player_pool_{side_label}'] = np.mean(pool_sizes) if pool_sizes else 0
            f[f'player_experience_{side_label}'] = np.mean(experiences) if experiences else 0

            # Star player (best KDA) and weakest player
            if kdas:
                f[f'star_player_kda_{side_label}'] = max(kdas)
                f[f'weakest_player_kda_{side_label}'] = min(kdas)
            else:
                f[f'star_player_kda_{side_label}'] = np.nan
                f[f'weakest_player_kda_{side_label}'] = np.nan

        # Player diff features
        for feat in ['avg_player_kda', 'avg_player_streak', 'avg_player_trajectory',
                     'avg_champ_comfort', 'avg_champ_games', 'avg_player_pool',
                     'star_player_kda', 'weakest_player_kda', 'player_experience']:
            va = f.get(f'{feat}_a', np.nan)
            vb = f.get(f'{feat}_b', np.nan)
            if pd.notna(va) and pd.notna(vb):
                f[f'{feat}_diff'] = va - vb
            else:
                f[f'{feat}_diff'] = np.nan

        # === TEAM ROLLING STATS (same as V1) ===
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
            va = team_tracker.rolling_mean(ta, 10, sk)
            vb = team_tracker.rolling_mean(tb, 10, sk)
            safe = sk.replace(' ', '_')
            f[f'avg10_{safe}_diff'] = (va - vb) if (pd.notna(va) and pd.notna(vb)) else np.nan

        for sk in ['teamkills', 'teamdeaths', 'dragons', 'barons', 'towers', 'golddiffat15', 'earned gpm', 'dpm']:
            va = team_tracker.rolling_mean(ta, 20, sk)
            vb = team_tracker.rolling_mean(tb, 20, sk)
            safe = sk.replace(' ', '_')
            f[f'avg20_{safe}_diff'] = (va - vb) if (pd.notna(va) and pd.notna(vb)) else np.nan

        # Derived metrics (same as V1)
        ka = team_tracker.rolling_mean(ta, 10, 'teamkills') or 0
        da_t = team_tracker.rolling_mean(ta, 10, 'teamdeaths') or 1
        kb = team_tracker.rolling_mean(tb, 10, 'teamkills') or 0
        db_t = team_tracker.rolling_mean(tb, 10, 'teamdeaths') or 1
        f['kda_ratio_diff'] = (ka / max(da_t, 0.5)) - (kb / max(db_t, 0.5))

        dr_a = team_tracker.rolling_mean(ta, 10, 'dragons') or 0
        odr_a = team_tracker.rolling_mean(ta, 10, 'opp_dragons') or 0
        dr_b = team_tracker.rolling_mean(tb, 10, 'dragons') or 0
        odr_b = team_tracker.rolling_mean(tb, 10, 'opp_dragons') or 0
        f['dragon_control_diff'] = dr_a / max(dr_a + odr_a, 0.5) - dr_b / max(dr_b + odr_b, 0.5)

        tw_a = team_tracker.rolling_mean(ta, 10, 'towers') or 0
        otw_a = team_tracker.rolling_mean(ta, 10, 'opp_towers') or 0
        tw_b = team_tracker.rolling_mean(tb, 10, 'towers') or 0
        otw_b = team_tracker.rolling_mean(tb, 10, 'opp_towers') or 0
        f['tower_control_diff'] = tw_a / max(tw_a + otw_a, 0.5) - tw_b / max(tw_b + otw_b, 0.5)

        ba_a = team_tracker.rolling_mean(ta, 10, 'barons') or 0
        oba_a = team_tracker.rolling_mean(ta, 10, 'opp_barons') or 0
        ba_b = team_tracker.rolling_mean(tb, 10, 'barons') or 0
        oba_b = team_tracker.rolling_mean(tb, 10, 'opp_barons') or 0
        f['baron_control_diff'] = ba_a / max(ba_a + oba_a, 0.01) - ba_b / max(ba_b + oba_b, 0.01)

        vs_a = team_tracker.rolling_mean(ta, 10, 'vspm')
        vs_b = team_tracker.rolling_mean(tb, 10, 'vspm')
        f['vision_diff'] = ((vs_a or 0) - (vs_b or 0))

        gd15_a = team_tracker.rolling_mean(ta, 10, 'golddiffat15') or 0
        gd15_b = team_tracker.rolling_mean(tb, 10, 'golddiffat15') or 0
        f['composite_strength_diff'] = (
            0.3 * (ea - eb) / 200 +
            0.25 * (team_tracker.win_rate(ta, 10) - team_tracker.win_rate(tb, 10)) * 2 +
            0.2 * f['kda_ratio_diff'] +
            0.15 * f['dragon_control_diff'] * 2 +
            0.1 * (gd15_a - gd15_b) / 1000
        )

        feature_rows.append(f)

        # === UPDATE ALL TRACKERS ===
        ch_a = game_champions.get((gid, ta), [])
        ch_b = game_champions.get((gid, tb), [])
        team_tracker.add_game(ta, tb, row_a, ch_a, weight=bo_weight)
        team_tracker.add_game(tb, ta, row_b, ch_b, weight=bo_weight)
        team_tracker.update_roster(ta, pl_a)
        team_tracker.update_roster(tb, pl_b)
        elo.update(ta, tb, ra, tier)

        # Update player trackers
        for pstats_side, result_side in [(pstats_a, ra), (pstats_b, 1 - ra)]:
            for ps in pstats_side:
                pid = ps.get('playerid', '')
                if not pid: continue
                player_tracker.add_game(
                    pid, ps.get('playername', ''), ps.get('position', ''),
                    ps.get('champion', ''), result_side, ps
                )

        # Update series tracker
        series_tracker.record_game(gid, ta, tb, ra, game_num, date)
        fearless_detector.record_picks(series_key, ch_a + ch_b)

        if len(feature_rows) % 5000 == 0:
            print(f"  {len(feature_rows)} games processed...")

    print(f"\nTotal: {len(feature_rows)} games with enhanced features")
    return pd.DataFrame(feature_rows), elo, team_tracker, player_tracker


# ============================================================
# FEATURE COLUMNS (V2 - expanded)
# ============================================================

FEATURE_COLUMNS_V2 = [
    # V1 core
    'elo_diff', 'elo_expected', 'n_games_diff',
    'wr_diff_5', 'wr_diff_10', 'wr_diff_20', 'wr_diff_all',
    'wr_a_10', 'wr_b_10',
    'wwr_diff_10',  # NEW: weighted win rate
    'is_blue_a', 'side_wr_diff', 'h2h_a',
    'league_tier', 'is_playoffs', 'game_number', 'first_pick_a',
    'roster_overlap_diff', 'champ_div_diff',

    # NEW: Context
    'is_lan', 'bo_format',

    # NEW: Substitution
    'has_sub_a', 'has_sub_b', 'sub_diff',

    # NEW: Series
    'series_games_played', 'series_score_diff', 'is_elimination', 'series_momentum_a',

    # NEW: Fearless draft
    'is_fearless',

    # NEW: Player-level diffs
    'avg_player_kda_diff', 'avg_player_streak_diff', 'avg_player_trajectory_diff',
    'avg_champ_comfort_diff', 'avg_champ_games_diff', 'avg_player_pool_diff',
    'star_player_kda_diff', 'weakest_player_kda_diff', 'player_experience_diff',

    # Team rolling stats (V1)
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
    # Derived
    'kda_ratio_diff', 'dragon_control_diff', 'tower_control_diff',
    'baron_control_diff', 'vision_diff', 'composite_strength_diff',
]


def prepare_model_data(features_df, min_games=5):
    mask = (features_df['n_games_a'] >= min_games) & (features_df['n_games_b'] >= min_games)
    df = features_df[mask].copy()
    print(f"After filtering (>={min_games} games): {len(df)} (dropped {len(features_df)-len(df)})")

    avail = [c for c in FEATURE_COLUMNS_V2 if c in df.columns]
    print(f"Using {len(avail)} features")

    X = df[avail].values.astype(float)
    y = df['result'].values.astype(float)
    dates = df['date_numeric'].values.astype(float)

    X = np.nan_to_num(X, nan=0.0)
    X = np.clip(X, -1e6, 1e6)
    return X, y, dates, avail, df


if __name__ == "__main__":
    team_df, player_df = load_all_data('data')
    game_players, game_champions, game_player_stats, game_player_ids = build_game_data(player_df)
    del player_df
    gc.collect()

    features_df, elo_sys, team_tracker, player_tracker = build_features(
        team_df, game_players, game_champions, game_player_stats, game_player_ids)
    del team_df
    gc.collect()

    features_df.to_csv('data/engineered_features_v2.csv', index=False)
    print(f"\nSaved: data/engineered_features_v2.csv ({features_df.shape})")
    print(f"Date range: {features_df['date'].min()} to {features_df['date'].max()}")

    # Quick check on new features
    cols = [c for c in features_df.columns if 'player' in c or 'sub' in c or 'series' in c or 'fearless' in c]
    print(f"\nNew V2 features ({len(cols)}):")
    for c in cols:
        print(f"  {c}: mean={features_df[c].mean():.4f}, non-null={features_df[c].notna().sum()}")
