"""
Leakage tests for the V4 feature builder (stdlib unittest; needs numpy + pandas).

Run from the repo root:
    python -m unittest tests.test_no_lookahead -v

The property under test: a game's features may depend only on games that
finished before it. So changing the RESULT of a game, or of any later game,
must leave that game's features untouched. The first version of
v4_enhancements.py failed this (it filled its trackers with every game before
querying them), which inflated the reported V4 accuracy.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import v4_enhancements as v4  # noqa: E402

TEAMS = ["Alpha", "Bravo", "Cobra", "Delta"]
CHAMPS = ["Ahri", "Azir", "Gnar", "Jax", "Vi", "Sejuani", "Varus", "Jinx", "Rakan", "Nami",
          "Orianna", "Sion", "Lee Sin", "Ashe", "Lulu"]


def make_games(seed=0, n=120):
    """Synthetic schedule: one game every 6 hours, a new patch every 30 games."""
    rng = np.random.RandomState(seed)
    games = []
    for i in range(n):
        a, b = rng.choice(len(TEAMS), 2, replace=False)
        res_a = int(rng.rand() < 0.5)
        sides = []
        for team, res in ((TEAMS[a], res_a), (TEAMS[b], 1 - res_a)):
            picks = list(rng.choice(CHAMPS, 5, replace=False))
            bans = list(rng.choice(CHAMPS, 5, replace=False))
            sides.append({"team": team, "result": res, "champs": picks, "picks": picks, "bans": bans})
        games.append({"gameid": f"g{i:04d}", "date": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=6 * i),
                      "patch": 16.01 + (i // 30) * 0.01, "sides": sides})
    return games


def flip_result(game):
    for side in game["sides"]:
        side["result"] = 1 - side["result"]


def same(x, y):
    """Deep equality that treats NaN == NaN."""
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
        return True
    return x == y


class TestNoLookAhead(unittest.TestCase):
    def test_own_result_does_not_change_own_features(self):
        base = v4.build_pre_game_features(make_games())
        for k in (5, 40, 119):
            games = make_games()
            flip_result(games[k])
            feats = v4.build_pre_game_features(games)
            self.assertTrue(same(feats[games[k]["gameid"]], base[games[k]["gameid"]]),
                            f"features of game {k} changed when its own result was flipped")

    def test_future_results_do_not_change_past_features(self):
        base = v4.build_pre_game_features(make_games())
        games = make_games()
        for g in games[60:]:
            flip_result(g)
        feats = v4.build_pre_game_features(games)
        for g in games[:61]:          # game 60 itself must also be unchanged
            self.assertTrue(same(feats[g["gameid"]], base[g["gameid"]]), g["gameid"])

    def test_past_results_do_change_later_features(self):
        # Guard against a trivially "leak-free" builder that ignores results.
        base = v4.build_pre_game_features(make_games())
        games = make_games()
        for g in games[:60]:
            flip_result(g)
        feats = v4.build_pre_game_features(games)
        changed = sum(not same(feats[g["gameid"]], base[g["gameid"]]) for g in games[60:])
        self.assertGreater(changed, 30)

    def test_first_game_has_no_history(self):
        games = make_games()
        first = v4.build_pre_game_features(games)[games[0]["gameid"]]
        for team_feats in first["teams"].values():
            self.assertEqual(team_feats["team_patch_games"], 0)
            self.assertTrue(np.isnan(team_feats["patch_wr_delta"]))
            self.assertTrue(np.isnan(team_feats["avg_champ_wr"]))

    def test_features_attach_by_team_name(self):
        games = make_games()
        feats = v4.build_pre_game_features(games)
        g = games[50]
        a, b = g["sides"]
        # Feature table lists the teams in the opposite order to the raw file.
        df = pd.DataFrame([{"gameid": g["gameid"], "team_a": b["team"], "team_b": a["team"]}])
        out = v4.add_patch_meta_features(df, feats)
        self.assertEqual(out.loc[0, "team_patch_games_a"], feats[g["gameid"]]["teams"][b["team"]]["team_patch_games"])
        self.assertEqual(out.loc[0, "team_patch_games_b"], feats[g["gameid"]]["teams"][a["team"]]["team_patch_games"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
