"""
build_coaches.py — rebuild data/coaches.tsv (team, year, coach) from Leaguepedia.

Usage (from the repo root):
    python src/build_coaches.py            # uses the cached download if present
    python src/build_coaches.py --refresh  # re-download from Leaguepedia

Source: Leaguepedia's public Cargo API, table RosterChanges (every staff
join/leave with a role). Content is CC BY-SA 3.0, so neither the download nor
coaches.tsv is committed; this script reproduces them.

How a team's coach for a year is chosen
---------------------------------------
1. Keep roster changes whose role is "Head Coach" or plain "Coach"
   (assistant, strategic and positional coaches are ignored).
2. Pair each person's Join with their next Leave at the same team to get
   tenure intervals. A departure with no recorded Join is treated as having
   started before the download window. A Join with no Leave stays open until
   someone else joins that team in an equal or higher role.
3. For each team and year, pick the person with the highest-ranked role
   (Head Coach over Coach), breaking ties by days in charge that year.

Known limits: one coach per team-year (a mid-season change keeps whoever
served longer), promotions inside a staff are only seen if Leaguepedia logged
them as a roster change, and team names must match Oracle's Elixir exactly.
The script prints how many team-seasons in the match data it could cover.
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime

import pandas as pd

API = "https://lol.fandom.com/api.php"
USER_AGENT = "lol-esports-predictor research script (github.com/Hget15/lol-esports-predictor)"
WINDOW_START = date(2018, 1, 1)
FAR_FUTURE = date(2100, 1, 1)
ROLE_RANK = {"Head Coach": 2, "Coach": 1}


def fetch_roster_changes(cache_path):
    """Page through RosterChanges politely (the API rate-limits anonymous clients)."""
    rows, offset, wait = [], 0, 15
    while True:
        params = {
            "action": "cargoquery", "format": "json", "tables": "RosterChanges",
            "fields": "Date_Sort,Player,Direction,Team,Roles",
            # Roles is a list field, so it needs HOLDS; a plain LIKE makes the API error out.
            "where": f"Roles HOLDS LIKE '%Coach' AND Date_Sort >= '{WINDOW_START.isoformat()}'",
            "order_by": "Date_Sort,Team,Player,Direction", "limit": "500", "offset": str(offset),
        }
        req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode("utf-8"))
        if "error" in d:
            if wait > 600:
                raise SystemExit(f"Leaguepedia API keeps refusing: {d['error']}")
            print(f"  offset {offset}: {d['error'].get('code')} — waiting {wait}s")
            time.sleep(wait)
            wait *= 2
            continue
        wait = 15
        page = [r["title"] for r in d.get("cargoquery", [])]
        rows.extend(page)
        print(f"  offset {offset}: +{len(page)} rows")
        if len(page) < 500:
            break
        offset += 500
        time.sleep(12)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    return rows


def role_rank(roles):
    """Rank of the best head-coach-type role in a comma-separated role string (0 = not a head coach)."""
    return max((ROLE_RANK.get(r.strip(), 0) for r in str(roles).split(",")), default=0)


def tenures(rows):
    """(team, person, rank, start, end) intervals from paired Join/Leave events."""
    events = defaultdict(list)
    for r in rows:
        rank = role_rank(r.get("Roles"))
        raw_date = (r.get("Date Sort") or r.get("Date_Sort") or "")[:10]
        if rank == 0 or not raw_date or not r.get("Team") or not r.get("Player"):
            continue
        events[(r["Team"], r["Player"])].append((datetime.strptime(raw_date, "%Y-%m-%d").date(), r.get("Direction"), rank))

    out = []
    joins_by_team = defaultdict(list)                       # team -> [(day, person, rank)]
    for (team, person), evs in events.items():
        # Leaguepedia logs a contract renewal as a Leave and a Join on the same
        # day, so on equal dates the Leave must be processed first; otherwise a
        # renewal would end the tenure instead of continuing it.
        evs.sort(key=lambda e: (e[0], e[1] != "Leave"))
        open_start, open_rank = None, 0
        for i, (day, direction, rank) in enumerate(evs):
            if direction == "Join":
                joins_by_team[team].append((day, person, rank))
                if open_start is None:
                    open_start, open_rank = day, rank
                else:
                    open_rank = max(open_rank, rank)        # role change while already on staff
            elif direction == "Leave":
                if open_start is not None:
                    out.append((team, person, max(open_rank, rank), open_start, day))
                elif i == 0:
                    # First thing we see is a departure: they joined before the download window.
                    out.append((team, person, rank, WINDOW_START, day))
                open_start, open_rank = None, 0
        if open_start is not None:
            out.append((team, person, open_rank, open_start, FAR_FUTURE))

    # Some departures were never logged with a coaching role, which would leave
    # a long-gone coach "in charge" forever. Close any still-open tenure when a
    # different person later joins the same team in an equal or higher role.
    closed = []
    for team, person, rank, start, end in out:
        if end == FAR_FUTURE:
            successors = [d for d, p, r in joins_by_team[team] if p != person and r >= rank and d > start]
            if successors:
                end = min(successors)
        closed.append((team, person, rank, start, end))
    return closed


def coach_by_team_year(intervals, years):
    best = {}
    for team, person, rank, start, end in intervals:
        for y in years:
            lo, hi = max(start, date(y, 1, 1)), min(end, date(y, 12, 31))
            days = (hi - lo).days
            if days <= 0:
                continue
            key = (team, y)
            if key not in best or (rank, days) > best[key][:2]:
                best[key] = (rank, days, person)
    return {k: v[2] for k, v in best.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    cache = os.path.join(args.data_dir, "leaguepedia_roster_changes.json")
    if args.refresh or not os.path.exists(cache):
        print("Downloading coaching roster changes from Leaguepedia...")
        rows = fetch_roster_changes(cache)
    else:
        rows = json.load(open(cache, encoding="utf-8"))
    print(f"{len(rows)} coaching roster changes")

    years = list(range(2021, 2027))
    mapping = coach_by_team_year(tenures(rows), years)

    # Coverage against the match data, if the V1 feature table exists.
    feat = os.path.join(args.data_dir, "engineered_features.csv")
    if os.path.exists(feat):
        g = pd.read_csv(feat, usecols=["team_a", "team_b", "date"], low_memory=False)
        g["year"] = pd.to_datetime(g["date"]).dt.year
        seasons = pd.concat([g[["team_a", "year"]].rename(columns={"team_a": "team"}),
                             g[["team_b", "year"]].rename(columns={"team_b": "team"})])
        counts = seasons.value_counts().reset_index(name="games")
        counts["covered"] = [(t, y) in mapping for t, y in zip(counts["team"], counts["year"])]
        mapping = {(t, y): mapping[(t, y)] for t, y in zip(counts["team"], counts["year"]) if (t, y) in mapping}
        share = counts.loc[counts["covered"], "games"].sum() / counts["games"].sum()
        print(f"Coverage: {int(counts['covered'].sum())} of {len(counts)} team-seasons, "
              f"{share:.0%} of team-games in the match data")

    out = pd.DataFrame([{"team": t, "year": y, "coach": c} for (t, y), c in sorted(mapping.items())])
    path = os.path.join(args.data_dir, "coaches.tsv")
    out.to_csv(path, sep="\t", index=False)
    print(f"Wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
