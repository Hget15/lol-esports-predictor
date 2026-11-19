# Data

The raw match data is **not committed** to this repo. It comes from **Oracle's Elixir**,
which asks that the files be downloaded from their site rather than redistributed, and the
full set is ~450 MB (well over GitHub's file-size limits). The CSVs are gitignored — drop
them into this folder and every script will find them.

## How to get it

1. Go to **https://oracleselixir.com/tools/downloads**
2. Download the yearly match-data CSVs (2021–2026).
3. Place them in this `data/` directory with their original names:

```
data/
├── 2021_LoL_esports_match_data_from_OraclesElixir.csv
├── 2022_LoL_esports_match_data_from_OraclesElixir.csv
├── 2023_LoL_esports_match_data_from_OraclesElixir.csv
├── 2024_LoL_esports_match_data_from_OraclesElixir.csv
├── 2025_LoL_esports_match_data_from_OraclesElixir.csv
├── 2026_LoL_esports_match_data_from_OraclesElixir.csv
└── coaches.tsv          # coach per team and year — built by `python src/build_coaches.py`
```

Each CSV has 165 columns and 12 rows per game (10 player rows + 2 team-summary rows).
See the root `worklog.md` for the full column reference.

## Coach data

`coaches.tsv` is not downloaded by hand. `python src/build_coaches.py` (run from the repo
root, after the V1 feature script) pulls coaching-staff changes from Leaguepedia's public
API and writes one head coach per team and year. The API is rate-limited, so the first run
takes several minutes; the download is cached in `data/leaguepedia_roster_changes.json`.
Leaguepedia content is CC BY-SA 3.0, so neither file is committed.

## Regenerating engineered features

Feature CSVs (`engineered_features_v*.csv`) are also gitignored — they are produced from the
raw data by the feature-engineering scripts in `../src/`. See the root `README.md` for the
run order.

> Oracle's Elixir data © Tim Sevenhuysen / oracleselixir.com. Please credit the source and
> follow their terms when using it.
