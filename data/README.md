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
└── coaches.tsv          # coach↔team↔year mappings (scraped from Leaguepedia; ~230 rows)
```

Each CSV has 165 columns and 12 rows per game (10 player rows + 2 team-summary rows).
See the root `worklog.md` for the full column reference.

## Regenerating engineered features

Feature CSVs (`engineered_features_v*.csv`) are also gitignored — they are produced from the
raw data by the feature-engineering scripts in `../src/`. See the root `README.md` for the
run order.

> Oracle's Elixir data © Tim Sevenhuysen / oracleselixir.com. Please credit the source and
> follow their terms when using it.
