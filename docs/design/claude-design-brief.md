# Brief: LoL Esports Predictor — live demo web page

Design and build a single-page web app: a **pro League of Legends match predictor**. A visitor picks two pro teams and gets a win probability, an explanation of what drove it, and a player-by-player comparison. It will be deployed as a portfolio demo for a game-design role at Riot, so it must look deliberate and polished, and it must read in 30 seconds to someone who has never seen it. Desktop first, but it must work at 390px wide.

**Reference mockups (match these):** https://claude.ai/artifact/6fvvdwuR7QS2BWC8VmYNWQ — page 1 has the chosen direction ("A · Broadcast") as a desktop artboard and a phone artboard. Build exactly that layout and system; the other page holds rejected directions — ignore it.

## Tech
React + Tailwind (Vite). No UI or chart libraries. Fonts from Google Fonts: **Barlow Condensed** (500/600/700) for the wordmark, section labels, team names and every numeral; **IBM Plex Sans** (400/500/600) for body text. Fallbacks `"Arial Narrow", "Segoe UI", sans-serif` and `"Segoe UI", system-ui, sans-serif`. All logic in one pure-function module (`model.js`); components only render. Output a runnable project.

## Visual system
Dark, flat, broadcast-graphic feel. **No gradients, no drop shadows, no emoji, no rounded cards with a colored left border.** Radii 3–4px, 1px borders, inline SVG stroke icons only (chevron, swap arrows, code brackets).

| Token | Hex | Use |
|---|---|---|
| bg | `#0b0e14` | page |
| panel | `#11161f` | cards |
| line | `#1f2733` | hairlines, bar tracks |
| line-strong | `#2a3442` | chip/field borders |
| ink | `#e8ecf1` | body text |
| ink-muted | `#8b95a5` | secondary text |
| ink-soft | `#b7c0cc` | stat text |
| gold | `#c8aa6e` | the model's own accents: section labels, driver values and bars, links |
| teal | `#0ac8b9` | focus rings only |
| blue | `#4c8dff` | Team A |
| red | `#ff6161` | Team B |
| role colors | top `#ef4444` · jng `#22c55e` · mid `#3b82f6` · bot `#f59e0b` · sup `#a855f7` | role pills |

Section labels: Barlow Condensed 14px, weight 600, uppercase, letter-spacing 0.2em, gold. Body 12–15px. Hit targets ≥ 44px on every control. Focus state: 2px teal outline, 2px offset. Toggle on = blue fill with dark text; off = 1px `line-strong` border with muted text.

## Page, top to bottom (max width 1000px, centered, 40px side padding)
1. **Header** — wordmark "LOL ESPORTS PREDICTOR" + small gold "V4" tag on the left; three bordered chips on the right: `51,088 GAMES · 2021–26`, `AUC 0.797`, `130 FEATURES`. Hairline below.
2. **Intro** — one muted sentence: "Who wins a professional League of Legends match? Pick two teams. The model, trained from scratch on five seasons of pro play, gives you a probability — and shows its reasoning."
3. **Matchup** — two equal cards: "TEAM A · BLUE SIDE" (blue label) and "TEAM B · RED SIDE" (red label). Each has a styled **native `<select>`** of all teams (sorted A–Z, team name in 30px Barlow Condensed, chevron icon) and beneath it `Elo +77.6 · Win rate 82%`. Below the cards, a row of toggle buttons: **"<Team A> on blue side"** (on by default, check icon), **"International (LAN)"** (off by default), and on the far right **"Swap sides"** (swap icon) which exchanges the two teams.
4. **Probability hero** — one panel: 88px numerals `63%` (blue) left and `37%` (red) right with the team names in small caps beneath, "WIN PROBABILITY" in gold letterspaced caps between them, and a 10px two-color split bar underneath whose widths animate (500ms) when they change. This is the only motion on the page.
5. **What's driving this** (left half) — four rows, each: label, a gold bar scaled to the largest contribution, and the signed value in gold Barlow Condensed. Rows: `Recent form  +15 pts WR → +0.23`, `Elo edge  +16.6 → +0.15`, `Blue side → +0.15`, `International event → 0` (muted when zero). Footer text: "Log-odds contributions toward Gen.G. They sum to +0.53, which is 63%. Form on the current patch matters more than raw rating — the model's central finding."
6. **Player head-to-head** (right half) — label + score `4 – 1` (blue/red digits). Column caption `Gen.G · KDA / DPM / WR` and `T1 · KDA / DPM / WR`. Five rows: role pill (TOP/JNG/MID/BOT/SUP in the role color), Team A player (name, `KDA · DPM · WR%`, top-3 champions in 10px muted), Team B player likewise. The row is tinted blue `#0f1a2e` when A's player wins it and red `#2a1116` when B's does; the winner's name is white and semibold, the loser's muted. Never indicate the winner by color alone.
7. **About the model** — a collapsed 44px row ("ABOUT THE MODEL · V1 → V4", chevron). Expanded: a small table — V1 · AUC 0.688 · 63 features · "Elo, rolling stats, game context"; V2 · 0.698 · 79 · "+ player tracking, series momentum"; V3 · 0.698 · 106 · "+ coaches, travel, regional playstyle"; V4 · 0.797 · 130 · "+ patch meta, champion meta" — plus the sentence "The page uses a calibrated Elo + form proxy of the full model; the real model needs a full 130-feature vector per game."
8. **Footer** — hairline above; muted caveat "Probabilities come from a calibrated Elo + form proxy of the full 130-feature model. Rosters as of March 2026. Data: Oracle's Elixir." and a gold link "SEE THE CODE" (code icon) to https://github.com/Hget15/lol-esports-predictor.

**Phone (< 720px):** 16px side padding; chips wrap; the two team cards stack; toggles wrap; hero numerals 56px; Drivers and Head-to-head stack; each H2H row becomes the role pill on top with the two players in a 2-column grid beneath.

## Behavior
- **Default state on load:** Team A = Gen.G, Team B = T1, A on blue side, LAN off. The page never loads empty.
- State mirrors to the URL query string and is read back on load: `?a=<team>&b=<team>&blue=a|b&lan=1` (team names URI-encoded). Unknown, identical or malformed values silently fall back to the default state.
- Same team in both selects → hide sections 4–6 and show one inline note: "Pick two different teams."
- If a team lacks a role in its roster, omit that H2H row and score only the rows shown.

## The model (implement exactly — it is calibrated)
Constants: `D = 250`, `WR_W = 1.5`, `BLUE_B = 0.15`, `LAN_B = 0.05`. Each team has `elo` (number) and `wr` (0–1).

```
base  = 1 / (1 + 10 ** (-(a.elo - b.elo) / D)), clamped to [0.001, 0.999]
elo   = ln(base / (1 - base))
form  = WR_W * (a.wr - b.wr)
side  = blueIsA ? BLUE_B : 0
lan   = lanEvent ? LAN_B : 0
total = elo + form + side + lan
p(A wins) = 1 / (1 + e ** -total)
```
The four terms are what the "What's driving this" rows display. Check: Gen.G (elo 77.6, wr 0.82) vs T1 (elo 61.0, wr 0.67), A on blue, no LAN → **0.629**.

Head-to-head rule per role: compare A vs B on five stats — `kda`, `dpm`, `wr`, `cs`, `pool`; A wins the row if A is strictly higher on **3 or more**, otherwise B wins it.

## Data
Every team record has this shape (the full set of 50 teams, rosters as of March 2026, is in the repo at `app/lol_predictor_v4.jsx` as the `T` constant — copy it verbatim into `data/teams.json`):

```
"Gen.G": { elo: 77.6, wr: 0.82, r: {
  top: { n: "Kiin",   kda: 4.2,  dpm: 791, g: 20, wr: 75, cs: 8.9,  dp: 26.7, gp: 20.6, pool: 10, ch: ["Zaahen","Renekton","Gnar"] },
  jng: { n: "Canyon", kda: 5.97, dpm: 450, g: 20, wr: 75, cs: 7.7,  dp: 14.5, gp: 19.7, pool: 9,  ch: ["Vi","Pantheon","Ambessa"] },
  mid: { n: "Chovy",  kda: 7.34, dpm: 820, g: 20, wr: 75, cs: 9.9,  dp: 26.9, gp: 22.7, pool: 12, ch: ["Ahri","Galio","Mel"] },
  bot: { n: "Ruler",  kda: 5.31, dpm: 733, g: 20, wr: 75, cs: 10.5, dp: 24.2, gp: 27.5, pool: 10, ch: ["Yunara","Ashe","Sivir"] },
  sup: { n: "Duro",   kda: 4.25, dpm: 232, g: 20, wr: 75, cs: 0.9,  dp: 7.7,  gp: 9.6,  pool: 10, ch: ["Rakan","Seraphine","Neeko"] } } }
"T1": { elo: 61.0, wr: 0.67, r: {
  top: { n: "Doran", kda: 2.85, dpm: 764, g: 20, wr: 60, cs: 8.8,  dp: 24.9, gp: 20.4, pool: 15, ch: ["Kennen","Jayce","Gnar"] },
  jng: { n: "Oner",  kda: 4.0,  dpm: 553, g: 20, wr: 60, cs: 7.0,  dp: 18.0, gp: 20.7, pool: 13, ch: ["Xin Zhao","Vi","Pantheon"] },
  mid: { n: "Faker", kda: 2.98, dpm: 674, g: 20, wr: 60, cs: 9.1,  dp: 22.1, gp: 20.9, pool: 9,  ch: ["Azir","Ryze","Galio"] },
  bot: { n: "Peyz",  kda: 3.79, dpm: 861, g: 20, wr: 60, cs: 10.2, dp: 27.3, gp: 27.7, pool: 10, ch: ["Aphelios","Varus","Kai'Sa"] },
  sup: { n: "Keria", kda: 4.6,  dpm: 240, g: 20, wr: 60, cs: 1.0,  dp: 7.8,  gp: 10.3, pool: 13, ch: ["Thresh","Rakan","Lulu"] } } }
```
(`n` name, `kda`, `dpm` damage/min, `wr` win rate %, `cs` CS/min, `pool` champion-pool size, `ch` top champions.) With these two teams the head-to-head is Gen.G 4 – 1 T1 (Keria is T1's only winning row).

## Do not
Invent extra sections, stats or copy; use gradients, glows, shadows or emoji; use Inter/Roboto/Arial; put team logos or any Riot branding on the page; change the model constants.
