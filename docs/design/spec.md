# Live demo: LoL Esports Predictor web app — design spec

**Date:** 2026-09-15
**Status:** approved by owner (direction A "Broadcast" chosen from three canvas explorations)

## 1. Goal

Turn the existing single-file React component (`app/lol_predictor_v4.jsx`) into a
**deployed, polished demo** a recruiter can open from a link, redesigned in the
"Broadcast" direction. The predictor is the hero; a small amount of context around it
(what it is, how good the model is, what drives a prediction) lets a cold visitor
understand it in ~30 seconds. Nothing about the model's numbers changes.

Out of scope: embedding result charts, the designer's-lens essay, a searchable team
combobox, running the real 130-feature model in the browser (see §5).

## 2. Stack and repository layout

- **Vite + React 18 + Tailwind 4** (`@tailwindcss/vite`). No UI or chart libraries.
- Fonts from Google Fonts via `<link>` in `index.html`: **Barlow Condensed** (500/600/700)
  for display/numerals, **IBM Plex Sans** (400/500/600) for body. Fallbacks:
  `"Arial Narrow", "Segoe UI", sans-serif` and `"Segoe UI", system-ui, sans-serif`.
- `app/` becomes the Vite project root:

```
app/
├── index.html
├── package.json               # scripts: dev, build, preview, test
├── vite.config.js             # base: '/lol-esports-predictor/'
├── README.md                  # npm install && npm run dev
├── legacy/
│   ├── lol_predictor_v2.jsx   # moved, unchanged
│   └── lol_predictor_v4.jsx   # moved, unchanged
└── src/
    ├── main.jsx
    ├── App.jsx
    ├── index.css              # @import "tailwindcss"; theme tokens (§4)
    ├── model.js               # constants, predict(), explain(), compareRosters()
    ├── model.test.js          # Vitest
    ├── data/teams.json        # 50 teams, extracted verbatim from the V4 JSX
    ├── data/versions.json     # V1–V4 table for "About the model"
    └── components/
        ├── Header.jsx
        ├── Matchup.jsx        # two TeamSelect + toggles + swap
        ├── TeamSelect.jsx
        ├── Probability.jsx
        ├── Drivers.jsx
        ├── HeadToHead.jsx     # renders PlayerRow ×5
        ├── AboutModel.jsx     # collapsible version table
        └── Footer.jsx
```

- `results/` and `src/` (the Python) are untouched. `.gitignore` gains
  `app/node_modules/` and `app/dist/`.

## 3. Page structure and behavior

Single page, max content width 1000px centered, 40px side padding on desktop.
Order top to bottom:

1. **Header** — wordmark "LOL ESPORTS PREDICTOR" + "V4" tag; three stat chips:
   `51,088 GAMES · 2021–26`, `AUC 0.797`, `130 FEATURES`.
2. **Intro** — one sentence: what this is and that it shows its reasoning.
3. **Matchup** — two team cards side by side (Team A · Blue side / Team B · Red side).
   Each holds a styled native `<select>` of the 50 teams (sorted) and the team's
   `Elo` and `Win rate` beneath. Between/below: toggle buttons **"<A> on blue side"**
   and **"International (LAN)"**, and a **Swap sides** control that exchanges A and B.
4. **Probability** — hero: `<pA>%` (blue) and `<pB>%` (red) in 88px condensed
   numerals with team names beneath, "WIN PROBABILITY" label between, and a 10px split
   bar whose widths animate on change (`transition: width 500ms`).
5. **Drivers** ("What's driving this") — four rows, one per factor in §5, each with
   label, a bar scaled to the largest absolute contribution, and the signed log-odds
   value in gold. Footer line: "Log-odds contributions toward <A>. They sum to
   <total>, which is <pA>%." plus the one-sentence finding.
6. **Head-to-head** — score `<aWins> – <bWins>` and five rows (TOP/JNG/MID/BOT/SUP):
   role pill, A player (name, `KDA · DPM · WR`, top-3 champions), B player likewise.
   The winning side's row is tinted (blue `#0f1a2e` / red `#2a1116`) and its
   name is bold + white; the losing side is muted. Ties tint nothing.
7. **About the model** — a collapsed row ("About the model ▾"); expanded, it shows the
   V1–V4 table (version, AUC, #features, what was added) and the proxy caveat.
8. **Footer** — caveat sentence and "SEE THE CODE" link to the GitHub repo.

**Default state:** `Gen.G` vs `T1`, A on blue side, not LAN — the page never loads
empty.

**Shareable URL:** state mirrors to the query string on every change and is read on
load: `?a=<team>&b=<team>&blue=a|b&lan=1`. Team names are URI-encoded. Unknown or
identical team names, or malformed values, fall back to the default state silently.

**Same team twice:** results (§3.4–3.6) are replaced by a single inline note
"Pick two different teams." Toggles stay enabled.

**Missing roster data:** if either team lacks a position, that H2H row is omitted and
the score counts only rows shown (mirrors current behavior).

## 4. Visual system (direction A)

Tokens in `index.css` via Tailwind `@theme`:

| Token | Value | Use |
|---|---|---|
| `--color-bg` | `#0b0e14` | page background |
| `--color-panel` | `#11161f` | cards |
| `--color-line` | `#1f2733` | hairlines, bar tracks |
| `--color-line-strong` | `#2a3442` | chip and field borders |
| `--color-ink` | `#e8ecf1` | body text |
| `--color-ink-muted` | `#8b95a5` | secondary text |
| `--color-ink-soft` | `#b7c0cc` | stat text |
| `--color-gold` | `#c8aa6e` | model accents: labels, driver values/bars, link |
| `--color-teal` | `#0ac8b9` | reserved secondary accent (focus rings) |
| `--color-blue` | `#4c8dff` | Team A |
| `--color-red` | `#ff6161` | Team B |
| role colors | top `#ef4444`, jng `#22c55e`, mid `#3b82f6`, bot `#f59e0b`, sup `#a855f7` | role pills |

- Radii 3–4px; 1px borders; **no gradients, no shadows, no left-border accents.**
- Display type (`Barlow Condensed`) for the wordmark, section labels (uppercase,
  letter-spacing 0.2em), team names, and every numeral. Body (`IBM Plex Sans`) 12–15px.
- Hit targets ≥ 44px for selects and toggle buttons.
- Focus: 2px teal outline offset 2px. Toggle "on" = blue fill/dark text; "off" =
  bordered/muted. Selects: dark field, chevron icon, hover border `#3a4658`.
- Icons: inline SVG stroke icons only (chevron, swap, code), no emoji.
- Motion: only the probability bar width transition.

**Responsive (< 720px):** 16px side padding; header chips wrap; matchup cards stack;
toggles wrap; hero numerals 56px; Drivers and Head-to-head stack; H2H row becomes
role pill on top, then A and B blocks stacked. Verified at 390px wide before shipping.

## 5. Model (`model.js`) — pure functions, no React

Constants exactly as in the V4 component: `D = 250`, `WR_W = 1.5`, `BLUE_B = 0.15`,
`LAN_B = 0.05`.

```
predict(a, b, blueIsA, lan) -> p in (0,1)   // unchanged formula
explain(a, b, blueIsA, lan) -> {
  elo:  logit(clamp(1 / (1 + 10^(-(a.elo - b.elo)/D)), .001, .999)),
  form: WR_W * (a.wr - b.wr),
  side: blueIsA ? BLUE_B : 0,
  lan:  lan ? LAN_B : 0,
  total, p            // total = sum; p = sigmoid(total) === predict(...)
}
compareRosters(a, b) -> { rows: [{pos, pA, pB, winner: 'a'|'b'|'t'}], aWins, bWins }
```

`compareRosters` uses the V4 rule: per position, count stats where A > B across
KDA, DPM, WR, CS/min, pool; A wins the row with ≥ 3, otherwise B (ties → B, as today).

The side and LAN bonuses apply to Team A exactly as the calibrated proxy defines
them; this is documented in "About the model" as a simplification of the full model.

## 6. Deployment

- `.github/workflows/deploy.yml`: on push to `main` (paths `app/**` and the workflow),
  `npm ci`, `npm test`, `npm run build` in `app/`, then `actions/upload-pages-artifact`
  + `actions/deploy-pages`. Pages source = GitHub Actions.
- `vite.config.js` sets `base: '/lol-esports-predictor/'`.
- Live URL: `https://hget15.github.io/lol-esports-predictor/`. The README gets a
  "Live demo" link under the title and the structure tree is updated.

## 7. Testing

- **Vitest** (`app/src/model.test.js`):
  - `predict(GenG, T1, true, false)` ≈ 0.629 (±0.001) — pins the formula.
  - `explain(...).p === predict(...)` and `total` equals the sum of the parts.
  - Symmetry without bonuses: `predict(a,b,false,false) + predict(b,a,false,false) ≈ 1`.
  - `compareRosters(GenG, T1)` → 4–1 with SUP the only T1 row.
  - URL parsing helper: bad/identical/unknown params → default state.
- CI runs `npm test` before `npm run build`; a failing test blocks the deploy.
- Manual: phone-width check in the browser pane; keyboard-only run-through
  (selects, toggles, swap, about).

## 8. Commits

All work lands on `main` in a few focused commits (scaffold + data extraction;
components + styling; deploy workflow + README). No AI co-author trailers.
