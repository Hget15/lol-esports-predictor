# LoL Esports Predictor — live demo

```
npm install
npm run dev      # http://localhost:5173/lol-esports-predictor/
npm test         # pins Gen.G vs T1 → 0.629, 4–1 H2H, URL parsing
npm run build    # dist/
```

- `src/model.js` — all logic (formula, head-to-head rule, URL state, formatting). Pure functions.
- `src/App.jsx` — render-only components.
- `src/data/teams.json` — 50 teams, verbatim from `app/legacy/lol_predictor_v4.jsx`.
- `src/data/versions.json` — V1–V4 table.

`vite.config.js` sets `base: "/lol-esports-predictor/"` for GitHub Pages.
