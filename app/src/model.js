// Pure model + presentation logic. No React, no DOM.
export const D = 250;
export const WR_W = 1.5;
export const BLUE_B = 0.15;
export const LAN_B = 0.05;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export function explain(a, b, blueIsA, lanEvent) {
  const base = clamp(1 / (1 + 10 ** (-(a.elo - b.elo) / D)), 0.001, 0.999);
  const elo = Math.log(base / (1 - base));
  const form = WR_W * (a.wr - b.wr);
  const side = blueIsA ? BLUE_B : 0;
  const lan = lanEvent ? LAN_B : 0;
  const total = elo + form + side + lan;
  return { elo, form, side, lan, total, p: 1 / (1 + Math.exp(-total)) };
}

export function predict(a, b, blueIsA, lanEvent) {
  return explain(a, b, blueIsA, lanEvent).p;
}

export const POS = ["top", "jng", "mid", "bot", "sup"];
export const POS_COLOR = { top: "#ef4444", jng: "#22c55e", mid: "#3b82f6", bot: "#f59e0b", sup: "#a855f7" };
const H2H_STATS = ["kda", "dpm", "wr", "cs", "pool"];

export function compareRosters(a, b) {
  let aWins = 0, bWins = 0;
  const rows = [];
  for (const pos of POS) {
    const pA = a.r?.[pos], pB = b.r?.[pos];
    if (!pA || !pB) continue;
    const aHigher = H2H_STATS.filter(k => pA[k] > pB[k]).length;
    const winner = aHigher >= 3 ? "a" : "b";
    if (winner === "a") aWins++; else bWins++;
    rows.push({ pos, label: pos.toUpperCase(), color: POS_COLOR[pos], pA, pB, winner });
  }
  return { rows, aWins, bWins };
}

// ---- formatting ----
export const fmtSigned = (n, digits = 2) => {
  const k = 10 ** digits;
  const v = Math.round((n + Math.sign(n) * 1e-9) * k) / k; // epsilon guards 0.22499999… → 0.23
  if (v === 0) return "0";
  return (v > 0 ? "+" : "−") + Math.abs(v).toFixed(digits);
};
export const fmtElo = elo => (elo >= 0 ? "+" : "−") + Math.abs(elo).toFixed(1);
export const fmtPct = wr => Math.round(wr * 100) + "%";
export const fmtKda = kda => kda.toFixed(1);

export function driverRows(a, b, ex) {
  const max = Math.max(...[ex.elo, ex.form, ex.side, ex.lan].map(Math.abs), 1e-9);
  const row = (label, sub, v) => ({ label, sub, value: fmtSigned(v), width: Math.abs(v) / max * 100, muted: Math.abs(v) < 1e-9 });
  return [
    row("Recent form", fmtSigned((a.wr - b.wr) * 100, 0) + " pts WR", ex.form),
    row("Elo edge", fmtSigned(a.elo - b.elo, 1), ex.elo),
    row("Blue side", "", ex.side),
    row("International event", "", ex.lan),
  ];
}

// ---- state ----
export const DEFAULT_STATE = { a: "Gen.G", b: "T1", blueIsA: true, lan: false };

export function parseState(search, teams) {
  try {
    const q = new URLSearchParams(search);
    const a = q.get("a"), b = q.get("b");
    if (!a || !b || a === b || !teams[a] || !teams[b]) return { ...DEFAULT_STATE };
    const blue = q.get("blue");
    return { a, b, blueIsA: blue !== "b", lan: q.get("lan") === "1" };
  } catch {
    return { ...DEFAULT_STATE };
  }
}

export function serializeState(s) {
  const q = new URLSearchParams();
  q.set("a", s.a);
  q.set("b", s.b);
  q.set("blue", s.blueIsA ? "a" : "b");
  if (s.lan) q.set("lan", "1");
  return "?" + q.toString();
}

// Everything the page renders, derived from state + data.
export function derive(state, teams) {
  const a = teams[state.a], b = teams[state.b];
  const same = state.a === state.b;
  if (!a || !b || same) return { a, b, same: true };
  const ex = explain(a, b, state.blueIsA, state.lan);
  const pA = Math.round(ex.p * 100);
  return {
    a, b, same: false, ex, pA, pB: 100 - pA,
    total: fmtSigned(ex.total),
    drivers: driverRows(a, b, ex),
    h2h: compareRosters(a, b),
  };
}
