import { describe, it, expect } from "vitest";
import teams from "./data/teams.json";
import { predict, explain, compareRosters, parseState, DEFAULT_STATE } from "./model.js";

const G = teams["Gen.G"], T = teams["T1"];

describe("model", () => {
  it("Gen.G vs T1, A on blue, no LAN → 0.629", () => {
    expect(predict(G, T, true, false)).toBeCloseTo(0.629, 3);
  });
  it("explain sums to total and matches predict", () => {
    const e = explain(G, T, true, false);
    expect(e.elo + e.form + e.side + e.lan).toBeCloseTo(e.total, 12);
    expect(e.p).toBe(predict(G, T, true, false));
  });
  it("symmetric without bonuses", () => {
    expect(predict(G, T, false, false) + predict(T, G, false, false)).toBeCloseTo(1, 12);
  });
  it("head-to-head is 4–1 with SUP the only T1 row", () => {
    const r = compareRosters(G, T);
    expect(r.aWins).toBe(4);
    expect(r.bWins).toBe(1);
    expect(r.rows.find(x => x.pos === "sup").winner).toBe("b");
  });
  it("bad URL params fall back to default", () => {
    expect(parseState("?a=Nope&b=T1", teams)).toEqual(DEFAULT_STATE);
    expect(parseState("?a=T1&b=T1", teams)).toEqual(DEFAULT_STATE);
    expect(parseState("?a=T1&b=Gen.G&blue=b&lan=1", teams)).toEqual({ a: "T1", b: "Gen.G", blueIsA: false, lan: true });
  });
});
