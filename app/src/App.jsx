import { useEffect, useMemo, useState } from "react";
import teams from "./data/teams.json";
import versions from "./data/versions.json";
import { DEFAULT_STATE, parseState, serializeState, derive, fmtElo, fmtPct, fmtKda } from "./model.js";

const REPO = "https://github.com/Hget15/lol-esports-predictor";
const TEAM_NAMES = Object.keys(teams).sort((x, y) => x.localeCompare(y));

// ---- icons (inline stroke SVG) ----
const Icon = ({ d, size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{d}</svg>
);
const Chevron = props => <Icon {...props} d={<path d="M6 9l6 6 6-6" />} />;
const Check = props => <Icon {...props} d={<path d="M5 12l5 5L20 7" />} />;
const Swap = props => <Icon {...props} d={<><path d="M17 3l4 4-4 4" /><path d="M21 7H8" /><path d="M7 21l-4-4 4-4" /><path d="M3 17h13" /></>} />;
const Code = props => <Icon {...props} d={<><path d="M16 18l6-6-6-6" /><path d="M8 6l-6 6 6 6" /></>} />;

const Label = ({ children, className = "" }) => (
  <div className={`font-display text-[14px] font-semibold uppercase tracking-[0.2em] text-gold ${className}`}>{children}</div>
);

// ---- sections ----
function Header() {
  return (
    <header className="flex flex-wrap items-center justify-between gap-4 border-b border-line py-6">
      <div className="flex items-baseline gap-3">
        <span className="font-display text-[26px] font-bold uppercase leading-none tracking-[0.08em] text-ink">LoL Esports Predictor</span>
        <span className="font-display text-[14px] font-semibold tracking-[0.1em] text-gold">V4</span>
      </div>
      <ul className="flex flex-wrap gap-2">
        {["51,088 GAMES · 2021–26", "AUC 0.797", "130 FEATURES"].map(c => (
          <li key={c} className="rounded-[3px] border border-line-strong px-3 py-1.5 font-display text-[13px] font-semibold tracking-[0.06em] text-ink-soft">{c}</li>
        ))}
      </ul>
    </header>
  );
}

function TeamCard({ side, name, team, onChange }) {
  const isA = side === "a";
  return (
    <div className="rounded-[4px] border border-line bg-panel p-6">
      <Label className={isA ? "text-blue" : "text-red"}>{isA ? "Team A · Blue side" : "Team B · Red side"}</Label>
      <div className="relative mt-4">
        <select
          aria-label={isA ? "Team A" : "Team B"}
          value={name}
          onChange={e => onChange(e.target.value)}
          className="h-16 w-full cursor-pointer appearance-none rounded-[4px] border border-line-strong bg-bg pl-4 pr-14 font-display text-[30px] font-semibold text-ink hover:border-line-hover"
        >
          {TEAM_NAMES.map(n => <option key={n} value={n}>{n}</option>)}
        </select>
        <span className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-ink-muted"><Chevron size={18} /></span>
      </div>
      <div className="mt-3 flex gap-4 text-[13px] text-ink-muted">
        <span>Elo <b className="font-display text-[15px] font-semibold text-ink">{fmtElo(team.elo)}</b></span>
        <span>Win rate <b className="font-display text-[15px] font-semibold text-ink">{fmtPct(team.wr)}</b></span>
      </div>
    </div>
  );
}

function Toggle({ on, onClick, icon, children, short }) {
  const base = "inline-flex h-11 items-center gap-2.5 whitespace-nowrap rounded-[3px] px-4 font-display text-[15px] font-semibold tracking-[0.02em]";
  const cls = on ? "bg-blue text-bg" : "border border-line-strong text-ink-muted hover:border-line-hover hover:text-ink";
  return (
    <button type="button" aria-pressed={on} onClick={onClick} className={`${base} ${cls}`}>
      {icon}
      <span className="hidden md:inline">{children}</span>
      <span className="md:hidden">{short ?? children}</span>
    </button>
  );
}

function Matchup({ state, set, a, b }) {
  return (
    <section>
      <div className="grid gap-4 md:grid-cols-2">
        <TeamCard side="a" name={state.a} team={a} onChange={v => set({ a: v })} />
        <TeamCard side="b" name={state.b} team={b} onChange={v => set({ b: v })} />
      </div>
      <div className="mt-5 flex flex-wrap items-center gap-2.5">
        <Toggle on={state.blueIsA} onClick={() => set({ blueIsA: !state.blueIsA })} icon={state.blueIsA ? <Check /> : null} short={`${state.a} on blue`}>{state.a} on blue side</Toggle>
        <Toggle on={state.lan} onClick={() => set({ lan: !state.lan })} icon={state.lan ? <Check /> : null} short="LAN">International (LAN)</Toggle>
        <button type="button" onClick={() => set({ a: state.b, b: state.a })} className="ml-auto inline-flex h-11 items-center gap-2.5 whitespace-nowrap rounded-[3px] border border-line-strong px-4 font-display text-[15px] font-semibold text-ink hover:border-line-hover">
          <Swap /><span className="hidden md:inline">Swap sides</span><span className="md:hidden">Swap</span>
        </button>
      </div>
    </section>
  );
}

function Probability({ pA, pB, nameA, nameB }) {
  const big = "font-display text-[56px] font-bold leading-none md:text-[88px]";
  const pct = "font-display text-[28px] font-bold md:text-[40px]";
  const team = "font-display text-[16px] font-semibold uppercase tracking-[0.15em] mt-1";
  return (
    <section className="rounded-[4px] border border-line bg-panel px-7 py-7">
      <Label className="md:hidden">Win probability</Label>
      <div className="flex items-end justify-between md:items-center">
        <div className="text-blue">
          <div className={big}>{pA}<span className={pct}>%</span></div>
          <div className={team}>{nameA}</div>
        </div>
        <Label className="hidden md:block">Win probability</Label>
        <div className="text-right text-red">
          <div className={big}>{pB}<span className={pct}>%</span></div>
          <div className={team}>{nameB}</div>
        </div>
      </div>
      <div className="mt-5 flex h-2.5 overflow-hidden rounded-[3px] bg-line" role="img" aria-label={`${nameA} ${pA}%, ${nameB} ${pB}%`}>
        <div className="h-full bg-blue transition-[width] duration-500" style={{ width: `${pA}%` }} />
        <div className="h-full bg-red transition-[width] duration-500" style={{ width: `${pB}%` }} />
      </div>
    </section>
  );
}

function Drivers({ drivers, nameA, total, pA }) {
  return (
    <section className="rounded-[4px] border border-line bg-panel p-6">
      <Label>What's driving this</Label>
      <div className="mt-4 grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_48px] items-center gap-x-4 gap-y-4 md:grid-cols-[170px_minmax(0,1fr)_48px]">
        {drivers.map(d => (
          <div key={d.label} className="contents">
            <div className={`text-[14px] ${d.muted ? "text-ink-muted" : "text-ink"}`}>
              <span className="md:hidden">{d.label === "International event" ? "International" : d.label}</span>
              <span className="hidden md:inline">{d.label}</span>
              {d.sub && <span className="hidden text-ink-muted md:inline"> {d.sub}</span>}
            </div>
            <div className="h-2 rounded-[2px] bg-line"><div className="h-full rounded-[2px] bg-gold" style={{ width: `${d.width}%` }} /></div>
            <div className={`text-right font-display text-[18px] font-semibold ${d.muted ? "text-ink-muted" : "text-gold"}`}>{d.value}</div>
          </div>
        ))}
      </div>
      <p className="mt-6 border-t border-line pt-4 text-[12px] leading-relaxed text-ink-muted">
        <span className="hidden md:inline">Log-odds contributions toward {nameA}. They sum to {total}, which is {pA}%. Form on the current patch matters more than raw rating — the model's central finding.</span>
        <span className="md:hidden">Log-odds toward {nameA}, summing to {total} = {pA}%. Form on the current patch beats the rating gap.</span>
      </p>
    </section>
  );
}

function Player({ p, win, align }) {
  const right = align === "right";
  return (
    <div className={`min-w-0 ${right ? "text-right md:text-left" : ""}`}>
      <div className={`truncate text-[14px] ${win ? "font-semibold text-white" : "text-ink-muted"}`}>{p.n}</div>
      <div className={`font-display text-[13px] tracking-[0.02em] ${win ? "text-ink-soft" : "text-ink-muted"}`}>{fmtKda(p.kda)} · {p.dpm} · {p.wr}%</div>
      <div className="mt-0.5 hidden truncate text-[10px] text-ink-muted md:block">{p.ch.join(" · ")}</div>
    </div>
  );
}

function HeadToHead({ h2h, nameA, nameB }) {
  return (
    <section className="rounded-[4px] border border-line bg-panel p-6">
      <div className="flex items-center justify-between">
        <Label><span className="md:hidden">Head-to-head</span><span className="hidden md:inline">Player head-to-head</span></Label>
        <div className="font-display text-[24px] font-bold leading-none">
          <span className="text-blue">{h2h.aWins}</span><span className="text-ink-muted"> – </span><span className="text-red">{h2h.bWins}</span>
        </div>
      </div>
      <div className="mt-2 hidden grid-cols-[48px_1fr_1fr] gap-x-3 font-display text-[11px] font-medium uppercase tracking-[0.12em] text-ink-muted md:grid">
        <span />
        <span>{nameA} · KDA / DPM / WR</span>
        <span>{nameB} · KDA / DPM / WR</span>
      </div>
      <div className="mt-2 font-display text-[11px] font-medium uppercase tracking-[0.12em] text-ink-muted md:hidden">KDA · DPM · WR</div>
      <ul className="mt-3 flex flex-col gap-1.5">
        {h2h.rows.map(r => {
          const aWin = r.winner === "a";
          return (
            <li key={r.pos} className={`grid gap-x-3 gap-y-1.5 rounded-[3px] px-3 py-2.5 md:grid-cols-[48px_1fr_1fr] md:items-center ${aWin ? "bg-tint-a" : "bg-tint-b"}`}>
              <div className="font-display text-[12px] font-bold tracking-[0.12em]" style={{ color: r.color }} aria-label={`${r.label}, ${aWin ? r.pA.n : r.pB.n} wins`}>{r.label}</div>
              <div className="contents md:contents">
                <div className="grid grid-cols-2 gap-3 md:contents">
                  <Player p={r.pA} win={aWin} />
                  <Player p={r.pB} win={!aWin} align="right" />
                </div>
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function AboutModel() {
  const [open, setOpen] = useState(false);
  return (
    <section className="rounded-[4px] border border-line bg-panel">
      <button type="button" aria-expanded={open} onClick={() => setOpen(o => !o)} className="flex h-11 w-full items-center justify-between px-6 text-left">
        <Label><span className="md:hidden">About the model</span><span className="hidden md:inline">About the model · V1 → V4</span></Label>
        <span className={`text-ink-muted transition-transform duration-200 ${open ? "rotate-180" : ""}`}><Chevron size={18} /></span>
      </button>
      {open && (
        <div className="border-t border-line px-6 pb-6 pt-4">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="font-display text-[11px] font-medium uppercase tracking-[0.12em] text-ink-muted">
                <th className="py-1.5 text-left font-medium">Version</th>
                <th className="py-1.5 text-right font-medium">AUC</th>
                <th className="py-1.5 text-right font-medium">Features</th>
                <th className="py-1.5 pl-6 text-left font-medium">Added</th>
              </tr>
            </thead>
            <tbody>
              {versions.map(v => (
                <tr key={v.v} className="border-t border-line">
                  <td className="py-2 font-display text-[15px] font-semibold text-ink">{v.v}</td>
                  <td className="py-2 text-right font-display text-[15px] font-semibold text-gold">{v.auc}</td>
                  <td className="py-2 text-right font-display text-[15px] text-ink-soft">{v.features}</td>
                  <td className="py-2 pl-6 text-ink-soft">{v.added}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-4 text-[12px] leading-relaxed text-ink-muted">The page uses a calibrated Elo + form proxy of the full model; the real model needs a full 130-feature vector per game.</p>
        </div>
      )}
    </section>
  );
}

function Footer() {
  return (
    <footer className="flex flex-wrap items-start justify-between gap-4 border-t border-line pt-5 text-[12px] leading-relaxed text-ink-muted">
      <p className="max-w-[640px] m-0">Probabilities come from a calibrated Elo + form proxy of the full 130-feature model. Rosters as of March 2026. Data: Oracle's Elixir.</p>
      <a href={REPO} target="_blank" rel="noreferrer" className="inline-flex h-11 items-center gap-2 font-display text-[14px] font-semibold uppercase tracking-[0.15em] text-gold no-underline hover:text-ink">
        <Code /> See the code
      </a>
    </footer>
  );
}

export default function App() {
  const [state, setState] = useState(() => parseState(window.location.search, teams));
  const set = patch => setState(s => ({ ...s, ...patch }));
  useEffect(() => { window.history.replaceState(null, "", serializeState(state)); }, [state]);
  const d = useMemo(() => derive(state, teams), [state]);

  return (
    <main className="mx-auto flex max-w-[1080px] flex-col gap-5 px-4 pb-16 md:px-10">
      <Header />
      <p className="m-0 max-w-[640px] text-[15px] leading-relaxed text-ink-muted">
        <span className="hidden md:inline">Who wins a professional League of Legends match? Pick two teams. The model, trained from scratch on five seasons of pro play, gives you a probability — and shows its reasoning.</span>
        <span className="md:hidden">Pick two pro teams. The model gives a probability — and shows its reasoning.</span>
      </p>
      <Matchup state={state} set={set} a={d.a} b={d.b} />
      {d.same ? (
        <p className="m-0 rounded-[4px] border border-line bg-panel px-6 py-5 text-[15px] text-ink-muted">Pick two different teams.</p>
      ) : (
        <>
          <Probability pA={d.pA} pB={d.pB} nameA={state.a} nameB={state.b} />
          <div className="grid gap-5 md:grid-cols-2">
            <Drivers drivers={d.drivers} nameA={state.a} total={d.total} pA={d.pA} />
            <HeadToHead h2h={d.h2h} nameA={state.a} nameB={state.b} />
          </div>
        </>
      )}
      <AboutModel />
      <Footer />
    </main>
  );
}
