"use client";

import type { CSSProperties, FormEvent } from "react";
import { useCallback, useEffect, useMemo, useState } from "react";

type Features = { avg_reaction_time_ms: number; max_movement_speed: number; click_interval_std: number; aim_snap_ratio: number };
type ReviewSession = { session_id: string; player_id: string; cheat_risk_score: number; player_cheat_risk_score?: number; created_at: string | null; features: Features; reasons: string[] };
type TestPlayerOptions = { reaction: "typical" | "fast"; movement: "typical" | "fast"; clicks: "natural" | "regular"; aim: "natural" | "high" };

const sampleSessions: ReviewSession[] = [
  { session_id: "match-1048-player-7", player_id: "player-7", cheat_risk_score: 91, player_cheat_risk_score: 73, created_at: "Just now", reasons: ["Fast Reactions", "High Aim Snapping"], features: { avg_reaction_time_ms: 112, max_movement_speed: 8.8, click_interval_std: 19, aim_snap_ratio: 0.81 } },
  { session_id: "match-1048-player-12", player_id: "player-12", cheat_risk_score: 85, player_cheat_risk_score: 61, created_at: "4 min ago", reasons: ["Movement Above Human Cap"], features: { avg_reaction_time_ms: 226, max_movement_speed: 11.4, click_interval_std: 46, aim_snap_ratio: 0.18 } },
  { session_id: "match-1047-player-3", player_id: "player-3", cheat_risk_score: 76, player_cheat_risk_score: 50, created_at: "11 min ago", reasons: ["Highly Regular Clicks"], features: { avg_reaction_time_ms: 241, max_movement_speed: 7.1, click_interval_std: 4, aim_snap_ratio: 0.16 } },
  { session_id: "match-1047-player-19", player_id: "player-19", cheat_risk_score: 65, player_cheat_risk_score: 43, created_at: "18 min ago", reasons: ["Multiple Unusual Patterns"], features: { avg_reaction_time_ms: 164, max_movement_speed: 9.2, click_interval_std: 16, aim_snap_ratio: 0.52 } },
];

const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const riskTone = (score: number) => score >= 80 ? "high" : score >= 60 ? "medium" : "low";
const displayTime = (value: string | null) => !value || value.includes("ago") || value === "Just now" ? value ?? "Recent" : new Intl.DateTimeFormat("en", { hour: "numeric", minute: "2-digit" }).format(new Date(value));

function values(mean: number, spread: number, count: number) {
  return Array.from({ length: count }, (_, index) => Math.max(1, mean + Math.sin(index * 1.7) * spread));
}

function createTelemetry(options: TestPlayerOptions) {
  const reaction = options.reaction === "fast" ? [120, 10] : [245, 55];
  const movement = options.movement === "fast" ? [8.2, 11.5] : [5.2, 7.4];
  const click = options.clicks === "regular" ? [108, 4] : [210, 52];
  const intervals = values(click[0], click[1], 24);
  return {
    reaction_times_ms: values(reaction[0], reaction[1], 24),
    movement_speeds: [movement[1], ...values(movement[0], 0.45, 29)],
    click_timestamps_ms: intervals.reduce<number[]>((times, interval) => [...times, (times.at(-1) ?? 0) + interval], [0]),
    aim_movements: Array.from({ length: 30 }, (_, index) => index < (options.aim === "high" ? 24 : 5)),
  };
}

export default function Home() {
  const [sessions, setSessions] = useState<ReviewSession[]>(sampleSessions);
  const [selectedId, setSelectedId] = useState(sampleSessions[0].session_id);
  const [source, setSource] = useState<"demo" | "live">("demo");
  const [sessionLimit, setSessionLimit] = useState(20);
  const [playerScore, setPlayerScore] = useState<number | null>(sampleSessions[0].player_cheat_risk_score ?? null);
  const [testPlayer, setTestPlayer] = useState<TestPlayerOptions>({ reaction: "typical", movement: "typical", clicks: "natural", aim: "natural" });
  const [importMessage, setImportMessage] = useState("");
  const selected = sessions.find((session) => session.session_id === selectedId) ?? sessions[0];

  const loadSessions = useCallback((limit: number) => {
    fetch(`${apiUrl}/review/flagged-sessions?limit=${limit}&minimum_score=50`)
      .then((response) => response.ok ? response.json() : Promise.reject())
      .then((data: ReviewSession[]) => {
        if (data.length) { setSessions(data); setSelectedId(data[0].session_id); setSource("live"); }
        else { setSessions(sampleSessions); setSelectedId(sampleSessions[0].session_id); setSource("demo"); }
      })
      .catch(() => { setSessions(sampleSessions); setSelectedId(sampleSessions[0].session_id); setSource("demo"); });
  }, []);

  useEffect(() => { loadSessions(sessionLimit); }, [loadSessions, sessionLimit]);

  useEffect(() => {
    if (source === "demo") { setPlayerScore(selected.player_cheat_risk_score ?? null); return; }
    fetch(`${apiUrl}/players/${encodeURIComponent(selected.player_id)}/cheat_risk`)
      .then((response) => response.ok ? response.json() : Promise.reject())
      .then((data: { player_cheat_risk_score: number }) => setPlayerScore(data.player_cheat_risk_score))
      .catch(() => setPlayerScore(null));
  }, [selected, source]);

  async function submitPlayer(options: TestPlayerOptions) {
    const unique = `${Date.now()}-${Math.random().toString(16).slice(2, 7)}`;
    const response = await fetch(`${apiUrl}/events`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: `dashboard-${unique}`, player_id: `test-player-${unique}`, ...createTelemetry(options) }) });
    if (!response.ok) throw new Error("Unable to queue test player");
  }

  async function addRandomPlayers() {
    setImportMessage("Adding 10 simulated players…");
    const scenarios: TestPlayerOptions[] = [
      { reaction: "fast", movement: "fast", clicks: "regular", aim: "high" },
      { reaction: "typical", movement: "fast", clicks: "natural", aim: "natural" },
      { reaction: "typical", movement: "typical", clicks: "regular", aim: "natural" },
      { reaction: "fast", movement: "typical", clicks: "natural", aim: "high" },
    ];
    try { await Promise.all(Array.from({ length: 10 }, (_, index) => submitPlayer(scenarios[index % scenarios.length]))); setImportMessage("10 players added. Refreshing queue…"); window.setTimeout(() => { loadSessions(sessionLimit); setImportMessage("10 simulated players added."); }, 1200); }
    catch { setImportMessage("Could not add players. Make sure the API and worker are running."); }
  }

  async function createTestPlayer(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setImportMessage("Adding test player…");
    try { await submitPlayer(testPlayer); setImportMessage("Test player added. Refreshing queue…"); window.setTimeout(() => { loadSessions(sessionLimit); setImportMessage("Test player added."); }, 1200); }
    catch { setImportMessage("Could not add the test player. Make sure the API and worker are running."); }
  }

  const summary = useMemo(() => ({ high: sessions.filter((item) => item.cheat_risk_score >= 80).length, average: Math.round(sessions.reduce((total, item) => total + item.cheat_risk_score, 0) / sessions.length) }), [sessions]);
  const safety = 100 - Math.round(selected.cheat_risk_score);

  return <main className="app-shell">
    <header><div><p className="eyebrow">Review queue</p><h1>Anti-Cheat Review</h1><p>Review flagged player sessions and their recent history.</p></div><span className={`data-status ${source}`}>{source === "live" ? "Live Data" : "Demo Data"}</span></header>
    <section className="summary"><SummaryCard label="Flagged Sessions" value={sessions.length} detail={source === "live" ? "Current review queue" : "Showing demo sessions"} /><SummaryCard label="High Risk" value={summary.high} detail="Risk score of 80 or higher" danger /><SummaryCard label="Average Risk" value={summary.average} detail="Across displayed sessions" /></section>
    <section className="review-controls card">
      <label className="queue-size">Queue size<select value={sessionLimit} onChange={(event) => setSessionLimit(Number(event.target.value))} aria-label="Number of flagged sessions to show"><option value={12}>12 sessions</option><option value={20}>20 sessions</option><option value={50}>50 sessions</option><option value={100}>100 sessions</option></select></label>
      <div className="random-import"><span>Add test data</span><button type="button" onClick={addRandomPlayers}>Add 10 random players</button></div>
      <p className="lookup-message">{importMessage || "Add simulated players to populate the live review queue."}</p>
    </section>
    <form className="test-player card" onSubmit={createTestPlayer}><div><h2>Create Test Player</h2><p>Choose behavior signals; the app assigns the player ID automatically.</p></div><label>Reaction Time<select value={testPlayer.reaction} onChange={(event) => setTestPlayer({ ...testPlayer, reaction: event.target.value as TestPlayerOptions["reaction"] })}><option value="typical">Typical</option><option value="fast">Fast</option></select></label><label>Movement<select value={testPlayer.movement} onChange={(event) => setTestPlayer({ ...testPlayer, movement: event.target.value as TestPlayerOptions["movement"] })}><option value="typical">Typical</option><option value="fast">Fast</option></select></label><label>Click Regularity<select value={testPlayer.clicks} onChange={(event) => setTestPlayer({ ...testPlayer, clicks: event.target.value as TestPlayerOptions["clicks"] })}><option value="natural">Natural</option><option value="regular">Highly Regular</option></select></label><label>Aim Snapping<select value={testPlayer.aim} onChange={(event) => setTestPlayer({ ...testPlayer, aim: event.target.value as TestPlayerOptions["aim"] })}><option value="natural">Natural</option><option value="high">High</option></select></label><button type="submit">Add test player</button></form>
    <section className="review-layout">
      <article className="players card"><div className="section-title"><div><h2>Flagged Players</h2><p>Select a player to review the session.</p></div><span>{sessions.length} Sessions</span></div><div className="player-grid">{sessions.map((session) => <button key={session.session_id} type="button" className={`player-card ${selected.session_id === session.session_id ? "selected" : ""}`} onClick={() => setSelectedId(session.session_id)}><span className="player-avatar" aria-hidden="true">{session.player_id.replace("player-", "P")}</span><div><p>{session.player_id}</p><span>{displayTime(session.created_at)}</span></div><strong className={riskTone(session.cheat_risk_score)}>{Math.round(session.cheat_risk_score)}<small>Risk</small></strong><i aria-hidden="true" /></button>)}</div></article>
      <article className="session-detail card"><div className="detail-heading"><div><p>Selected Session</p><h2>{selected.player_id}</h2><span>{selected.session_id}</span></div><div className="risk-circle" style={{ "--risk": selected.cheat_risk_score } as CSSProperties}><div><strong>{Math.round(selected.cheat_risk_score)}%</strong><span>Cheat Risk</span></div></div></div><div className="safety"><span>Safety Score</span><strong>{safety}%</strong><p>Based on this session&apos;s behavior.</p></div><section className="reasons"><h3>Why It Was Flagged</h3><div>{selected.reasons.map((reason) => <span key={reason}>{reason}</span>)}</div></section><section className="signals"><Signal label="Reaction Time" value={`${Math.round(selected.features.avg_reaction_time_ms)} ms`} note="Lower can be suspicious" alert={selected.features.avg_reaction_time_ms < 150} /><Signal label="Movement Speed" value={selected.features.max_movement_speed.toFixed(1)} note="Human cap: 9.5" alert={selected.features.max_movement_speed > 9.5} /><Signal label="Click Regularity" value={`${Math.round(selected.features.click_interval_std)} ms`} note="Low means regular clicks" alert={selected.features.click_interval_std < 12} /><Signal label="Aim Snapping" value={`${Math.round(selected.features.aim_snap_ratio * 100)}%`} note="Higher can be suspicious" alert={selected.features.aim_snap_ratio > 0.55} /></section><section className="history"><div><h3>Player History</h3><p>Rolling score from recent sessions.</p></div><strong>{playerScore === null ? "—" : `${Math.round(playerScore)}%`}</strong></section></article>
    </section>
  </main>;
}

function SummaryCard({ label, value, detail, danger = false }: { label: string; value: number; detail: string; danger?: boolean }) { return <article><span>{label}</span><strong className={danger ? "danger" : ""}>{value}</strong><small>{detail}</small></article>; }
function Signal({ label, value, note, alert }: { label: string; value: string; note: string; alert: boolean }) { return <article className={alert ? "alert" : ""}><span>{label}</span><strong>{value}</strong><small>{note}</small></article>; }
