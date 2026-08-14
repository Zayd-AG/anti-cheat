"use client";

import type { CSSProperties } from "react";
import { useEffect, useMemo, useState } from "react";

type Features = { avg_reaction_time_ms: number; max_movement_speed: number; click_interval_std: number; aim_snap_ratio: number };
type ReviewSession = { session_id: string; player_id: string; cheat_risk_score: number; player_cheat_risk_score?: number; created_at: string | null; features: Features; reasons: string[] };

const sampleSessions: ReviewSession[] = [
  { session_id: "match-1048-player-7", player_id: "player-7", cheat_risk_score: 91, player_cheat_risk_score: 73, created_at: "Just now", reasons: ["Fast Reactions", "High Aim Snapping"], features: { avg_reaction_time_ms: 112, max_movement_speed: 8.8, click_interval_std: 19, aim_snap_ratio: 0.81 } },
  { session_id: "match-1048-player-12", player_id: "player-12", cheat_risk_score: 85, player_cheat_risk_score: 61, created_at: "4 min ago", reasons: ["Movement Above Human Cap"], features: { avg_reaction_time_ms: 226, max_movement_speed: 11.4, click_interval_std: 46, aim_snap_ratio: 0.18 } },
  { session_id: "match-1047-player-3", player_id: "player-3", cheat_risk_score: 76, player_cheat_risk_score: 50, created_at: "11 min ago", reasons: ["Highly Regular Clicks"], features: { avg_reaction_time_ms: 241, max_movement_speed: 7.1, click_interval_std: 4, aim_snap_ratio: 0.16 } },
  { session_id: "match-1047-player-19", player_id: "player-19", cheat_risk_score: 65, player_cheat_risk_score: 43, created_at: "18 min ago", reasons: ["Multiple Unusual Patterns"], features: { avg_reaction_time_ms: 164, max_movement_speed: 9.2, click_interval_std: 16, aim_snap_ratio: 0.52 } },
];

const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const riskTone = (score: number) => score >= 80 ? "high" : score >= 60 ? "medium" : "low";
const displayTime = (value: string | null) => !value || value.includes("ago") || value === "Just now" ? value ?? "Recent" : new Intl.DateTimeFormat("en", { hour: "numeric", minute: "2-digit" }).format(new Date(value));

export default function Home() {
  const [sessions, setSessions] = useState<ReviewSession[]>(sampleSessions);
  const [selectedId, setSelectedId] = useState(sampleSessions[0].session_id);
  const [source, setSource] = useState<"demo" | "live">("demo");
  const [playerScore, setPlayerScore] = useState<number | null>(sampleSessions[0].player_cheat_risk_score ?? null);
  const selected = sessions.find((session) => session.session_id === selectedId) ?? sessions[0];

  useEffect(() => {
    fetch(`${apiUrl}/review/flagged-sessions?limit=12&minimum_score=50`)
      .then((response) => response.ok ? response.json() : Promise.reject())
      .then((data: ReviewSession[]) => { if (data.length) { setSessions(data); setSelectedId(data[0].session_id); setSource("live"); } })
      .catch(() => setSource("demo"));
  }, []);

  useEffect(() => {
    // Demo cards do not exist in PostgreSQL, so do not make requests that would
    // create noisy 404s in the API log.
    if (source === "demo") { setPlayerScore(selected.player_cheat_risk_score ?? null); return; }
    fetch(`${apiUrl}/players/${selected.player_id}/cheat_risk`)
      .then((response) => response.ok ? response.json() : Promise.reject())
      .then((data) => setPlayerScore(data.player_cheat_risk_score))
      .catch(() => setPlayerScore(null));
  }, [selected, source]);

  const summary = useMemo(() => ({
    high: sessions.filter((item) => item.cheat_risk_score >= 80).length,
    average: Math.round(sessions.reduce((total, item) => total + item.cheat_risk_score, 0) / sessions.length),
  }), [sessions]);
  const safety = 100 - Math.round(selected.cheat_risk_score);

  return <main className="app-shell">
    <header><div><p className="eyebrow">Review queue</p><h1>Anti-Cheat Review</h1><p>Review flagged player sessions and their recent history.</p></div><span className={`data-status ${source}`}>{source === "live" ? "Live Data" : "Demo Data"}</span></header>
    <section className="summary"><SummaryCard label="Flagged Sessions" value={sessions.length} detail="Current review queue" /><SummaryCard label="High Risk" value={summary.high} detail="Risk score of 80 or higher" danger /><SummaryCard label="Average Risk" value={summary.average} detail="Across flagged sessions" /></section>
    <section className="review-layout">
      <article className="players card"><div className="section-title"><div><h2>Flagged Players</h2><p>Select a player to review the session.</p></div><span>{sessions.length} Sessions</span></div><div className="player-grid">{sessions.map((session) => <button key={session.session_id} type="button" className={`player-card ${selected.session_id === session.session_id ? "selected" : ""}`} onClick={() => setSelectedId(session.session_id)}><span className="player-avatar" aria-hidden="true">{session.player_id.replace("player-", "P")}</span><div><p>{session.player_id}</p><span>{displayTime(session.created_at)}</span></div><strong className={riskTone(session.cheat_risk_score)}>{Math.round(session.cheat_risk_score)}<small>Risk</small></strong><i aria-hidden="true" /></button>)}</div></article>
      <article className="session-detail card"><div className="detail-heading"><div><p>Selected Session</p><h2>{selected.player_id}</h2><span>{selected.session_id}</span></div><div className="risk-circle" style={{ "--risk": selected.cheat_risk_score } as CSSProperties}><div><strong>{Math.round(selected.cheat_risk_score)}%</strong><span>Cheat Risk</span></div></div></div><div className="safety"><span>Safety Score</span><strong>{safety}%</strong><p>Based on this session&apos;s behavior.</p></div><section className="reasons"><h3>Why It Was Flagged</h3><div>{selected.reasons.map((reason) => <span key={reason}>{reason}</span>)}</div></section><section className="signals"><Signal label="Reaction Time" value={`${Math.round(selected.features.avg_reaction_time_ms)} ms`} note="Lower can be suspicious" alert={selected.features.avg_reaction_time_ms < 150} /><Signal label="Movement Speed" value={selected.features.max_movement_speed.toFixed(1)} note="Human cap: 9.5" alert={selected.features.max_movement_speed > 9.5} /><Signal label="Click Regularity" value={`${Math.round(selected.features.click_interval_std)} ms`} note="Low means regular clicks" alert={selected.features.click_interval_std < 12} /><Signal label="Aim Snapping" value={`${Math.round(selected.features.aim_snap_ratio * 100)}%`} note="Higher can be suspicious" alert={selected.features.aim_snap_ratio > 0.55} /></section><section className="history"><div><h3>Player History</h3><p>Rolling score from recent sessions.</p></div><strong>{playerScore === null ? "—" : `${Math.round(playerScore)}%`}</strong></section></article>
    </section>
  </main>;
}

function SummaryCard({ label, value, detail, danger = false }: { label: string; value: number; detail: string; danger?: boolean }) { return <article><span>{label}</span><strong className={danger ? "danger" : ""}>{value}</strong><small>{detail}</small></article>; }
function Signal({ label, value, note, alert }: { label: string; value: string; note: string; alert: boolean }) { return <article className={alert ? "alert" : ""}><span>{label}</span><strong>{value}</strong><small>{note}</small></article>; }
