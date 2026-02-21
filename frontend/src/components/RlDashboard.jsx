import React, { useCallback, useEffect, useMemo, useState } from 'react'
import ActiveSet from './ActiveSet.jsx'
import Hand from './Hand.jsx'
import HistoryPanel from './HistoryPanel.jsx'
import ScorePanel from './ScorePanel.jsx'
import {
  N_SHOW,
  N_SCOUT,
  ORIENT_FLIP,
  ORIENT_KEEP,
  decodeScout,
  decodeShow,
  isOrientationAction,
} from '../utils/actionEncoding.js'

function apiUrl(path) {
  return `${window.location.origin}${path}`
}

function runApiPath(runId, suffix) {
  const encoded = encodeURIComponent(runId)
  return `/rl/runs/${encoded}${suffix}`
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined) return 'n/a'
  const num = Number(value)
  if (Number.isNaN(num)) return String(value)
  return num.toFixed(digits)
}

function TrendBar({ value }) {
  const pct = Math.max(0, Math.min(1, Number(value) || 0))
  return (
    <div className="trend-bar-wrap">
      <div className="trend-bar-fill" style={{ width: `${pct * 100}%` }} />
      <span>{fmt(value, 3)}</span>
    </div>
  )
}

function decodeActionLabel(action) {
  if (action === null || action === undefined) return 'No action'
  if (isOrientationAction(action)) {
    return action === ORIENT_FLIP ? 'ORIENT flip hand' : 'ORIENT keep hand'
  }
  if (action < N_SHOW) {
    const [start, end] = decodeShow(action)
    return `SHOW [${start}..${end}]`
  }
  if (action < N_SHOW + N_SCOUT) {
    const [side, flip, insert] = decodeScout(action)
    return `SCOUT ${side === 0 ? 'left' : 'right'} ${flip ? 'flip' : 'keep'} -> pos ${insert}`
  }
  return `Unknown(${action})`
}

function ReplayBoard({ state, candidateSeat }) {
  if (!state) return null
  const opponentSeat =
    candidateSeat === 'player_0'
      ? 'player_1'
      : candidateSeat === 'player_1'
        ? 'player_0'
        : null

  return (
    <div className="replay-board-wrap">
      <ScorePanel state={state} mySeat={candidateSeat || 'spectator'} />
      {opponentSeat ? (
        <div className="rl-card">
          <h3>Opponent Hand ({opponentSeat})</h3>
          <Hand
            cards={state?.hands?.[opponentSeat] || []}
            mode="default"
            isMyTurn={false}
          />
        </div>
      ) : null}
      <ActiveSet
        cards={state.active_set || []}
        owner={state.active_owner}
        mySeat={candidateSeat || 'spectator'}
        actionMask={[]}
        scoutMode={false}
      />
      {candidateSeat ? (
        <div className="rl-card">
          <h3>Candidate Hand ({candidateSeat})</h3>
          <Hand
            cards={state?.hands?.[candidateSeat] || []}
            mode="default"
            isMyTurn={false}
          />
        </div>
      ) : null}
      <HistoryPanel history={state?.history || []} mySeat={candidateSeat || 'spectator'} />
    </div>
  )
}

function ReplayPanel({ replayPayload }) {
  const [idx, setIdx] = useState(0)

  useEffect(() => {
    setIdx(0)
  }, [replayPayload?.game?.game_id])

  const game = replayPayload?.game || null
  const replay = replayPayload?.replay || null
  const steps = replay?.steps || []
  const step = steps[idx] || null
  const state = step?.state || null
  const maxIdx = Math.max(0, steps.length - 1)

  const preserveViewport = useCallback((fn) => {
    const x = window.scrollX
    const y = window.scrollY
    fn()
    requestAnimationFrame(() => {
      window.scrollTo(x, y)
    })
  }, [])

  const jumpTo = useCallback((nextIdx) => {
    preserveViewport(() => {
      setIdx(Math.max(0, Math.min(maxIdx, nextIdx)))
    })
  }, [maxIdx, preserveViewport])

  const moveBy = useCallback((delta) => {
    preserveViewport(() => {
      setIdx((v) => Math.max(0, Math.min(maxIdx, v + delta)))
    })
  }, [maxIdx, preserveViewport])

  useEffect(() => {
    if (!steps.length) return undefined
    function onKeyDown(e) {
      const targetTag = e.target?.tagName?.toLowerCase?.() || ''
      const isTyping = targetTag === 'input' || targetTag === 'textarea' || e.target?.isContentEditable
      if (isTyping) return

      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        moveBy(-1)
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        moveBy(1)
      } else if (e.key === 'Home') {
        e.preventDefault()
        jumpTo(0)
      } else if (e.key === 'End') {
        e.preventDefault()
        jumpTo(maxIdx)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [steps.length, moveBy, jumpTo, maxIdx])

  if (!replayPayload) {
    return <div className="rl-card"><h3>Replay</h3><p>Select a game row to replay.</p></div>
  }
  if (!game || !replay || !Array.isArray(steps)) {
    return (
      <div className="rl-card">
        <h3>Replay</h3>
        <p>Replay data unavailable for the selected game.</p>
      </div>
    )
  }

  return (
    <div className="rl-card">
      <h3>Replay: {game.game_id}</h3>
      <p>
        Evaluated checkpoint: <span className="mono">{game.candidate_snapshot || 'n/a'}</span>
      </p>
      <p>
        Opponent: <span className="mono">
          {game.opponent_type === 'history_checkpoint'
            ? `${game.opponent_type} (${game.opponent_snapshot || 'unknown'})`
            : game.opponent_type || 'random'}
        </span>
      </p>
      <p>
        Candidate seat: <span className="mono">{game.candidate_seat || 'n/a'}</span>
      </p>
      <p>
        Step {idx + 1} / {steps.length}
      </p>
      <div className="action-row replay-nav-row">
        <button type="button" className="btn btn-secondary replay-arrow-btn" onMouseDown={e => e.preventDefault()} onClick={() => jumpTo(0)} disabled={idx <= 0}>|&lt;</button>
        <button type="button" className="btn btn-secondary replay-arrow-btn" onMouseDown={e => e.preventDefault()} onClick={() => moveBy(-1)} disabled={idx <= 0}>&larr;</button>
        <button type="button" className="btn btn-secondary replay-arrow-btn" onMouseDown={e => e.preventDefault()} onClick={() => moveBy(1)} disabled={idx >= maxIdx}>&rarr;</button>
        <button type="button" className="btn btn-secondary replay-arrow-btn" onMouseDown={e => e.preventDefault()} onClick={() => jumpTo(maxIdx)} disabled={idx >= maxIdx}>&gt;|</button>
        <span className="replay-nav-hint mono">Keys: ←/→, Home/End</span>
      </div>

      {step ? (
        <>
          <p>Actor: <span className="mono">{step.actor || 'start'}</span></p>
          <p>Action: <span className="mono">{decodeActionLabel(step.action)}</span></p>
          <ReplayBoard state={state} candidateSeat={game?.candidate_seat || null} />
        </>
      ) : null}

      <h4 style={{ marginTop: 10 }}>Action Timeline</h4>
      <div className="rl-table-wrap" style={{ maxHeight: 220 }}>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Actor</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((s, i) => (
              <tr key={i} style={{ background: i === idx ? 'rgba(93,232,197,0.12)' : 'transparent' }} onClick={() => jumpTo(i)}>
                <td>{i + 1}</td>
                <td>{s.actor || 'start'}</td>
                <td>{decodeActionLabel(s.action)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function RlDashboard() {
  const [runs, setRuns] = useState([])
  const [runId, setRunId] = useState('')
  const [summary, setSummary] = useState(null)
  const [trainRows, setTrainRows] = useState([])
  const [evalRows, setEvalRows] = useState([])
  const [evalGames, setEvalGames] = useState([])
  const [snapshots, setSnapshots] = useState([])
  const [status, setStatus] = useState(null)
  const [events, setEvents] = useState([])
  const [selectedGameId, setSelectedGameId] = useState('')
  const [replayPayload, setReplayPayload] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let cancelled = false
    async function loadRuns() {
      try {
        const res = await fetch(apiUrl('/rl/runs'))
        const data = await res.json()
        if (cancelled) return
        const list = data.runs || []
        setRuns(list)
        if (!runId && list.length > 0) setRunId(list[0].run_id)
      } catch (err) {
        if (!cancelled) setError(`Failed to load runs: ${err}`)
      }
    }
    loadRuns()
    return () => {
      cancelled = true
    }
  }, [runId])

  useEffect(() => {
    if (!runId) return
    let cancelled = false
    async function loadRunData() {
      setLoading(true)
      setError(null)
      try {
        const [s, t, e, g, snap, st, ev] = await Promise.all([
          fetch(apiUrl(runApiPath(runId, '/summary'))).then(r => r.json()),
          fetch(apiUrl(runApiPath(runId, '/train?limit=5000'))).then(r => r.json()),
          fetch(apiUrl(runApiPath(runId, '/evolution?limit=5000'))).then(r => r.json()),
          fetch(apiUrl(runApiPath(runId, '/eval-games?limit=5000'))).then(r => r.json()),
          fetch(apiUrl(runApiPath(runId, '/snapshots'))).then(r => r.json()),
          fetch(apiUrl(runApiPath(runId, '/status'))).then(r => r.json()),
          fetch(apiUrl(runApiPath(runId, '/events?limit=500'))).then(r => r.json()),
        ])

        if (cancelled) return
        setSummary(s)
        setTrainRows(t.rows || [])
        setEvalRows(e.rows || [])
        setEvalGames(g.rows || [])
        setSnapshots(snap.snapshots || [])
        setStatus(st || null)
        setEvents(ev.rows || [])
        setSelectedGameId('')
        setReplayPayload(null)
      } catch (err) {
        if (!cancelled) setError(`Failed to load run data: ${err}`)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    loadRunData()
    return () => {
      cancelled = true
    }
  }, [runId])

  useEffect(() => {
    if (!runId || !selectedGameId) return
    let cancelled = false
    async function loadReplay() {
      try {
        const runPath = encodeURIComponent(runId)
        const gamePath = encodeURIComponent(selectedGameId)
        const res = await fetch(apiUrl(`/rl/runs/${runPath}/eval-games/${gamePath}`))
        const data = await res.json()
        if (!res.ok) {
          throw new Error(data?.detail || `HTTP ${res.status}`)
        }
        if (!cancelled) {
          setReplayPayload(data)
          setError(null)
        }
      } catch (err) {
        if (!cancelled) {
          setReplayPayload(null)
          setError(`Failed to load replay: ${err}`)
        }
      }
    }
    loadReplay()
    return () => {
      cancelled = true
    }
  }, [runId, selectedGameId])

  const latestEval = summary?.latest_eval_summary || null
  const trainingState = status?.training_state || summary?.training_state || {}
  const timeoutState = status?.timeouts || summary?.timeouts || {}

  const randomRows = useMemo(
    () => evalRows.filter(r => r.opponent_type === 'random'),
    [evalRows]
  )

  const historyRows = useMemo(
    () => evalRows.filter(r => r.opponent_type === 'history_checkpoint'),
    [evalRows]
  )

  return (
    <div className="rl-dashboard">
      <header className="rl-header">
        <h1>RL Evolution Dashboard</h1>
        <a href="/" className="btn btn-secondary">Back To Game</a>
      </header>

      <section className="rl-controls">
        <label htmlFor="run-select">Run</label>
        <select id="run-select" value={runId} onChange={e => setRunId(e.target.value)}>
          {runs.map(r => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id} (iter {r.latest_iteration || 0})
            </option>
          ))}
        </select>
      </section>

      {error ? <div className="error-msg">{error}</div> : null}
      {loading ? <div className="rl-card">Loading...</div> : null}

      {!loading && latestEval ? (
        <section className="rl-grid">
          <div className="rl-card">
            <h3>Latest Evolution Summary</h3>
            <p>Iteration: {latestEval.iteration}</p>
            <p>Win vs random</p>
            <TrendBar value={latestEval.win_rate_vs_random} />
            <p>Win vs history (avg)</p>
            <TrendBar value={latestEval.win_rate_vs_history_avg} />
            <p>Win vs history (min)</p>
            <TrendBar value={latestEval.win_rate_vs_history_min} />
            <p>Mean score diff vs random: {fmt(latestEval.mean_score_diff_vs_random)}</p>
          </div>

          <div className="rl-card">
            <h3>Latest Train Stats</h3>
            <p>Episode return mean: {fmt(summary?.latest_train?.episode_return_mean)}</p>
            <p>Player 0 return: {fmt(summary?.latest_train?.per_agent_returns?.player_0)}</p>
            <p>Player 1 return: {fmt(summary?.latest_train?.per_agent_returns?.player_1)}</p>
            <p>Timesteps total: {fmt(summary?.latest_train?.timesteps_total, 0)}</p>
            <p>Snapshots retained: {snapshots.length}</p>
            <p>Eval games: {evalGames.length}</p>
          </div>

          <div className="rl-card">
            <h3>Training Status</h3>
            <p>Phase: <span className="mono">{trainingState.phase || 'unknown'}</span></p>
            <p>Active iteration: {trainingState.active_iteration ?? 'n/a'}</p>
            <p>Active eval opponent: <span className="mono">{trainingState.active_eval_opponent || 'n/a'}</span></p>
            <p>Last heartbeat: <span className="mono">{trainingState.last_heartbeat_at || 'n/a'}</span></p>
            <p>Total eval timeouts: {timeoutState.total_eval_timeouts ?? 0}</p>
            <p>Consecutive eval timeouts: {timeoutState.consecutive_eval_timeouts ?? 0}</p>
            <p>Resume count: {status?.resume_count ?? summary?.manifest?.resume_count ?? 0}</p>
          </div>
        </section>
      ) : null}

      <section className="rl-card">
        <h3>Random Opponent Trend</h3>
        <div className="rl-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Iter</th>
                <th>Win</th>
                <th>Draw</th>
                <th>Loss</th>
                <th>Score Diff</th>
                <th>Status</th>
                <th>Attempts</th>
              </tr>
            </thead>
            <tbody>
              {randomRows.slice(-50).map((r, idx) => (
                <tr key={`${r.iteration}-${idx}`}>
                  <td>{r.iteration}</td>
                  <td>{fmt(r.win_rate)}</td>
                  <td>{fmt(r.draw_rate)}</td>
                  <td>{fmt(r.loss_rate)}</td>
                  <td>{fmt(r.mean_score_diff)}</td>
                  <td><span className={`status-badge status-${(r.status || 'ok').replace(/_/g, '-')}`}>{r.status || 'ok'}</span></td>
                  <td>{r.attempts || 1}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rl-card">
        <h3>Evaluated Games (Click row to replay)</h3>
        <div className="rl-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Game ID</th>
                <th>Iter</th>
                <th>Opponent</th>
                <th>Snapshot</th>
                <th>Outcome</th>
                <th>Score Diff</th>
                <th>Candidate</th>
                <th>Opponent</th>
              </tr>
            </thead>
            <tbody>
              {evalGames.slice(-300).reverse().map((r) => (
                <tr
                  key={r.game_id}
                  style={{ background: selectedGameId === r.game_id ? 'rgba(124,106,245,0.12)' : 'transparent', cursor: 'pointer' }}
                  onClick={() => setSelectedGameId(r.game_id)}
                >
                  <td className="mono">{r.game_id}</td>
                  <td>{r.iteration}</td>
                  <td>{r.opponent_type}</td>
                  <td>{r.opponent_snapshot || 'random'}</td>
                  <td>{r.outcome}</td>
                  <td>{fmt(r.score_diff)}</td>
                  <td>{fmt(r.candidate_score)}</td>
                  <td>{fmt(r.opponent_score)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <ReplayPanel replayPayload={replayPayload} />

      <section className="rl-card">
        <h3>History Checkpoint Matchups</h3>
        <div className="rl-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Iter</th>
                <th>Opponent Snapshot</th>
                <th>Win</th>
                <th>Draw</th>
                <th>Loss</th>
                <th>Score Diff</th>
                <th>Status</th>
                <th>Attempts</th>
              </tr>
            </thead>
            <tbody>
              {historyRows.slice(-200).map((r, idx) => (
                <tr key={`${r.iteration}-${r.opponent_snapshot}-${idx}`}>
                  <td>{r.iteration}</td>
                  <td>{r.opponent_snapshot || 'n/a'}</td>
                  <td>{fmt(r.win_rate)}</td>
                  <td>{fmt(r.draw_rate)}</td>
                  <td>{fmt(r.loss_rate)}</td>
                  <td>{fmt(r.mean_score_diff)}</td>
                  <td><span className={`status-badge status-${(r.status || 'ok').replace(/_/g, '-')}`}>{r.status || 'ok'}</span></td>
                  <td>{r.attempts || 1}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rl-card">
        <h3>Events</h3>
        <div className="rl-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Event</th>
                <th>Iter</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {events.slice(-200).reverse().map((row, idx) => (
                <tr key={`${row.timestamp}-${idx}`}>
                  <td className="mono">{row.timestamp}</td>
                  <td>{row.event}</td>
                  <td>{row.iteration ?? 'n/a'}</td>
                  <td className="mono">{JSON.stringify(row.details || {})}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rl-card">
        <h3>Snapshots</h3>
        <div className="rl-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Snapshot</th>
                <th>Iteration</th>
                <th>Created</th>
                <th>Path</th>
              </tr>
            </thead>
            <tbody>
              {snapshots.map(s => (
                <tr key={s.snapshot_id}>
                  <td>{s.snapshot_id}</td>
                  <td>{s.iteration}</td>
                  <td>{s.created_at}</td>
                  <td className="mono">{s.path}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
