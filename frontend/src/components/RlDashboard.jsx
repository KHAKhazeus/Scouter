import React, { useEffect, useMemo, useState } from 'react'

function apiUrl(path) {
  return `${window.location.origin}${path}`
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

export default function RlDashboard() {
  const [runs, setRuns] = useState([])
  const [runId, setRunId] = useState('')
  const [summary, setSummary] = useState(null)
  const [trainRows, setTrainRows] = useState([])
  const [evalRows, setEvalRows] = useState([])
  const [snapshots, setSnapshots] = useState([])
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
        const [s, t, e, snap] = await Promise.all([
          fetch(apiUrl(`/rl/runs/${runId}/summary`)).then(r => r.json()),
          fetch(apiUrl(`/rl/runs/${runId}/train?limit=5000`)).then(r => r.json()),
          fetch(apiUrl(`/rl/runs/${runId}/evolution?limit=5000`)).then(r => r.json()),
          fetch(apiUrl(`/rl/runs/${runId}/snapshots`)).then(r => r.json())
        ])

        if (cancelled) return
        setSummary(s)
        setTrainRows(t.rows || [])
        setEvalRows(e.rows || [])
        setSnapshots(snap.snapshots || [])
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

  const latestEval = summary?.latest_eval_summary || null

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
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

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
