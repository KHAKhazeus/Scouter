import React, { useEffect, useState } from 'react'

const SEATS = ['player_0', 'player_1', 'spectator']
const HUMAN_SEATS = ['player_0', 'player_1']

export default function LobbyScreen({ onJoined }) {
  const [gameId, setGameId] = useState('')
  const [seat, setSeat] = useState('player_0')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const [agents, setAgents] = useState([])
  const [agentId, setAgentId] = useState('')
  const [agentSeat, setAgentSeat] = useState('player_0')
  const [agentLoadStatus, setAgentLoadStatus] = useState('idle') // idle | loading | loaded | error
  const [agentLoadMessage, setAgentLoadMessage] = useState('')
  const [agentLoadMeta, setAgentLoadMeta] = useState(null)
  const [loadedAgentId, setLoadedAgentId] = useState('')

  useEffect(() => {
    let cancelled = false
    async function loadAgents() {
      try {
        const res = await fetch('/agents/deployed')
        if (!res.ok) return
        const data = await res.json()
        if (cancelled) return
        const list = data.agents || []
        setAgents(list)
        if (!agentId && list.length > 0) setAgentId(list[0].agent_id)
      } catch {
        // Keep lobby usable even if agent API is unavailable.
      }
    }
    loadAgents()
    return () => {
      cancelled = true
    }
  }, [agentId])

  useEffect(() => {
    setAgentLoadStatus('idle')
    setAgentLoadMessage('')
    setAgentLoadMeta(null)
    setLoadedAgentId('')
  }, [agentId])

  async function handleCreate() {
    setLoading(true)
    setError('')
    try {
      const res = await fetch('/game', { method: 'POST' })
      if (!res.ok) throw new Error('Failed to create game')
      const data = await res.json()
      onJoined(data.game_id, seat)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleJoin() {
    if (!gameId.trim()) {
      setError('Enter a game ID')
      return
    }
    setError('')
    onJoined(gameId.trim(), seat)
  }

  async function handlePlayAgent() {
    if (!agentId) {
      setError('No deployed agent selected')
      return
    }
    if (agentLoadStatus !== 'loaded' || loadedAgentId !== agentId) {
      setError('Load agent first before starting.')
      return
    }
    setLoading(true)
    setError('')
    try {
      const res = await fetch('/agent-game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agent_id: agentId,
          human_seat: agentSeat,
          num_rounds: 2,
          reward_mode: 'raw',
        })
      })
      if (!res.ok) {
        const msg = await res.text()
        throw new Error(`Failed to create agent game: ${msg}`)
      }
      const data = await res.json()
      onJoined(data.game_id, agentSeat)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleLoadAgent() {
    if (!agentId) {
      setError('No deployed agent selected')
      return
    }
    setError('')
    setAgentLoadStatus('loading')
    setAgentLoadMessage('')
    setAgentLoadMeta(null)
    try {
      const res = await fetch('/agents/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_id: agentId })
      })
      if (!res.ok) {
        const body = await res.text()
        throw new Error(body || `Failed to load agent ${agentId}`)
      }
      const data = await res.json()
      const device = data.device || 'unknown'
      const seconds = typeof data.load_seconds === 'number' ? data.load_seconds.toFixed(2) : null
      const prefix = data.already_loaded ? 'Agent already loaded' : 'Agent loaded successfully'
      const timing = seconds ? ` (${seconds}s)` : ''
      setAgentLoadStatus('loaded')
      setLoadedAgentId(agentId)
      setAgentLoadMeta(data)
      setAgentLoadMessage(`${prefix} on ${device}${timing}.`)
    } catch (e) {
      setAgentLoadStatus('error')
      setLoadedAgentId('')
      setAgentLoadMeta(null)
      setAgentLoadMessage(e.message || 'Failed to load agent.')
    }
  }

  return (
    <div className="lobby">
      <h1>Scout</h1>
      <p className="tagline">The double-number card game — 2 players, 2 rounds.</p>

      <div className="lobby-card">
        <h2>Play Vs Agent</h2>
        {agents.length === 0 ? (
          <p className="tagline" style={{ marginBottom: 12 }}>No deployed agents found in <code>deployed_agents/</code>.</p>
        ) : (
          <>
            <div className="input-row" style={{ marginBottom: 10 }}>
              <select
                value={agentId}
                onChange={e => setAgentId(e.target.value)}
                style={{
                  flex: 1,
                  background: 'var(--surface2)',
                  border: '1px solid var(--border)',
                  borderRadius: 8,
                  padding: '10px 14px',
                  color: 'var(--text)',
                }}
              >
                {agents.map(a => (
                  <option key={a.agent_id} value={a.agent_id}>
                    {a.name || a.agent_id} ({a.agent_id})
                  </option>
                ))}
              </select>
            </div>
            <div className="seat-picker">
              {HUMAN_SEATS.map(s => (
                <button
                  key={s}
                  className={`seat-btn${agentSeat === s ? ' selected' : ''}`}
                  onClick={() => setAgentSeat(s)}
                >
                  {s === 'player_0' ? 'You = Player 1' : 'You = Player 2'}
                </button>
              ))}
            </div>
            <button
              className="btn btn-secondary btn-full"
              onClick={handleLoadAgent}
              disabled={loading || !agentId || agentLoadStatus === 'loading'}
              style={{ marginBottom: 8 }}
            >
              {agentLoadStatus === 'loading' ? 'Loading Agent…' : 'Load Agent'}
            </button>
            {agentLoadStatus === 'loaded' && (
              <p className="success-msg" style={{ marginBottom: 8 }}>
                {agentLoadMessage}
                {agentLoadMeta?.checkpoint_path ? ` Checkpoint: ${agentLoadMeta.checkpoint_path}` : ''}
              </p>
            )}
            {agentLoadStatus === 'error' && (
              <p className="error-msg" style={{ marginBottom: 8 }}>
                {agentLoadMessage}
              </p>
            )}
            <button
              className="btn btn-primary btn-full"
              onClick={handlePlayAgent}
              disabled={
                loading ||
                !agentId ||
                agentLoadStatus !== 'loaded' ||
                loadedAgentId !== agentId
              }
            >
              {loading ? 'Starting…' : 'Start Vs Agent (2 Rounds)'}
            </button>
          </>
        )}
      </div>

      {/* Create new game */}
      <div className="lobby-card">
        <h2>Create New Game</h2>
        <div className="seat-picker">
          {SEATS.map(s => (
            <button
              key={s}
              className={`seat-btn${seat === s ? ' selected' : ''}`}
              onClick={() => setSeat(s)}
            >
              {s === 'spectator' ? 'Spectator' : s === 'player_0' ? 'Player 1' : 'Player 2'}
            </button>
          ))}
        </div>
        <button
          className="btn btn-primary btn-full"
          onClick={handleCreate}
          disabled={loading}
        >
          {loading ? 'Creating…' : 'Create Game'}
        </button>
      </div>

      {/* Join existing game */}
      <div className="lobby-card">
        <h2>Join Existing Game</h2>
        <div className="input-row">
          <input
            placeholder="Game ID (e.g. a3f1bc2e)"
            value={gameId}
            onChange={e => setGameId(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleJoin()}
          />
        </div>
        <div className="seat-picker">
          {SEATS.map(s => (
            <button
              key={s}
              className={`seat-btn${seat === s ? ' selected' : ''}`}
              onClick={() => setSeat(s)}
            >
              {s === 'spectator' ? 'Spectator' : s === 'player_0' ? 'Player 1' : 'Player 2'}
            </button>
          ))}
        </div>
        <button className="btn btn-secondary btn-full" onClick={handleJoin}>
          Join Game
        </button>
      </div>

      {error && <p className="error-msg">{error}</p>}

      <p style={{ marginTop: 20, fontSize: '0.82rem', color: 'var(--muted)' }}>
        Two players must connect to the same game ID to begin. Share the game ID
        with your opponent after creating.
      </p>
    </div>
  )
}
