import React, { useState } from 'react'

const SEATS = ['player_0', 'player_1', 'spectator']

export default function LobbyScreen({ onJoined }) {
  const [gameId, setGameId] = useState('')
  const [seat, setSeat] = useState('player_0')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

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
    // The GameBoard will initiate the WebSocket connection; just pass along
    onJoined(gameId.trim(), seat)
  }

  return (
    <div className="lobby">
      <h1>Scout</h1>
      <p className="tagline">The double-number card game — 2 players, 2 rounds.</p>

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
