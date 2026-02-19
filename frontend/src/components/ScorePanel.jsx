import React from 'react'

function ChipDots({ count, max = 3 }) {
  return (
    <div className="chips">
      {Array.from({ length: max }).map((_, i) => (
        <div key={i} className={`chip-dot${i < count ? '' : ' empty'}`} />
      ))}
    </div>
  )
}

/**
 * ScorePanel — shows chips, collected counts, and cumulative scores for both players.
 *
 * Props:
 *   state      - game state dict from server
 *   mySeat     - "player_0" | "player_1" | "spectator"
 */
export default function ScorePanel({ state, mySeat }) {
  if (!state) return null

  const players = ['player_0', 'player_1']
  const chips = state.scout_chips || {}
  const collected = state.collected || {}
  const scores = state.cumulative_scores || {}

  return (
    <div className="score-panel">
      {players.map((p, i) => {
        const isMe = p === mySeat
        const label = `Player ${i + 1}`
        return (
          <React.Fragment key={p}>
            {i > 0 && <div className="score-divider" />}
            <div className="score-player">
              <span className="label">{label}</span>
              <span className="name">
                {p === state.current_player ? '▶ ' : ''}{label}
                {isMe && <span className="you-badge">you</span>}
              </span>
              <ChipDots count={chips[p] ?? 0} />
              <span className="stats">
                Collected: {collected[p] ?? 0} &nbsp;|&nbsp; Score: {scores[p] ?? 0}
              </span>
            </div>
          </React.Fragment>
        )
      })}

      <div style={{ marginLeft: 'auto', fontSize: '0.82rem', color: 'var(--muted)' }}>
        Round {(state.round ?? 0) + 1}/2
      </div>
    </div>
  )
}
