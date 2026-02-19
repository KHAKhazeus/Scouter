import React, { useRef, useEffect } from 'react'

function playerLabel(p, mySeat) {
  const num = p === 'player_0' ? '1' : '2'
  const you = p === mySeat ? ' (you)' : ''
  return `P${num}${you}`
}

function formatEntry(entry, mySeat) {
  const p = playerLabel(entry.player, mySeat)

  if (entry.action === 'show') {
    const vals = entry.cards.join(', ')
    const type = entry.set_type === 'match' ? 'match' : entry.set_type === 'run' ? 'run' : ''
    let text = `${p} showed [${vals}] (${entry.size}-card ${type})`
    if (entry.collected) {
      text += ` — collected ${entry.collected} card(s) from ${playerLabel(entry.beat, mySeat)}'s set`
    }
    return { text, cls: 'hist-show' }
  }

  if (entry.action === 'scout') {
    const side = entry.side
    const val = entry.card_value
    const flip = entry.flipped ? ` (flipped from ${entry.original_value})` : ''
    return {
      text: `${p} scouted ${side} card → ${val}${flip}, chips left: ${entry.chips_remaining}`,
      cls: 'hist-scout',
    }
  }

  if (entry.action === 'round_end') {
    const scores = Object.entries(entry.scores)
      .map(([a, s]) => `${playerLabel(a, mySeat)}: ${s >= 0 ? '+' : ''}${s}`)
      .join(', ')
    const rawReason = entry.reason || 'Round ended'
    const reason = rawReason
      .replace(/player_0/g, playerLabel('player_0', mySeat))
      .replace(/player_1/g, playerLabel('player_1', mySeat))
    return { text: `── Round ${entry.round + 1} End ── ${reason}. Scores: ${scores}`, cls: 'hist-round-end' }
  }

  if (entry.action === 'game_over') {
    const w = playerLabel(entry.winner, mySeat)
    const scores = Object.entries(entry.final_scores)
      .map(([a, s]) => `${playerLabel(a, mySeat)}: ${s}`)
      .join(', ')
    return { text: `══ Game Over ══ Winner: ${w} | Final: ${scores}`, cls: 'hist-game-over' }
  }

  return { text: JSON.stringify(entry), cls: '' }
}

export default function HistoryPanel({ history = [], mySeat }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history.length])

  if (history.length === 0) {
    return (
      <div className="history-panel">
        <h4 className="history-title">Move History</h4>
        <p className="history-empty">No moves yet.</p>
      </div>
    )
  }

  return (
    <div className="history-panel">
      <h4 className="history-title">Move History</h4>
      <div className="history-list">
        {history.map((entry, i) => {
          const { text, cls } = formatEntry(entry, mySeat)
          return (
            <div key={i} className={`history-entry ${cls}`}>
              <span className="history-idx">{i + 1}</span>
              <span className="history-text">{text}</span>
            </div>
          )
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
