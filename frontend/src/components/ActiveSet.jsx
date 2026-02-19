import React from 'react'
import Card from './Card.jsx'
import { decodeScout } from '../utils/actionEncoding.js'

/**
 * ActiveSet — renders the current active set on the table.
 *
 * Props:
 *   cards         - array of [a, b, flipped] tuples
 *   owner         - "player_0" | "player_1" | null
 *   mySeat        - current player seat
 *   actionMask    - Int8Array / number[] of valid action booleans
 *   scoutMode     - bool: are we in the process of choosing where to insert?
 *   onScout       - fn(side: 0|1, flip: 0|1) — called when a scoutable end card is clicked
 */
export default function ActiveSet({ cards, owner, mySeat, actionMask, scoutMode, onScout }) {
  if (!cards || cards.length === 0) {
    return (
      <div className="active-set-area">
        <span className="active-set-label">Active Set</span>
        <span className="empty-set-hint">No active set — first player must Show.</span>
      </div>
    )
  }

  // Determine which ends are scoutable in the current mask
  const leftScoutable = actionMask
    ? Array.from(actionMask).some((v, idx) => {
        if (!v) return false
        try { const [s] = decodeScout(idx); return s === 0 } catch { return false }
      })
    : false

  const rightScoutable = actionMask
    ? Array.from(actionMask).some((v, idx) => {
        if (!v) return false
        try { const [s] = decodeScout(idx); return s === 1 } catch { return false }
      })
    : false

  return (
    <div className="active-set-area">
      <span className="active-set-label">
        Active Set — owned by {owner ?? '?'}
      </span>
      <div className="active-set-cards">
        {cards.map(([a, b, flipped], i) => {
          const isLeft = i === 0
          const isRight = i === cards.length - 1
          const scoutable = (isLeft && leftScoutable) || (isRight && rightScoutable)
          const side = isLeft ? 0 : 1

          return (
            <Card
              key={i}
              a={a}
              b={b}
              flipped={flipped === 1}
              scoutable={scoutable}
              disabled={!scoutable}
              onClick={scoutable ? () => onScout && onScout(side) : undefined}
            />
          )
        })}
      </div>
      {scoutMode && (
        <span style={{ fontSize: '0.8rem', color: 'var(--warn)' }}>
          Click a card end to Scout, then choose where to insert in your hand.
        </span>
      )}
    </div>
  )
}
