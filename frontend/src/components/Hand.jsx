import React from 'react'
import Card from './Card.jsx'
import { encodeScout } from '../utils/actionEncoding.js'

/**
 * Hand — renders the player's own hand with selection logic.
 *
 * Selection model (managed by parent GameBoard):
 *   • selStart / selEnd  – inclusive range of currently-selected cards (or null)
 *   • isSelectionValid   – whether the selected range is a legal show
 *   • onCardClick(idx)   – parent handles all selection state transitions
 *
 * Modes:
 *   default      – card clicking for Show selection
 *   scout-insert – insert-slot markers for placing a scouted card
 *
 * Props:
 *   cards           - [[a, b, flipped], ...]
 *   selStart        - number | null
 *   selEnd          - number | null
 *   isSelectionValid - bool
 *   mode            - "default" | "scout-insert"
 *   scoutSide       - 0 | 1
 *   scoutFlip       - 0 | 1
 *   onCardClick     - fn(idx) — called on card click in default mode
 *   onInsert        - fn(actionIndex) — called when user picks an insert slot
 *   isMyTurn        - bool
 */
export default function Hand({
  cards = [],
  selStart = null,
  selEnd = null,
  isSelectionValid = false,
  mode = 'default',
  scoutSide,
  scoutFlip,
  onCardClick,
  onInsert,
  isMyTurn,
}) {
  // Derived range (normalise so lo <= hi)
  const rangeStart = selStart !== null && selEnd !== null ? Math.min(selStart, selEnd) : null
  const rangeEnd   = selStart !== null && selEnd !== null ? Math.max(selStart, selEnd) : null
  const hasSelection = rangeStart !== null

  function label() {
    if (!isMyTurn) return 'Your Hand'
    if (mode === 'scout-insert') return 'Your Hand — click a gap to insert the scouted card'
    if (!hasSelection) return 'Your Hand — click a card to start selecting'
    if (isSelectionValid) return `Cards ${rangeStart + 1}–${rangeEnd + 1} selected ✓ — click Confirm Show, extend, or click a selected card to shrink`
    return `Cards ${rangeStart + 1}–${rangeEnd + 1} — not a valid show yet, keep clicking to adjust`
  }

  if (mode === 'scout-insert') {
    return (
      <div className="hand-area">
        <div className="hand-label">{label()}</div>
        <div className="insert-markers">
          <InsertSlot pos={0} scoutSide={scoutSide} scoutFlip={scoutFlip} onInsert={onInsert} />
          {cards.map(([a, b, flipped], i) => (
            <React.Fragment key={i}>
              <Card a={a} b={b} flipped={flipped === 1} disabled />
              <InsertSlot pos={i + 1} scoutSide={scoutSide} scoutFlip={scoutFlip} onInsert={onInsert} />
            </React.Fragment>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="hand-area">
      <div className="hand-label">{label()}</div>
      <div className="hand-cards">
        {cards.map(([a, b, flipped], i) => {
          const inRange = hasSelection && i >= rangeStart && i <= rangeEnd
          const isEdge  = inRange && (i === rangeStart || i === rangeEnd)
          const isMiddle = inRange && !isEdge

          // Determine card state classes
          const selected = inRange && isSelectionValid
          const selectedInvalid = inRange && !isSelectionValid
          const canClick = isMyTurn && mode === 'default'
          const disabled = !canClick

          return (
            <Card
              key={i}
              a={a}
              b={b}
              flipped={flipped === 1}
              selected={selected}
              selectedInvalid={selectedInvalid}
              isEdge={isEdge && hasSelection}
              disabled={disabled}
              onClick={canClick ? () => onCardClick(i) : undefined}
              showIndex={i + 1}
            />
          )
        })}
      </div>
    </div>
  )
}

function InsertSlot({ pos, scoutSide, scoutFlip, onInsert }) {
  const [hover, setHover] = React.useState(false)
  function handleClick() {
    onInsert && onInsert(encodeScout(scoutSide, scoutFlip, pos))
  }
  return (
    <div
      className={`insert-slot${hover ? ' active' : ''}`}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={handleClick}
      title={`Insert at position ${pos + 1}`}
    />
  )
}
