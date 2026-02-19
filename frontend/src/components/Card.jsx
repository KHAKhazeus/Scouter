import React from 'react'

/**
 * Scout card component.
 *
 * Props:
 *   a, b             - the two face values (integers 1–10)
 *   flipped          - bool: if true, b is active (up); else a is active
 *   selected         - bool: highlighted green (valid Show selection)
 *   selectedInvalid  - bool: highlighted orange (selected but not yet a legal show)
 *   isEdge           - bool: this card is the start or end of the selection range
 *   scoutable        - bool: end card of the active set (Scout affordance)
 *   disabled         - bool: greyed out
 *   onClick          - click handler
 *   showIndex        - optional 1-based position shown below the card
 *   isBack           - bool: render as a face-down card (opponent hand)
 */
export default function Card({
  a,
  b,
  flipped = false,
  selected = false,
  selectedInvalid = false,
  isEdge = false,
  scoutable = false,
  disabled = false,
  onClick,
  showIndex,
  isBack = false,
}) {
  const activeVal = flipped ? b : a
  const inactiveVal = flipped ? a : b

  let cls = 'scout-card'
  if (isBack) cls += ' back-face'
  else {
    if (selected) cls += ' selected'
    else if (selectedInvalid) cls += ' selected-invalid'
    else if (scoutable) cls += ' scoutable-end'
    if (isEdge) cls += ' sel-edge'
    if (disabled) cls += ' disabled'
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
      <div className={cls} onClick={!disabled ? onClick : undefined} title={isBack ? '?' : `${activeVal} / ${inactiveVal}`}>
        {isBack ? (
          <div className="card-back-icon">♠</div>
        ) : (
          <>
            <div className="card-half top active-val">{activeVal}</div>
            <div className="card-half bottom inactive-val">{inactiveVal}</div>
          </>
        )}
      </div>
      {showIndex !== undefined && (
        <span style={{ fontSize: '0.68rem', color: 'var(--muted)' }}>{showIndex}</span>
      )}
    </div>
  )
}
