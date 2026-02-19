import React from 'react'

/**
 * ActionPanel — shows contextual action buttons and status hints.
 *
 * Props:
 *   isMyTurn          - bool
 *   mode              - "default" | "scout-pick" | "scout-insert"
 *   canScout          - bool
 *   hasSelection      - bool (cards selected in hand)
 *   isSelectionValid  - bool (selected range is a legal show)
 *   onConfirmShow     - fn() — send the selected show action
 *   onClearSelection  - fn() — cancel current card selection
 *   onScoutMode       - fn()
 *   onCancel          - fn() — cancel scout mode
 *   currentPlayer     - string
 *   mySeat            - string
 */
export default function ActionPanel({
  isMyTurn,
  mode,
  canScout,
  hasSelection,
  isSelectionValid,
  onConfirmShow,
  onClearSelection,
  onScoutMode,
  onCancel,
  currentPlayer,
  mySeat,
}) {
  if (!isMyTurn) {
    return (
      <div className="action-panel">
        <h3>Waiting</h3>
        <span className="action-hint">
          {currentPlayer === 'player_0' ? 'Player 1' : 'Player 2'}'s turn…
        </span>
      </div>
    )
  }

  if (mode === 'scout-pick') {
    return (
      <div className="action-panel">
        <h3>Scout — Pick a card</h3>
        <p className="action-hint">
          Click the <strong>left</strong> or <strong>right</strong> end of the active
          set to take that card. You will choose its orientation next.
        </p>
        <div className="action-row">
          <button className="btn btn-secondary" onClick={onCancel}>Cancel</button>
        </div>
      </div>
    )
  }

  if (mode === 'scout-insert') {
    return (
      <div className="action-panel">
        <h3>Scout — Choose insert position</h3>
        <p className="action-hint">
          Click a gap in your hand to insert the scouted card there.
        </p>
        <div className="action-row">
          <button className="btn btn-secondary" onClick={onCancel}>Cancel</button>
        </div>
      </div>
    )
  }

  // Default mode
  return (
    <div className="action-panel">
      <h3>Your Turn</h3>

      {/* Show action area */}
      <div className="action-row" style={{ flexWrap: 'wrap', gap: 8 }}>
        {!hasSelection && (
          <span className="action-hint">
            Click cards in your hand one by one to build a selection. Click a
            selected card again to shrink it. Click elsewhere to start over.
          </span>
        )}

        {hasSelection && !isSelectionValid && (
          <span className="action-hint" style={{ color: 'var(--warn)' }}>
            Selection is not yet a valid show — extend or adjust it.
          </span>
        )}

        {hasSelection && isSelectionValid && (
          <>
            <button className="btn btn-primary" onClick={onConfirmShow}>
              Confirm Show
            </button>
            <button className="btn btn-secondary" onClick={onClearSelection}>
              Cancel selection
            </button>
          </>
        )}

        {hasSelection && !isSelectionValid && (
          <button className="btn btn-secondary" onClick={onClearSelection}>
            Clear selection
          </button>
        )}
      </div>

      {/* Scout action */}
      {canScout && (
        <div className="action-row">
          <button className="btn btn-secondary" onClick={onScoutMode}>
            Scout (take from active set)
          </button>
        </div>
      )}
      {!canScout && (
        <span className="action-hint" style={{ color: 'var(--muted)' }}>
          Scout not available (no chips or no active set).
        </span>
      )}
    </div>
  )
}
