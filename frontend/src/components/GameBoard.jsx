import React, { useState, useCallback, useMemo, useEffect, useRef } from 'react'
import { useGameSocket } from '../hooks/useGameSocket.js'
import ScorePanel from './ScorePanel.jsx'
import ActiveSet from './ActiveSet.jsx'
import Hand from './Hand.jsx'
import ActionPanel from './ActionPanel.jsx'
import Card from './Card.jsx'
import HistoryPanel from './HistoryPanel.jsx'
import RoundSummary from './RoundSummary.jsx'
import {
  ORIENT_FLIP,
  ORIENT_KEEP,
  encodeShow,
  encodeScout,
  validScoutsFromMask,
} from '../utils/actionEncoding.js'

/**
 * GameBoard — main game UI.
 *
 * Props:
 *   gameId  - string
 *   seat    - "player_0" | "player_1" | "spectator"
 *   onLeave - fn()
 */
export default function GameBoard({ gameId, seat, onLeave }) {
  const { state, connected, sendAction, sendChat, messages, error } =
    useGameSocket(gameId, seat)

  const gameOver    = state?.game_over ?? false
  const isMyTurn = state?.current_player === seat && !gameOver
  const myHand   = state?.hands?.[seat] ?? []
  const oppSeat  = seat === 'player_0' ? 'player_1' : seat === 'player_1' ? 'player_0' : null
  const oppHandSize = oppSeat
    ? (state?.hands?.[oppSeat] === null ? '?' : state?.hands?.[oppSeat]?.length ?? 0)
    : 0

  const actionMask = state?.action_mask ?? []

  // ── Card selection state ──────────────────────────────────────────────────
  // selStart / selEnd are indices into myHand; null = nothing selected.
  const [selStart, setSelStart] = useState(null)
  const [selEnd,   setSelEnd]   = useState(null)

  // Clear selection whenever the state updates (own turn starts / ends)
  const prevPlayerRef = useRef(null)
  useEffect(() => {
    if (state?.current_player !== prevPlayerRef.current) {
      setSelStart(null)
      setSelEnd(null)
      prevPlayerRef.current = state?.current_player
    }
  }, [state?.current_player])

  const hasSelection    = selStart !== null && selEnd !== null
  const rangeStart      = hasSelection ? Math.min(selStart, selEnd) : null
  const rangeEnd        = hasSelection ? Math.max(selStart, selEnd) : null
  const isSelectionValid = useMemo(() => {
    if (!hasSelection) return false
    const idx = encodeShow(rangeStart, rangeEnd)
    return actionMask[idx] === 1
  }, [hasSelection, rangeStart, rangeEnd, actionMask])

  /**
   * Card click logic:
   * - No selection → start selection at this card
   * - Click range boundary → shrink from that side (cancel if single)
   * - Click one past a boundary → extend
   * - Click anywhere else → start fresh selection at clicked card
   */
  function handleCardClick(idx) {
    if (!isMyTurn) return

    if (!hasSelection) {
      setSelStart(idx)
      setSelEnd(idx)
      return
    }

    const lo = rangeStart
    const hi = rangeEnd

    if (idx === lo && idx === hi) {
      // Single card — second click cancels
      setSelStart(null); setSelEnd(null)
    } else if (idx === lo) {
      // Shrink from left
      setSelStart(lo + 1)
    } else if (idx === hi) {
      // Shrink from right
      setSelEnd(hi - 1)
    } else if (idx === hi + 1) {
      // Extend right
      setSelEnd(hi + 1)
    } else if (idx === lo - 1) {
      // Extend left
      setSelStart(lo - 1)
    } else {
      // Anywhere else → start fresh
      setSelStart(idx); setSelEnd(idx)
    }
  }

  function handleConfirmShow() {
    if (!isSelectionValid) return
    sendAction(encodeShow(rangeStart, rangeEnd))
    setSelStart(null); setSelEnd(null)
  }

  function handleClearSelection() {
    setSelStart(null); setSelEnd(null)
  }

  // ── Scout state ───────────────────────────────────────────────────────────
  // mode: "default" | "scout-pick" | "scout-flip" | "scout-insert"
  const [mode,      setMode]      = useState('default')
  const [scoutSide, setScoutSide] = useState(null)
  const [scoutFlip, setScoutFlip] = useState(null)

  function resetMode() {
    setMode('default'); setScoutSide(null); setScoutFlip(null)
  }

  useEffect(() => { resetMode() }, [state?.current_player])  // eslint-disable-line

  const canScout = useMemo(() => {
    if (!isMyTurn) return false
    if (!state?.active_set?.length) return false
    const chips = state?.scout_chips?.[seat] ?? 0
    if (chips <= 0) return false
    return validScoutsFromMask(actionMask).length > 0
  }, [isMyTurn, state, seat, actionMask])

  function handleScoutMode()        { setMode('scout-pick') }
  function handleScoutPick(side)    { setScoutSide(side); setMode('scout-flip') }
  function handleFlipChoice(flip)   { setScoutFlip(flip); setMode('scout-insert') }
  function handleInsert(actionIdx)  { sendAction(actionIdx); resetMode(); setSelStart(null); setSelEnd(null) }

  // ── Round-start flip modal ────────────────────────────────────────────────
  // Show whenever the backend says this player can still flip their hand.
  // "flipModalDone" is reset whenever the round (or game) changes.
  const [flipModalDone, setFlipModalDone] = useState(false)
  const prevRoundRef = useRef(null)
  useEffect(() => {
    if (state?.round !== prevRoundRef.current) {
      setFlipModalDone(false)
      prevRoundRef.current = state?.round
    }
  }, [state?.round])

  const orientationPending = seat !== 'spectator' && (state?.orientation_pending?.[seat] ?? false)
  const canFlip = orientationPending && isMyTurn
  const showFlipModal = canFlip && !flipModalDone

  // ── Round summary banner ─────────────────────────────────────────────────
  // Track how many round results we've dismissed so we only show the latest new one.
  const [dismissedRounds, setDismissedRounds] = useState(0)
  const roundResults = state?.round_results ?? []
  const latestRoundResult = roundResults.length > dismissedRounds ? roundResults[roundResults.length - 1] : null
  const showRoundSummary = latestRoundResult && !gameOver

  // ── Chat ──────────────────────────────────────────────────────────────────
  const [chatText, setChatText] = useState('')
  function handleSendChat(e) {
    e.preventDefault()
    if (chatText.trim()) { sendChat(chatText.trim()); setChatText('') }
  }

  // Scout flip-card preview
  const activeSetCards = state?.active_set ?? []
  let scoutCardPreview = null
  if (mode === 'scout-flip' && scoutSide !== null && activeSetCards.length > 0) {
    const raw = scoutSide === 0 ? activeSetCards[0] : activeSetCards[activeSetCards.length - 1]
    const [a, b, flipped] = raw
    scoutCardPreview = {
      a, b,
      flipped: flipped === 1,
      activeVal:   flipped ? b : a,
      otherVal:    flipped ? a : b,
    }
  }

  // ── Waiting screen ────────────────────────────────────────────────────────
  if (!state) {
    return (
      <div className="game-board">
        <div className="top-bar">
          <span className="brand">Scout</span>
          <button className="btn btn-secondary" onClick={onLeave}>Leave</button>
        </div>
        <div className="waiting-overlay">
          <div className="spinner" />
          <p style={{ color: 'var(--muted)' }}>
            {connected ? `Waiting for opponent… Game ID: ${gameId}` : 'Connecting…'}
          </p>
          <p style={{ fontSize: '0.82rem', color: 'var(--muted)' }}>
            Share this Game ID with your opponent: <strong>{gameId}</strong>
          </p>
          {error && <p className="error-msg">{error}</p>}
        </div>
      </div>
    )
  }

  const winner      = state?.winner
  const finalScores = state?.cumulative_scores ?? {}

  return (
    <div className={`game-board${mode.startsWith('scout') ? ' scout-mode' : ''}`}>

      {/* Top bar */}
      <div className="top-bar">
        <span className="brand">Scout</span>
        <span className="round-info">
          Game: <code>{gameId}</code> &nbsp;|&nbsp; You: <strong>{seat}</strong>
        </span>
        <button
          className="btn btn-secondary"
          style={{ fontSize: '0.8rem', padding: '6px 12px' }}
          onClick={onLeave}
        >Leave</button>
      </div>

      {/* Score panel */}
      <ScorePanel state={state} mySeat={seat} />

      {/* Status bar */}
      <div className={`status-bar${isMyTurn ? ' my-turn' : ''}`}>
        {gameOver
          ? `Game Over! Winner: ${winner ?? 'tie'}`
          : isMyTurn
            ? 'Your turn!'
            : `Waiting for ${state.current_player}…`}
        {error && <span className="error-msg"> &nbsp; {error}</span>}
      </div>

      {/* ── Round summary banner (between rounds) ────────────────────────── */}
      {showRoundSummary && (
        <RoundSummary
          roundResult={latestRoundResult}
          mySeat={seat}
          onDismiss={() => setDismissedRounds(roundResults.length)}
        />
      )}

      {/* ── Round-start flip modal ─────────────────────────────────────────── */}
      {showFlipModal && (
        <div className="flip-modal">
          <div className="flip-modal-header">
            <span className="flip-modal-title">
              Round {state.round + 1} — Choose your hand orientation
            </span>
            <p className="flip-modal-hint">
              You have one chance to flip your entire hand before playing.
              Flipping reverses the order and shows each card from the other side.
            </p>
          </div>

          <div className="flip-modal-hands">
            {/* Current orientation */}
            <div className="flip-option">
              <div className="flip-option-label">Current</div>
              <div className="flip-preview-cards">
                {myHand.map(([a, b, f], i) => (
                  <Card key={i} a={a} b={b} flipped={f === 1} />
                ))}
              </div>
            </div>

            {/* Flipped orientation (reverse + flip each) */}
            <div className="flip-option">
              <div className="flip-option-label">Flipped</div>
              <div className="flip-preview-cards">
                {[...myHand].reverse().map(([a, b, f], i) => (
                  <Card key={i} a={a} b={b} flipped={f === 0} />
                ))}
              </div>
            </div>
          </div>

          <div className="flip-modal-actions">
            <button
              className="btn btn-secondary"
              onClick={() => {
                sendAction(ORIENT_FLIP)
                setFlipModalDone(true)
              }}
            >
              Flip all cards
            </button>
            <button
              className="btn btn-primary"
              onClick={() => {
                sendAction(ORIENT_KEEP)
                setFlipModalDone(true)
              }}
            >
              Confirm &amp; Play
            </button>
          </div>
        </div>
      )}

      {/* Opponent's hand */}
      <div className="opp-hand-area">
        <span className="hand-label" style={{ marginBottom: 0 }}>
          {oppSeat ?? 'Opponent'} — {oppHandSize} card(s)
        </span>
        <div className="opp-hand-cards">
          {seat === 'spectator' && oppSeat && state.hands[oppSeat]
            ? state.hands[oppSeat].map(([a, b, f], i) => (
                <Card key={i} a={a} b={b} flipped={f === 1} />
              ))
            : Array.from({ length: oppHandSize }).map((_, i) => (
                <Card key={i} isBack />
              ))}
        </div>
      </div>

      {/* Active set */}
      <ActiveSet
        cards={state.active_set}
        owner={state.active_owner}
        mySeat={seat}
        actionMask={actionMask}
        scoutMode={mode === 'scout-pick'}
        onScout={handleScoutPick}
      />

      {/* Scout flip-choice modal (inline) */}
      {mode === 'scout-flip' && scoutCardPreview && (
        <div style={{
          background: 'var(--surface2)',
          border: '1px solid var(--warn)',
          borderRadius: 10,
          padding: '16px 20px',
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          alignItems: 'center',
        }}>
          <p style={{ fontSize: '0.9rem', color: 'var(--text)' }}>
            Which side up when you insert this card?
          </p>
          <div style={{ display: 'flex', gap: 24, alignItems: 'center' }}>
            <div style={{ textAlign: 'center' }}>
              <Card a={scoutCardPreview.a} b={scoutCardPreview.b} flipped={scoutCardPreview.flipped} />
              <p style={{ fontSize: '0.75rem', color: 'var(--muted)', marginTop: 4 }}>
                {scoutCardPreview.activeVal} up (keep)
              </p>
              <button className="btn btn-secondary" style={{ marginTop: 6 }} onClick={() => handleFlipChoice(0)}>
                Keep {scoutCardPreview.activeVal}
              </button>
            </div>
            <div style={{ textAlign: 'center' }}>
              <Card a={scoutCardPreview.a} b={scoutCardPreview.b} flipped={!scoutCardPreview.flipped} />
              <p style={{ fontSize: '0.75rem', color: 'var(--muted)', marginTop: 4 }}>
                {scoutCardPreview.otherVal} up (flip)
              </p>
              <button className="btn btn-secondary" style={{ marginTop: 6 }} onClick={() => handleFlipChoice(1)}>
                Flip to {scoutCardPreview.otherVal}
              </button>
            </div>
          </div>
          <button className="btn btn-secondary" onClick={resetMode}>Cancel</button>
        </div>
      )}

      {/* Player's own hand */}
      {seat !== 'spectator' && (
        <Hand
          cards={myHand}
          selStart={selStart}
          selEnd={selEnd}
          isSelectionValid={isSelectionValid}
          mode={mode === 'scout-insert' ? 'scout-insert' : 'default'}
          scoutSide={scoutSide}
          scoutFlip={scoutFlip}
          onCardClick={handleCardClick}
          onInsert={handleInsert}
          isMyTurn={isMyTurn}
        />
      )}

      {/* Action panel */}
      {seat !== 'spectator' && (
        <ActionPanel
          isMyTurn={isMyTurn}
          mode={mode}
          canScout={canScout}
          hasSelection={hasSelection}
          isSelectionValid={isSelectionValid}
          onConfirmShow={handleConfirmShow}
          onClearSelection={handleClearSelection}
          onScoutMode={handleScoutMode}
          onCancel={resetMode}
          currentPlayer={state.current_player}
          mySeat={seat}
        />
      )}

      {/* History + Chat side-by-side */}
      <div className="bottom-panels">
        <HistoryPanel history={state?.history ?? []} mySeat={seat} />
        <div className="chat-panel">
          <h4 className="history-title">Chat</h4>
          <div className="chat-area">
            {messages.map((m, i) => (
              <div key={i} className="chat-msg">
                {m.sys
                  ? <span className="sys">{m.text}</span>
                  : <><span className="from">{m.from}:</span>{m.text}</>}
              </div>
            ))}
          </div>
          <form className="chat-input-row" onSubmit={handleSendChat}>
            <input
              value={chatText}
              onChange={e => setChatText(e.target.value)}
              placeholder="Chat…"
            />
            <button type="submit" className="btn btn-secondary" style={{ padding: '7px 14px' }}>
              Send
            </button>
          </form>
        </div>
      </div>

      {/* Game over overlay with score breakdown */}
      {gameOver && (() => {
        const pLabel = (p) => p === seat ? 'You' : `P${p === 'player_0' ? '1' : '2'}`
        const humanReason = (r) => r
          ? r.replace(/player_0/g, pLabel('player_0')).replace(/player_1/g, pLabel('player_1'))
          : ''
        const lastRR = roundResults[roundResults.length - 1]
        const endReason = lastRR?.reason ? humanReason(lastRR.reason) : ''

        return (
          <div className="end-overlay">
            <div className="end-card">
              <h2>{winner ? (winner === seat ? 'You Win!' : 'You Lose') : 'Game Over'}</h2>
              {endReason && <p className="end-reason">{endReason}</p>}
              <p className="subtitle">Final Scores</p>
              <div className="scores">
                {Object.entries(finalScores).map(([p, pts]) => (
                  <div key={p} className="score-box">
                    <span className="pts">{pts}</span>
                    <span className="name">{pLabel(p)}{p === winner ? ' \u{1F3C6}' : ''}</span>
                  </div>
                ))}
              </div>

              {/* Round-by-round breakdown */}
              {roundResults.length > 0 && (
                <div className="round-breakdown">
                  {roundResults.map((rr, ri) => {
                    const rrReason = humanReason(rr.reason)
                    return (
                      <div key={ri} className="round-block">
                        <h4>Round {rr.round + 1}</h4>
                        {rrReason && <p className="round-block-reason">{rrReason}</p>}
                        <table className="breakdown-table">
                          <thead>
                            <tr>
                              <th></th>
                              <th>Collected</th>
                              <th>Hand (-)</th>
                              <th>Chips (+)</th>
                              <th>Total</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(rr.breakdown).map(([p, bd]) => (
                              <tr key={p} className={p === winner ? 'winner-row' : ''}>
                                <td className="player-col">
                                  {pLabel(p)}
                                  {bd.exempt && <span className="exempt-badge">ended round</span>}
                                </td>
                                <td>+{bd.collected}</td>
                                <td>{bd.exempt ? '0 (exempt)' : bd.hand_penalty}</td>
                                <td>+{bd.unspent_chips}</td>
                                <td className="total-col">{bd.total >= 0 ? '+' : ''}{bd.total}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )
                  })}
                </div>
              )}

              <button className="btn btn-primary" onClick={onLeave}>Back to Lobby</button>
            </div>
          </div>
        )
      })()}
    </div>
  )
}
