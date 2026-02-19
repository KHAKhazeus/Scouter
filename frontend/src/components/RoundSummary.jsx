import React from 'react'

function pLabel(p, mySeat) {
  const num = p === 'player_0' ? '1' : '2'
  return p === mySeat ? 'You' : `P${num}`
}

function humanReason(reason, mySeat) {
  if (!reason) return ''
  return reason
    .replace(/player_0/g, pLabel('player_0', mySeat))
    .replace(/player_1/g, pLabel('player_1', mySeat))
}

export default function RoundSummary({ roundResult, mySeat, onDismiss }) {
  if (!roundResult) return null

  const rr = roundResult
  const reason = humanReason(rr.reason, mySeat)

  return (
    <div className="round-summary-banner">
      <div className="round-summary-header">
        <h3>Round {rr.round + 1} Complete</h3>
        {reason && <p className="round-summary-reason">{reason}</p>}
      </div>

      <table className="breakdown-table">
        <thead>
          <tr>
            <th></th>
            <th>Collected</th>
            <th>Hand (-)</th>
            <th>Chips (+)</th>
            <th>Round Total</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(rr.breakdown).map(([p, bd]) => (
            <tr key={p} className={bd.exempt ? 'exempt-row' : ''}>
              <td className="player-col">
                {pLabel(p, mySeat)}
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

      {onDismiss && (
        <button className="btn btn-primary" onClick={onDismiss}>
          Continue to Round {rr.round + 2}
        </button>
      )}
    </div>
  )
}
