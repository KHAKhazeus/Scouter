import React, { useState } from 'react'
import LobbyScreen from './components/LobbyScreen.jsx'
import GameBoard from './components/GameBoard.jsx'
import RlDashboard from './components/RlDashboard.jsx'

export default function App() {
  const isDashboard = window.location.pathname.startsWith('/rl-dashboard')
  const [session, setSession] = useState(null)
  // session: { gameId, seat }

  function handleJoined(gameId, seat) {
    setSession({ gameId, seat })
  }

  function handleLeave() {
    setSession(null)
  }

  return (
    <div className="app-root">
      {isDashboard ? (
        <RlDashboard />
      ) : (
        <>
      {session === null ? (
        <LobbyScreen onJoined={handleJoined} />
      ) : (
        <GameBoard
          gameId={session.gameId}
          seat={session.seat}
          onLeave={handleLeave}
        />
      )}
        </>
      )}
    </div>
  )
}
