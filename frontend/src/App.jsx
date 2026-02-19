import React, { useState } from 'react'
import LobbyScreen from './components/LobbyScreen.jsx'
import GameBoard from './components/GameBoard.jsx'

export default function App() {
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
      {session === null ? (
        <LobbyScreen onJoined={handleJoined} />
      ) : (
        <GameBoard
          gameId={session.gameId}
          seat={session.seat}
          onLeave={handleLeave}
        />
      )}
    </div>
  )
}
