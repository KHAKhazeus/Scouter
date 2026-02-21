/**
 * useGameSocket — manages the WebSocket connection to the Scout backend.
 *
 * Returns:
 *   state       - latest game state dict from server (or null)
 *   connected   - bool
 *   seat        - confirmed seat string ("player_0" | "player_1" | "spectator")
 *   sendAction  - fn(actionIndex: int)
 *   sendChat    - fn(text: string)
 *   messages    - chat/system messages array [{from, text, sys}]
 *   error       - latest error string or null
 */

import { useEffect, useRef, useState, useCallback } from 'react'

export function useGameSocket(gameId, requestedSeat) {
  const [state, setState] = useState(null)
  const [connected, setConnected] = useState(false)
  const [seat, setSeat] = useState(requestedSeat)
  const [messages, setMessages] = useState([])
  const [error, setError] = useState(null)
  const wsRef = useRef(null)

  const addMessage = useCallback((msg) => {
    setMessages(prev => [...prev.slice(-99), msg])
  }, [])

  useEffect(() => {
    if (!gameId) return

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const host = window.location.host
    const url = `${protocol}://${host}/game/${gameId}/ws?seat=${requestedSeat}`

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      setError(null)
      addMessage({ sys: true, text: `Connected to game ${gameId} as ${requestedSeat}.` })
    }

    ws.onmessage = (evt) => {
      let data
      try {
        data = JSON.parse(evt.data)
      } catch {
        return
      }

      switch (data.type) {
        case 'state':
          setState(data)
          break
        case 'joined':
          setSeat(data.seat)
          addMessage({ sys: true, text: `Joined as ${data.seat}. Available seats: ${data.available_seats.join(', ') || 'none'}` })
          break
        case 'error':
          setError(data.message)
          addMessage({ sys: true, text: `Error: ${data.message}` })
          break
        case 'chat':
          addMessage({ from: data.from, text: data.text })
          break
        case 'pong':
          break
        default:
          break
      }
    }

    ws.onclose = () => {
      setConnected(false)
      addMessage({ sys: true, text: 'Disconnected.' })
    }

    ws.onerror = () => {
      setError('WebSocket connection error.')
    }

    return () => {
      ws.close()
    }
  }, [gameId, requestedSeat]) // eslint-disable-line react-hooks/exhaustive-deps

  const sendAction = useCallback((actionIndex) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'action', action: actionIndex }))
    }
  }, [])

  const sendChat = useCallback((text) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'chat', text }))
    }
  }, [])

  return { state, connected, seat, sendAction, sendChat, messages, error }
}
