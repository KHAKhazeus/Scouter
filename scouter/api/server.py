"""FastAPI server: HTTP endpoints + WebSocket game sessions."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from scouter.api.game_manager import GameManager
from scouter.env.scout_env import AGENTS, ScoutEnv

app = FastAPI(title="Scout Game Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = GameManager()

# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@app.post("/game")
async def create_game() -> dict:
    """Create a new game session and return its ID."""
    session = manager.create_session()
    return {"game_id": session.game_id, "available_seats": AGENTS}


@app.get("/game/{game_id}/state")
async def get_state(game_id: str) -> dict:
    """Return the full public state of a game (for reconnect / spectators)."""
    session = manager.get_session(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return session.state_for_seat(None)


@app.get("/games")
async def list_games() -> dict:
    """List all active game sessions."""
    return {"games": manager.list_sessions()}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/game/{game_id}/ws")
async def game_ws(websocket: WebSocket, game_id: str, seat: str = "spectator") -> None:
    """WebSocket endpoint for a game session.

    Query params:
      seat: "player_0" | "player_1" | "spectator"  (default: "spectator")

    Client → Server messages:
      {"type": "action", "action": <int>}
      {"type": "set_orientation", "flip_hand": <bool>}   # pre-round flip (not yet impl)

    Server → Client messages (broadcast after each action):
      {"type": "state", ...}   — see GameSession.state_for_seat()
      {"type": "error", "message": <str>}
      {"type": "joined", "seat": <str>, "available_seats": [...]}
    """
    session = manager.get_session(game_id)
    if session is None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Game not found"})
        await websocket.close()
        return

    await websocket.accept()

    # Assign seat
    assigned_seat: str | None = None

    if seat in AGENTS:
        if session.seat_taken(seat):
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Seat {seat} is already taken. "
                    f"Available: {session.available_seats()}",
                }
            )
            await websocket.close()
            return
        session.players[seat] = websocket
        assigned_seat = seat
    else:
        # Spectator
        session.spectators.append(websocket)
        assigned_seat = None

    await websocket.send_json(
        {
            "type": "joined",
            "seat": assigned_seat or "spectator",
            "available_seats": session.available_seats(),
        }
    )

    # Start game when both seats are filled
    if not session.started and session.all_seats_filled():
        session.env.reset()
        session.started = True
        await session.broadcast()

    # If game already running (reconnect), send current state
    elif session.started:
        state = session.state_for_seat(assigned_seat)
        await websocket.send_json({"type": "state", **state})

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "action":
                if not session.started:
                    await session.send_error(websocket, "Game has not started yet.")
                    continue

                # Verify this is the acting player's turn
                current = session.env.agent_selection
                if assigned_seat != current:
                    await session.send_error(
                        websocket,
                        f"It is {current}'s turn, not yours ({assigned_seat}).",
                    )
                    continue

                action = data.get("action")
                if action is None or not isinstance(action, int):
                    await session.send_error(websocket, "Invalid action format.")
                    continue

                try:
                    session.env.step(action)
                except ValueError as exc:
                    await session.send_error(websocket, str(exc))
                    continue

                # Auto-process dead steps so game_over is reflected immediately
                while session.env.agents:
                    a = session.env.agent_selection
                    if session.env.terminations.get(a) or session.env.truncations.get(a):
                        session.env.step(None)
                    else:
                        break

                await session.broadcast()

            elif msg_type == "flip_hand":
                if not session.started:
                    await session.send_error(websocket, "Game has not started yet.")
                    continue
                if assigned_seat not in AGENTS:
                    await session.send_error(websocket, "Spectators cannot flip hands.")
                    continue
                ok = session.env.flip_player_hand(assigned_seat)
                if not ok:
                    await session.send_error(
                        websocket,
                        "Hand flip is only allowed before the first action of a round.",
                    )
                    continue
                await session.broadcast()

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "chat":
                # Simple chat relay
                text = str(data.get("text", ""))[:200]
                sender = assigned_seat or "spectator"
                for ws in list(session.players.values()) + list(session.spectators):
                    try:
                        await ws.send_json(
                            {"type": "chat", "from": sender, "text": text}
                        )
                    except Exception:
                        pass

    except WebSocketDisconnect:
        if assigned_seat and assigned_seat in session.players:
            del session.players[assigned_seat]
        elif websocket in session.spectators:
            session.spectators.remove(websocket)
