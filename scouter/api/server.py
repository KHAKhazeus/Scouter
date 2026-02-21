"""FastAPI server: HTTP endpoints + WebSocket game sessions."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from scouter.api.game_manager import GameManager
from scouter.api.rl_monitor import router as rl_monitor_router
from scouter.env.scout_env import AGENTS, ScoutEnv
from scouter.rl.agent_pool import AgentPool
from scouter.rl.rllib_wrapper import flatten_obs_dict

app = FastAPI(title="Scout Game Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = GameManager()
app.include_router(rl_monitor_router)
agent_pool = AgentPool(Path("deployed_agents"))
# Use uvicorn logger so inference traces are always visible in standard server logs.
logger = logging.getLogger("uvicorn.error")

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@app.post("/game")
async def create_game() -> dict:
    """Create a new game session and return its ID."""
    session = manager.create_session()
    return {"game_id": session.game_id, "available_seats": AGENTS}


@app.get("/agents/deployed")
async def list_deployed_agents() -> dict:
    return {"agents": agent_pool.list_deployed()}


@app.post("/agents/load")
async def load_agent(payload: dict) -> dict:
    agent_id = str(payload.get("agent_id", "")).strip()
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id is required")

    available = {a["agent_id"] for a in agent_pool.list_deployed()}
    if agent_id not in available:
        raise HTTPException(status_code=404, detail=f"Unknown agent_id: {agent_id}")

    started = time.monotonic()
    try:
        info = agent_pool.preload(agent_id)
    except Exception as exc:
        logger.exception("[agent-load] failed agent_id=%s error=%r", agent_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to load agent {agent_id}") from exc

    elapsed = time.monotonic() - started
    logger.info(
        "[agent-load] ok agent_id=%s already_loaded=%s device=%s load_seconds=%.3f",
        info["agent_id"],
        info["already_loaded"],
        info["device"],
        elapsed,
    )
    return {"ok": True, "load_seconds": elapsed, **info}


@app.post("/agent-game")
async def create_agent_game(payload: dict) -> dict:
    agent_id = str(payload.get("agent_id", "")).strip()
    human_seat = str(payload.get("human_seat", "player_0"))
    num_rounds = int(payload.get("num_rounds", 2))
    reward_mode = str(payload.get("reward_mode", "raw"))

    if human_seat not in AGENTS:
        raise HTTPException(status_code=400, detail="human_seat must be player_0 or player_1")
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id is required")

    available = {a["agent_id"] for a in agent_pool.list_deployed()}
    if agent_id not in available:
        raise HTTPException(status_code=404, detail=f"Unknown agent_id: {agent_id}")

    session = manager.create_session()
    session.env = ScoutEnv(num_rounds=num_rounds, reward_mode=reward_mode)
    session.agent_id = agent_id
    session.agent_seat = AGENTS[1 - AGENTS.index(human_seat)]
    return {
        "game_id": session.game_id,
        "human_seat": human_seat,
        "agent_seat": session.agent_seat,
        "agent_id": session.agent_id,
    }


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
      {"type": "action", "action": <int>} for orientation/show/scout

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
            "agent_session": session.agent_id is not None,
            "agent_id": session.agent_id,
            "agent_seat": session.agent_seat,
        }
    )

    # Start game when both seats are filled
    if not session.started and session.all_seats_filled():
        session.env.reset()
        session.started = True
        await _auto_play_agent(session)
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

                await _auto_play_agent(session)
                await session.broadcast()

            elif msg_type == "flip_hand":
                await session.send_error(
                    websocket,
                    "flip_hand is deprecated. Use orientation action from action_mask.",
                )

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


async def _auto_play_agent(session) -> None:
    """Advance turns while it is the configured agent's turn."""
    if not session.agent_id or not session.agent_seat:
        return

    while session.env.agents and session.env.agent_selection == session.agent_seat:
        agent = session.env.agent_selection
        if session.env.terminations.get(agent) or session.env.truncations.get(agent):
            session.env.step(None)
            continue

        raw_obs = session.env.observe(agent)
        obs = flatten_obs_dict(raw_obs)
        source = "model"
        meta: dict | None = None
        try:
            action, meta = agent_pool.compute_action(
                agent_id=session.agent_id,
                obs=obs,
                return_info=True,
            )
            source = str(meta.get("source", "model")) if meta else "model"
        except Exception as exc:
            # Fallback to the first valid action if model inference fails.
            mask = obs["action_mask"]
            valid = [i for i, m in enumerate(mask) if m > 0]
            action = int(valid[0]) if valid else 0
            source = "fallback_first_valid"
            meta = {"error": repr(exc)}

        raw_id = meta.get("raw_argmax_action") if isinstance(meta, dict) else None
        device = meta.get("device") if isinstance(meta, dict) else None
        turn_idx = len(session.env._history) if hasattr(session.env, "_history") else -1
        msg = (
            f"[agent-infer] agent_id={session.agent_id} seat={agent} "
            f"action={action} raw_argmax={raw_id} source={source} device={device} turn={turn_idx}"
        )
        logger.info(msg)
        await _broadcast_system_chat(session, msg)
        session.env.step(action)

        while session.env.agents:
            a = session.env.agent_selection
            if session.env.terminations.get(a) or session.env.truncations.get(a):
                session.env.step(None)
            else:
                break


async def _broadcast_system_chat(session, text: str) -> None:
    payload = {"type": "chat", "from": "system", "text": text[:240]}
    for ws in list(session.players.values()) + list(session.spectators):
        try:
            await ws.send_json(payload)
        except Exception:
            pass


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    async def frontend_index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/rl-dashboard", include_in_schema=False)
    async def frontend_dashboard() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")
