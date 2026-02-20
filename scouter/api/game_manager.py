"""Game session management: seat assignment, WebSocket registry, state broadcast."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from scouter.env.scout_env import AGENTS, ScoutEnv


@dataclass
class GameSession:
    """Holds one active Scout game and its connected WebSocket clients."""

    game_id: str
    env: ScoutEnv = field(default_factory=ScoutEnv)
    # seat → WebSocket mapping; "spectator" key may have multiple, so we use a list
    players: dict[str, WebSocket] = field(default_factory=dict)
    spectators: list[WebSocket] = field(default_factory=list)
    started: bool = False
    agent_id: str | None = None
    agent_seat: str | None = None

    def seat_taken(self, seat: str) -> bool:
        return seat in self.players or seat == self.agent_seat

    def available_seats(self) -> list[str]:
        return [s for s in AGENTS if not self.seat_taken(s)]

    def all_seats_filled(self) -> bool:
        if self.agent_seat in AGENTS:
            human_seat = AGENTS[1 - AGENTS.index(self.agent_seat)]
            return human_seat in self.players
        return all(a in self.players for a in AGENTS)

    def player_for_ws(self, ws: WebSocket) -> str | None:
        for seat, w in self.players.items():
            if w is ws:
                return seat
        return None

    def state_for_seat(self, seat: str | None) -> dict[str, Any]:
        """Return the game state, hiding the opponent's hand unless spectator."""
        rich = self.env.get_rich_state()
        if seat is None:
            # Spectator sees all hands
            return rich

        # Hide opponent's cards
        opp = AGENTS[1 - AGENTS.index(seat)] if seat in AGENTS else None
        hands = dict(rich["hands"])
        if opp:
            hands[opp] = None  # hidden
        return {
            **rich,
            "hands": hands,
            "my_seat": seat,
            "agent_session": self.agent_id is not None,
            "agent_seat": self.agent_seat,
            "agent_id": self.agent_id,
        }

    async def broadcast(self) -> None:
        """Send the current state to all connected clients."""
        tasks = []

        for seat, ws in list(self.players.items()):
            msg = {"type": "state", **self.state_for_seat(seat)}
            tasks.append(_safe_send(ws, msg))

        for ws in list(self.spectators):
            msg = {"type": "state", **self.state_for_seat(None)}
            tasks.append(_safe_send(ws, msg))

        if tasks:
            await asyncio.gather(*tasks)

    async def send_error(self, ws: WebSocket, message: str) -> None:
        await _safe_send(ws, {"type": "error", "message": message})


async def _safe_send(ws: WebSocket, data: dict) -> None:
    try:
        await ws.send_json(data)
    except Exception:
        pass


class GameManager:
    """Registry of all active game sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, GameSession] = {}

    def create_session(self) -> GameSession:
        game_id = str(uuid.uuid4())[:8]
        session = GameSession(game_id=game_id)
        self._sessions[game_id] = session
        return session

    def get_session(self, game_id: str) -> GameSession | None:
        return self._sessions.get(game_id)

    def remove_session(self, game_id: str) -> None:
        self._sessions.pop(game_id, None)

    def list_sessions(self) -> list[dict]:
        return [
            {
                "game_id": s.game_id,
                "started": s.started,
                "available_seats": s.available_seats(),
                "spectator_count": len(s.spectators),
                "agent_session": s.agent_id is not None,
                "agent_id": s.agent_id,
                "agent_seat": s.agent_seat,
            }
            for s in self._sessions.values()
        ]
