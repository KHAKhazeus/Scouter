from scouter.api.game_manager import GameSession
from scouter.env.scout_env import AGENTS


def test_agent_session_seat_accounting():
    s = GameSession(game_id="g1")
    s.agent_id = "agent_x"
    s.agent_seat = AGENTS[1]

    assert s.seat_taken(AGENTS[1])
    assert AGENTS[1] not in s.available_seats()
    assert not s.all_seats_filled()

    # Simulate human occupying the non-agent seat.
    s.players[AGENTS[0]] = object()  # dummy marker
    assert s.all_seats_filled()
