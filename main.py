"""CLI entry points for the Scout game server and demo."""

from __future__ import annotations

import argparse
import random
import sys


def cmd_serve(args):
    """Start the FastAPI WebSocket server."""
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: uv add 'uvicorn[standard]'")
        sys.exit(1)

    uvicorn.run(
        "scouter.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


def cmd_demo(args):
    """Play a complete random game and print the outcome."""
    import numpy as np

    from scouter.env.game_logic import decode_scout, decode_show, is_show_action
    from scouter.env.scout_env import AGENTS, ScoutEnv

    env = ScoutEnv(render_mode="ansi")
    seed = args.seed if args.seed is not None else random.randint(0, 9999)
    print(f"=== Scout Demo (seed={seed}) ===\n")
    env.reset(seed=seed)

    prev_round = env._round
    step = 0
    while env.agents:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
        else:
            obs = env.observe(agent)
            mask = obs["action_mask"]
            valid = np.where(mask)[0]
            action = int(np.random.choice(valid))

            hand = env._hands[agent]
            hand_vals = [c.value for c in hand]
            beaten_count = 0
            if is_show_action(action):
                s, e = decode_show(action)
                shown = hand_vals[s : e + 1]
                active_before = [c.value for c in env._active_set]
                if env._active_set and env._active_owner is not None:
                    beaten_count = len(env._active_set)
                print(f"[Step {step+1}] {agent} SHOW cards at [{s}..{e}] = {shown}  "
                      f"(active was {active_before})")
            else:
                side, flip, ins = decode_scout(action)
                side_label = "left" if side == 0 else "right"
                active_vals = [c.value for c in env._active_set]
                card_val = active_vals[0] if side == 0 else active_vals[-1]
                print(f"[Step {step+1}] {agent} SCOUT {side_label} (val={card_val}"
                      f"{', flip' if flip else ''}) insert@{ins}  "
                      f"chips: {env._chips[agent]}→{env._chips[agent]-1}")

            env.step(action)

            if beaten_count > 0:
                print(f"         → {agent} collected {beaten_count} card(s) from beaten set")

            if env._round != prev_round or not env.agents:
                rr_list = env._round_results
                if rr_list and rr_list[-1]["round"] == prev_round:
                    rr = rr_list[-1]
                    print(f"\n  ── Round {rr['round']+1} End ──  (ender: {rr['round_ender']})")
                    for a in AGENTS:
                        bd = rr["breakdown"][a]
                        exempt_str = " (exempt)" if bd["exempt"] else ""
                        print(f"    {a}: collected={bd['collected']}, "
                              f"hand_penalty={bd['hand_penalty']}{exempt_str}, "
                              f"chips_bonus=+{bd['unspent_chips']}, "
                              f"total={bd['total']}")
                    print()
                prev_round = env._round
        step += 1

    print("=== Game Over ===")
    info = env.final_info
    scores = info.get("final_scores", {})
    winner = info.get("winner")
    for a, s in scores.items():
        tag = " <-- WINNER" if a == winner else ""
        print(f"  {a}: {s} pts{tag}")
    print(f"\nCompleted in {step} steps.")


def cmd_test(args):
    """Run the test suite via pytest."""
    try:
        import pytest
    except ImportError:
        print("pytest not installed. Run: uv add --dev pytest")
        sys.exit(1)
    sys.exit(pytest.main(["tests/", "-v"] + (args.extra or [])))


def main():
    parser = argparse.ArgumentParser(
        description="Scout card game — server, demo, and test runner."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = sub.add_parser("serve", help="Start the FastAPI WebSocket server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    # demo
    p_demo = sub.add_parser("demo", help="Play a random game and print results")
    p_demo.add_argument("--seed", type=int, default=None)
    p_demo.set_defaults(func=cmd_demo)

    # test
    p_test = sub.add_parser("test", help="Run the test suite")
    p_test.add_argument("extra", nargs="*", help="Extra args passed to pytest")
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
