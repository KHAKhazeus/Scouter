"""LangGraph agent stub for Scout.

Provides a ready-to-use agent graph with clearly defined hook points:

    [format_obs] → [call_llm] → [parse_action] → [validate_action]
                         ↑ retry on invalid               |
                         └────────────────────────────────┘

Usage
-----
1. Install langgraph: ``pip install langgraph``
2. Provide an ``llm_call_fn`` — an async function ``prompt -> str``.
3. Call ``build_agent(llm_call_fn)`` to get a compiled LangGraph ``CompiledGraph``.
4. Use ``run_agent(graph, env)`` to play a full game.

The LLM call node is intentionally a swappable async function so teams can
plug in OpenAI, vLLM (via AsyncLLMEngine), Anthropic, etc.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from scouter.env.scout_env import AGENTS, ScoutEnv
from scouter.rl.action_parser import parse_action
from scouter.rl.obs_formatter import format_observation

AsyncLLMFn = Callable[[str], "asyncio.Coroutine[Any, Any, str]"]

# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    """State passed through the LangGraph nodes."""

    prompt: str = ""
    completion: str = ""
    action: int = -1
    attempt: int = 0
    max_attempts: int = 3
    hand_size: int = 0
    action_mask: list[int] = field(default_factory=list)
    valid: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def node_format_obs(state: AgentState, obs: dict, agent: str, active_owner: str | None) -> AgentState:
    """Format the current observation into an LLM prompt."""
    state.prompt = format_observation(obs, agent, active_owner)
    state.hand_size = int(obs["hand_size"])
    state.action_mask = obs["action_mask"].tolist()
    state.attempt = 0
    return state


async def node_call_llm(state: AgentState, llm_fn: AsyncLLMFn) -> AgentState:
    """Call the LLM asynchronously and store the completion."""
    state.completion = await llm_fn(state.prompt)
    return state


def node_parse_action(state: AgentState) -> AgentState:
    """Parse the LLM completion into a flat action integer."""
    state.action = parse_action(state.completion, state.hand_size)
    return state


def node_validate_action(state: AgentState) -> AgentState:
    """Check the parsed action is legal."""
    a = state.action
    if (
        a == -1
        or a < 0
        or a >= len(state.action_mask)
        or not state.action_mask[a]
    ):
        state.valid = False
        state.attempt += 1
        if state.attempt < state.max_attempts:
            state.error = (
                f"Action {a!r} is invalid. "
                "Re-read the legal actions list and try again.\n"
            )
            state.prompt = state.prompt + state.error
        else:
            # Give up; caller will fall back to random
            state.valid = False
    else:
        state.valid = True
        state.error = ""
    return state


# ---------------------------------------------------------------------------
# Graph builder (requires langgraph)
# ---------------------------------------------------------------------------


def build_agent(
    llm_fn: AsyncLLMFn,
    max_attempts: int = 3,
):
    """Build and compile a LangGraph agent for Scout.

    Parameters
    ----------
    llm_fn:
        Async function ``prompt -> completion``.
    max_attempts:
        Maximum re-prompt attempts on invalid actions.

    Returns
    -------
    A compiled LangGraph ``CompiledGraph`` object.
    Raises ``ImportError`` if ``langgraph`` is not installed.
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError as exc:
        raise ImportError(
            "langgraph is required for build_agent(). "
            "Install with: pip install langgraph"
        ) from exc

    # We use a simple dict-based state graph
    from typing import TypedDict

    class GraphState(TypedDict):
        prompt: str
        completion: str
        action: int
        attempt: int
        hand_size: int
        action_mask: list
        valid: bool
        error: str

    async def _call_llm(state: GraphState) -> GraphState:
        completion = await llm_fn(state["prompt"])
        return {**state, "completion": completion}

    def _parse(state: GraphState) -> GraphState:
        action = parse_action(state["completion"], state["hand_size"])
        return {**state, "action": action}

    def _validate(state: GraphState) -> GraphState:
        a = state["action"]
        mask = state["action_mask"]
        attempt = state["attempt"]
        if a == -1 or a >= len(mask) or not mask[a]:
            attempt += 1
            valid = False
            error = (
                f"[Attempt {attempt} invalid — action={a}] "
                "Check legal actions and retry.\n"
            )
            prompt = state["prompt"] + error
        else:
            valid = True
            error = ""
            prompt = state["prompt"]
        return {
            **state,
            "valid": valid,
            "error": error,
            "attempt": attempt,
            "prompt": prompt,
        }

    def _should_retry(state: GraphState) -> str:
        if state["valid"]:
            return END
        if state["attempt"] >= max_attempts:
            return END
        return "call_llm"

    graph = StateGraph(GraphState)
    graph.add_node("call_llm", _call_llm)
    graph.add_node("parse", _parse)
    graph.add_node("validate", _validate)

    graph.set_entry_point("call_llm")
    graph.add_edge("call_llm", "parse")
    graph.add_edge("parse", "validate")
    graph.add_conditional_edges("validate", _should_retry)

    return graph.compile()


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------


async def run_agent(
    llm_fn: AsyncLLMFn,
    env: ScoutEnv | None = None,
    seed: int | None = None,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Play one complete 2-round Scout game using the LangGraph agent.

    Parameters
    ----------
    llm_fn:
        Async LLM call function.
    env:
        Optional pre-created ScoutEnv. If None, a fresh one is created.
    seed:
        Random seed.
    max_attempts:
        Max retry attempts per action.

    Returns
    -------
    dict with "final_scores", "winner", "trajectory" (list of step dicts).
    """
    graph = build_agent(llm_fn, max_attempts=max_attempts)

    if env is None:
        env = ScoutEnv()
    env.reset(seed=seed)

    trajectory = []

    while env.agents:
        agent = env.agent_selection
        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            env.step(None)
            continue

        active_owner_idx = int(obs["active_set_owner"])
        active_owner = AGENTS[active_owner_idx] if env._active_set else None

        init_state = {
            "prompt": format_observation(obs, agent, active_owner),
            "completion": "",
            "action": -1,
            "attempt": 0,
            "hand_size": int(obs["hand_size"]),
            "action_mask": obs["action_mask"].tolist(),
            "valid": False,
            "error": "",
        }

        result = await graph.ainvoke(init_state)
        action = result["action"]

        # Fallback to random if agent gave up
        if action == -1 or not obs["action_mask"][action]:
            valid = np.where(obs["action_mask"])[0]
            action = int(np.random.choice(valid)) if len(valid) > 0 else 0

        env.step(action)

        trajectory.append(
            {
                "agent": agent,
                "action": action,
                "prompt": result["prompt"],
                "completion": result["completion"],
                "attempts": result["attempt"] + 1,
            }
        )

    final_info = env.infos.get(AGENTS[0], {})
    return {
        "final_scores": final_info.get("final_scores", {}),
        "winner": final_info.get("winner"),
        "trajectory": trajectory,
    }
