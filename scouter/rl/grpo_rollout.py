"""Trajectory collector for TRL GRPO training.

Generates (prompt, completion, reward) tuples by running full or partial
Scout game episodes with a provided policy function.

Compatible with TRL GRPOTrainer's expected dataset format:
    [{"prompt": str, "completion": str, "reward": float}, ...]

Supports both synchronous and async (vLLM AsyncLLMEngine) policy functions.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from scouter.env.scout_env import AGENTS, ScoutEnv
from scouter.rl.action_parser import parse_action
from scouter.rl.obs_formatter import format_observation

PolicyFn = Callable[[str], str]
AsyncPolicyFn = Callable[[str], "asyncio.Coroutine[Any, Any, str]"]


def collect_grpo_dataset(
    num_episodes: int,
    policy_fn: PolicyFn,
    reward_fn: Callable[[dict], float] | None = None,
    seed: int | None = None,
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    """Collect GRPO-format trajectories by running complete game episodes.

    Parameters
    ----------
    num_episodes:
        Number of complete 2-round games to run.
    policy_fn:
        Callable ``prompt -> completion`` (synchronous). Use this for local or
        synchronous LLM inference (e.g. OpenAI client, HuggingFace pipeline).
    reward_fn:
        Optional custom reward function. Receives a dict with keys:
        ``{"agent", "action", "obs", "next_obs", "done", "round_scores",
           "cumulative_scores"}``.
        Defaults to the per-step delta score (0 during a round, final round
        score on round end).
    seed:
        Optional random seed for reproducibility.
    max_retries:
        Number of times to re-prompt the LLM if it produces an invalid action
        before falling back to a random valid action.

    Returns
    -------
    list[dict]
        Each entry has keys ``"prompt"``, ``"completion"``, ``"reward"``.
    """
    import numpy as np

    records: list[dict[str, Any]] = []
    env = ScoutEnv()

    for ep in range(num_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        env.reset(seed=ep_seed)

        while env.agents:
            agent = env.agent_selection
            obs, reward, terminated, truncated, info = env.last(observe=False)

            if terminated or truncated:
                env.step(None)
                continue

            obs = env.observe(agent)

            # Format the observation into a prompt
            active_owner_idx = int(obs["active_set_owner"])
            active_owner = AGENTS[active_owner_idx] if env._active_set else None
            prompt = format_observation(obs, agent, active_owner)

            # Query policy with retries on invalid parse
            completion = ""
            action = -1
            hand_size = int(obs["hand_size"])
            for attempt in range(max_retries):
                completion = policy_fn(prompt)
                action = parse_action(completion, hand_size)
                if action != -1 and obs["action_mask"][action]:
                    break
                # If invalid, append error hint and retry
                prompt = prompt + f"\n[Attempt {attempt+1} was invalid. Try again.]\n"
                action = -1

            # Fall back to random valid action if LLM still failed
            if action == -1 or not obs["action_mask"][action]:
                valid = np.where(obs["action_mask"])[0]
                action = int(np.random.choice(valid)) if len(valid) > 0 else 0

            env.step(action)

            # Collect reward: use last(observe=False) to get cumulative reward
            # without triggering observe() which may crash if agents list changed.
            _, step_reward, step_done, _, step_info = env.last(observe=False)
            if reward_fn is not None:
                r = reward_fn(
                    {
                        "agent": agent,
                        "action": action,
                        "obs": obs,
                        "done": step_done,
                        "round_scores": step_info.get("round_scores"),
                        "cumulative_scores": step_info.get("final_scores"),
                    }
                )
            else:
                r = float(step_reward)

            records.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "reward": r,
                }
            )

    return records


async def collect_grpo_dataset_async(
    num_episodes: int,
    policy_fn: AsyncPolicyFn,
    reward_fn: Callable[[dict], float] | None = None,
    seed: int | None = None,
    max_retries: int = 3,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Async variant of collect_grpo_dataset for vLLM AsyncLLMEngine.

    Batches ``batch_size`` prompt requests per LLM call for throughput.
    Each batch is a list of concurrent prompts, all awaited together.
    """
    import numpy as np

    records: list[dict[str, Any]] = []
    env = ScoutEnv()
    pending_prompts: list[tuple[str, str, dict, int]] = []  # (agent, prompt, obs, ep)

    async def flush_batch(
        batch: list[tuple[str, str, dict, int, int]]
    ) -> list[tuple[str, int]]:
        """Run a batch of async policy calls, return (completion, action) pairs."""
        results = await asyncio.gather(
            *[policy_fn(item[1]) for item in batch], return_exceptions=True
        )
        out = []
        for item, result in zip(batch, results):
            agent, prompt, obs, ep, _ = item
            hand_size = int(obs["hand_size"])
            if isinstance(result, Exception):
                completion = ""
            else:
                completion = str(result)
            action = parse_action(completion, hand_size)
            if action == -1 or not obs["action_mask"][action]:
                valid = np.where(obs["action_mask"])[0]
                action = int(np.random.choice(valid)) if len(valid) > 0 else 0
            out.append((completion, action))
        return out

    # Sequential episodes for correctness; batching is within a single step
    for ep in range(num_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        env.reset(seed=ep_seed)

        while env.agents:
            agent = env.agent_selection
            _, reward, terminated, truncated, info = env.last(observe=False)

            if terminated or truncated:
                env.step(None)
                continue

            obs = env.observe(agent)
            active_owner_idx = int(obs["active_set_owner"])
            active_owner = AGENTS[active_owner_idx] if env._active_set else None
            prompt = format_observation(obs, agent, active_owner)

            # Single async call
            results = await flush_batch([(agent, prompt, obs, ep, 0)])
            completion, action = results[0]

            env.step(action)

            _, step_reward, step_done, _, step_info = env.last(observe=False)
            if reward_fn is not None:
                r = reward_fn(
                    {
                        "agent": agent,
                        "action": action,
                        "obs": obs,
                        "done": step_done,
                        "round_scores": step_info.get("round_scores"),
                        "cumulative_scores": step_info.get("final_scores"),
                    }
                )
            else:
                r = float(step_reward)

            records.append(
                {"prompt": prompt, "completion": completion, "reward": r}
            )

    return records
