"""Train PPO self-play on Scout via PettingZooEnv + action masking."""

from __future__ import annotations

import argparse
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from scouter.rl.rllib_wrapper import ScoutActionMaskRLModule, register_scout_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO self-play for Scout.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--reward-mode", choices=["raw", "score_diff"], default="score_diff")
    parser.add_argument("--num-env-runners", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=4000)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--num-learners", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PPOConfig:
    env_name = "scout_v0"
    register_scout_env(env_name)

    config = (
        PPOConfig()
        .framework("torch")
        .environment(
            env=env_name,
            env_config={"num_rounds": args.num_rounds, "reward_mode": args.reward_mode},
            disable_env_checking=True,
        )
        .env_runners(num_env_runners=args.num_env_runners)
        .learners(
            num_learners=args.num_learners,
            num_gpus_per_learner=args.num_gpus,
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda *_a, **_k: "shared_policy",
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "shared_policy": RLModuleSpec(
                        module_class=ScoutActionMaskRLModule,
                        model_config={
                            "head_fcnet_hiddens": [256, 256],
                            "head_fcnet_activation": "relu",
                        },
                    ),
                }
            )
        )
        .training(
            train_batch_size_per_learner=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            vf_loss_coeff=0.5,
        )
    )

    if args.seed is not None:
        config = config.debugging(seed=args.seed)
    return config


def main() -> None:
    args = parse_args()
    algo = build_config(args).build_algo()

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i in range(args.iterations):
            result = algo.train()
            reward_mean = result.get("env_runners", {}).get(
                "episode_return_mean",
                result.get("episode_reward_mean"),
            )
            per_agent_returns = result.get("env_runners", {}).get(
                "agent_episode_returns_mean", {}
            )
            print(
                f"iter={i + 1} reward_mean={reward_mean} "
                f"timesteps={result.get('num_env_steps_sampled_lifetime')} "
                f"per_agent_returns={per_agent_returns}"
            )
    finally:
        save_result = algo.save(str(checkpoint_dir)) if checkpoint_dir else algo.save()
        if hasattr(save_result, "checkpoint"):
            checkpoint_path = save_result.checkpoint.path
        elif hasattr(save_result, "path"):
            checkpoint_path = save_result.path
        else:
            checkpoint_path = str(save_result)
        print(f"checkpoint={checkpoint_path}")
        algo.stop()


if __name__ == "__main__":
    main()
