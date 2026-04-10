"""Evaluate a saved PPO model on LunarLander-v3 without opening a GUI window."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


ENV_ID = "LunarLander-v3"
DEFAULT_MODEL_PATH = Path("models") / "ppo_lunarlander.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO on LunarLander-v3.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved SB3 model zip file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes. Default prints a 10 episode average.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Base random seed for evaluation episodes.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=200.0,
        help="Episode reward at or above this value is counted as a success.",
    )
    parser.add_argument(
        "--crash-threshold",
        type=float,
        default=-100.0,
        help="Episode reward at or below this value is counted as a crash.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1.")

    env = gym.make(ENV_ID)
    model = PPO.load(args.model_path, env=env)

    rewards: list[float] = []
    successes = 0
    crashes = 0

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)

        rewards.append(episode_reward)

        # LunarLander does not expose a universal "landed/crashed" flag through
        # info, so these counts use common reward thresholds as a simple signal.
        if episode_reward >= args.success_threshold:
            successes += 1
            result = "success"
        elif episode_reward <= args.crash_threshold:
            crashes += 1
            result = "crash"
        else:
            result = "other"

        print(
            f"[eval] episode={episode:02d} "
            f"reward={episode_reward:8.2f} result={result}"
        )

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))

    print(f"[eval] episodes={args.episodes}")
    print(f"[eval] mean_reward={mean_reward:.2f} std_reward={std_reward:.2f}")
    print(f"[eval] successes={successes} crashes={crashes}")

    env.close()


if __name__ == "__main__":
    main()
