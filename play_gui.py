"""Run a saved PPO LunarLander-v3 model with Gymnasium's human renderer."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import sleep

import gymnasium as gym
from stable_baselines3 import PPO


ENV_ID = "LunarLander-v3"
DEFAULT_MODEL_PATH = Path("models") / "ppo_lunarlander.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch PPO play LunarLander-v3.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved SB3 model zip file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="How many GUI episodes to play.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=200,
        help="Base random seed for GUI episodes.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional pause after each step if rendering is too fast.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1.")

    # The user asked specifically for render_mode="human"; this opens a pygame
    # window so the policy can be checked visually.
    env = gym.make(ENV_ID, render_mode="human")
    model = PPO.load(args.model_path, env=env)

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)

            if args.sleep > 0:
                sleep(args.sleep)

        print(f"[gui] episode={episode:02d} reward={episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
