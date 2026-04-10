"""Train a PPO agent on Gymnasium LunarLander-v3.

This file is intentionally small: it creates the environment, trains a PPO
model, prints progress through Stable-Baselines3 logs, and optionally saves the
trained model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


ENV_ID = "LunarLander-v3"
DEFAULT_MODEL_PATH = Path("models") / "ppo_lunarlander"


def model_zip_path(path: Path) -> Path:
    """Return the filename Stable-Baselines3 will write for a model save."""

    return path if path.suffix == ".zip" else path.with_suffix(".zip")


class ProgressCallback(BaseCallback):
    """Print a compact progress line every fixed number of environment steps."""

    def __init__(self, print_freq: int = 10_000) -> None:
        super().__init__()
        self.print_freq = print_freq
        self._last_print_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print_step >= self.print_freq:
            self._last_print_step = self.num_timesteps
            print(f"[train] timesteps={self.num_timesteps}")
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v3.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps. Try 100000, 300000, or 1000000.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model. .zip is added by SB3 if omitted.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run a quick training smoke test without writing a model file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible first experiments.",
    )
    return parser.parse_args()


def make_env(seed: int) -> Monitor:
    """Create the training environment with Monitor so episode rewards are logged."""

    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    return Monitor(env)


def main() -> None:
    args = parse_args()
    if args.timesteps < 1:
        raise ValueError("--timesteps must be at least 1.")

    env = make_env(args.seed)

    # PPO is a stable beginner-friendly default in Stable-Baselines3. It works
    # well with LunarLander-v3's discrete action space while needing few knobs.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log="runs",
    )

    print(f"[train] environment={ENV_ID}")
    print(f"[train] algorithm=PPO")
    print(f"[train] total_timesteps={args.timesteps}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=ProgressCallback(print_freq=10_000),
        progress_bar=True,
    )

    if args.no_save:
        print("[train] --no-save was set; model was not saved.")
    else:
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.model_path)
        print(f"[train] model saved to {model_zip_path(args.model_path)}")

    env.close()


if __name__ == "__main__":
    main()
