"""Train PPO while occasionally showing the current policy in a GUI window.

The training environment stays non-rendered for speed. For the first N
completed training episodes, a separate LunarLander-v3 environment with
render_mode="human" runs one demonstration episode using the current model.
No model file is saved.
"""

from __future__ import annotations

import argparse
from collections import deque

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


ENV_ID = "LunarLander-v3"


class VisualEpisodeCallback(BaseCallback):
    """Print rewards every episode and occasionally render the current policy."""

    def __init__(
        self,
        render_first_episodes: int = 10,
        render_every_episodes: int = 0,
        mean_window: int = 100,
        demo_seed: int = 10_000,
    ) -> None:
        super().__init__()
        self.render_first_episodes = render_first_episodes
        self.render_every_episodes = render_every_episodes
        self.demo_seed = demo_seed
        self.episode_count = 0
        self.recent_rewards: deque[float] = deque(maxlen=mean_window)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode_info = info.get("episode")
            if episode_info is None:
                continue

            self.episode_count += 1
            episode_reward = float(episode_info["r"])
            self.recent_rewards.append(episode_reward)
            mean_reward = float(np.mean(self.recent_rewards))

            print(
                f"[train_visual] episode={self.episode_count:04d} "
                f"reward={episode_reward:8.2f} "
                f"mean_last_{len(self.recent_rewards):03d}={mean_reward:8.2f} "
                f"timesteps={self.num_timesteps}"
            )

            should_render_first = self.episode_count <= self.render_first_episodes
            should_render_periodic = (
                self.render_every_episodes > 0
                and self.episode_count % self.render_every_episodes == 0
            )

            if should_render_first or should_render_periodic:
                self._run_gui_demo()

        return True

    def _run_gui_demo(self) -> None:
        """Open a human-rendered environment and show one current-policy episode."""

        print(
            f"[train_visual] GUI demo at episode {self.episode_count} "
            f"(timesteps={self.num_timesteps})"
        )

        demo_env = gym.make(ENV_ID, render_mode="human")
        obs, _ = demo_env.reset(seed=self.demo_seed + self.episode_count)
        terminated = False
        truncated = False
        demo_reward = 0.0

        while not (terminated or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = demo_env.step(action)
            demo_reward += float(reward)

        demo_env.close()
        print(f"[train_visual] GUI demo reward={demo_reward:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on LunarLander-v3 with occasional GUI demos."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps. No model is saved.",
    )
    parser.add_argument(
        "--render-first-episodes",
        type=int,
        default=10,
        help="Run a GUI demo after each of the first N completed training episodes.",
    )
    parser.add_argument(
        "--render-every-episodes",
        type=int,
        default=0,
        help="Also run one GUI demo every N completed episodes. Use 0 to disable.",
    )
    parser.add_argument(
        "--mean-window",
        type=int,
        default=100,
        help="Rolling reward window size printed to the console.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the training environment and model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps < 1:
        raise ValueError("--timesteps must be at least 1.")
    if args.render_first_episodes < 0:
        raise ValueError("--render-first-episodes must be 0 or greater.")
    if args.render_every_episodes < 0:
        raise ValueError("--render-every-episodes must be 0 or greater.")
    if args.mean_window < 1:
        raise ValueError("--mean-window must be at least 1.")

    env = gym.make(ENV_ID)
    env.reset(seed=args.seed)
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
    )

    print(f"[train_visual] environment={ENV_ID}")
    print("[train_visual] algorithm=PPO")
    print(f"[train_visual] total_timesteps={args.timesteps}")
    print(
        "[train_visual] rendering one GUI demo after each of the first "
        f"{args.render_first_episodes} training episodes"
    )
    if args.render_every_episodes > 0:
        print(
            "[train_visual] also rendering one GUI demo every "
            f"{args.render_every_episodes} training episodes"
        )
    print("[train_visual] model saving is disabled")

    model.learn(
        total_timesteps=args.timesteps,
        callback=VisualEpisodeCallback(
            render_first_episodes=args.render_first_episodes,
            render_every_episodes=args.render_every_episodes,
            mean_window=args.mean_window,
        ),
        progress_bar=True,
    )

    env.close()
    print("[train_visual] training complete; no model was saved")


if __name__ == "__main__":
    main()
