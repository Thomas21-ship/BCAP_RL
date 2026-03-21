import json
import os
from typing import Any, Dict, Tuple

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from training.env_setup import build_env


def evaluate_model(
    model: DQN,
    seed: int,
    env_cfg: Dict[str, Any],
    episodes: int,
) -> Tuple[float, float]:
    eval_env = build_env(seed, env_cfg)
    rewards, lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=episodes,
        return_episode_rewards=True,
        deterministic=True,
    )
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    mean_length = float(np.mean(lengths)) if lengths else 0.0
    return mean_reward, mean_length


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        episodes: int,
        eval_interval_steps: int,
        log_path: str,
        seed: int,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.episodes = episodes
        self.eval_interval_steps = eval_interval_steps
        self.log_path = log_path
        self.seed = seed
        self._last_eval_step = 0

        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_interval_steps:
            return True
        self._last_eval_step = self.num_timesteps

        rewards, lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.episodes,
            return_episode_rewards=True,
            deterministic=True,
        )
        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        mean_length = float(np.mean(lengths)) if lengths else 0.0

        record = {
            "step": int(self.num_timesteps),
            "mean_reward": mean_reward,
            "mean_length": mean_length,
            "seed": int(self.seed),
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return True


def build_eval_callback(
    env_cfg: Dict[str, Any],
    episodes: int,
    eval_interval_steps: int,
    log_path: str,
    seed: int,
) -> PeriodicEvalCallback:
    eval_env = build_env(seed, env_cfg)
    return PeriodicEvalCallback(
        eval_env=eval_env,
        episodes=episodes,
        eval_interval_steps=eval_interval_steps,
        log_path=log_path,
        seed=seed,
    )
