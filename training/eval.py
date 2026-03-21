import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from training.env_setup import build_env


def _evaluate_once(
    model: DQN,
    seed: int,
    env_cfg: Dict[str, Any],
    episodes: int,
) -> Tuple[float, float]:
    eval_env = build_env(seed, env_cfg)
    eval_env = Monitor(eval_env, filename=None)
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


def evaluate_model(
    model: DQN,
    seeds: List[int],
    env_cfg: Dict[str, Any],
    episodes: int,
) -> Tuple[float, float, float, float]:
    rewards = []
    lengths = []
    for seed in seeds:
        mean_reward, mean_length = _evaluate_once(model, seed, env_cfg, episodes)
        rewards.append(mean_reward)
        lengths.append(mean_length)
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    mean_length = float(np.mean(lengths)) if lengths else 0.0
    std_reward = float(np.std(rewards)) if rewards else 0.0
    std_length = float(np.std(lengths)) if lengths else 0.0
    return mean_reward, mean_length, std_reward, std_length


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        env_cfg: Dict[str, Any],
        episodes: int,
        eval_interval_steps: int,
        log_path: str,
        seeds: List[int],
        run_id: str,
        config_hash: str,
    ):
        super().__init__()
        self.env_cfg = env_cfg
        self.episodes = episodes
        self.eval_interval_steps = eval_interval_steps
        self.log_path = log_path
        self.seeds = list(seeds)
        self.run_id = run_id
        self.config_hash = config_hash
        self._last_eval_step = 0

        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_interval_steps:
            return True
        self._last_eval_step = self.num_timesteps

        rewards = []
        lengths = []
        for seed in self.seeds:
            mean_reward, mean_length = _evaluate_once(
                self.model, seed, self.env_cfg, self.episodes
            )
            rewards.append(mean_reward)
            lengths.append(mean_length)

        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        mean_length = float(np.mean(lengths)) if lengths else 0.0
        std_reward = float(np.std(rewards)) if rewards else 0.0
        std_length = float(np.std(lengths)) if lengths else 0.0

        record = {
            "run_id": self.run_id,
            "step": int(self.num_timesteps),
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "std_length": std_length,
            "seeds": self.seeds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_hash": self.config_hash,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return True


def build_eval_callback(
    env_cfg: Dict[str, Any],
    episodes: int,
    eval_interval_steps: int,
    log_path: str,
    seeds: List[int],
    run_id: str,
    config_hash: str,
) -> PeriodicEvalCallback:
    return PeriodicEvalCallback(
        env_cfg=env_cfg,
        episodes=episodes,
        eval_interval_steps=eval_interval_steps,
        log_path=log_path,
        seeds=seeds,
        run_id=run_id,
        config_hash=config_hash,
    )
