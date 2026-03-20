from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from port_env import PortEnv


DEFAULT_CONFIG = {
    "seed": 42,
    "total_timesteps": 500_000,
    "model_save_path": "models/dqn_antwerp_port",
    "dqn": {
        "learning_rate": 3e-4,
        "buffer_size": 200_000,
        "learning_starts": 10_000,
        "batch_size": 64,
        "gamma": 0.99,
        "tau": 1.0,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 10_000,
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    "env": {
        "invalid_action_penalty": 0.25,
        "terminate_on_invalid_action": False,
        "waiting_ship_penalty": 0.02,
        "long_wait_penalty": 0.002,
        "long_wait_threshold": 16,
    },
}


def _deep_merge(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict:
    parser = argparse.ArgumentParser(description="Train DQN on AntwerpPortEnv (minimal mode).")
    parser.add_argument("--config", default="training_config.json", help="Path to JSON config file")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps from config")
    args = parser.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        cfg = _deep_merge(cfg, file_cfg)

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.timesteps is not None:
        cfg["total_timesteps"] = int(args.timesteps)

    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_env_config(env: PortEnv, env_cfg: dict) -> None:
    for key, value in env_cfg.items():
        if hasattr(env, key):
            setattr(env, key, value)


def main() -> None:
    cfg = load_config()
    seed = int(cfg["seed"])
    total_timesteps = int(cfg["total_timesteps"])
    model_save_path = str(cfg["model_save_path"])
    dqn_cfg = dict(cfg.get("dqn", DEFAULT_CONFIG["dqn"]))
    env_cfg = dict(cfg.get("env", {}))

    seed_everything(seed)

    env = PortEnv()
    apply_env_config(env, env_cfg)
    env.reset(seed=seed)

    print("[1/3] Checking environment...")
    check_env(env)

    print("[2/3] Building model...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=float(dqn_cfg.get("learning_rate", 3e-4)),
        buffer_size=int(dqn_cfg.get("buffer_size", 200_000)),
        learning_starts=int(dqn_cfg.get("learning_starts", 10_000)),
        batch_size=int(dqn_cfg.get("batch_size", 64)),
        gamma=float(dqn_cfg.get("gamma", 0.99)),
        tau=float(dqn_cfg.get("tau", 1.0)),
        train_freq=int(dqn_cfg.get("train_freq", 4)),
        gradient_steps=int(dqn_cfg.get("gradient_steps", 1)),
        target_update_interval=int(dqn_cfg.get("target_update_interval", 10_000)),
        exploration_fraction=float(dqn_cfg.get("exploration_fraction", 0.2)),
        exploration_initial_eps=float(dqn_cfg.get("exploration_initial_eps", 1.0)),
        exploration_final_eps=float(dqn_cfg.get("exploration_final_eps", 0.05)),
        verbose=1,
        seed=seed,
    )

    print("[3/3] Training and saving...")
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Saved model to: {model_save_path}")


if __name__ == "__main__":
    main()
