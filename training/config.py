from __future__ import annotations

import json
import os
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
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
        "waiting_ship_penalty": 0.02,
        "long_wait_penalty": 0.002,
        "long_wait_threshold": 16,
    },
    "grading": {
        "enabled": True,
        "episodes": 10,
        "eval_interval_steps": 10_000,
        "log_path": "metrics/training_eval.jsonl",
    },
}


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        cfg = _deep_merge(cfg, file_cfg)
    return cfg
