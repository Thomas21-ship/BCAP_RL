# =============================================================================
# train.py - Training the RL Agent on the Antwerp Port Simulation
# =============================================================================

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime, timezone

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.env_checker import check_env

from port_env import AntwerpPortEnv


DEFAULT_CONFIG = {
    "seed": 42,
    "total_timesteps": 500_000,
    "model_save_path": "models/ppo_antwerp_port",
    "log_dir": "logs/",
    "eval_freq": 10_000,
    "checkpoint_freq": 50_000,
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
    },
    "env": {
        "invalid_action_penalty": 0.25,
        "terminate_on_invalid_action": False,
        "waiting_ship_penalty": 0.02,
        "long_wait_penalty": 0.002,
        "long_wait_threshold": 16,
    },
    "early_stopping": {
        "enabled": True,
        "max_no_improvement_evals": 6,
        "min_evals": 5,
    },
    "kpi": {
        "enabled": True,
        "output_dir": "metrics",
    },
}


def _deep_merge(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config() -> dict:
    parser = argparse.ArgumentParser(description="Train PPO on AntwerpPortEnv")
    parser.add_argument("--config", default="training_config.json", help="Path to JSON config file")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps from config")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        config = _deep_merge(config, file_cfg)

    if args.seed is not None:
        config["seed"] = args.seed
    if args.timesteps is not None:
        config["total_timesteps"] = args.timesteps

    config["_config_path"] = args.config
    return config


def apply_env_config(env: AntwerpPortEnv, env_cfg: dict) -> None:
    for key, value in env_cfg.items():
        if hasattr(env, key):
            setattr(env, key, value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_kpis(test_env: AntwerpPortEnv, total_reward: float, total_steps: int,
                 cranes_sum: float, invalid_steps: int, seed: int) -> dict:
    vessels = test_env.vessels
    departed = sum(1 for v in vessels if v.status == "departed")
    waiting = sum(1 for v in vessels if v.status == "waiting")
    docked = sum(1 for v in vessels if v.status == "docked")

    docking_waits = [
        max(0.0, float(v.docking_step - v.arrival_time))
        for v in vessels
        if hasattr(v, "docking_step")
    ]
    unresolved_waits = [
        max(0.0, float(test_env.current_step - v.arrival_time))
        for v in vessels
        if v.status == "waiting" and v.arrival_time <= test_env.current_step
    ]

    completed_pct = [
        ((v.workload - v.containers_remaining) / v.workload) * 100.0 if v.workload > 0 else 0.0
        for v in vessels
    ]

    crane_util = cranes_sum / max(1, total_steps * test_env.total_cranes_limit)
    invalid_rate = invalid_steps / max(1, total_steps)

    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": int(seed),
        "total_reward": float(total_reward),
        "total_steps": int(total_steps),
        "vessels_total": int(len(vessels)),
        "vessels_departed": int(departed),
        "vessels_docked": int(docked),
        "vessels_waiting": int(waiting),
        "completion_ratio": float(departed / max(1, len(vessels))),
        "mean_docking_wait_steps": float(np.mean(docking_waits)) if docking_waits else None,
        "max_docking_wait_steps": float(np.max(docking_waits)) if docking_waits else None,
        "mean_unresolved_wait_steps": float(np.mean(unresolved_waits)) if unresolved_waits else 0.0,
        "avg_processed_pct": float(np.mean(completed_pct)) if completed_pct else 0.0,
        "crane_utilization": float(crane_util),
        "invalid_action_rate": float(invalid_rate),
    }


def save_kpis(kpi_cfg: dict, metrics: dict) -> tuple[str, str]:
    out_dir = kpi_cfg.get("output_dir", "metrics")
    os.makedirs(out_dir, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"run_{run_id}.json")
    csv_path = os.path.join(out_dir, "runs.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    headers = list(metrics.keys())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    return json_path, csv_path


def main() -> None:
    cfg = load_config()
    seed = int(cfg["seed"])
    total_timesteps = int(cfg["total_timesteps"])
    model_save_path = cfg["model_save_path"]
    log_dir = cfg["log_dir"]
    eval_freq = int(cfg["eval_freq"])
    checkpoint_freq = int(cfg["checkpoint_freq"])

    print("=" * 60)
    print("ANTWERP PORT RL - TRAINING SCRIPT")
    print("=" * 60)
    print(f"Run seed: {seed}")
    print(f"Config path: {cfg.get('_config_path')}")

    seed_everything(seed)

    print("\n[1/5] Creating the port environment...")
    env = AntwerpPortEnv()
    apply_env_config(env, cfg.get("env", {}))
    env.reset(seed=seed)
    print("      Environment created.")

    print("\n[2/5] Running environment safety check (check_env)...")
    check_env(env, warn=True)
    print("      Safety check complete.")

    print("\n[3/5] Creating the PPO agent...")
    os.makedirs("models", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=cfg["ppo"]["learning_rate"],
        n_steps=int(cfg["ppo"]["n_steps"]),
        batch_size=int(cfg["ppo"]["batch_size"]),
        n_epochs=int(cfg["ppo"]["n_epochs"]),
        gamma=float(cfg["ppo"]["gamma"]),
        seed=seed,
    )
    print("      PPO agent created.")

    print("\n[4/5] Setting up callbacks...")
    eval_env = AntwerpPortEnv()
    apply_env_config(eval_env, cfg.get("env", {}))
    eval_env.reset(seed=seed + 1)

    stop_after_no_improve = None
    early_cfg = cfg.get("early_stopping", {})
    if early_cfg.get("enabled", False):
        stop_after_no_improve = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=int(early_cfg.get("max_no_improvement_evals", 6)),
            min_evals=int(early_cfg.get("min_evals", 5)),
            verbose=1,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_save_path + "_best",
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        callback_after_eval=stop_after_no_improve,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=model_save_path + "_checkpoints/",
        name_prefix="ppo_port",
    )

    print("      Callbacks ready.")

    print("\n[5/5] Starting training...")
    print(f"      Total timesteps: {total_timesteps:,}")
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

    print("\nTraining complete.")
    model.save(model_save_path)
    print(f"Final model saved to: {model_save_path}.zip")

    best_model_file = os.path.join(model_save_path + "_best", "best_model.zip")
    eval_model = model
    if os.path.exists(best_model_file):
        print(f"Best model found. Using for final evaluation: {best_model_file}")
        eval_model = PPO.load(best_model_file)

    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION - One Full Simulated Week")
    print("=" * 60)

    test_env = AntwerpPortEnv()
    apply_env_config(test_env, cfg.get("env", {}))
    obs, _ = test_env.reset(seed=seed + 2)

    total_reward = 0.0
    total_steps = 0
    cranes_sum = 0.0
    invalid_steps = 0

    while True:
        action, _ = eval_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

        total_reward += reward
        total_steps += 1
        cranes_sum += float(info.get("cranes_in_use", 0))
        invalid_steps += int(bool(info.get("invalid_action", False)))

        if terminated or truncated:
            break

    departed = sum(1 for v in test_env.vessels if v.status == "departed")
    waiting = sum(1 for v in test_env.vessels if v.status == "waiting")
    docked = sum(1 for v in test_env.vessels if v.status == "docked")

    print(f"Steps completed      : {total_steps} / {test_env.max_steps}")
    print(f"Total reward         : {total_reward:.2f}")
    print(f"Vessels departed     : {departed} / {len(test_env.vessels)}")
    print(f"Still docked at end  : {docked}")
    print(f"Still waiting at end : {waiting}")

    print("\nVessel-by-vessel breakdown:")
    for v in test_env.vessels:
        processed = v.workload - v.containers_remaining
        pct = (processed / v.workload * 100.0) if v.workload > 0 else 0.0
        print(
            f"  Vessel {v.id:>2} | length={v.length:>2} blocks | "
            f"workload={v.workload:>5.0f} | status={v.status:<9} | "
            f"processed={pct:.0f}%"
        )

    if cfg.get("kpi", {}).get("enabled", True):
        metrics = compute_kpis(
            test_env=test_env,
            total_reward=total_reward,
            total_steps=total_steps,
            cranes_sum=cranes_sum,
            invalid_steps=invalid_steps,
            seed=seed,
        )
        metrics["model_save_path"] = model_save_path
        metrics["used_best_model"] = bool(os.path.exists(best_model_file))
        json_path, csv_path = save_kpis(cfg.get("kpi", {}), metrics)
        print("\nKPI artifacts saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV : {csv_path}")

    print("\n" + "=" * 60)
    print("Training and evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
