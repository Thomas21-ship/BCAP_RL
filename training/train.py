import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _add_repo_root_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _reward_plot_path(metrics_dir: Path) -> Path:
    return metrics_dir / "training_rewards.png"


def _config_hash(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_id(cfg: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = _config_hash(cfg)[:6]
    return f"{timestamp}_{short_hash}"


def main() -> None:
    _add_repo_root_to_path()

    from stable_baselines3.common.env_checker import check_env

    from env.port_env import PortEnv
    from training.config import load_config
    from training.env_setup import apply_env_config
    from training.eval import evaluate_model
    from training.model import train_model

    cfg = load_config("training_config.json")
    seed = int(cfg["seed"])
    total_timesteps = int(cfg["total_timesteps"])
    env_cfg = dict(cfg.get("env", {}))
    grading_cfg = dict(cfg.get("grading", {}))

    # Check environment contract
    env = PortEnv()
    apply_env_config(env, env_cfg)
    env.reset(seed=seed)
    try:
        check_env(env)
        print("Environment OK")
    except RuntimeError as exc:
        if "Invalid action" in str(exc):
            print("Environment check skipped: random action can be invalid for this env.")
        else:
            raise

    # Run metadata and artifact directories
    run_id = _run_id(cfg)
    config_hash = _config_hash(cfg)
    metrics_dir = Path("metrics") / "runs" / run_id
    models_dir = Path("models") / run_id
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Configure per-run paths
    model_save_path = str(models_dir / "dqn_antwerp_port")
    eval_log_path = str(metrics_dir / "eval.jsonl")
    cfg["model_save_path"] = model_save_path
    cfg.setdefault("grading", {})
    cfg["grading"]["log_path"] = eval_log_path

    eval_seeds = [seed + 1000 + i for i in range(5)]

    # Train and save model
    model, _, episode_rewards = train_model(
        cfg,
        seed_override=seed,
        timesteps_override=total_timesteps,
        run_id=run_id,
        config_hash=config_hash,
        eval_seeds=eval_seeds,
    )

    model.save(model_save_path)
    print("Saved model to:", model_save_path)
    print("Episodes recorded:", len(episode_rewards))
    print("Run ID:", run_id)

    # Plot training episode rewards (saved to disk)
    if episode_rewards:
        plt.figure(figsize=(8, 4))
        plt.plot(episode_rewards)
        plt.title("Training Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.tight_layout()
        plot_path = _reward_plot_path(metrics_dir)
        plt.savefig(plot_path, dpi=150)
        print("Saved reward plot to:", str(plot_path))
    else:
        print("No episode rewards recorded. Train longer to capture full episodes.")

    # Optional evaluation
    if bool(grading_cfg.get("enabled", False)):
        episodes = int(grading_cfg.get("episodes", 10))
        mean_reward, mean_length, std_reward, std_length = evaluate_model(
            model, eval_seeds, env_cfg, episodes
        )
        print(f"Mean reward: {mean_reward:.3f} (std {std_reward:.3f})")
        print(f"Mean episode length: {mean_length:.2f} (std {std_length:.2f})")


if __name__ == "__main__":
    main()
