from typing import Any, Dict, List, Tuple

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from training.config import DEFAULT_CONFIG
from training.eval import build_eval_callback
from training.env_setup import build_env
from training.seed import seed_everything


def build_model(env, dqn_cfg: Dict[str, Any], seed: int) -> DQN:
    return DQN(
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


def train_model(
    cfg: Dict[str, Any],
    seed_override: int | None = None,
    timesteps_override: int | None = None,
) -> Tuple[DQN, Dict[str, Any], List[float]]:
    seed = int(seed_override if seed_override is not None else cfg["seed"])
    total_timesteps = int(
        timesteps_override if timesteps_override is not None else cfg["total_timesteps"]
    )
    dqn_cfg = dict(cfg.get("dqn", DEFAULT_CONFIG["dqn"]))
    env_cfg = dict(cfg.get("env", {}))
    grading_cfg = dict(cfg.get("grading", {}))

    seed_everything(seed)
    env = Monitor(build_env(seed, env_cfg), filename=None)

    model = build_model(env, dqn_cfg, seed)
    callback = None
    eval_interval_steps = int(grading_cfg.get("eval_interval_steps", 0) or 0)
    log_path = grading_cfg.get("log_path")
    if eval_interval_steps > 0 and log_path:
        episodes = int(grading_cfg.get("episodes", 10))
        callback = build_eval_callback(
            env_cfg=env_cfg,
            episodes=episodes,
            eval_interval_steps=eval_interval_steps,
            log_path=str(log_path),
            seed=seed + 1000,
        )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    episode_rewards = list(env.get_episode_rewards())

    return model, cfg, episode_rewards
