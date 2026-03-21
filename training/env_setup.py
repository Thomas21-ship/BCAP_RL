from typing import Any, Dict

from env.port_env import PortEnv


def apply_env_config(env: PortEnv, env_cfg: Dict[str, Any]) -> None:
    for key, value in env_cfg.items():
        if hasattr(env, key):
            setattr(env, key, value)


def build_env(seed: int, env_cfg: Dict[str, Any]) -> PortEnv:
    env = PortEnv()
    apply_env_config(env, env_cfg)
    env.reset(seed=seed)
    return env
