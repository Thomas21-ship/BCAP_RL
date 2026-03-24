import os
import sys
from pathlib import Path

import numpy as np

def _add_repo_root_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_add_repo_root_to_path()

from env.port_env import PortEnv, encode_action, decode_action

SEED = 42
ROLLOUT_STEPS = 50


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    env = PortEnv()
    obs, info = env.reset(seed=SEED)

    # Spaces & shapes
    assert_true(obs.shape == (37,), f"Expected obs shape (37,), got {obs.shape}")
    assert_true(obs.dtype == np.float32, f"Expected obs dtype float32, got {obs.dtype}")
    assert_true(
        np.all(obs >= 0.0) and np.all(obs <= 1.0),
        "Observation values must be in [0, 1]",
    )
    assert_true(env.action_space.n > 0, "Action space must be non-empty")

    # Reset/step contract
    step_out = env.step(env.no_op_action)
    assert_true(len(step_out) == 5, f"Expected 5 outputs from step, got {len(step_out)}")
    obs2, reward, terminated, truncated, info2 = step_out
    assert_true(obs2.shape == (37,), f"Expected obs shape (37,), got {obs2.shape}")

    # Encode/decode sanity
    noop = encode_action(env, vessel_slot=env.no_op_slot, quay_position=0, cranes=0)
    decoded = decode_action(env, noop)
    assert_true(decoded == (env.no_op_slot, 0, 0), f"No-op decode mismatch: {decoded}")

    # Invariants
    assert_true(
        0 <= env.cranes_in_use <= env.total_cranes_limit,
        "cranes_in_use out of bounds",
    )
    assert_true(
        np.all((env.quay_map == 0) | (env.quay_map == 1)),
        "quay_map must be 0/1",
    )
    current_step_before = env.current_step
    env.step(env.no_op_action)
    assert_true(
        env.current_step == current_step_before + 1,
        "current_step must increment by 1 per step",
    )

    # Random rollout
    env.reset(seed=SEED)
    total_reward = 0.0
    terminated = False
    truncated = False
    for _ in range(ROLLOUT_STEPS):
        action = env.action_space.sample()
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except RuntimeError as exc:
            if "Invalid action" in str(exc):
                continue
            raise
        total_reward += reward
        assert_true(obs.shape == (37,), f"Expected obs shape (37,), got {obs.shape}")
        assert_true(
            np.all(obs >= 0.0) and np.all(obs <= 1.0),
            "Observation values must be in [0, 1]",
        )
        assert_true(
            0 <= env.cranes_in_use <= env.total_cranes_limit,
            "cranes_in_use out of bounds",
        )
        if terminated or truncated:
            break

    print("PASS: PortEnv readiness checks succeeded")
    print(
        "Random rollout: steps="
        + str(_ + 1)
        + f", total_reward={total_reward:.3f}, terminated={terminated}, truncated={truncated}"
    )


if __name__ == "__main__":
    main()
