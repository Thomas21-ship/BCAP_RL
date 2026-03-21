import numpy as np


def encode_action(env, vessel_slot: int, quay_position: int, cranes: int) -> int:
    if vessel_slot == env.no_op_slot:
        return int(env.no_op_action)
    if vessel_slot < 0 or vessel_slot >= env.num_vessel_slots:
        raise ValueError(f"vessel_slot out of bounds: {vessel_slot}")
    if quay_position < 0 or quay_position >= env.quay_size:
        raise ValueError(f"quay_position out of bounds: {quay_position}")
    if cranes < 0 or cranes > env.total_cranes_limit:
        raise ValueError(f"cranes out of bounds: {cranes}")
    return int(vessel_slot * env.actions_per_vessel + quay_position * env.crane_choices + cranes)


def decode_action(env, action: int):
    action = int(action)
    if action < 0 or action >= env.action_space.n:
        raise ValueError(f"Action {action} out of bounds [0, {env.action_space.n - 1}]")
    if action == env.no_op_action:
        return env.no_op_slot, 0, 0

    vessel_slot = action // env.actions_per_vessel
    remainder = action % env.actions_per_vessel
    quay_position = remainder // env.crane_choices
    cranes = remainder % env.crane_choices
    return vessel_slot, quay_position, cranes


def get_observation(env):
    quay_obs = env.quay_map.copy().astype(np.float32)

    vessel_obs = np.zeros(env.num_vessel_slots * 4, dtype=np.float32)
    status_map = {"none": 0.0, "waiting": 1 / 3, "docked": 2 / 3, "departed": 1.0}

    for i in range(env.num_vessel_slots):
        base = i * 4
        if i < len(env.vessels):
            v = env.vessels[i]
            vessel_obs[base + 0] = np.clip(v.arrival_time / env.max_steps, 0.0, 1.0)
            vessel_obs[base + 1] = np.clip(v.length / env.quay_size, 0.0, 1.0)
            vessel_obs[base + 2] = np.clip(v.containers_remaining / env.max_workload, 0.0, 1.0)
            vessel_obs[base + 3] = status_map.get(v.status, 0.0)

    if env.total_cranes_limit > 0:
        cranes_norm = np.clip(env.cranes_in_use / env.total_cranes_limit, 0.0, 1.0)
    else:
        cranes_norm = 0.0

    return np.concatenate([quay_obs, vessel_obs, np.array([cranes_norm], dtype=np.float32)])


__all__ = ["encode_action", "decode_action", "get_observation"]
