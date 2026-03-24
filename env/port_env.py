import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .vessel import VesselQueue


def encode_action(env, vessel_slot: int, quay_position: int, cranes: int) -> int:
    if vessel_slot == env.no_op_slot:
        return int(env.no_op_action)
    if vessel_slot < 0 or vessel_slot >= env.num_vessel_slots:
        raise ValueError(f"vessel_slot out of bounds: {vessel_slot}")
    if quay_position < 0 or quay_position >= env.quay_size:
        raise ValueError(f"quay_position out of bounds: {quay_position}")
    if cranes < 0 or cranes > env.total_cranes_limit:
        raise ValueError(f"cranes out of bounds: {cranes}")
    return int(
        vessel_slot * env.actions_per_vessel
        + quay_position * env.crane_choices
        + cranes
    )


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
    status_map = {"none": 0.0, "on_the_way": 0.0, "waiting": 0.5, "at_berth": 1.0}

    for i in range(env.num_vessel_slots):
        base = i * 4
        if i < len(env.vessel_queue.queue):
            v = env.vessel_queue.queue[i]
            vessel_obs[base + 0] = np.clip(v.arrival_time / env.max_steps, 0.0, 1.0)
            vessel_obs[base + 1] = np.clip(v.length / env.quay_size, 0.0, 1.0)
            vessel_obs[base + 2] = np.clip(
                v.containers_remaining / env.max_workload, 0.0, 1.0
            )
            vessel_obs[base + 3] = status_map.get(v.status, 0.0)

    if env.total_cranes_limit > 0:
        cranes_norm = np.clip(env.cranes_in_use / env.total_cranes_limit, 0.0, 1.0)
    else:
        cranes_norm = 0.0

    return np.concatenate(
        [quay_obs, vessel_obs, np.array([cranes_norm], dtype=np.float32)]
    )


def _enforce_crane_invariants(env):
    env.cranes_in_use = int(
        np.clip(env.cranes_in_use, 0, env.total_cranes_limit)
    )
    if getattr(env, "debug", False):
        assigned = sum(
            v.cranes_assigned for v in env.vessel_queue.queue if v.status == "at_berth"
        )
        assert assigned == env.cranes_in_use, (
            f"cranes_in_use mismatch: {env.cranes_in_use} vs {assigned}"
        )


def reset_env(env, seed=None, options=None):
    # This line handles the random seed properly (Gymnasium standard practice)
    gym.Env.reset(env, seed=seed)

    # Reset the clock
    env.current_step = 0

    # Reset per-episode IDs to keep identity bounded and easy to read in logs.
    env.vessel_id_counter = 0

    # Clear the quay
    env.quay_map = np.zeros(env.quay_size, dtype=np.float32)

    # Generate starting vessels
    env.vessel_queue = VesselQueue(
        capacity=env.num_vessel_slots,
        rng=env.np_random,
        current_time=env.current_step,
        arrival_scale=env.arrival_scale,
        min_length=env.min_vessel_length,
        max_length=env.max_vessel_length,
        min_workload=env.min_workload,
        max_workload=env.max_workload,
    )

    # Reset crane tracking
    env.cranes_in_use = 0
    env._terminated = False

    observation = get_observation(env)
    info = {}
    return observation, info


def step_env(env, action):
    if getattr(env, "_terminated", False):
        observation, info = reset_env(env, seed=None, options=None)
        info["autoreset"] = True
        return observation, 0.0, False, False, info

    # 1. Decode action
    try:
        vessel_slot, quay_position, cranes = decode_action(env, action)
    except ValueError:
        vessel_slot, quay_position, cranes = env.no_op_slot, 0, 0
        invalid_action = True
        invalid_reason = "action_out_of_bounds"
    else:
        invalid_action = False
        invalid_reason = ""

    reward = 0.0
    terminated = False

    # 2. Try to execute the action
    if vessel_slot == env.no_op_slot:
        pass
    elif vessel_slot < len(env.vessel_queue.queue):
        vessel = env.vessel_queue.queue[vessel_slot]
        if vessel.status == "at_berth":
            desired_cranes = int(np.clip(cranes, 0, env.total_cranes_limit))
            current_cranes = vessel.cranes_assigned
            delta = desired_cranes - current_cranes

            if delta > 0:
                cranes_available = env.total_cranes_limit - env.cranes_in_use
                if cranes_available >= delta:
                    vessel.cranes_assigned = desired_cranes
                    env.cranes_in_use += delta
                    vessel.max_cranes_assigned = max(
                        getattr(vessel, "max_cranes_assigned", 0),
                        vessel.cranes_assigned,
                    )
                else:
                    invalid_action = True
                    invalid_reason = "insufficient_cranes_for_reallocation"
            elif delta < 0:
                vessel.cranes_assigned = desired_cranes
                env.cranes_in_use += delta
        else:
            can_dock = (
                vessel.status == "waiting"
                and vessel.arrival_time <= 0
            )

            if can_dock:
                end_position = quay_position + vessel.length

                if end_position <= env.quay_size:
                    blocks_needed = env.quay_map[quay_position:end_position]
                    quay_is_free = np.all(blocks_needed == 0)

                    if quay_is_free:
                        cranes_available = env.total_cranes_limit - env.cranes_in_use
                        cranes_to_assign = min(cranes, cranes_available)

                        if cranes_to_assign > 0:
                            vessel.status = "at_berth"
                            vessel.cranes_assigned = cranes_to_assign
                            vessel.max_cranes_assigned = cranes_to_assign
                            vessel.docking_position = quay_position
                            vessel.docking_step = env.current_step

                            env.quay_map[quay_position:end_position] = 1.0
                            env.cranes_in_use += cranes_to_assign

                            reward += 1.0
                        else:
                            invalid_action = True
                            invalid_reason = "no_cranes_assigned_or_available"
                    else:
                        invalid_action = True
                        invalid_reason = "quay_blocks_not_free"
                else:
                    invalid_action = True
                    invalid_reason = "quay_position_out_of_bounds"
            else:
                invalid_action = True
                invalid_reason = "vessel_not_ready_to_dock"
    else:
        invalid_action = True
        invalid_reason = "invalid_vessel_slot"

    if invalid_action:
        reward -= env.invalid_action_penalty

    # 3. Process docked vessels
    for vessel in env.vessel_queue.queue:
        if vessel.status == "at_berth":
            containers_processed = vessel.cranes_assigned * env.crane_rate
            vessel.containers_remaining -= containers_processed
            vessel.containers_remaining = max(0, vessel.containers_remaining)
            reward += containers_processed * 0.001

    # 4. Check departures
    finished_indices = []
    for i, vessel in enumerate(env.vessel_queue.queue):
        if vessel.status == "at_berth" and vessel.is_finished():
            pos = vessel.docking_position
            env.quay_map[pos:pos + vessel.length] = 0
            env.cranes_in_use -= vessel.cranes_assigned
            vessel.cranes_assigned = 0
            vessel.departure_step = env.current_step
            reward += 5.0
            finished_indices.append(i)

    for i in reversed(finished_indices):
        del env.vessel_queue.queue[i]
        env.vessel_queue.enqueue(env.vessel_queue._generate_random_vessel())

    # 5. Penalise idle cranes
    idle_cranes = env.total_cranes_limit - env.cranes_in_use
    waiting_ships = sum(
        1 for v in env.vessel_queue.queue
        if v.status == "waiting" and v.arrival_time <= 0
    )
    if waiting_ships > 0 and idle_cranes > 0:
        reward -= idle_cranes * 0.1

    # 6. Penalise queueing delay
    waiting_vessels = [
        v for v in env.vessel_queue.queue
        if v.status == "waiting" and v.arrival_time <= 0
    ]
    if waiting_vessels:
        reward -= env.waiting_ship_penalty * len(waiting_vessels)
        for v in waiting_vessels:
            wait_duration = -v.arrival_time
            excess_wait = max(0.0, wait_duration - env.long_wait_threshold)
            reward -= env.long_wait_penalty * excess_wait

    # 7. Advance the clock
    env.vessel_queue.advance_time(steps=1)
    env.current_step += 1

    # 8. Check if the week is over
    if env.current_step >= env.max_steps:
        terminated = True
        env._terminated = True

    truncated = False

    _enforce_crane_invariants(env)

    observation = get_observation(env)
    info = {
        "step": env.current_step,
        "cranes_in_use": env.cranes_in_use,
        "invalid_action": invalid_action,
        "invalid_reason": invalid_reason,
    }

    return observation, reward, terminated, truncated, info


class PortEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # --- PHYSICAL CONSTANTS ---
        self.quay_size = 20           # 20 blocks of 20m each = 400m of quay
        self.total_cranes_limit = 3   # We only have 3 cranes total across ALL docked ships
        self.crane_rate = 7.5         # Each crane unloads 7.5 containers per 15-min step
        self.max_steps = 96           # One day = 24 hours x 4 steps/hour
        self.invalid_action_penalty = 0.25
        self.waiting_ship_penalty = 0.02
        self.long_wait_penalty = 0.002
        self.long_wait_threshold = 16  # 4 hours (16 x 15-minute steps)

        # --- ACTION SPACE ---
        self.num_vessel_slots = 4
        self.crane_choices = self.total_cranes_limit + 1  # 0..3
        self.actions_per_vessel = self.quay_size * self.crane_choices
        self.num_docking_actions = self.num_vessel_slots * self.actions_per_vessel

        self.no_op_action = self.num_docking_actions
        self.no_op_slot = self.num_vessel_slots

        self.action_space = spaces.Discrete(self.no_op_action + 1)

        # --- OBSERVATION SPACE ---
        obs_size = self.quay_size + (self.num_vessel_slots * 4) + 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )

        # --- VESSEL GENERATION LIMITS ---
        self.min_vessel_length = 2
        self.max_vessel_length = 10
        self.min_workload = 50
        self.max_workload = 2000
        self.arrival_scale = 20

        # --- VESSEL ID COUNTER ---
        self.vessel_id_counter = 0
        self.vessel_queue = None

        # --- DEBUG ---
        self.debug = False

    def reset(self, seed=None, options=None):
        return reset_env(self, seed=seed, options=options)

    def step(self, action):
        return step_env(self, action)

__all__ = [
    "PortEnv",
    "encode_action",
    "decode_action",
    "get_observation",
    "reset_env",
    "step_env",
]
