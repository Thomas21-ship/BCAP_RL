import gymnasium as gym
import numpy as np

from ship_generator import generate_single_vessel
from ship_manager import Vessel
from .port_env_spec import decode_action


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
    env.vessels = []
    last_arrival = 0
    for _ in range(env.num_vessel_slots):
        raw = generate_single_vessel(
            last_arrival_time=last_arrival,
            min_length=env.min_vessel_length,
            max_length=env.max_vessel_length,
            min_workload=env.min_workload,
            max_workload=env.max_workload,
            arrival_scale=env.arrival_scale,
        )
        env.vessel_id_counter += 1
        v = Vessel(
            vessel_id=env.vessel_id_counter,
            length=int(np.round(raw[0])),
            workload=raw[1],
            arrival_time=raw[2],
        )
        env.vessels.append(v)
        last_arrival = raw[2]

    # Reset crane tracking
    env.cranes_in_use = 0

    observation = env._get_observation()
    info = {}
    return observation, info


def step_env(env, action):
    # 1. Decode action
    vessel_slot, quay_position, cranes = decode_action(env, action)

    reward = 0.0
    terminated = False
    invalid_action = False
    invalid_reason = ""

    # 2. Try to execute the action
    if vessel_slot == env.no_op_slot:
        pass
    elif vessel_slot < len(env.vessels):
        vessel = env.vessels[vessel_slot]
        if vessel.status == "docked":
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
                and vessel.arrival_time <= env.current_step
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
                            vessel.status = "docked"
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
        raise RuntimeError(f"Invalid action: {invalid_reason}")

    # 3. Process docked vessels
    for vessel in env.vessels:
        if vessel.status == "docked":
            containers_processed = vessel.cranes_assigned * env.crane_rate
            vessel.containers_remaining -= containers_processed
            vessel.containers_remaining = max(0, vessel.containers_remaining)
            reward += containers_processed * 0.001

    # 4. Check departures
    for vessel in env.vessels:
        if vessel.status == "docked" and vessel.is_finished():
            pos = vessel.docking_position
            env.quay_map[pos:pos + vessel.length] = 0
            env.cranes_in_use -= vessel.cranes_assigned
            vessel.status = "departed"
            vessel.cranes_assigned = 0
            vessel.departure_step = env.current_step
            reward += 5.0

    # 5. Penalise idle cranes
    idle_cranes = env.total_cranes_limit - env.cranes_in_use
    waiting_ships = sum(
        1 for v in env.vessels
        if v.status == "waiting" and v.arrival_time <= env.current_step
    )
    if waiting_ships > 0 and idle_cranes > 0:
        reward -= idle_cranes * 0.1

    # 6. Penalise queueing delay
    waiting_vessels = [
        v for v in env.vessels
        if v.status == "waiting" and v.arrival_time <= env.current_step
    ]
    if waiting_vessels:
        reward -= env.waiting_ship_penalty * len(waiting_vessels)
        for v in waiting_vessels:
            wait_duration = env.current_step - v.arrival_time
            excess_wait = max(0.0, wait_duration - env.long_wait_threshold)
            reward -= env.long_wait_penalty * excess_wait

    # 7. Advance the clock
    env.current_step += 1

    # 8. Check if the week is over
    if env.current_step >= env.max_steps:
        terminated = True

    truncated = False

    observation = env._get_observation()
    info = {
        "step": env.current_step,
        "cranes_in_use": env.cranes_in_use,
        "invalid_action": invalid_action,
        "invalid_reason": invalid_reason,
    }

    return observation, reward, terminated, truncated, info


__all__ = ["reset_env", "step_env"]
