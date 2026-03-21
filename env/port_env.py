import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .port_env_dynamics import reset_env, step_env
from .port_env_spec import decode_action, encode_action, get_observation


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
        self.observation_space = spaces.Box(low=0, high=1, shape=(37,), dtype=np.float32)

        # --- VESSEL GENERATION LIMITS ---
        self.min_vessel_length = 2
        self.max_vessel_length = 10
        self.min_workload = 50
        self.max_workload = 2000
        self.arrival_scale = 20

        # --- VESSEL ID COUNTER ---
        self.vessel_id_counter = 0

    def _decode_action(self, action):
        return decode_action(self, action)

    def _encode_action(self, vessel_slot, quay_position, cranes):
        return encode_action(self, vessel_slot, quay_position, cranes)

    def _get_observation(self):
        return get_observation(self)

    def reset(self, seed=None, options=None):
        return reset_env(self, seed=seed, options=options)

    def step(self, action):
        return step_env(self, action)


__all__ = ["PortEnv"]
