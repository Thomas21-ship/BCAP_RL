import gymnasium as gym
from gymnasium import spaces
import numpy as np

# This is our 'Blueprint' for the Port
class AntwerpPortEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # --- PHYSICAL VARIABLES ---
        self.quay_size = 40           # 800m / 20m blocks
        self.total_cranes_limit = 7   # Maximum cranes in the pool
        self.crane_rate = 7.5         # Containers moved per 15 mins
        self.max_steps = 672          # Steps in 1 week
        
        # --- ACTION SPACE ---
        # We tell the AI it has 3201 possible 'buttons' to press.
        # We will explain later how one button = (Ship + Position + Cranes)
        self.action_space = spaces.Discrete(3201)

        # --- OBSERVATION SPACE ---
        # The AI 'sees' 80 numbers (40 quay blocks + 10 ships with 4 stats each)
        # We set them between 0 and 1 so the AI learns easier.
        self.observation_space = spaces.Box(low=0, high=1, shape=(80,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # This function starts a new 'Week'
        pass

    def step(self, action):
        # This function moves the clock forward by 15 minutes
        pass