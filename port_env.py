import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ship_generator import generate_single_vessel
from ship_manager import Vessel


class AntwerpPortEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # --- PHYSICAL CONSTANTS ---
        # These never change during training. Think of them as the rules of the port.
        self.quay_size = 40           # 40 blocks of 20m each = 800m of quay
        self.total_cranes_limit = 7   # We only have 7 cranes total across ALL docked ships
        self.crane_rate = 7.5         # Each crane unloads 7.5 containers per 15-min step
        self.max_steps = 672          # One week = 7 days x 24 hours x 4 steps/hour
        self.invalid_action_penalty = 0.25
        self.terminate_on_invalid_action = False
        self.waiting_ship_penalty = 0.02
        self.long_wait_penalty = 0.002
        self.long_wait_threshold = 16  # 4 hours (16 x 15-minute steps)

        # --- ACTION SPACE ---
        # Docking actions are encoded as:
        #   action = vessel_slot * (quay_size * crane_choices) + quay_position * crane_choices + cranes
        # where:
        #   vessel_slot   in [0, 9]
        #   quay_position in [0, 39]
        #   cranes        in [0, 7]
        #
        # We also reserve one dedicated NO-OP action at the end.
        self.num_vessel_slots = 10
        self.crane_choices = self.total_cranes_limit + 1  # 0..7
        self.actions_per_vessel = self.quay_size * self.crane_choices
        self.num_docking_actions = self.num_vessel_slots * self.actions_per_vessel
        self.no_op_action = self.num_docking_actions
        self.no_op_slot = self.num_vessel_slots
        self.action_space = spaces.Discrete(self.no_op_action + 1)

        # --- OBSERVATION SPACE ---
        # 80 numbers between 0 and 1 that describe the current state of the port.
        # First 40: quay block occupancy (0.0 = free, 1.0 = occupied)
        # Next  40: 10 ships x 4 stats each (arrival, length, workload, status)
        self.observation_space = spaces.Box(low=0, high=1, shape=(80,), dtype=np.float32)

        # --- VESSEL ID COUNTER ---
        # We assign IDs here (not in the generator) so every vessel ever created
        # gets a unique number across the whole simulation.
        self.vessel_id_counter = 0

    # =========================================================================
    # HELPER: Decode Action
    # =========================================================================
    def _decode_action(self, action):
        """
        Converts one integer action into three separate decisions.

        Docking encoding formula:
            action = vessel_slot * actions_per_vessel
                     + quay_position * crane_choices
                     + cranes

        One additional action index is reserved for NO-OP:
            action == no_op_action -> (no_op_slot, 0, 0)

        Returns:
            vessel_slot   (0..9 = which ship, 10 = do nothing)
            quay_position (0..39 = which block on the quay)
            cranes        (0..7 = how many cranes to assign)
        """
        action = int(action)
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Action {action} out of bounds [0, {self.action_space.n - 1}]")

        if action == self.no_op_action:
            return self.no_op_slot, 0, 0

        vessel_slot = action // self.actions_per_vessel
        remainder = action % self.actions_per_vessel
        quay_position = remainder // self.crane_choices
        cranes = remainder % self.crane_choices

        return vessel_slot, quay_position, cranes

    # =========================================================================
    # HELPER: Build Observation
    # =========================================================================
    def _get_observation(self):
        """
        Packages the current state of the port into 80 numbers between 0 and 1.
        This is what the AI 'sees' each step — like taking a photograph of the port.

        Structure:
          obs[0:40]  = quay occupancy map (0 = free block, 1 = occupied block)
          obs[40:80] = 10 vessels x 4 stats each:
                         [0] arrival_time  / max_steps          (0–1)
                         [1] length        / quay_size          (0–1)
                         [2] containers_remaining / 5000        (0–1)
                         [3] status code   / 3                  (0=none,
                                                                  0.33=waiting,
                                                                  0.67=docked,
                                                                  1=departed)
        """
        # Part 1: Quay map — a 40-item list. 0 = free, 1 = occupied.
        quay_obs = self.quay_map.copy().astype(np.float32)

        # Part 2: Vessel stats — loop over 10 vessel slots
        vessel_obs = np.zeros(40, dtype=np.float32)
        status_map = {"none": 0.0, "waiting": 1/3, "docked": 2/3, "departed": 1.0}

        for i in range(10):                        # 10 vessel slots
            base = i * 4                           # Each vessel takes 4 slots in the array
            if i < len(self.vessels):
                v = self.vessels[i]
                vessel_obs[base + 0] = np.clip(v.arrival_time / self.max_steps, 0.0, 1.0)
                vessel_obs[base + 1] = np.clip(v.length / self.quay_size, 0.0, 1.0)
                vessel_obs[base + 2] = np.clip(v.containers_remaining / 5000, 0.0, 1.0)
                vessel_obs[base + 3] = status_map.get(v.status, 0.0)
            # If there are fewer than 10 vessels, the remaining slots stay 0

        return np.concatenate([quay_obs, vessel_obs])

    # =========================================================================
    # RESET — Start a fresh simulated week
    # =========================================================================
    def reset(self, seed=None, options=None):
        """
        Called at the start of every new episode (simulated week).
        Clears the quay, resets the clock, and generates 10 new vessels.

        Gymnasium requires reset() to return: (observation, info)
        """
        # This line handles the random seed properly (Gymnasium standard practice)
        super().reset(seed=seed)

        # --- Reset the clock ---
        self.current_step = 0

        # Reset per-episode IDs to keep identity bounded and easy to read in logs.
        # IDs are no longer used in observations, so this only affects bookkeeping.
        self.vessel_id_counter = 0

        # --- Clear the quay ---
        # quay_map is a list of 40 numbers.
        # 0.0 = free block, 1.0 = occupied block.
        self.quay_map = np.zeros(self.quay_size, dtype=np.float32)

        # --- Generate 10 starting vessels ---
        # We create vessels with staggered arrival times using the Poisson process
        # from ship_generator.py. Each vessel's arrival is relative to the last.
        self.vessels = []
        last_arrival = 0
        for _ in range(10):
            raw = generate_single_vessel(last_arrival)
            # raw = [length, workload, arrival_time]

            self.vessel_id_counter += 1
            v = Vessel(
                vessel_id=self.vessel_id_counter,
                length=int(np.round(raw[0])),   # Round to whole blocks
                workload=raw[1],
                arrival_time=raw[2]
            )
            self.vessels.append(v)
            last_arrival = raw[2]

        # --- Reset crane tracking ---
        # We'll track how many cranes are currently in use across all docked ships
        self.cranes_in_use = 0

        # Build and return the first observation
        observation = self._get_observation()
        info = {}
        return observation, info

    # =========================================================================
    # STEP — Advance the simulation by 15 minutes
    # =========================================================================
    def step(self, action):
        """
        The AI has chosen an action. We:
          1. Decode the action into (vessel, position, cranes)
          2. Try to execute it:
             - waiting vessel: attempt docking
             - docked vessel: reallocate cranes (target absolute crane count)
          3. Process all docked ships (cranes work, containers reduce)
          4. Check for departures (ship finished = leaves quay)
          5. Calculate reward
          6. Build and return the new observation

        Gymnasium requires step() to return:
            (observation, reward, terminated, truncated, info)
        """

        # --- 1. DECODE THE ACTION ---
        vessel_slot, quay_position, cranes = self._decode_action(action)

        reward = 0.0       # Start with zero reward this step
        terminated = False # Will become True when the week ends
        invalid_action = False
        invalid_reason = ""

        # --- 2. TRY TO EXECUTE THE ACTION ---
        # no_op_slot means "do nothing" — the AI chose to skip this step
        if vessel_slot == self.no_op_slot:
            pass
        elif vessel_slot < len(self.vessels):
            vessel = self.vessels[vessel_slot]
            if vessel.status == "docked":
                # Reallocation path: cranes value is desired ABSOLUTE crane count.
                desired_cranes = int(np.clip(cranes, 0, self.total_cranes_limit))
                current_cranes = vessel.cranes_assigned
                delta = desired_cranes - current_cranes

                if delta > 0:
                    cranes_available = self.total_cranes_limit - self.cranes_in_use
                    if cranes_available >= delta:
                        vessel.cranes_assigned = desired_cranes
                        self.cranes_in_use += delta
                        vessel.max_cranes_assigned = max(
                            getattr(vessel, "max_cranes_assigned", 0),
                            vessel.cranes_assigned,
                        )
                    else:
                        invalid_action = True
                        invalid_reason = "insufficient_cranes_for_reallocation"
                elif delta < 0:
                    # Reducing assigned cranes always feasible; release to pool.
                    vessel.cranes_assigned = desired_cranes
                    self.cranes_in_use += delta
                # delta == 0: no change, valid action
            else:
                # Docking path: vessel must be waiting and already arrived.
                can_dock = (
                    vessel.status == "waiting"
                    and vessel.arrival_time <= self.current_step
                )

                if can_dock:
                    # Check the quay has enough consecutive free blocks
                    end_position = quay_position + vessel.length

                    if end_position <= self.quay_size:
                        # Check all blocks in that range are free
                        blocks_needed = self.quay_map[quay_position:end_position]
                        quay_is_free = np.all(blocks_needed == 0)

                        if quay_is_free:
                            # Check we have enough cranes available
                            cranes_available = self.total_cranes_limit - self.cranes_in_use
                            cranes_to_assign = min(cranes, cranes_available)

                            if cranes_to_assign > 0:
                                # --- DOCK THE VESSEL ---
                                vessel.status = "docked"
                                vessel.cranes_assigned = cranes_to_assign
                                vessel.max_cranes_assigned = cranes_to_assign
                                vessel.docking_position = quay_position  # Store where it's parked
                                vessel.docking_step = self.current_step

                                # Mark those quay blocks as occupied (binary occupancy map).
                                self.quay_map[quay_position:end_position] = 1.0
                                self.cranes_in_use += cranes_to_assign

                                # Small reward for successfully docking a ship
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

        # --- 3. PROCESS ALL DOCKED VESSELS (cranes work) ---
        for vessel in self.vessels:
            if vessel.status == "docked":
                # Each crane removes crane_rate containers this step
                containers_processed = vessel.cranes_assigned * self.crane_rate
                vessel.containers_remaining -= containers_processed
                vessel.containers_remaining = max(0, vessel.containers_remaining)

                # Reward for throughput — the more containers processed, the better
                reward += containers_processed * 0.001

        # --- 4. CHECK FOR DEPARTURES ---
        for vessel in self.vessels:
            if vessel.status == "docked" and vessel.is_finished():
                # Free up the quay blocks this vessel was occupying
                pos = vessel.docking_position
                self.quay_map[pos:pos + vessel.length] = 0

                # Return its cranes to the pool
                self.cranes_in_use -= vessel.cranes_assigned

                # Mark the vessel as departed
                vessel.status = "departed"
                vessel.cranes_assigned = 0
                vessel.departure_step = self.current_step

                # Reward for completing a vessel
                reward += 5.0

        # --- 5. PENALISE IDLE CRANES ---
        # If cranes are sitting unused while ships are waiting, that's bad
        idle_cranes = self.total_cranes_limit - self.cranes_in_use
        waiting_ships = sum(1 for v in self.vessels
                            if v.status == "waiting" and v.arrival_time <= self.current_step)
        if waiting_ships > 0 and idle_cranes > 0:
            reward -= idle_cranes * 0.1   # Small penalty per idle crane

        # --- 6. PENALISE QUEUEING DELAY ---
        # Penalise each arrived ship that is still waiting, plus additional pressure
        # when waiting duration exceeds a threshold.
        waiting_vessels = [
            v for v in self.vessels
            if v.status == "waiting" and v.arrival_time <= self.current_step
        ]
        if waiting_vessels:
            reward -= self.waiting_ship_penalty * len(waiting_vessels)
            for v in waiting_vessels:
                wait_duration = self.current_step - v.arrival_time
                excess_wait = max(0.0, wait_duration - self.long_wait_threshold)
                reward -= self.long_wait_penalty * excess_wait

        if invalid_action:
            reward -= self.invalid_action_penalty
            if self.terminate_on_invalid_action:
                terminated = True

        # --- 7. ADVANCE THE CLOCK ---
        self.current_step += 1

        # --- 8. CHECK IF THE WEEK IS OVER ---
        if self.current_step >= self.max_steps:
            terminated = True

        # Gymnasium also has 'truncated' (used for time limits set externally).
        # We handle our own time limit via terminated, so truncated is always False.
        truncated = False

        observation = self._get_observation()
        info = {
            "step": self.current_step,
            "cranes_in_use": self.cranes_in_use,
            "invalid_action": invalid_action,
            "invalid_reason": invalid_reason,
        }

        return observation, reward, terminated, truncated, info


# =============================================================================
# QUICK TEST — Run this file directly to check it works
# =============================================================================
if __name__ == "__main__":
    print("=== Testing AntwerpPortEnv ===\n")

    env = AntwerpPortEnv()

    # reset() starts a new week and gives us the first observation
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")       # Should be (80,)
    print(f"First 5 quay values: {obs[:5]}")       # Should all be 0.0 (empty quay)
    print(f"Vessels generated: {len(env.vessels)}")

    print("\nFirst 3 vessels:")
    for v in env.vessels[:3]:
        print(f"  {v}")

    # Take 3 random steps to check the engine runs without errors
    print("\nTaking 3 random steps...")
    for i in range(3):
        random_action = env.action_space.sample()   # Pick a random action
        obs, reward, terminated, truncated, info = env.step(random_action)
        print(f"  Step {i+1}: reward={reward:.3f}, terminated={terminated}, info={info}")

    print("\n✅ Environment runs without errors!")

