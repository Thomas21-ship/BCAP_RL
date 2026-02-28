import unittest

import numpy as np

from port_env import AntwerpPortEnv
from ship_generator import generate_single_vessel


class TestPortEnv(unittest.TestCase):
    def test_action_decode_mapping_and_noop(self):
        env = AntwerpPortEnv()
        seen = set()

        for action in range(env.action_space.n):
            vessel_slot, quay_pos, cranes = env._decode_action(action)
            if action == env.no_op_action:
                self.assertEqual(vessel_slot, env.no_op_slot)
                self.assertEqual(quay_pos, 0)
                self.assertEqual(cranes, 0)
            else:
                self.assertGreaterEqual(vessel_slot, 0)
                self.assertLess(vessel_slot, env.num_vessel_slots)
                self.assertGreaterEqual(quay_pos, 0)
                self.assertLess(quay_pos, env.quay_size)
                self.assertGreaterEqual(cranes, 0)
                self.assertLessEqual(cranes, env.total_cranes_limit)
                seen.add((vessel_slot, quay_pos, cranes))

        self.assertEqual(len(seen), env.num_docking_actions)

    def test_observation_shape_and_bounds(self):
        env = AntwerpPortEnv()

        for _ in range(5):
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (80,))
            self.assertGreaterEqual(float(obs.min()), 0.0)
            self.assertLessEqual(float(obs.max()), 1.0)

            done = False
            while not done:
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                self.assertEqual(obs.shape, (80,))
                self.assertGreaterEqual(float(obs.min()), 0.0)
                self.assertLessEqual(float(obs.max()), 1.0)
                done = terminated or truncated

    def test_crane_conservation(self):
        env = AntwerpPortEnv()
        env.reset()

        for _ in range(300):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            self.assertGreaterEqual(env.cranes_in_use, 0)
            self.assertLessEqual(env.cranes_in_use, env.total_cranes_limit)
            if terminated or truncated:
                break

    def test_terminates_at_max_steps(self):
        env = AntwerpPortEnv()
        env.reset()

        steps = 0
        while True:
            _, _, terminated, truncated, info = env.step(env.no_op_action)
            steps += 1
            if terminated or truncated:
                break

        self.assertEqual(steps, env.max_steps)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["step"], env.max_steps)


class TestGenerator(unittest.TestCase):
    def test_generate_single_vessel_ranges(self):
        last_arrival = 0.0
        for _ in range(200):
            length, workload, arrival_time = generate_single_vessel(last_arrival)
            self.assertGreaterEqual(length, 4.0)
            self.assertLessEqual(length, 20.0)
            self.assertGreaterEqual(workload, 100.0)
            self.assertLessEqual(workload, 5000.0)
            self.assertGreaterEqual(arrival_time, last_arrival)
            last_arrival = arrival_time


if __name__ == "__main__":
    unittest.main()
