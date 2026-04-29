import os
import sys
import unittest

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from airraid.airraid_torch import AirRaidConfig, AirRaidEnv


class AirRaidEnvTests(unittest.TestCase):
    def test_airraid_single_agent_reset_and_step_shapes(self):
        env = AirRaidEnv(
            AirRaidConfig(num_agents=3, single_agent_mode=True, max_steps=20),
            num_envs=2,
            device=torch.device("cpu"),
            seed=0,
        )
        try:
            obs, info = env.reset()
            self.assertEqual(obs.shape, (2, env.obs_shape[1]))
            self.assertEqual(info, {})
            action = torch.zeros((2,), dtype=torch.int64)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            self.assertEqual(next_obs.shape, obs.shape)
            self.assertEqual(reward.shape, (2,))
            self.assertEqual(terminated.shape, (2,))
            self.assertEqual(truncated.shape, (2,))
            self.assertIn("reward_components", step_info)
        finally:
            env.close()

    def test_airraid_multi_agent_reset_and_step_shapes(self):
        env = AirRaidEnv(
            AirRaidConfig(num_agents=3, max_enemies=3, single_agent_mode=False, max_steps=20),
            num_envs=2,
            device=torch.device("cpu"),
            seed=1,
        )
        try:
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (2, 3, env.per_agent_obs_dim))
            action = torch.zeros((2, 3, 2), dtype=torch.int64)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            self.assertEqual(next_obs.shape, obs.shape)
            self.assertEqual(reward.shape, (2, 3))
            self.assertEqual(terminated.shape, (2,))
            self.assertEqual(truncated.shape, (2,))
            self.assertEqual(step_info["enemy_visible"].shape, (2, 3))
            self.assertIn("reward_no_bid_sum", step_info)
        finally:
            env.close()

    def test_airraid_visible_object_positions_stay_in_bounds(self):
        env = AirRaidEnv(
            AirRaidConfig(num_agents=3, single_agent_mode=True, max_steps=100),
            num_envs=1,
            device=torch.device("cpu"),
            seed=2,
        )
        try:
            env.reset()
            max_visible_enemies = 0
            for _ in range(120):
                _, _, terminated, truncated, info = env.step(torch.tensor([0], dtype=torch.int64))
                enemy_visible = info["enemy_visible"][0]
                enemy_raw = info["enemy_raw"][0]
                max_visible_enemies = max(max_visible_enemies, int(enemy_visible.sum().item()))
                for slot in range(enemy_visible.shape[0]):
                    if enemy_visible[slot].item():
                        x, y = enemy_raw[slot].tolist()
                        self.assertGreaterEqual(x, 0)
                        self.assertLessEqual(x, 159)
                        self.assertGreaterEqual(y, 0)
                        self.assertLess(y, 210)
                player_missile = info["player_missile_raw"][0].tolist()
                if player_missile != [0.0, 0.0]:
                    self.assertGreaterEqual(player_missile[0], 0)
                    self.assertLessEqual(player_missile[0], 159)
                    self.assertGreaterEqual(player_missile[1], 0)
                    self.assertLess(player_missile[1], 210)
                enemy_missile = info["enemy_missile_raw"][0].tolist()
                if enemy_missile != [0.0, 0.0]:
                    self.assertGreaterEqual(enemy_missile[0], 0)
                    self.assertLessEqual(enemy_missile[0], 159)
                    self.assertGreaterEqual(enemy_missile[1], 0)
                    self.assertLess(enemy_missile[1], 210)
                if terminated.item() or truncated.item():
                    env.partial_reset(terminated | truncated)
            self.assertLessEqual(max_visible_enemies, 3)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
