#!/usr/bin/env python3
"""Record a single rollout video from a specific checkpoint."""

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bidding_gridworld.bidding_gridworld_torch import (
    BiddingGridworld,
    BiddingGridworldConfig,
    evaluate_multi_agent_policy,
)
from bidding_gridworld.bidding_ppo import SharedAgent

EXP_DIR = Path("logs/gridworld_all_methods_comparison/bidding_cmp_all_pay_20260228_232515")
ITER = 390
OUTPUT_PATH = EXP_DIR / "rollouts" / f"iter_{ITER}_ep_0_new.mp4"
FPS = 2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config_path = EXP_DIR / "config" / "training_config.json"
    checkpoint_path = EXP_DIR / "checkpoints" / f"iter_{ITER}" / "agent.pt"

    with open(config_path) as f:
        cfg = json.load(f)

    env_config = BiddingGridworldConfig(
        grid_size=cfg["grid_size"],
        num_agents=cfg["num_agents"],
        bid_upper_bound=cfg["bid_upper_bound"],
        bid_penalty=cfg["bid_penalty"],
        target_reward=cfg["target_reward"],
        max_steps=cfg["max_steps"],
        action_window=cfg["action_window"],
        distance_reward_scale=cfg["distance_reward_scale"],
        target_expiry_steps=cfg["target_expiry_steps"],
        target_expiry_penalty=cfg["target_expiry_penalty"],
        moving_targets=cfg["moving_targets"],
        direction_change_prob=cfg["direction_change_prob"],
        target_move_interval=cfg["target_move_interval"],
        window_bidding=cfg["window_bidding"],
        window_penalty=cfg["window_penalty"],
        visible_targets=cfg["visible_targets"],
        single_agent_mode=False,
        bidding_mechanism=cfg["bidding_mechanism"],
    )
    env = BiddingGridworld(env_config, num_envs=1, device=device, seed=cfg["seed"])

    include_reached = not cfg["moving_targets"]
    agent = SharedAgent(
        obs_dim=env.per_agent_obs_dim,
        num_actions_per_agent=3 if cfg["window_bidding"] else 2,
        window_bidding=cfg["window_bidding"],
        actor_hidden_sizes=tuple(cfg["actor_hidden_sizes"]),
        critic_hidden_sizes=tuple(cfg["critic_hidden_sizes"]),
        use_target_attention_pooling=cfg["use_target_attention_pooling"],
        target_embed_dim=cfg["target_embed_dim"],
        target_encoder_hidden_sizes=tuple(cfg["target_encoder_hidden_sizes"]),
        include_target_reached=include_reached,
    ).to(device)

    agent.set_bid_head(cfg["bid_upper_bound"])
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    agent.eval()

    def policy_fn(obs):
        obs_tensor = obs if torch.is_tensor(obs) else torch.tensor(obs, dtype=torch.float32)
        obs_tensor = obs_tensor.to(device)
        n_agents, obs_dim = obs_tensor.shape
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        return action

    eval_stats = evaluate_multi_agent_policy(
        env=env,
        policy_fn=policy_fn,
        num_episodes=1,
        target_expiry_penalty=cfg["target_expiry_penalty"],
        verbose=True,
    )

    episode_data = eval_stats["episode_data_list"][0]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    env.create_competition_gif(episode_data, OUTPUT_PATH, fps=FPS)
    print(f"Saved: {OUTPUT_PATH}")

    env.close()


if __name__ == "__main__":
    main()
