#!/usr/bin/env python3
"""
Record video rollouts for the best-performing checkpoint from each method
in logs/assault_all_methods_comparison (excluding dwn, winner_pays_others_reward,
and single_agent).
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from assault.assault_torch import AssaultConfig, AssaultEnv
from assault.assault_bidding_ppo import AssaultSharedAgent

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path("logs/assault_all_methods_comparison")

EXPERIMENTS = [
    {
        "name": "winner_pays_global_obs",
        "exp_dir": BASE_DIR / "assault_cmp_winner_pays_20260228_233104",
        "best_iter": 130,
    },
    {
        "name": "all_pay_global_obs",
        "exp_dir": BASE_DIR / "assault_cmp_all_pay_20260228_233104",
        "best_iter": 70,
    },
    {
        "name": "all_pay_local_obs",
        "exp_dir": BASE_DIR / "assault_ppo_multiagent_localobs_20260303_203429",
        "best_iter": 130,
    },
]

NUM_VIDEO_EPISODES = 2
# ============================================================================


def record_videos(exp_cfg: dict, device: torch.device) -> None:
    exp_dir = exp_cfg["exp_dir"]
    best_iter = exp_cfg["best_iter"]
    name = exp_cfg["name"]

    config_path = exp_dir / "config" / "training_config.json"
    model_path = exp_dir / "checkpoints" / f"iter_{best_iter}" / "agent.pt"
    output_dir = exp_dir / "rollout_videos"
    output_dir.mkdir(exist_ok=True)

    with open(config_path) as f:
        config = json.load(f)

    print(f"\n{'='*60}")
    print(f"Method: {name}  |  iter {best_iter}")
    print(f"Model:  {model_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    env_config = AssaultConfig(
        num_agents=config["num_agents"],
        max_enemies=config["max_enemies"],
        bid_upper_bound=config.get("bid_upper_bound", 0),
        bid_penalty=config.get("bid_penalty", 0.0),
        action_window=config.get("action_window", 1),
        window_bidding=config.get("window_bidding", False),
        window_penalty=config.get("window_penalty", 0.0),
        enemy_destroy_reward=config["enemy_destroy_reward"],
        hit_penalty=config["hit_penalty"],
        life_loss_penalty=config["life_loss_penalty"],
        raw_score_scale=config.get("raw_score_scale", 0.0),
        fire_while_hot_penalty=config.get("fire_while_hot_penalty", 0.0),
        max_steps=config["max_steps"],
        hud=config["hud"],
        single_agent_mode=False,
        allow_variable_enemies=config["allow_variable_enemies"],
        allow_sideward_fire=config.get("allow_sideward_fire", True),
        bidding_mechanism=config.get("bidding_mechanism", "all_pay"),
        only_own_enemy=config.get("only_own_enemy", False),
    )

    env = AssaultEnv(
        env_config,
        num_envs=1,
        device=device,
        seed=config["seed"],
        render_mode="rgb_array",
        render_oc_overlay=False,
    )

    agent = AssaultSharedAgent(
        obs_dim=env.per_agent_obs_dim,
        action_space_n=env.action_space_n,
        bid_upper_bound=config["bid_upper_bound"],
        window_bidding=config["window_bidding"],
        action_window=config["action_window"],
        actor_hidden_sizes=tuple(config["actor_hidden_sizes"]),
        critic_hidden_sizes=tuple(config["critic_hidden_sizes"]),
    ).to(device)

    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.eval()
    obs_dim = env.per_agent_obs_dim

    base_seed = config["seed"]
    for ep_idx in range(NUM_VIDEO_EPISODES):
        obs, _ = env.reset(seed=base_seed + ep_idx)
        done = False
        frames = []

        frame = env.render(env_idx=0, show_agent_overlay=True)
        if frame is not None:
            frames.append(frame)

        while not done:
            with torch.no_grad():
                flat_obs = obs.reshape(-1, obs_dim)
                action, _, _, _ = agent.get_action_and_value(flat_obs)
                action = action.reshape(1, config["num_agents"], -1)

            obs, reward, terminated, truncated, info = env.step(action)

            frame = env.render(env_idx=0, show_agent_overlay=True)
            if frame is not None:
                frames.append(frame)

            done = bool(terminated.item() or truncated.item())

        ep_score = float(info.get("score", torch.tensor(0.0)).item()) if isinstance(info, dict) else 0.0
        print(f"  Episode {ep_idx}: {len(frames)} frames, score={ep_score:.0f}")

        if frames:
            video_path = output_dir / f"ep_{ep_idx}.mp4"
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"  Saved: {video_path}")

    env.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for exp_cfg in EXPERIMENTS:
        record_videos(exp_cfg, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
