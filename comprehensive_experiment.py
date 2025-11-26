"""
Comprehensive training and evaluation script for Zero-Sum DQN on BiddingGridworld.

This script:
1. Trains a single agent using Zero-Sum DQN
2. Saves learned model and training logs
3. Records rollout videos/MP4s after training
4. Deploys multiple instances of the same agent to compete in the original environment
5. Records and saves competition rollouts

All outputs are organized in timestamped subdirectories within the logs folder.
"""

import os
import sys
import json
import time
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.bidding_gridworld import BiddingGridworld, MovingTargetBiddingGridworld
from src.zero_sum_wrapper import ZeroSumBiddingWrapper
from src.zero_sum_dqn import ZeroSumDQN, ZeroSumDQNPolicy

# Alias for cleaner code
ObsParser = ZeroSumBiddingWrapper


class ComprehensiveTrainer:
    """Comprehensive trainer for Zero-Sum DQN experiments."""
    
    def __init__(self, base_log_dir: str = "logs", experiment_name: str = "", use_moving_targets: bool = False):
        """Initialize the trainer with timestamped logging directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not experiment_name:
            target_type = "moving" if use_moving_targets else "static"
            experiment_name = f"15x15grid_3agents_6actwin_denserewards_expiry40_{target_type}_dqn"
        self.log_dir = Path(base_log_dir) / f"experiment_{experiment_name}_{timestamp}"
        self.use_moving_targets = use_moving_targets
        
        # Create subdirectories
        self.training_dir = self.log_dir / "training"
        self.models_dir = self.log_dir / "models"
        self.checkpoints_dir = self.log_dir / "checkpoints"
        self.rollouts_dir = self.log_dir / "rollouts"
        self.competition_dir = self.log_dir / "competition"
        self.plots_dir = self.log_dir / "plots"

        for dir_path in [self.training_dir, self.models_dir, self.checkpoints_dir,
                        self.rollouts_dir, self.competition_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Experiment directory: {self.log_dir}")
        
        # Training parameters
        self.grid_size = 15
        self.num_agents = 3
        self.bid_upper_bound = 4  # Increased for richer bidding strategy (0-4 = 5 values)
        self.training_timesteps = 2_000_000
        self.evaluation_interval = 200_000  # Evaluate and record every 200k steps
        self.learning_rate = 0.001  # Standard DQN learning rate
        self.discount_factor = 0.99
        self.epsilon = 0.5
        self.min_epsilon = 0.01

        # Environment parameters
        self.target_reward = 50.0  # Reward for reaching a target (increased for stronger signal)
        self.bid_penalty = 0.1  # Penalty multiplier for bids (reduced to encourage bidding)
        self.max_steps = 300  # Maximum steps per episode (default: grid_size * 10)
        self.action_window = 6  # Number of steps a winning agent controls the action
        self.distance_reward_scale = 0.1  # Reward scaling for distance improvements (dense reward)
        self.target_expiry_steps = 40  # Maximum steps before target expires (None = disabled)
        self.target_expiry_penalty = 50.0  # Penalty for not reaching target before expiry
        self.direction_change_prob = 0.1  # Probability of targets changing direction (moving targets only)
        self.target_move_interval = 1  # Number of steps between target movements (moving targets only)

        # Define target positions: corners and center
        self.target_positions = [
            (0, 5),  # Top-right
            (5, 0),  # Bottom-left
            (5, 5),  # Bottom-right
        ]
        
        # Save experiment configuration
        self.save_config()
    
    def save_config(self):
        """Save experiment configuration."""
        config = {
            "grid_size": self.grid_size,
            "num_agents": self.num_agents,
            "target_positions": self.target_positions,
            "bid_upper_bound": self.bid_upper_bound,
            "training_timesteps": self.training_timesteps,
            "evaluation_interval": self.evaluation_interval,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "target_reward": self.target_reward,
            "bid_penalty": self.bid_penalty,
            "max_steps": self.max_steps,
            "action_window": self.action_window,
            "distance_reward_scale": self.distance_reward_scale,
            "target_expiry_steps": self.target_expiry_steps,
            "target_expiry_penalty": self.target_expiry_penalty,
            "direction_change_prob": self.direction_change_prob,
            "target_move_interval": self.target_move_interval,
            "use_moving_targets": self.use_moving_targets,
            "timestamp": datetime.now().isoformat(),
            "description": f"Zero-Sum DQN: Single agent trained and deployed as {self.num_agents} instances on 15x15 grid" +
                          (" (moving targets)" if self.use_moving_targets else "") +
                          (f" with distance rewards (scale={self.distance_reward_scale})" if self.distance_reward_scale > 0 else "") +
                          f". Evaluations every {self.evaluation_interval} timesteps",
            "training_approach": "single_agent_multiple_instances"
        }

        with open(self.log_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def _get_env_kwargs(self, target_agent_id: Optional[int] = None, for_zero_sum_wrapper: bool = True) -> Dict:
        """
        Get environment kwargs for creating environments.

        Args:
            target_agent_id: Target agent ID for zero-sum wrapper (None for multi-agent env)
            for_zero_sum_wrapper: If True, includes wrapper-specific params. If False, returns kwargs for direct env creation.

        Returns:
            Dictionary of environment kwargs
        """
        env_kwargs = {
            "grid_size": self.grid_size,
            "num_agents": self.num_agents,
            "target_positions": self.target_positions,
            "bid_upper_bound": self.bid_upper_bound,
            "bid_penalty": self.bid_penalty,
            "target_reward": self.target_reward,
            "max_steps": self.max_steps,
            "action_window": self.action_window,
            "distance_reward_scale": self.distance_reward_scale,
            "target_expiry_steps": self.target_expiry_steps,
            "target_expiry_penalty": self.target_expiry_penalty,
        }

        # Add target_agent_id for zero-sum wrapper
        if for_zero_sum_wrapper and target_agent_id is not None:
            env_kwargs["target_agent_id"] = target_agent_id

        # Add environment class and moving target specific parameters
        if self.use_moving_targets:
            if for_zero_sum_wrapper:
                env_kwargs["env_class"] = MovingTargetBiddingGridworld
            env_kwargs["direction_change_prob"] = self.direction_change_prob
            env_kwargs["target_move_interval"] = self.target_move_interval

        return env_kwargs

    def train_agent(self, target_agent_id: int) -> Tuple[ZeroSumDQN, List[float], Dict]:
        """Train a single agent using Zero-Sum DQN."""
        print(f"\n{'='*60}")
        print(f"Training Agent {target_agent_id} (Target: {target_agent_id})")
        print(f"{'='*60}")

        # Create environment
        env_kwargs = self._get_env_kwargs(target_agent_id=target_agent_id, for_zero_sum_wrapper=True)
        env = ZeroSumBiddingWrapper(**env_kwargs)
        
        # Calculate action space size
        action_space_size = 4 * (self.bid_upper_bound + 1)
        
        # Network architecture parameters
        policy_kwargs = {
            "net_arch": [128, 128],  # Two hidden layers with 128 units each
            "activation_fn": torch.nn.ReLU,
            "normalize_images": False
        }
        
        # Create DQN agent
        agent = ZeroSumDQN(
            policy=ZeroSumDQNPolicy,
            env=env,
            protagonist_actions=action_space_size,
            adversary_actions=action_space_size,
            learning_rate=self.learning_rate,
            gamma=self.discount_factor,
            exploration_initial_eps=self.epsilon,
            exploration_final_eps=self.min_epsilon,
            learning_starts=1000,
            batch_size=32,
            buffer_size=100000,  # Larger buffer for better experience replay
            target_update_interval=1000,
            policy_kwargs=policy_kwargs,
            verbose=1
        )
        
        # Training tracking
        episode_rewards = []
        episode_lengths = []
        success_rates = []
        training_log = []
        
        print(f"Starting training for {self.training_timesteps} timesteps...")
        print(f"Evaluations and rollouts will be recorded every {self.evaluation_interval} steps")
        start_time = time.time()

        # Train in chunks with periodic evaluations
        num_chunks = self.training_timesteps // self.evaluation_interval
        timesteps_trained = 0

        for chunk in range(num_chunks):
            print(f"\n{'='*60}")
            print(f"Training chunk {chunk + 1}/{num_chunks} ({self.evaluation_interval} timesteps)")
            print(f"Total timesteps so far: {timesteps_trained}")
            print(f"{'='*60}")

            # Train for one chunk
            agent.learn(total_timesteps=self.evaluation_interval, reset_num_timesteps=False)
            timesteps_trained += self.evaluation_interval

            # Save intermediate model
            print(f"\nSaving model at {timesteps_trained} timesteps...")
            intermediate_model_path = self.models_dir / f"agent_{target_agent_id}_model_{timesteps_trained}.zip"
            agent.save(str(intermediate_model_path))

            # Record rollouts for this checkpoint
            print(f"\nRecording rollouts at {timesteps_trained} timesteps...")
            for target_id in range(self.num_agents):
                self.record_rollout(
                    agent,
                    target_id,
                    num_episodes=2,  # Fewer episodes for intermediate evaluations
                    filename_prefix=f"train_step{timesteps_trained}_rollout"
                )

            # Run cooperative evaluation
            print(f"\nRunning cooperative evaluation at {timesteps_trained} timesteps...")
            agents = [agent] * self.num_agents
            self.run_competition(
                agents,
                num_episodes=3,  # Fewer episodes for intermediate evaluations
                filename_prefix=f"train_step{timesteps_trained}"
            )

        # Evaluate trained agent to get episode rewards for plotting
        num_eval_episodes = 5
        for episode in range(num_eval_episodes):
            obs, info = env.reset()
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            
            while (not terminated) and (not truncated):
                action, _ = agent.predict(obs, deterministic=True)
                
                next_obs, reward, terminated, truncated, info = env.step(action[0])
                
                protagonist_reward = reward
                total_reward += protagonist_reward
                obs = next_obs
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Calculate success rate (moving window)
            window_size = min(100, episode + 1)
            recent_rewards = episode_rewards[-window_size:]
            success_rate = sum(1 for r in recent_rewards if r > 5) / len(recent_rewards)
            success_rates.append(success_rate)
            
            # Log progress every episode during evaluation
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)

            log_entry = {
                "episode": episode + 1,
                "avg_reward": avg_reward,
                "avg_length": avg_length,
                "success_rate": success_rate,
                "exploration_rate": agent.exploration_rate if hasattr(agent, 'exploration_rate') else 0.0,
                "num_timesteps": agent.num_timesteps if hasattr(agent, 'num_timesteps') else 0,
                "elapsed_time": time.time() - start_time
            }
            training_log.append(log_entry)

            print(f"Eval Episode {episode + 1}/{num_eval_episodes}: "
                  f"Reward: {total_reward:.2f}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Success Rate: {success_rate:.2%}")
        
        env.close()
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save training log
        log_file = self.training_dir / f"agent_{target_agent_id}_training_log.json"
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        # Save detailed metrics
        metrics = {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "success_rates": success_rates,
            "final_stats": {
                "num_timesteps": agent.num_timesteps if hasattr(agent, 'num_timesteps') else 0,
                "exploration_rate": agent.exploration_rate if hasattr(agent, 'exploration_rate') else 0.0,
                "learning_rate": agent.learning_rate if hasattr(agent, 'learning_rate') else self.learning_rate
            },
            "training_time": training_time
        }
        
        metrics_file = self.training_dir / f"agent_{target_agent_id}_metrics.pkl"
        with open(metrics_file, 'wb') as f:
            pickle.dump(metrics, f)
        
        return agent, episode_rewards, metrics
    
    def save_model(self, agent: ZeroSumDQN, target_agent_id: int):
        """Save trained model."""
        model_path = self.models_dir / f"agent_{target_agent_id}_model.zip"
        agent.save(str(model_path))
        print(f"Model saved: {model_path}")
    
    def plot_training_results(self, rewards_0: List[float], rewards_1: List[float]):
        """Plot training results for both agents."""
        # Check if we have enough data for meaningful plots
        min_episodes = 100
        if len(rewards_0) < min_episodes or len(rewards_1) < min_episodes:
            print(f"Skipping training plots: insufficient episodes ({len(rewards_0)} episodes, need {min_episodes}+)")
            print("Note: Training was done via DQN.learn() which doesn't provide per-episode rewards.")
            print("Eval episodes are too few for training progress visualization.")
            return

        print("Creating training plots...")

        def moving_average(data, window=100):
            return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]

        # Use last N episodes for distribution and comparison plots
        last_n = min(500, len(rewards_0))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Raw rewards
        episodes = range(len(rewards_0))
        ax1.plot(episodes, rewards_0, alpha=0.3, label='Agent 0', color='blue')
        ax1.plot(episodes, rewards_1, alpha=0.3, label='Agent 1', color='red')
        ax1.plot(episodes, moving_average(rewards_0), label='Agent 0 (MA)', color='blue', linewidth=2)
        ax1.plot(episodes, moving_average(rewards_1), label='Agent 1 (MA)', color='red', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress: Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Reward distributions
        ax2.hist(rewards_0[-last_n:], alpha=0.6, bins=20, label='Agent 0', color='blue')
        ax2.hist(rewards_1[-last_n:], alpha=0.6, bins=20, label='Agent 1', color='red')
        ax2.set_xlabel('Episode Reward')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Reward Distribution (Last {last_n} Episodes)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Success rates
        success_rates_0 = [sum(1 for r in rewards_0[max(0, i-99):i+1] if r > 5) / min(100, i+1) 
                          for i in range(len(rewards_0))]
        success_rates_1 = [sum(1 for r in rewards_1[max(0, i-99):i+1] if r > 5) / min(100, i+1) 
                          for i in range(len(rewards_1))]
        
        ax3.plot(episodes, success_rates_0, label='Agent 0', color='blue', linewidth=2)
        ax3.plot(episodes, success_rates_1, label='Agent 1', color='red', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Success Rate Over Training')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Comparative performance
        win_0 = sum(1 for r0, r1 in zip(rewards_0[-last_n:], rewards_1[-last_n:]) if r0 > r1)
        win_1 = sum(1 for r0, r1 in zip(rewards_0[-last_n:], rewards_1[-last_n:]) if r1 > r0)
        ties = last_n - win_0 - win_1
        
        ax4.bar(['Agent 0 Wins', 'Agent 1 Wins', 'Ties'], [win_0, win_1, ties],
                color=['blue', 'red', 'gray'], alpha=0.7)
        ax4.set_ylabel('Count')
        ax4.set_title(f'Head-to-Head Performance (Last {last_n} Episodes)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / "training_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved: {plot_path}")
    
    def record_rollout(self, agent: ZeroSumDQN, target_agent_id: int,
                      num_episodes: int = 5, filename_prefix: str = "rollout"):
        """Record rollout videos/MP4s for a trained agent."""
        print(f"Recording rollouts for Agent {target_agent_id}...")

        # Create environment
        env_kwargs = self._get_env_kwargs(target_agent_id=target_agent_id, for_zero_sum_wrapper=True)
        env = ZeroSumBiddingWrapper(**env_kwargs)

        for episode in range(num_episodes):
            states = []
            actions = []
            rewards = []
            step_details = []
            
            obs, info = env.reset()
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            
            # Record episode
            while (not terminated) and (not truncated):
                # obs format from zero-sum wrapper:
                # [agent_row, agent_col, target_row, target_col, target_reached, target_step_counter] (6 elements)

                # Extract positions using utility methods
                row, col = ObsParser.extract_agent_position(obs, self.grid_size)
                target_row = ObsParser.denormalize_position(obs[2], self.grid_size)
                target_col = ObsParser.denormalize_position(obs[3], self.grid_size)
                target_reached = int(obs[4])
                # target_step_counter = obs[5]  # Not used in visualization

                # For visualization, track all targets
                targets_reached = [0] * self.num_agents
                targets_reached[target_agent_id] = target_reached

                # Track actual target position (important for moving targets!)
                target_position = (target_row, target_col)

                states.append({
                    "agent_position": (row, col),
                    "target_position": target_position,  # Actual position from observation
                    "targets_reached": targets_reached,
                    "step": steps
                })
                
                action_idx, _ = agent.predict(obs, deterministic=True)
                action = action_idx[0]  # Extract the discrete action
                actions.append(action)
                
                next_obs, episode_reward, terminated, truncated, info = env.step(action)
                rewards.append(episode_reward)
                total_reward += episode_reward
                
                # Store step details including bids and actions for visualization
                # Convert discrete action back to readable format using utility methods
                prot_idx, adv_idx = ObsParser.split_discrete_action(action, env.actions_per_player)
                prot_action = ObsParser.action_idx_to_direction_bid(prot_idx, env._bid_upper_bound)
                adv_action = ObsParser.action_idx_to_direction_bid(adv_idx, env._bid_upper_bound)
                
                step_details.append({
                    "bids": info.get("bids", {}),
                    "winning_agent": info.get("winning_agent", -1),
                    "window_agent": info.get("window_agent", None),
                    "window_steps_remaining": info.get("window_steps_remaining", 0),
                    "bid_penalty_applied": info.get("bid_penalty_applied", False),
                    "protagonist_action": prot_action,
                    "adversary_action": adv_action
                })
                
                obs = next_obs
                steps += 1
            
            # Save episode data
            episode_data = {
                "target_agent_id": target_agent_id,
                "episode": episode,
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "step_details": step_details,
                "total_reward": total_reward,
                "success": total_reward > 5,
                "steps": steps
            }
            
            rollout_file = self.rollouts_dir / f"{filename_prefix}_agent_{target_agent_id}_ep_{episode}.json"
            with open(rollout_file, 'w') as f:
                json.dump(episode_data, f, indent=2, default=str)
            
            # Create visualization
            self.create_rollout_mp4(episode_data, rollout_file.with_suffix('.gif'))
        
        env.close()
    
    def _direction_to_string(self, direction: int) -> str:
        """Convert direction number to readable string."""
        # Must match BiddingGridworld: 0=Left, 1=Right, 2=Up, 3=Down
        directions = ["Left", "Right", "Up", "Down"]
        return directions[direction] if 0 <= direction <= 3 else "Unknown"
    
    def create_rollout_mp4(self, episode_data: Dict, gif_path: Path):
        """Create animated GIF of episode rollout."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            ax.clear()
            
            if frame >= len(episode_data["states"]):
                return
            
            state = episode_data["states"][frame]
            agent_pos = state["agent_position"]
            target_pos = state["target_position"]  # Actual target position from observation
            targets_reached = state["targets_reached"]
            target_id = episode_data["target_agent_id"]

            # Create grid
            ax.set_xlim(-0.5, self.grid_size - 0.5)
            ax.set_ylim(-0.5, self.grid_size - 0.5)
            ax.set_aspect('equal')

            # Draw grid lines
            for i in range(self.grid_size + 1):
                ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
                ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)

            # Draw the tracked target (protagonist's target) with actual position
            target_colors = ['lightblue', 'lightcoral', 'lightyellow']
            edge_colors = ['blue', 'red', 'orange']

            # Determine if this agent is currently in control
            winning_agent = None
            if frame < len(episode_data.get("step_details", [])):
                step_detail = episode_data["step_details"][frame]
                winning_agent = step_detail.get("winning_agent", -1)

            # Use thicker border and glow effect for controlling agent's target
            is_controlling = (winning_agent == target_id)
            edge_width = 4 if is_controlling else 2
            edge_color = 'gold' if is_controlling else edge_colors[target_id % 3]

            if targets_reached[target_id] == 0:
                ax.add_patch(plt.Rectangle((target_pos[1] - 0.4, target_pos[0] - 0.4),
                                         0.8, 0.8, facecolor=target_colors[target_id % 3],
                                         edgecolor=edge_color, linewidth=edge_width))
                ax.text(target_pos[1], target_pos[0], str(target_id), ha='center', va='center',
                       fontsize=12, fontweight='bold')
                # Add control indicator
                if is_controlling:
                    ax.text(target_pos[1], target_pos[0] - 0.6, '⚡', ha='center', va='center',
                           fontsize=10, color='gold')
            else:
                ax.add_patch(plt.Rectangle((target_pos[1] - 0.4, target_pos[0] - 0.4),
                                         0.8, 0.8, facecolor='lightgreen', edgecolor='green', linewidth=2))
                ax.text(target_pos[1], target_pos[0], '✓', ha='center', va='center', fontsize=12, fontweight='bold')

            # Draw agent with colored ring showing which target it's moving towards
            # Add outer ring in controlling agent's color
            if winning_agent is not None and winning_agent >= 0:
                ring_color = edge_colors[winning_agent % 3] if winning_agent != target_id else 'gold'
                ax.add_patch(plt.Circle((agent_pos[1], agent_pos[0]), 0.35,
                                       facecolor='none', edgecolor=ring_color, linewidth=3))

            ax.add_patch(plt.Circle((agent_pos[1], agent_pos[0]), 0.3, facecolor='yellow', edgecolor='orange'))
            ax.text(agent_pos[1], agent_pos[0], 'A', ha='center', va='center', fontsize=10, fontweight='bold')

            # Title and info
            target_id = episode_data["target_agent_id"]
            step = state["step"]
            reward = episode_data["rewards"][frame] if frame < len(episode_data["rewards"]) else 0

            # Add bid and action information if available
            title = f'Agent {target_id}'
            if self.use_moving_targets:
                title += ' (Moving Target)'
            title += f' - Step {step} - Reward: {reward:.2f}\n'
            title += f'Total Reward: {sum(episode_data["rewards"][:frame+1]):.2f}'

            if frame < len(episode_data.get("step_details", [])):
                step_detail = episode_data["step_details"][frame]
                prot_action = step_detail.get("protagonist_action", {})
                adv_action = step_detail.get("adversary_action", {})
                window_steps = step_detail.get("window_steps_remaining", 0)
                window_agent = step_detail.get("window_agent", None)
                bid_penalty_applied = step_detail.get("bid_penalty_applied", False)

                if prot_action and adv_action:
                    prot_dir = self._direction_to_string(prot_action.get("direction", -1))
                    prot_bid = prot_action.get("bid", 0)
                    adv_dir = self._direction_to_string(adv_action.get("direction", -1))
                    adv_bid = adv_action.get("bid", 0)

                    winner = step_detail.get("winning_agent", -1)

                    # Determine if this is a new bid or window continuation
                    if bid_penalty_applied:
                        # New bid just won
                        title += f'\n🎯 NEW BID: Protagonist: {prot_dir} ({prot_bid}) | Adversary: {adv_dir} ({adv_bid})'
                        if winner is None:
                            title += f'\n❌ No Movement (all bid 0)'
                        elif winner == target_id:
                            title += f'\n✅ Protagonist WINS! ⚡ (Window: {self.action_window} steps)'
                        else:
                            title += f'\n✅ Adversary WINS (Window: {self.action_window} steps)'
                    elif window_agent is not None:
                        # Continuing an action window
                        controller = "Protagonist" if window_agent == target_id else "Adversary"
                        controller_dir = prot_dir if window_agent == target_id else adv_dir
                        title += f'\n🔒 WINDOW CONTROL: {controller} moving {controller_dir}'
                        title += f'\n⏱️  Steps remaining in window: {window_steps}'
                        title += f'\n💭 Bids ignored (window active)'
                    else:
                        # Shouldn't happen, but fallback
                        title += f'\nProtagonist: {prot_dir} ({prot_bid}) | Adversary: {adv_dir} ({adv_bid})'
            
            ax.set_title(title, fontsize=12)
            
            ax.set_xticks(range(self.grid_size))
            ax.set_yticks(range(self.grid_size))
            ax.invert_yaxis()  # Make (0,0) top-left
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(episode_data["states"]) + 5, 
                                     interval=500, repeat=True)
        
        # Save GIF
        try:
            anim.save(str(gif_path), writer='pillow', fps=1)
            print(f"Rollout GIF saved: {gif_path}")
        except Exception as e:
            print(f"Warning: Could not save GIF {gif_path}: {e}")
        
        plt.close()
    
    def load_models(self) -> Tuple[ZeroSumDQN, ZeroSumDQN]:
        """Load both trained models."""
        print("Loading trained models...")
        
        # Load models using DQN.load
        agent_0 = ZeroSumDQN.load(str(self.models_dir / "agent_0_model.zip"))
        agent_1 = ZeroSumDQN.load(str(self.models_dir / "agent_1_model.zip"))
        
        return agent_0, agent_1
    
    def run_competition(self, agents: List[ZeroSumDQN],
                       num_episodes: int = 10, filename_prefix: str = ""):
        """Run cooperative evaluation where all agents try to reach their respective targets."""
        print(f"\n{'='*60}")
        print(f"COOPERATIVE EVALUATION: All {self.num_agents} Agents Pursuing Their Targets")
        print(f"{'='*60}")

        # Create environment based on moving targets flag
        env_class = MovingTargetBiddingGridworld if self.use_moving_targets else BiddingGridworld
        env_kwargs = self._get_env_kwargs(target_agent_id=None, for_zero_sum_wrapper=False)
        env = env_class(**env_kwargs)
        
        competition_results = []
        
        for episode in range(num_episodes):
            print(f"\nCooperative Episode {episode + 1}/{num_episodes}")
            
            episode_log = {
                "episode": episode,
                "states": [],
                "actions": [],
                "rewards": [],
                "step_details": []
            }
            
            obs, info = env.reset()
            total_rewards = {f"agent_{i}": 0 for i in range(self.num_agents)}
            steps = 0
            terminated = False
            truncated = False

            while (not terminated) and (not truncated):
                # obs from BiddingGridworld is (2 + 2*num_agents + num_agents + num_agents)-element array:
                # [agent_row_norm, agent_col_norm,
                #  target0_row_norm, target0_col_norm, ..., targetN_row_norm, targetN_col_norm,
                #  target0_reached, ..., targetN_reached,
                #  target0_step_counter_norm, ..., targetN_step_counter_norm]

                # Create wrapped observations for each agent using utility method
                wrapped_observations = []
                for agent_id in range(self.num_agents):
                    wrapped_obs = ObsParser.get_observation_for_agent(obs, agent_id, self.num_agents)
                    wrapped_observations.append(wrapped_obs)
                
                # Get actions from all agents pursuing their respective targets
                agent_actions = {}

                for agent_id in range(self.num_agents):
                    action_idx, _ = agents[agent_id].predict(wrapped_observations[agent_id], deterministic=True)

                    # Convert action index to direction/bid format using utility method
                    action_dict = ObsParser.extract_protagonist_action(
                        action_idx[0],
                        agents[agent_id].adversary_actions,
                        self.bid_upper_bound
                    )

                    agent_actions[f"agent_{agent_id}"] = action_dict

                # Format actions for original environment
                action = agent_actions
                
                # Execute step
                next_obs, rewards, terminated, truncated, step_info = env.step(action)

                # Extract position and target status from observation array using utility methods
                agent_row, agent_col = ObsParser.extract_agent_position(obs, self.grid_size)
                agent_position = agent_row * self.grid_size + agent_col

                # Extract all target reached statuses
                targets_reached = ObsParser.extract_target_reached_flags(obs, self.num_agents)

                # Log step details
                targets_status = {f"target_{i}_reached": bool(targets_reached[i]) for i in range(self.num_agents)}

                step_detail = {
                    "step": steps,
                    "agent_pos": agent_position,
                    "targets_reached": targets_reached.copy(),
                    "actions": action,
                    "rewards": rewards,
                    "winning_agent": step_info["winning_agent"],
                    "window_agent": step_info.get("window_agent", None),
                    "window_steps_remaining": step_info.get("window_steps_remaining", 0),
                    "bid_penalty_applied": step_info.get("bid_penalty_applied", False),
                    "bids": step_info["bids"],
                    "targets_status": targets_status
                }

                episode_log["states"].append(obs.copy())
                episode_log["actions"].append(action)
                episode_log["rewards"].append(rewards)
                episode_log["step_details"].append(step_detail)

                for i in range(self.num_agents):
                    total_rewards[f"agent_{i}"] += rewards[f"agent_{i}"]

                # Show which agent wins the bidding and current target status
                target_status_str = ", ".join([f"{i}={'✓' if targets_reached[i] else '✗'}" for i in range(self.num_agents)])
                bids_str = ", ".join([f"{i}={step_info['bids'][f'agent_{i}']}" for i in range(self.num_agents)])
                print(f"  Step {steps + 1}: Agent {step_info['winning_agent']} wins bid "
                      f"(Bids: {bids_str}) "
                      f"| Targets: {target_status_str} | Rewards: {rewards}")
                
                obs = next_obs
                steps += 1
                
                if terminated or truncated:
                    break
            
            # Episode summary - success is measured by target achievement
            # Extract final target status from observation array using utility method
            final_targets_reached = ObsParser.extract_target_reached_flags(obs, self.num_agents)

            all_targets_reached = all(t == 1 for t in final_targets_reached)
            individual_successes = {f"agent_{i}_success": final_targets_reached[i] == 1 for i in range(self.num_agents)}

            episode_log["total_rewards"] = total_rewards
            episode_log["all_targets_reached"] = all_targets_reached
            episode_log.update(individual_successes)
            episode_log["final_targets"] = final_targets_reached.copy()
            episode_log["steps"] = steps

            # Determine cooperation outcome
            num_successes = sum(individual_successes.values())
            if all_targets_reached:
                cooperation_outcome = "success"
            elif num_successes > 0:
                cooperation_outcome = "partial"
            else:
                cooperation_outcome = "failure"

            episode_log["cooperation_outcome"] = cooperation_outcome

            competition_results.append(episode_log)

            # Build outcome message
            if all_targets_reached:
                outcome_msg = "ALL TARGETS REACHED!"
            else:
                target_statuses = [f"Target {i}: {'✓' if final_targets_reached[i] else '✗'}" for i in range(self.num_agents)]
                outcome_msg = f"Partial success ({', '.join(target_statuses)})"

            print(f"  Episode Result: {outcome_msg}")
            rewards_str = ", ".join([f"Agent {i}: {total_rewards[f'agent_{i}']:.2f}" for i in range(self.num_agents)])
            print(f"  Final Scores - {rewards_str}")
            
            # Save individual episode
            prefix = f"{filename_prefix}_" if filename_prefix else ""
            episode_file = self.competition_dir / f"{prefix}cooperative_episode_{episode}.json"
            with open(episode_file, 'w') as f:
                json.dump(episode_log, f, indent=2, default=str)

            # Create cooperative rollout MP4
            self.create_competition_mp4(episode_log, episode_file.with_suffix('.gif'))
        
        env.close()
        
        # Save overall cooperation results
        overall_results = {
            "episodes": competition_results,
            "summary": self.analyze_cooperation_results(competition_results)
        }

        prefix = f"{filename_prefix}_" if filename_prefix else ""
        results_file = self.competition_dir / f"{prefix}cooperation_results.json"
        with open(results_file, 'w') as f:
            json.dump(overall_results, f, indent=2, default=str)

        print(f"\nCooperation results saved: {results_file}")
        return overall_results
    
    def analyze_cooperation_results(self, results: List[Dict]) -> Dict:
        """Analyze cooperation results focusing on target achievement."""
        all_targets_success = sum(1 for r in results if r["all_targets_reached"])

        # Count individual successes for each agent
        individual_successes = {}
        for i in range(self.num_agents):
            individual_successes[f"agent_{i}"] = sum(1 for r in results if r[f"agent_{i}_success"])

        # Complete failure = no targets reached
        complete_failures = sum(1 for r in results if sum(r[f"agent_{i}_success"] for i in range(self.num_agents)) == 0)

        # Collect rewards for each agent
        agent_rewards = {}
        for i in range(self.num_agents):
            agent_rewards[f"agent_{i}"] = [r["total_rewards"][f"agent_{i}"] for r in results]

        avg_steps = np.mean([r["steps"] for r in results])

        summary = {
            "total_episodes": len(results),
            "all_targets_success": all_targets_success,
            "all_targets_success_rate": all_targets_success / len(results),
            "complete_failures": complete_failures,
            "complete_failure_rate": complete_failures / len(results),
            "avg_steps_to_completion": avg_steps,
            "cooperation_efficiency": all_targets_success / len(results)  # Key metric for cooperation
        }

        # Add individual agent statistics
        for i in range(self.num_agents):
            agent_key = f"agent_{i}"
            summary[f"{agent_key}_individual_success"] = individual_successes[agent_key]
            summary[f"{agent_key}_success_rate"] = individual_successes[agent_key] / len(results)
            summary[f"{agent_key}_avg_reward"] = np.mean(agent_rewards[agent_key])
            summary[f"{agent_key}_total_reward"] = sum(agent_rewards[agent_key])

        print(f"\n{'='*60}")
        print("COOPERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Episodes: {summary['total_episodes']}")
        print(f"All Targets Reached: {summary['all_targets_success']} ({summary['all_targets_success_rate']:.1%})")

        for i in range(self.num_agents):
            print(f"Agent {i} Target Success: {summary[f'agent_{i}_individual_success']} ({summary[f'agent_{i}_success_rate']:.1%})")

        print(f"Complete Failures: {summary['complete_failures']} ({summary['complete_failure_rate']:.1%})")

        for i in range(self.num_agents):
            print(f"Agent {i} avg reward: {summary[f'agent_{i}_avg_reward']:.2f}")

        print(f"Average steps: {summary['avg_steps_to_completion']:.1f}")
        print(f"Cooperation Efficiency: {summary['cooperation_efficiency']:.1%}")

        return summary
    
    def create_competition_mp4(self, episode_data: Dict, gif_path: Path):
        """Create animated GIF of cooperative episode."""
        fig, ax = plt.subplots(figsize=(10, 8))

        def animate(frame):
            ax.clear()

            if frame >= len(episode_data["states"]):
                return

            state = episode_data["states"][frame]
            # state is now a numpy array from BiddingGridworld
            # [agent_row_norm, agent_col_norm, t0_row, t0_col, ..., tN_row, tN_col, t0_reached, ..., tN_reached]

            # Extract agent position using utility method
            agent_row, agent_col = ObsParser.extract_agent_position(state, self.grid_size)
            agent_pos = agent_row * self.grid_size + agent_col

            # Extract actual target positions from observation (important for moving targets!)
            target_positions = ObsParser.extract_all_target_positions(state, self.num_agents, self.grid_size)

            # Extract targets_reached for all agents
            targets_reached = ObsParser.extract_target_reached_flags(state, self.num_agents)

            if frame < len(episode_data["step_details"]):
                step_detail = episode_data["step_details"][frame]
                action = episode_data["actions"][frame]
                rewards = episode_data["rewards"][frame]
            else:
                step_detail = None
                action = None
                rewards = None

            # Create grid
            ax.set_xlim(-0.5, self.grid_size - 0.5)
            ax.set_ylim(-0.5, self.grid_size - 0.5)
            ax.set_aspect('equal')

            # Draw grid lines
            for i in range(self.grid_size + 1):
                ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
                ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)

            # Draw targets with actual positions from observation
            target_colors = ['lightblue', 'lightcoral', 'lightyellow']
            edge_colors = ['blue', 'red', 'orange']

            # Determine which agent is currently in control
            winning_agent = None
            if step_detail:
                winning_agent = step_detail.get("winning_agent", -1)

            for i in range(self.num_agents):
                target_pos = target_positions[i]  # Use actual position from observation

                # Highlight the controlling agent's target
                is_controlling = (winning_agent == i)
                edge_width = 4 if is_controlling else 2
                edge_color = 'gold' if is_controlling else edge_colors[i]

                if targets_reached[i] == 0:
                    ax.add_patch(plt.Rectangle((target_pos[1] - 0.4, target_pos[0] - 0.4),
                                             0.8, 0.8, facecolor=target_colors[i],
                                             edgecolor=edge_color, linewidth=edge_width))
                    ax.text(target_pos[1], target_pos[0], str(i), ha='center', va='center', fontsize=12, fontweight='bold')
                    # Add control indicator
                    if is_controlling:
                        ax.text(target_pos[1], target_pos[0] - 0.6, '⚡', ha='center', va='center',
                               fontsize=10, color='gold')
                else:
                    ax.add_patch(plt.Rectangle((target_pos[1] - 0.4, target_pos[0] - 0.4),
                                             0.8, 0.8, facecolor='lightgreen', edgecolor='green', linewidth=2))
                    ax.text(target_pos[1], target_pos[0], '✓', ha='center', va='center', fontsize=12, fontweight='bold')

            # Draw agent with colored ring showing which target it's moving towards
            row = agent_pos // self.grid_size
            col = agent_pos % self.grid_size

            # Add outer ring in controlling agent's color
            if winning_agent is not None and winning_agent >= 0 and winning_agent < self.num_agents:
                ring_color = edge_colors[winning_agent]
                ax.add_patch(plt.Circle((col, row), 0.35,
                                       facecolor='none', edgecolor=ring_color, linewidth=3))

            ax.add_patch(plt.Circle((col, row), 0.3, facecolor='yellow', edgecolor='orange'))
            ax.text(col, row, 'A', ha='center', va='center', fontsize=10, fontweight='bold')

            # Title and info - focus on cooperation
            title = f'Cooperative Episode {episode_data["episode"]}'
            if self.use_moving_targets:
                title += ' (Moving Targets)'
            title += f' - Step {frame}\n'

            if step_detail:
                window_steps = step_detail.get("window_steps_remaining", 0)
                window_agent = step_detail.get("window_agent", None)
                bid_penalty_applied = step_detail.get("bid_penalty_applied", False)

                # Show current target status
                targets_status = ", ".join([f"{i}={'✓' if targets_reached[i] else '✗'}" for i in range(self.num_agents)])

                # Determine if this is a new bid or window continuation
                if bid_penalty_applied:
                    # New bidding round
                    winner_text = f'Agent {step_detail["winning_agent"]}' if step_detail["winning_agent"] is not None else 'No Movement'
                    title += f'🎯 NEW BID - Winner: {winner_text} | Targets: {targets_status}\n'
                    title += f'Window starts: {self.action_window} steps\n'
                elif window_agent is not None:
                    # Continuing action window
                    title += f'🔒 WINDOW CONTROL - Agent {window_agent} | Targets: {targets_status}\n'
                    title += f'⏱️  Steps remaining: {window_steps}\n'
                else:
                    # Fallback
                    winner_text = f'Agent {step_detail["winning_agent"]}' if step_detail["winning_agent"] is not None else 'No Movement'
                    title += f'Winner: {winner_text} | Targets: {targets_status}\n'

                # Show actions and bids for all agents
                actions = step_detail.get("actions", {})
                action_strs = []
                for i in range(self.num_agents):
                    agent_key = f"agent_{i}"
                    if agent_key in actions:
                        agent_dir = self._direction_to_string(actions[agent_key].get("direction", -1))
                        agent_bid = actions[agent_key].get("bid", 0)
                        # Highlight the controlling agent's action during window
                        if window_agent == i and not bid_penalty_applied:
                            action_strs.append(f"A{i}: {agent_dir}(bid:{agent_bid})→ACTIVE")
                        else:
                            action_strs.append(f"A{i}: {agent_dir}({agent_bid})")

                if action_strs:
                    if bid_penalty_applied:
                        title += f'Actions & Bids: {" | ".join(action_strs)}\n'
                    else:
                        title += f'Actions (bids ignored): {" | ".join(action_strs)}\n'

                # Show cumulative rewards for all agents up to this frame
                if rewards:
                    cumulative_rewards = {}
                    for i in range(self.num_agents):
                        agent_key = f"agent_{i}"
                        cumulative_rewards[agent_key] = sum(
                            episode_data["rewards"][f].get(agent_key, 0)
                            for f in range(frame + 1)
                        )
                    rewards_strs = [f"{i}={cumulative_rewards[f'agent_{i}']:.2f}" for i in range(self.num_agents)]
                    title += f'Cumulative Rewards: {", ".join(rewards_strs)}'
            
            ax.set_title(title, fontsize=11)
            
            ax.set_xticks(range(self.grid_size))
            ax.set_yticks(range(self.grid_size))
            ax.invert_yaxis()  # Make (0,0) top-left
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(episode_data["states"]) + 3, 
                                     interval=800, repeat=True)
        
        # Save GIF
        try:
            anim.save(str(gif_path), writer='pillow', fps=1)
            print(f"Cooperative GIF saved: {gif_path}")
        except Exception as e:
            print(f"Warning: Could not save GIF {gif_path}: {e}")
        
        plt.close()
    
    def run_evaluation_only(self, model_dir: str):
        """Load existing model and run evaluation only."""
        print(f"\n{'='*80}")
        print("LOADING MODEL AND RUNNING EVALUATION")
        print(f"{'='*80}")

        start_time = time.time()

        # Update models_dir to point to existing models
        self.models_dir = Path(model_dir) / "models"

        # Load single model
        print("\nPHASE 1: LOADING MODEL")
        model_path = self.models_dir / "agent_0_model.zip"
        print(f"Loading {model_path}...")
        agent = ZeroSumDQN.load(str(model_path))

        # Record rollouts for each target
        print("\nPHASE 2: RECORDING ROLLOUTS")
        print(f"Recording rollouts of the agent pursuing each target")
        for target_id in range(self.num_agents):
            self.record_rollout(agent, target_id, num_episodes=3, filename_prefix="eval_rollout")

        # Run competition (deploy multiple instances of same agent)
        print("\nPHASE 3: RUNNING COMPETITION")
        print(f"Deploying {self.num_agents} instances of the same trained agent")
        agents = [agent] * self.num_agents  # Use same agent multiple times
        competition_results = self.run_competition(agents, num_episodes=5)

        # Final summary
        total_time = time.time() - start_time

        final_summary = {
            "evaluation_completed": True,
            "total_time": total_time,
            "loaded_from": model_dir,
            "log_directory": str(self.log_dir),
            "cooperation_episodes": 5,
            "cooperation_summary": competition_results["summary"],
            "agent_deployment": "single_agent_multiple_instances",
            "num_instances": self.num_agents
        }

        summary_file = self.log_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)

        print(f"\n{'='*80}")
        print("EVALUATION COMPLETED!")
        print(f"{'='*80}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Results saved in: {self.log_dir}")
        print(f"Summary saved: {summary_file}")
        print(f"{'='*80}")

    def run_full_experiment(self):
        """Run the complete experiment pipeline."""
        print(f"\n{'='*80}")
        print("STARTING COMPREHENSIVE ZERO-SUM DQN EXPERIMENT")
        print(f"{'='*80}")

        start_time = time.time()

        # Phase 1: Train ONE agent (will be deployed as multiple instances)
        print("\nPHASE 1: TRAINING SINGLE AGENT")
        print("Training one agent that will be deployed as multiple instances during competition")
        agent, rewards, metrics = self.train_agent(target_agent_id=0)

        # Phase 2: Save final model
        print("\nPHASE 2: SAVING FINAL MODEL")
        self.save_model(agent, 0)

        # Note: Rollouts and cooperative evaluations were performed during training
        # every 200,000 timesteps. Results are saved in the rollouts/ and competition/ directories.

        # Phase 3: Final summary
        total_time = time.time() - start_time

        final_summary = {
            "experiment_completed": True,
            "total_time": total_time,
            "log_directory": str(self.log_dir),
            "training_timesteps": self.training_timesteps,
            "evaluation_interval": self.evaluation_interval,
            "num_evaluations": self.training_timesteps // self.evaluation_interval,
            "agent_deployment": "single_agent_multiple_instances",
            "num_instances": self.num_agents,
            "agent_final_performance": metrics["final_stats"],
            "note": f"Rollouts and cooperative evaluations performed every {self.evaluation_interval} timesteps during training"
        }

        summary_file = self.log_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)

        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"All results saved in: {self.log_dir}")
        print(f"Summary saved: {summary_file}")
        print(f"{'='*80}")


def main():
    """Main function to run the comprehensive experiment."""
    import argparse

    parser = argparse.ArgumentParser(description='Run Zero-Sum DQN experiment')
    parser.add_argument('--eval-only', type=str, default=None,
                       help='Path to existing experiment directory to load models from and run evaluation only')
    parser.add_argument('--moving-targets', action='store_true',
                       help='Use moving targets gridworld instead of static targets')
    parser.add_argument('--experiment-name', type=str, default="",
                       help='Custom experiment name (default: auto-generated)')

    args = parser.parse_args()

    trainer = ComprehensiveTrainer(
        experiment_name=args.experiment_name,
        use_moving_targets=args.moving_targets
    )

    if args.eval_only:
        trainer.run_evaluation_only(args.eval_only)
    else:
        trainer.run_full_experiment()


if __name__ == "__main__":
    main()
