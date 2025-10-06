"""
Comprehensive training and evaluation script for Zero-Sum DQN on BiddingGridworld.

This script:
1. Trains both agents using Zero-Sum DQN
2. Saves learned models and training logs
3. Records rollout videos/MP4s after training
4. Loads both policies and makes them compete in the original environment
5. Records and saves competition rollouts

All outputs are organized in timestamped subdirectories within the logs folder.
"""

import os
import sys
import json
import time
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.bidding_gridworld import BiddingGridworld
from src.zero_sum_wrapper import ZeroSumBiddingWrapper
from src.zero_sum_dqn import ZeroSumDQN, ZeroSumDQNPolicy


class ComprehensiveTrainer:
    """Comprehensive trainer for Zero-Sum DQN experiments."""
    
    def __init__(self, base_log_dir: str = "logs", experiment_name: str = ""):
        """Initialize the trainer with timestamped logging directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = experiment_name if experiment_name else "3x3grid_10000_iters_dqn_exp1"
        self.log_dir = Path(base_log_dir) / f"experiment_{experiment_name}_{timestamp}"
        
        # Create subdirectories
        self.training_dir = self.log_dir / "training"
        self.models_dir = self.log_dir / "models"
        self.rollouts_dir = self.log_dir / "rollouts"
        self.competition_dir = self.log_dir / "competition"
        self.plots_dir = self.log_dir / "plots"
        
        for dir_path in [self.training_dir, self.models_dir, self.rollouts_dir, 
                        self.competition_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Experiment directory: {self.log_dir}")
        
        # Training parameters
        self.grid_size = 3
        self.bid_upper_bound = 2
        self.training_timesteps = 10000
        self.learning_rate = 0.05
        self.discount_factor = 0.95
        self.epsilon = 0.5
        self.min_epsilon = 0.1
        
        # Save experiment configuration
        self.save_config()
    
    def save_config(self):
        """Save experiment configuration."""
        config = {
            "grid_size": self.grid_size,
            "bid_upper_bound": self.bid_upper_bound,
            "training_timesteps": self.training_timesteps,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "timestamp": datetime.now().isoformat(),
            "description": "Zero-Sum DQN training and competition experiment"
        }
        
        with open(self.log_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_agent(self, target_agent_id: int) -> Tuple[ZeroSumDQN, List[float], Dict]:
        """Train a single agent using Zero-Sum DQN."""
        print(f"\n{'='*60}")
        print(f"Training Agent {target_agent_id} (Target: {target_agent_id})")
        print(f"{'='*60}")
        
        # Create environment
        env = ZeroSumBiddingWrapper(
            target_agent_id=target_agent_id,
            grid_size=self.grid_size,
            bid_upper_bound=self.bid_upper_bound,
            bid_penalty=0.1,
            target_reward=10.0
        )
        
        # Calculate action space size
        action_space_size = 4 * (self.bid_upper_bound + 1)
        
        # Network architecture parameters
        policy_kwargs = {
            "net_arch": [64, 64],  # Two hidden layers with 128 units each
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
            buffer_size=50000,
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
        start_time = time.time()

        # Learn using the DQN's learn method
        agent.learn(total_timesteps=self.training_timesteps)

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
        print("Creating training plots...")
        
        def moving_average(data, window=100):
            return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]
        
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
        ax2.hist(rewards_0[-500:], alpha=0.6, bins=20, label='Agent 0', color='blue')
        ax2.hist(rewards_1[-500:], alpha=0.6, bins=20, label='Agent 1', color='red')
        ax2.set_xlabel('Episode Reward')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Reward Distribution (Last 500 Episodes)')
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
        win_0 = sum(1 for r0, r1 in zip(rewards_0[-500:], rewards_1[-500:]) if r0 > r1)
        win_1 = sum(1 for r0, r1 in zip(rewards_0[-500:], rewards_1[-500:]) if r1 > r0)
        ties = 500 - win_0 - win_1
        
        ax4.bar(['Agent 0 Wins', 'Agent 1 Wins', 'Ties'], [win_0, win_1, ties], 
                color=['blue', 'red', 'gray'], alpha=0.7)
        ax4.set_ylabel('Count')
        ax4.set_title('Head-to-Head Performance (Last 500 Episodes)')
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
        
        env = ZeroSumBiddingWrapper(
            target_agent_id=target_agent_id,
            grid_size=self.grid_size,
            bid_upper_bound=self.bid_upper_bound,
            bid_penalty=0.1,
            target_reward=10.0
        )
        
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
                # obs is now a numpy array: [agent_row, agent_col, target_row, target_col, target_reached]
                agent_row_norm = obs[0]
                agent_col_norm = obs[1]
                target_reached = int(obs[4])

                # Denormalize positions
                denom = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
                row = int(agent_row_norm * denom)
                col = int(agent_col_norm * denom)

                # For visualization, track both targets
                targets_reached = [0, 0]
                targets_reached[target_agent_id] = target_reached

                states.append({
                    "agent_position": (row, col),
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
                # Convert discrete action back to readable format using wrapper's conversion
                if hasattr(env, '_discrete_to_dict_action'):
                    action_dict = env._discrete_to_dict_action(action)
                    prot_action = action_dict["protagonist"]
                    adv_action = action_dict["adversary"] 
                else:
                    # Fallback - reconstruct from discrete action
                    actions_per_player = env.actions_per_player
                    prot_idx = action // actions_per_player
                    adv_idx = action % actions_per_player
                    directions_per_bid = env._bid_upper_bound + 1
                    prot_action = {
                        "direction": prot_idx // directions_per_bid,
                        "bid": prot_idx % directions_per_bid
                    }
                    adv_action = {
                        "direction": adv_idx // directions_per_bid,
                        "bid": adv_idx % directions_per_bid
                    }
                
                step_details.append({
                    "bids": info.get("bids", {}),
                    "winning_agent": info.get("winning_agent", -1),
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
            self.create_rollout_mp4(episode_data, rollout_file.with_suffix('.mp4'))
        
        env.close()
    
    def _direction_to_string(self, direction: int) -> str:
        """Convert direction number to readable string."""
        # Must match BiddingGridworld: 0=Left, 1=Right, 2=Up, 3=Down
        directions = ["Left", "Right", "Up", "Down"]
        return directions[direction] if 0 <= direction <= 3 else "Unknown"
    
    def create_rollout_mp4(self, episode_data: Dict, mp4_path: Path):
        """Create animated MP4 of episode rollout."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            ax.clear()
            
            if frame >= len(episode_data["states"]):
                return
            
            state = episode_data["states"][frame]
            agent_pos = state["agent_position"]
            targets_reached = state["targets_reached"]
            
            # Create grid
            ax.set_xlim(-0.5, self.grid_size - 0.5)
            ax.set_ylim(-0.5, self.grid_size - 0.5)
            ax.set_aspect('equal')
            
            # Draw grid lines
            for i in range(self.grid_size + 1):
                ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
                ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)
            
            # Draw targets
            target_0_pos = (self.grid_size - 1, self.grid_size - 1)  # Bottom-right
            target_1_pos = (0, self.grid_size - 1)  # Top-right
            
            if targets_reached[0] == 0:
                ax.add_patch(plt.Rectangle((target_0_pos[1] - 0.4, target_0_pos[0] - 0.4), 
                                         0.8, 0.8, facecolor='lightblue', edgecolor='blue'))
                ax.text(target_0_pos[1], target_0_pos[0], '0', ha='center', va='center', fontsize=12, fontweight='bold')
            else:
                ax.add_patch(plt.Rectangle((target_0_pos[1] - 0.4, target_0_pos[0] - 0.4), 
                                         0.8, 0.8, facecolor='lightgreen', edgecolor='green'))
                ax.text(target_0_pos[1], target_0_pos[0], '✓', ha='center', va='center', fontsize=12, fontweight='bold')
            
            if targets_reached[1] == 0:
                ax.add_patch(plt.Rectangle((target_1_pos[1] - 0.4, target_1_pos[0] - 0.4), 
                                         0.8, 0.8, facecolor='lightcoral', edgecolor='red'))
                ax.text(target_1_pos[1], target_1_pos[0], '1', ha='center', va='center', fontsize=12, fontweight='bold')
            else:
                ax.add_patch(plt.Rectangle((target_1_pos[1] - 0.4, target_1_pos[0] - 0.4), 
                                         0.8, 0.8, facecolor='lightgreen', edgecolor='green'))
                ax.text(target_1_pos[1], target_1_pos[0], '✓', ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Draw agent
            ax.add_patch(plt.Circle((agent_pos[1], agent_pos[0]), 0.3, facecolor='yellow', edgecolor='orange'))
            ax.text(agent_pos[1], agent_pos[0], 'A', ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Title and info
            target_id = episode_data["target_agent_id"]
            step = state["step"]
            reward = episode_data["rewards"][frame] if frame < len(episode_data["rewards"]) else 0
            
            # Add bid and action information if available
            title = f'Agent {target_id} - Step {step} - Reward: {reward:.2f}\n'
            title += f'Total Reward: {sum(episode_data["rewards"][:frame+1]):.2f}'
            
            if frame < len(episode_data.get("step_details", [])):
                step_detail = episode_data["step_details"][frame]
                prot_action = step_detail.get("protagonist_action", {})
                adv_action = step_detail.get("adversary_action", {})
                
                if prot_action and adv_action:
                    prot_dir = self._direction_to_string(prot_action.get("direction", -1))
                    prot_bid = prot_action.get("bid", 0)
                    adv_dir = self._direction_to_string(adv_action.get("direction", -1))
                    adv_bid = adv_action.get("bid", 0)
                    
                    winner = step_detail.get("winning_agent", -1)
                    title += f'\nProtagonist: {prot_dir} (bid: {prot_bid}) | Adversary: {adv_dir} (bid: {adv_bid})'
                    title += f'\nWinner: {"Protagonist" if winner == target_id else "Adversary" if winner != -1 else "Tie"}'
            
            ax.set_title(title, fontsize=12)
            
            ax.set_xticks(range(self.grid_size))
            ax.set_yticks(range(self.grid_size))
            ax.invert_yaxis()  # Make (0,0) top-left
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(episode_data["states"]) + 5, 
                                     interval=500, repeat=True)
        
        # Save MP4
        try:
            anim.save(str(mp4_path), writer='ffmpeg', fps=1)
            print(f"Rollout MP4 saved: {mp4_path}")
        except Exception as e:
            print(f"Warning: Could not save MP4 {mp4_path}: {e}")
        
        plt.close()
    
    def load_models(self) -> Tuple[ZeroSumDQN, ZeroSumDQN]:
        """Load both trained models."""
        print("Loading trained models...")
        
        # Load models using DQN.load
        agent_0 = ZeroSumDQN.load(str(self.models_dir / "agent_0_model.zip"))
        agent_1 = ZeroSumDQN.load(str(self.models_dir / "agent_1_model.zip"))
        
        return agent_0, agent_1
    
    def run_competition(self, agent_0: ZeroSumDQN, agent_1: ZeroSumDQN, 
                       num_episodes: int = 10):
        """Run cooperative evaluation where both agents try to reach their respective targets."""
        print(f"\n{'='*60}")
        print("COOPERATIVE EVALUATION: Both Agents Pursuing Their Targets")
        print(f"{'='*60}")
        
        env = BiddingGridworld(
            grid_size=self.grid_size,
            bid_upper_bound=self.bid_upper_bound,
            bid_penalty=0.1,
            target_reward=10.0
        )
        
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
            total_rewards = {"agent_0": 0, "agent_1": 0}
            steps = 0
            terminated = False
            truncated = False
            
            while (not terminated) and (not truncated):
                # obs from BiddingGridworld is 8-element array:
                # [agent_row_norm, agent_col_norm, target0_row_norm, target0_col_norm,
                #  target1_row_norm, target1_col_norm, target0_reached, target1_reached]

                # For agent 0: create observation focused on target 0
                # Format: [agent_row, agent_col, target_row, target_col, target_reached]
                wrapped_obs_0 = np.array([
                    obs[0],  # agent_row_norm
                    obs[1],  # agent_col_norm
                    obs[2],  # target0_row_norm
                    obs[3],  # target0_col_norm
                    obs[6],  # target0_reached
                ], dtype=np.float32)

                # For agent 1: create observation focused on target 1
                wrapped_obs_1 = np.array([
                    obs[0],  # agent_row_norm
                    obs[1],  # agent_col_norm
                    obs[4],  # target1_row_norm
                    obs[5],  # target1_col_norm
                    obs[7],  # target1_reached
                ], dtype=np.float32)
                
                # Get actions from both agents pursuing their respective targets
                action_0_idx, _ = agent_0.predict(wrapped_obs_0, deterministic=True)
                action_1_idx, _ = agent_1.predict(wrapped_obs_1, deterministic=True)
                
                # Convert action indices to direction/bid format
                directions_per_bid = self.bid_upper_bound + 1
                
                # Agent 0 action
                prot_idx_0 = action_0_idx[0] // agent_0.adversary_actions
                prot_direction_0 = prot_idx_0 // directions_per_bid
                prot_bid_0 = prot_idx_0 % directions_per_bid
                agent_0_action = {"direction": prot_direction_0, "bid": prot_bid_0}
                
                # Agent 1 action  
                prot_idx_1 = action_1_idx[0] // agent_1.adversary_actions
                prot_direction_1 = prot_idx_1 // directions_per_bid
                prot_bid_1 = prot_idx_1 % directions_per_bid
                agent_1_action = {"direction": prot_direction_1, "bid": prot_bid_1}
                
                # Format actions for original environment
                action = {
                    "agent_0": agent_0_action,
                    "agent_1": agent_1_action
                }
                
                # Execute step
                next_obs, rewards, terminated, truncated, step_info = env.step(action)

                # Extract position and target status from observation array
                # obs: [agent_row_norm, agent_col_norm, t0_row, t0_col, t1_row, t1_col, t0_reached, t1_reached]
                denom = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
                agent_row = int(obs[0] * denom)
                agent_col = int(obs[1] * denom)
                agent_position = agent_row * self.grid_size + agent_col
                targets_reached = [int(obs[6]), int(obs[7])]

                # Log step details
                step_detail = {
                    "step": steps,
                    "agent_pos": agent_position,
                    "targets_reached": targets_reached.copy(),
                    "actions": action,
                    "rewards": rewards,
                    "winning_agent": step_info["winning_agent"],
                    "bids": step_info["bids"],
                    "targets_status": {
                        "target_0_reached": bool(targets_reached[0]),
                        "target_1_reached": bool(targets_reached[1])
                    }
                }

                episode_log["states"].append(obs.copy())
                episode_log["actions"].append(action)
                episode_log["rewards"].append(rewards)
                episode_log["step_details"].append(step_detail)

                total_rewards["agent_0"] += rewards["agent_0"]
                total_rewards["agent_1"] += rewards["agent_1"]

                # Show which agent wins the bidding and current target status
                target_status = f"Targets: 0={'✓' if targets_reached[0] else '✗'}, 1={'✓' if targets_reached[1] else '✗'}"
                print(f"  Step {steps + 1}: Agent {step_info['winning_agent']} wins bid "
                      f"(Bids: 0={step_info['bids']['agent_0']}, 1={step_info['bids']['agent_1']}) "
                      f"| {target_status} | Rewards: {rewards}")
                
                obs = next_obs
                steps += 1
                
                if terminated or truncated:
                    break
            
            # Episode summary - success is measured by target achievement
            # Extract final target status from observation array
            final_targets_reached = [int(obs[6]), int(obs[7])]
            both_targets_reached = final_targets_reached[0] == 1 and final_targets_reached[1] == 1
            agent_0_success = final_targets_reached[0] == 1
            agent_1_success = final_targets_reached[1] == 1
            
            episode_log["total_rewards"] = total_rewards
            episode_log["both_targets_reached"] = both_targets_reached
            episode_log["agent_0_success"] = agent_0_success
            episode_log["agent_1_success"] = agent_1_success
            episode_log["final_targets"] = final_targets_reached.copy()
            episode_log["steps"] = steps
            episode_log["cooperation_outcome"] = "success" if both_targets_reached else "partial" if (agent_0_success or agent_1_success) else "failure"
            
            competition_results.append(episode_log)
            
            outcome_msg = f"BOTH TARGETS REACHED!" if both_targets_reached else \
                         f"Partial success (Target 0: {'✓' if agent_0_success else '✗'}, Target 1: {'✓' if agent_1_success else '✗'})"
            
            print(f"  Episode Result: {outcome_msg}")
            print(f"  Final Scores - Agent 0: {total_rewards['agent_0']:.2f}, Agent 1: {total_rewards['agent_1']:.2f}")
            
            # Save individual episode
            episode_file = self.competition_dir / f"cooperative_episode_{episode}.json"
            with open(episode_file, 'w') as f:
                json.dump(episode_log, f, indent=2, default=str)
            
            # Create cooperative rollout MP4
            self.create_competition_mp4(episode_log, episode_file.with_suffix('.mp4'))
        
        env.close()
        
        # Save overall cooperation results
        overall_results = {
            "episodes": competition_results,
            "summary": self.analyze_cooperation_results(competition_results)
        }
        
        results_file = self.competition_dir / "cooperation_results.json"
        with open(results_file, 'w') as f:
            json.dump(overall_results, f, indent=2, default=str)
        
        print(f"\nCooperation results saved: {results_file}")
        return overall_results
    
    def analyze_cooperation_results(self, results: List[Dict]) -> Dict:
        """Analyze cooperation results focusing on target achievement."""
        both_targets_success = sum(1 for r in results if r["both_targets_reached"])
        agent_0_successes = sum(1 for r in results if r["agent_0_success"])
        agent_1_successes = sum(1 for r in results if r["agent_1_success"])
        complete_failures = sum(1 for r in results if not r["agent_0_success"] and not r["agent_1_success"])
        
        agent_0_rewards = [r["total_rewards"]["agent_0"] for r in results]
        agent_1_rewards = [r["total_rewards"]["agent_1"] for r in results]
        
        avg_steps = np.mean([r["steps"] for r in results])
        
        summary = {
            "total_episodes": len(results),
            "both_targets_success": both_targets_success,
            "both_targets_success_rate": both_targets_success / len(results),
            "agent_0_individual_success": agent_0_successes,
            "agent_0_success_rate": agent_0_successes / len(results),
            "agent_1_individual_success": agent_1_successes,
            "agent_1_success_rate": agent_1_successes / len(results),
            "complete_failures": complete_failures,
            "complete_failure_rate": complete_failures / len(results),
            "agent_0_avg_reward": np.mean(agent_0_rewards),
            "agent_1_avg_reward": np.mean(agent_1_rewards),
            "agent_0_total_reward": sum(agent_0_rewards),
            "agent_1_total_reward": sum(agent_1_rewards),
            "avg_steps_to_completion": avg_steps,
            "cooperation_efficiency": both_targets_success / len(results)  # Key metric for cooperation
        }
        
        print(f"\n{'='*60}")
        print("COOPERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Episodes: {summary['total_episodes']}")
        print(f"Both Targets Reached: {summary['both_targets_success']} ({summary['both_targets_success_rate']:.1%})")
        print(f"Agent 0 Target Success: {summary['agent_0_individual_success']} ({summary['agent_0_success_rate']:.1%})")
        print(f"Agent 1 Target Success: {summary['agent_1_individual_success']} ({summary['agent_1_success_rate']:.1%})")
        print(f"Complete Failures: {summary['complete_failures']} ({summary['complete_failure_rate']:.1%})")
        print(f"Agent 0 avg reward: {summary['agent_0_avg_reward']:.2f}")
        print(f"Agent 1 avg reward: {summary['agent_1_avg_reward']:.2f}")
        print(f"Average steps: {summary['avg_steps_to_completion']:.1f}")
        print(f"Cooperation Efficiency: {summary['cooperation_efficiency']:.1%}")
        
        return summary
    
    def create_competition_mp4(self, episode_data: Dict, mp4_path: Path):
        """Create animated MP4 of cooperative episode."""
        fig, ax = plt.subplots(figsize=(10, 8))

        def animate(frame):
            ax.clear()

            if frame >= len(episode_data["states"]):
                return

            state = episode_data["states"][frame]
            # state is now a numpy array from BiddingGridworld
            # [agent_row_norm, agent_col_norm, t0_row, t0_col, t1_row, t1_col, t0_reached, t1_reached]
            denom = float(self.grid_size - 1) if self.grid_size > 1 else 1.0
            agent_row = int(state[0] * denom)
            agent_col = int(state[1] * denom)
            agent_pos = agent_row * self.grid_size + agent_col
            targets_reached = [int(state[6]), int(state[7])]
            
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
            
            # Draw targets
            target_0_pos = (self.grid_size - 1, self.grid_size - 1)  # Bottom-right
            target_1_pos = (0, self.grid_size - 1)  # Top-right
            
            if targets_reached[0] == 0:
                ax.add_patch(plt.Rectangle((target_0_pos[1] - 0.4, target_0_pos[0] - 0.4), 
                                         0.8, 0.8, facecolor='lightblue', edgecolor='blue'))
                ax.text(target_0_pos[1], target_0_pos[0], '0', ha='center', va='center', fontsize=12, fontweight='bold')
            else:
                ax.add_patch(plt.Rectangle((target_0_pos[1] - 0.4, target_0_pos[0] - 0.4), 
                                         0.8, 0.8, facecolor='lightgreen', edgecolor='green'))
                ax.text(target_0_pos[1], target_0_pos[0], '✓', ha='center', va='center', fontsize=12, fontweight='bold')
            
            if targets_reached[1] == 0:
                ax.add_patch(plt.Rectangle((target_1_pos[1] - 0.4, target_1_pos[0] - 0.4), 
                                         0.8, 0.8, facecolor='lightcoral', edgecolor='red'))
                ax.text(target_1_pos[1], target_1_pos[0], '1', ha='center', va='center', fontsize=12, fontweight='bold')
            else:
                ax.add_patch(plt.Rectangle((target_1_pos[1] - 0.4, target_1_pos[0] - 0.4), 
                                         0.8, 0.8, facecolor='lightgreen', edgecolor='green'))
                ax.text(target_1_pos[1], target_1_pos[0], '✓', ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Draw agent
            row = agent_pos // self.grid_size
            col = agent_pos % self.grid_size
            ax.add_patch(plt.Circle((col, row), 0.3, facecolor='yellow', edgecolor='orange'))
            ax.text(col, row, 'A', ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Title and info - focus on cooperation
            title = f'Cooperative Episode {episode_data["episode"]} - Step {frame}\n'
            
            if step_detail:
                # Show current target status
                targets_status = f"Targets: 0={'✓' if targets_reached[0] else '✗'}, 1={'✓' if targets_reached[1] else '✗'}"
                title += f'Movement Winner: Agent {step_detail["winning_agent"]} | {targets_status}\n'
                
                # Show actions and bids for both agents
                actions = step_detail.get("actions", {})
                if "agent_0" in actions and "agent_1" in actions:
                    agent_0_dir = self._direction_to_string(actions["agent_0"].get("direction", -1))
                    agent_0_bid = actions["agent_0"].get("bid", 0)
                    agent_1_dir = self._direction_to_string(actions["agent_1"].get("direction", -1))
                    agent_1_bid = actions["agent_1"].get("bid", 0)
                    
                    title += f'Agent 0: {agent_0_dir} (bid: {agent_0_bid}) | Agent 1: {agent_1_dir} (bid: {agent_1_bid})\n'
                else:
                    title += f'Bids: 0={step_detail["bids"]["agent_0"]}, 1={step_detail["bids"]["agent_1"]} | '
                
                title += f'Rewards: 0={rewards["agent_0"]:.2f}, 1={rewards["agent_1"]:.2f}'
            
            ax.set_title(title, fontsize=11)
            
            ax.set_xticks(range(self.grid_size))
            ax.set_yticks(range(self.grid_size))
            ax.invert_yaxis()  # Make (0,0) top-left
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(episode_data["states"]) + 3, 
                                     interval=800, repeat=True)
        
        # Save MP4
        try:
            anim.save(str(mp4_path), writer='ffmpeg', fps=1)
            print(f"Cooperative MP4 saved: {mp4_path}")
        except Exception as e:
            print(f"Warning: Could not save MP4 {mp4_path}: {e}")
        
        plt.close()
    
    def run_full_experiment(self):
        """Run the complete experiment pipeline."""
        print(f"\n{'='*80}")
        print("STARTING COMPREHENSIVE ZERO-SUM DQN EXPERIMENT")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Phase 1: Train both agents
        print("\nPHASE 1: TRAINING")
        agent_0, rewards_0, metrics_0 = self.train_agent(target_agent_id=0)
        agent_1, rewards_1, metrics_1 = self.train_agent(target_agent_id=1)
        
        # Phase 2: Save models
        print("\nPHASE 2: SAVING MODELS")
        self.save_model(agent_0, 0)
        self.save_model(agent_1, 1)
        
        # Phase 3: Create training plots
        print("\nPHASE 3: CREATING TRAINING PLOTS")
        self.plot_training_results(rewards_0, rewards_1)
        
        # Phase 4: Record rollouts
        print("\nPHASE 4: RECORDING TRAINING ROLLOUTS")
        self.record_rollout(agent_0, 0, num_episodes=3, filename_prefix="training_rollout")
        self.record_rollout(agent_1, 1, num_episodes=3, filename_prefix="training_rollout")
        
        # Phase 5: Cooperative evaluation
        print("\nPHASE 5: RUNNING COOPERATIVE EVALUATION")
        competition_results = self.run_competition(agent_0, agent_1, num_episodes=5)
        
        # Phase 6: Final summary
        total_time = time.time() - start_time
        
        final_summary = {
            "experiment_completed": True,
            "total_time": total_time,
            "log_directory": str(self.log_dir),
            "training_timesteps": self.training_timesteps,
            "cooperation_episodes": 5,
            "agent_0_final_performance": metrics_0["final_stats"],
            "agent_1_final_performance": metrics_1["final_stats"],
            "cooperation_summary": competition_results["summary"]
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
    trainer = ComprehensiveTrainer()
    trainer.run_full_experiment()


if __name__ == "__main__":
    main()
