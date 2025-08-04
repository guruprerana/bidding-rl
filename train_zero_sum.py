#!/usr/bin/env python3
"""
Training script for Zero-Sum Q-Learning on BiddingGridworld.

This script trains decentralized policies for the two agents by solving
separate zero-sum games for each agent using the ZeroSumQLearning algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.zero_sum_wrapper import ZeroSumBiddingWrapper
from src.zero_sum_qlearning import ZeroSumQLearning


def train_agent(
    target_agent_id: int,
    episodes: int = 1000,
    grid_size: int = 5,
    bid_upper_bound: int = 3,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon: float = 0.3,
    save_path: str = None
) -> Tuple[ZeroSumQLearning, List[float]]:
    """
    Train a single agent using Zero-Sum Q-Learning.
    
    Args:
        target_agent_id: Which agent (0 or 1) to train
        episodes: Number of training episodes
        grid_size: Size of the gridworld
        bid_upper_bound: Maximum bid value
        learning_rate: Learning rate for Q-learning
        discount_factor: Discount factor
        epsilon: Initial exploration rate
        save_path: Path to save the trained model
    
    Returns:
        Trained ZeroSumQLearning agent and episode rewards
    """
    print(f"Training Agent {target_agent_id}...")
    
    # Create environment
    env = ZeroSumBiddingWrapper(
        target_agent_id=target_agent_id,
        grid_size=grid_size,
        bid_upper_bound=bid_upper_bound,
        bid_penalty=0.1,
        target_reward=10.0
    )
    
    # Calculate action space size: 4 directions * (bid_upper_bound + 1) bids
    action_space_size = 4 * (bid_upper_bound + 1)
    observation_space_size = grid_size * grid_size * 2 * (grid_size * grid_size)  # Rough estimate
    
    # Create Q-learning agent
    agent = ZeroSumQLearning(
        protagonist_action_space_size=action_space_size,
        adversary_action_space_size=action_space_size,
        observation_space_size=observation_space_size,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon
    )
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100  # Prevent infinite episodes
        
        while steps < max_steps:
            # Get action from agent
            action = agent.get_action(obs, bid_upper_bound, training=True)
            
            # Take step in environment
            next_obs, rewards, terminated, truncated, info = env.step(action)
            
            # Update agent with protagonist's reward
            protagonist_reward = rewards["protagonist"]
            agent.update(obs, action, protagonist_reward, next_obs, terminated)
            
            total_reward += protagonist_reward
            obs = next_obs
            steps += 1
            
            if terminated or truncated:
                break
        
        # End episode
        agent.end_episode()
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            stats = agent.get_stats()
            print(f"Episode {episode + 1}/{episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Length: {avg_length:.1f}, "
                  f"Epsilon: {stats['epsilon']:.3f}, "
                  f"Q-table size: {stats['q_table_size']}")
    
    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agent.save_model(save_path)
    
    env.close()
    return agent, episode_rewards


def plot_training_results(
    agent_0_rewards: List[float], 
    agent_1_rewards: List[float],
    save_path: str = None
) -> None:
    """Plot training results for both agents."""
    
    def moving_average(data, window=100):
        """Compute moving average."""
        return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot raw rewards
    ax1.plot(agent_0_rewards, alpha=0.3, label='Agent 0 (raw)', color='blue')
    ax1.plot(agent_1_rewards, alpha=0.3, label='Agent 1 (raw)', color='red')
    
    # Plot moving averages
    ma_0 = moving_average(agent_0_rewards)
    ma_1 = moving_average(agent_1_rewards)
    ax1.plot(ma_0, label='Agent 0 (MA)', color='blue', linewidth=2)
    ax1.plot(ma_1, label='Agent 1 (MA)', color='red', linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Progress: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot reward distributions
    ax2.hist(agent_0_rewards[-500:], alpha=0.6, bins=20, label='Agent 0 (last 500)', color='blue')
    ax2.hist(agent_1_rewards[-500:], alpha=0.6, bins=20, label='Agent 1 (last 500)', color='red')
    ax2.set_xlabel('Episode Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution (Last 500 Episodes)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    """Main training loop for both agents."""
    print("=== Zero-Sum Q-Learning Training for BiddingGridworld ===\n")
    
    # Training parameters
    EPISODES = 2000
    GRID_SIZE = 5
    BID_UPPER_BOUND = 3
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.95
    EPSILON = 0.3
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train Agent 0
    print("Phase 1: Training Agent 0 (Target 0)")
    agent_0, rewards_0 = train_agent(
        target_agent_id=0,
        episodes=EPISODES,
        grid_size=GRID_SIZE,
        bid_upper_bound=BID_UPPER_BOUND,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=EPSILON,
        save_path='models/agent_0_zerosum_q.pkl'
    )
    
    print(f"Agent 0 training completed. Final stats: {agent_0.get_stats()}")
    
    # Train Agent 1
    print("\nPhase 2: Training Agent 1 (Target 1)")
    agent_1, rewards_1 = train_agent(
        target_agent_id=1,
        episodes=EPISODES,
        grid_size=GRID_SIZE,
        bid_upper_bound=BID_UPPER_BOUND,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=EPSILON,
        save_path='models/agent_1_zerosum_q.pkl'
    )
    
    print(f"Agent 1 training completed. Final stats: {agent_1.get_stats()}")
    
    # Plot results
    print("\nPlotting training results...")
    plot_training_results(rewards_0, rewards_1, save_path='training_results.png')
    
    # Summary
    print("\n=== Training Summary ===")
    print(f"Agent 0 - Final average reward (last 100): {np.mean(rewards_0[-100:]):.2f}")
    print(f"Agent 1 - Final average reward (last 100): {np.mean(rewards_1[-100:]):.2f}")
    print(f"Agent 0 - Success rate (last 100): {np.mean([r > 5 for r in rewards_0[-100:]]):.2%}")
    print(f"Agent 1 - Success rate (last 100): {np.mean([r > 5 for r in rewards_1[-100:]]):.2%}")
    
    print("\nTraining completed! Models saved in 'models/' directory.")


if __name__ == "__main__":
    main()
