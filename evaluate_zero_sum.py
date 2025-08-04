#!/usr/bin/env python3
"""
Evaluation script for trained Zero-Sum Q-Learning agents.

This script loads trained models and evaluates their performance
in the BiddingGridworld environment.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.zero_sum_wrapper import ZeroSumBiddingWrapper
from src.zero_sum_qlearning import ZeroSumQLearning


def evaluate_agent(
    agent_path: str,
    target_agent_id: int,
    episodes: int = 100,
    grid_size: int = 5,
    bid_upper_bound: int = 3,
    verbose: bool = False
) -> dict:
    """
    Evaluate a trained agent.
    
    Args:
        agent_path: Path to the saved agent model
        target_agent_id: Which agent (0 or 1) this is
        episodes: Number of evaluation episodes
        grid_size: Size of the gridworld
        bid_upper_bound: Maximum bid value
        verbose: Whether to print detailed episode information
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating Agent {target_agent_id} from {agent_path}...")
    
    # Create environment
    env = ZeroSumBiddingWrapper(
        target_agent_id=target_agent_id,
        grid_size=grid_size,
        bid_upper_bound=bid_upper_bound,
        bid_penalty=0.1,
        target_reward=10.0
    )
    
    # Load agent
    agent = ZeroSumQLearning.load_model(agent_path)
    
    episode_rewards = []
    episode_lengths = []
    target_reached = []
    total_bids = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        episode_bid_total = 0
        max_steps = 100
        
        if verbose and episode < 5:
            print(f"\nEpisode {episode + 1}:")
            print(f"Initial state: {obs}")
        
        while steps < max_steps:
            # Get action from agent (no exploration during evaluation)
            action = agent.get_action(obs, bid_upper_bound, training=False)
            
            if verbose and episode < 5:
                direction = action // (bid_upper_bound + 1)
                bid = action % (bid_upper_bound + 1)
                directions = ['Up', 'Down', 'Left', 'Right']
                print(f"  Step {steps + 1}: {directions[direction]}, Bid {bid}")
            
            # Take step
            next_obs, rewards, terminated, truncated, info = env.step(action)
            
            protagonist_reward = rewards["protagonist"]
            total_reward += protagonist_reward
            episode_bid_total += action % (bid_upper_bound + 1)  # Extract bid amount
            
            if verbose and episode < 5:
                print(f"    Reward: {protagonist_reward}, Terminated: {terminated}")
            
            obs = next_obs
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        target_reached.append(total_reward > 5)  # Assume positive reward means target reached
        total_bids.append(episode_bid_total)
    
    env.close()
    
    # Calculate metrics
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(target_reached),
        'mean_total_bid': np.mean(total_bids),
        'episode_rewards': episode_rewards
    }
    
    return metrics


def compare_agents(
    agent_0_path: str,
    agent_1_path: str,
    episodes: int = 100,
    grid_size: int = 5,
    bid_upper_bound: int = 3
) -> None:
    """Compare performance of both trained agents."""
    
    print("=== Agent Comparison ===\n")
    
    # Evaluate Agent 0
    metrics_0 = evaluate_agent(
        agent_0_path, 0, episodes, grid_size, bid_upper_bound
    )
    
    # Evaluate Agent 1
    metrics_1 = evaluate_agent(
        agent_1_path, 1, episodes, grid_size, bid_upper_bound
    )
    
    # Print comparison
    print("\n=== Evaluation Results ===")
    print(f"{'Metric':<20} {'Agent 0':<12} {'Agent 1':<12}")
    print("-" * 45)
    print(f"{'Mean Reward':<20} {metrics_0['mean_reward']:<12.2f} {metrics_1['mean_reward']:<12.2f}")
    print(f"{'Std Reward':<20} {metrics_0['std_reward']:<12.2f} {metrics_1['std_reward']:<12.2f}")
    print(f"{'Success Rate':<20} {metrics_0['success_rate']:<12.2%} {metrics_1['success_rate']:<12.2%}")
    print(f"{'Mean Length':<20} {metrics_0['mean_length']:<12.1f} {metrics_1['mean_length']:<12.1f}")
    print(f"{'Mean Total Bid':<20} {metrics_0['mean_total_bid']:<12.1f} {metrics_1['mean_total_bid']:<12.1f}")
    print(f"{'Min Reward':<20} {metrics_0['min_reward']:<12.2f} {metrics_1['min_reward']:<12.2f}")
    print(f"{'Max Reward':<20} {metrics_0['max_reward']:<12.2f} {metrics_1['max_reward']:<12.2f}")
    
    # Statistical significance test (simple)
    from scipy import stats
    
    try:
        t_stat, p_value = stats.ttest_ind(
            metrics_0['episode_rewards'], 
            metrics_1['episode_rewards']
        )
        print(f"\nStatistical test (reward difference):")
        print(f"T-statistic: {t_stat:.3f}")
        print(f"P-value: {p_value:.3f}")
        if p_value < 0.05:
            print("Significant difference in performance (p < 0.05)")
        else:
            print("No significant difference in performance (p >= 0.05)")
    except ImportError:
        print("\nSciPy not available for statistical tests")


def single_episode_demo(
    agent_path: str,
    target_agent_id: int,
    grid_size: int = 5,
    bid_upper_bound: int = 3
) -> None:
    """Run a single episode with detailed output for demonstration."""
    
    print(f"\n=== Single Episode Demo: Agent {target_agent_id} ===")
    
    # Create environment
    env = ZeroSumBiddingWrapper(
        target_agent_id=target_agent_id,
        grid_size=grid_size,
        bid_upper_bound=bid_upper_bound,
        bid_penalty=0.1,
        target_reward=10.0
    )
    
    # Load agent
    agent = ZeroSumQLearning.load_model(agent_path)
    
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    max_steps = 50
    
    print(f"Initial observation: {obs}")
    print(f"Agent {target_agent_id} is trying to reach their target\n")
    
    directions = ['Up', 'Down', 'Left', 'Right']
    
    while steps < max_steps:
        action = agent.get_action(obs, bid_upper_bound, training=False)
        direction = action // (bid_upper_bound + 1)
        bid = action % (bid_upper_bound + 1)
        
        print(f"Step {steps + 1}:")
        print(f"  Action: {directions[direction]}, Bid: {bid}")
        
        next_obs, rewards, terminated, truncated, info = env.step(action)
        
        protagonist_reward = rewards["protagonist"]
        total_reward += protagonist_reward
        
        print(f"  Reward: {protagonist_reward}")
        print(f"  New observation: {next_obs}")
        print(f"  Terminated: {terminated}")
        
        if terminated or truncated:
            print(f"\nEpisode finished after {steps + 1} steps")
            print(f"Total reward: {total_reward}")
            break
        
        obs = next_obs
        steps += 1
        print()
    
    env.close()


def main():
    """Main evaluation function."""
    
    # Check if models exist
    agent_0_path = 'models/agent_0_zerosum_q.pkl'
    agent_1_path = 'models/agent_1_zerosum_q.pkl'
    
    if not os.path.exists(agent_0_path) or not os.path.exists(agent_1_path):
        print("Error: Trained models not found!")
        print("Please run train_zero_sum.py first to train the agents.")
        return
    
    print("=== Zero-Sum Q-Learning Evaluation ===")
    
    # Compare agents
    compare_agents(agent_0_path, agent_1_path, episodes=200)
    
    # Single episode demos
    single_episode_demo(agent_0_path, 0)
    single_episode_demo(agent_1_path, 1)


if __name__ == "__main__":
    main()
