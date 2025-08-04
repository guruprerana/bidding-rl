#!/usr/bin/env python3
"""
Example demonstrating Zero-Sum Q-Learning with longer training.

This script shows how to train both agents and save their models.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.zero_sum_wrapper import ZeroSumBiddingWrapper
from src.zero_sum_qlearning import ZeroSumQLearning


def train_agent_example(target_agent_id: int, episodes: int = 500):
    """Train an agent with the Zero-Sum Q-Learning algorithm."""
    print(f"\nTraining Agent {target_agent_id} for {episodes} episodes...")
    
    # Create environment
    env = ZeroSumBiddingWrapper(
        target_agent_id=target_agent_id,
        grid_size=5,
        bid_upper_bound=3,
        bid_penalty=0.1,
        target_reward=10.0
    )
    
    # Calculate action space size: 4 directions * (bid_upper_bound + 1) bids
    action_space_size = 4 * (3 + 1)  # 16 actions
    observation_space_size = 25 * 2 * 25  # Rough estimate for discrete space
    
    # Create Q-learning agent
    agent = ZeroSumQLearning(
        protagonist_action_space_size=action_space_size,
        adversary_action_space_size=action_space_size,
        observation_space_size=observation_space_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3
    )
    
    episode_rewards = []
    successful_episodes = 0
    
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100
        
        while steps < max_steps:
            # Get action from agent
            action = agent.get_action(obs, 3, training=True)  # bid_upper_bound = 3
            
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
        
        # Count successful episodes (target reached)
        if total_reward > 5:  # Target reward - some penalties
            successful_episodes += 1
        
        # Print progress
        if (episode + 1) % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            recent_success = sum(1 for r in recent_rewards if r > 5)
            stats = agent.get_stats()
            
            print(f"  Episode {episode + 1}/{episodes}:")
            print(f"    Mean reward (last 100): {np.mean(recent_rewards):.3f}")
            print(f"    Success rate (last 100): {recent_success}%")
            print(f"    Epsilon: {stats['epsilon']:.3f}")
            print(f"    Q-table size: {stats['q_table_size']}")
    
    env.close()
    
    # Final statistics
    success_rate = (successful_episodes / episodes) * 100
    final_mean = np.mean(episode_rewards[-100:]) if episodes >= 100 else np.mean(episode_rewards)
    
    print(f"  Final Results:")
    print(f"    Overall success rate: {success_rate:.1f}%")
    print(f"    Final mean reward (last 100): {final_mean:.3f}")
    print(f"    Final Q-table size: {agent.get_stats()['q_table_size']}")
    
    return agent, episode_rewards


def demonstrate_trained_agent(agent, target_agent_id: int, episodes: int = 10):
    """Demonstrate a trained agent's performance."""
    print(f"\nDemonstrating trained Agent {target_agent_id} (no learning)...")
    
    env = ZeroSumBiddingWrapper(
        target_agent_id=target_agent_id,
        grid_size=5,
        bid_upper_bound=3,
        bid_penalty=0.1,
        target_reward=10.0
    )
    
    demo_rewards = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100
        
        print(f"  Demo Episode {episode + 1}:")
        
        while steps < max_steps:
            # Get action from agent (no training)
            action = agent.get_action(obs, 3, training=False)
            
            direction = action // 4
            bid = action % 4
            directions = ['Up', 'Down', 'Left', 'Right']
            
            if steps < 5:  # Show first few actions
                print(f"    Step {steps + 1}: {directions[direction]}, Bid {bid}")
            
            next_obs, rewards, terminated, truncated, info = env.step(action)
            
            protagonist_reward = rewards["protagonist"]
            total_reward += protagonist_reward
            obs = next_obs
            steps += 1
            
            if terminated or truncated:
                break
        
        demo_rewards.append(total_reward)
        result = "SUCCESS" if total_reward > 5 else "FAILED"
        print(f"    Result: {result} (Reward: {total_reward:.2f}, Steps: {steps})")
    
    env.close()
    
    success_count = sum(1 for r in demo_rewards if r > 5)
    print(f"  Demo Summary: {success_count}/{episodes} successful episodes")
    print(f"  Average reward: {np.mean(demo_rewards):.2f}")


def main():
    """Main example demonstrating Zero-Sum Q-Learning."""
    print("=== Zero-Sum Q-Learning Training Example ===")
    
    # Train both agents
    print("\n" + "="*50)
    print("TRAINING PHASE")
    print("="*50)
    
    agent_0, rewards_0 = train_agent_example(target_agent_id=0, episodes=500)
    agent_1, rewards_1 = train_agent_example(target_agent_id=1, episodes=500)
    
    # Demonstrate trained agents
    print("\n" + "="*50)
    print("DEMONSTRATION PHASE")
    print("="*50)
    
    demonstrate_trained_agent(agent_0, target_agent_id=0, episodes=10)
    demonstrate_trained_agent(agent_1, target_agent_id=1, episodes=10)
    
    # Save models
    print("\n" + "="*50)
    print("SAVING MODELS")
    print("="*50)
    
    os.makedirs('models', exist_ok=True)
    
    agent_0.save_model('models/agent_0_example.pkl')
    agent_1.save_model('models/agent_1_example.pkl')
    
    print("  Saved models:")
    print("    - models/agent_0_example.pkl")
    print("    - models/agent_1_example.pkl")
    
    # Show how to load models
    print("\n  Loading model example:")
    try:
        loaded_agent = ZeroSumQLearning.load_model('models/agent_0_example.pkl')
        print(f"    Successfully loaded agent with Q-table size: {loaded_agent.get_stats()['q_table_size']}")
    except Exception as e:
        print(f"    Error loading model: {e}")
    
    print("\n=== Training Complete ===")
    print("\nNext steps:")
    print("1. Run 'python train_zero_sum.py' for full training with plotting")
    print("2. Run 'python evaluate_zero_sum.py' to evaluate saved models")
    print("3. Modify parameters in the training scripts for different experiments")


if __name__ == "__main__":
    main()
