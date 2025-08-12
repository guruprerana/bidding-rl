# Multi-Objective RL via Zero-Sum Bidding Games

This project implements a novel multi-objective reinforcement learning algorithm that learns separate policies for each objective by framing them as zero-sum bidding games against an adversary.

## Core Algorithm

The key innovation is treating multi-objective RL as a competitive bidding process:
- Each objective has its own policy that bids for control
- Policies compete in zero-sum games to determine which objective gets to act
- This allows learning specialized policies for different objectives while maintaining overall coherence

## Project Structure

- `src/bidding_gridworld.py` - Core gridworld environment with multiple objectives
- `src/zero_sum_qlearning.py` - Q-Learning algorithm adapted for bidding-based multi-objective RL
- `src/zero_sum_wrapper.py` - Environment wrapper that implements the bidding mechanism
- `comprehensive_experiment.py` - Main experiment runner and evaluation suite

## Key Features

- **Multi-objective decomposition**: Each objective learns its own specialized policy
- **Zero-sum bidding mechanism**: Policies compete for control through strategic bidding
- **Adversarial training**: Objectives learn to bid against each other, promoting robustness
- **Gridworld testbed**: Multiple target objectives requiring different strategies
- **Comprehensive evaluation**: MP4 video recording of agent rollouts (FPS: 1)
- **Competition analysis**: Detailed logging of bidding strategies and outcomes

## Algorithm Benefits

- **Specialization**: Each objective develops focused strategies
- **Adaptability**: Bidding allows dynamic priority adjustment
- **Robustness**: Adversarial training prevents overfitting to single objectives
- **Interpretability**: Clear separation of objective-specific behaviors

## Running Experiments

The main experiment script trains competing objective policies, records their bidding behavior as MP4 videos, and evaluates the emergent multi-objective strategies.

## Video Output

- Rollout videos are saved as MP4 files with 1 FPS
- Training rollouts show individual objective policy performance
- Competition rollouts show bidding dynamics and multi-objective behavior