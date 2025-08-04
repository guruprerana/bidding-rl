# BiddingGridworld

A custom multi-objective gridworld environment using Gymnasium where agents must navigate and bid for resources, with Zero-Sum Q-Learning implementation for decentralized training.

## Environment Description

The BiddingGridworld is a 2-agent environment where:
- Agents navigate in a gridworld
- Each agent has a specific target to reach
- Agents must bid for the right to take actions
- Higher bids win, but cost resources

## Features

- **Discrete Action/Observation Spaces**: Optimized for tabular reinforcement learning
- **Bidding Mechanism**: Agents compete through bidding
- **Multi-Objective**: Multiple targets and competing objectives
- **Zero-Sum Game Theory**: Nash-Q learning for decentralized training
- **Configurable**: Grid size, bid limits, and rewards can be adjusted

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Test the System
```bash
python quick_test.py
```

### 2. Run Example Training
```bash
python example_training.py
```

### 3. Full Training with Plotting
```bash
python train_zero_sum.py
```

### 4. Evaluate Trained Models
```bash
python evaluate_zero_sum.py
```

## Usage

### Basic Environment

```python
from src.bidding_gridworld import BiddingGridworld

env = BiddingGridworld(grid_size=10, bid_upper_bound=5)
obs, info = env.reset()

for _ in range(100):
    # Random actions for both agents
    actions = {
        0: env.action_space.sample(),
        1: env.action_space.sample()
    }
    obs, rewards, terminated, truncated, info = env.step(actions)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Zero-Sum Wrapper for Single-Agent Training

```python
from src.zero_sum_wrapper import ZeroSumBiddingWrapper

env = ZeroSumBiddingWrapper(target_agent_id=0)
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, rewards, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Zero-Sum Q-Learning

```python
from src.zero_sum_qlearning import ZeroSumQLearning

# Create agent
agent = ZeroSumQLearning(
    protagonist_action_space_size=16,
    adversary_action_space_size=16,
    observation_space_size=1250,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=0.3
)

# Training loop
for episode in range(1000):
    obs, info = env.reset()
    while True:
        action = agent.get_action(obs, bid_upper_bound=3, training=True)
        next_obs, rewards, terminated, truncated, info = env.step(action)
        agent.update(obs, action, rewards["protagonist"], next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            break
    agent.end_episode()

# Save trained model
agent.save_model('trained_agent.pkl')

# Load trained model
loaded_agent = ZeroSumQLearning.load_model('trained_agent.pkl')
```

## Architecture

### Core Components

1. **BiddingGridworld** (`src/bidding_gridworld.py`): Main multi-agent environment
2. **ZeroSumBiddingWrapper** (`src/zero_sum_wrapper.py`): Single-agent wrapper for decentralized training
3. **ZeroSumQLearning** (`src/zero_sum_qlearning.py`): Nash-Q learning algorithm implementation

### Training Scripts

1. **quick_test.py**: Fast system verification (50 episodes each agent)
2. **example_training.py**: Medium training with demonstrations (500 episodes each agent)
3. **train_zero_sum.py**: Full training with plotting and model saving (2000 episodes each agent)
4. **evaluate_zero_sum.py**: Evaluation and comparison of trained models

## Files

- `src/bidding_gridworld.py`: Main multi-objective environment
- `src/zero_sum_wrapper.py`: Zero-sum wrapper for training
- `example_zero_sum.py`: Example usage demonstration

## Installation

```bash
pip install -r requirements.txt
```