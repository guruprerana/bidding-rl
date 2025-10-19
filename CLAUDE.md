# Multi-Agent Bidding RL via Zero-Sum Adversarial Training

This project implements a reinforcement learning algorithm that trains a single agent through zero-sum adversarial games, then deploys multiple instances of that agent to cooperatively navigate a shared environment via competitive bidding.

## Core Algorithm

The key innovation is learning a generalizable navigation policy through adversarial training:
- A single agent (protagonist) learns to navigate to targets while competing against an adversary
- Training uses zero-sum DQN where the protagonist and adversary have opposing rewards
- The trained agent is then deployed as multiple instances that compete for movement control via bidding
- Agents bid for control of a shared body, with the highest bidder determining the next action

## Project Structure

- `src/bidding_gridworld.py` - Core gridworld environment with bidding mechanism and action windows
- `src/zero_sum_wrapper.py` - Environment wrapper that creates zero-sum training scenarios
- `src/zero_sum_dqn.py` - Zero-Sum DQN algorithm for adversarial training
- `comprehensive_experiment.py` - Main experiment runner with training, evaluation, and visualization

## Key Features

- **Single-agent generalization**: One policy learns to handle multiple targets through normalized observations
- **Zero-sum adversarial training**: Agent learns robust navigation by competing against an adversary
- **Bidding mechanism**: Multiple instances compete for control through strategic bidding
- **Action windows**: Winning bidders maintain control for multiple steps, reducing bid penalties
- **Configurable parameters**: Easy adjustment of rewards, penalties, episode length, and action windows
- **Checkpoint saving**: Model checkpoints saved every 100,000 training steps
- **Comprehensive visualization**: GIF generation showing agent control, bidding outcomes, and target pursuit
- **15x15 gridworld**: Scalable environment for testing complex navigation scenarios

## Algorithm Benefits

- **Generalization**: Single policy handles multiple target positions
- **Robustness**: Adversarial training prevents overfitting and encourages diverse strategies
- **Efficiency**: Train once, deploy many times
- **Interpretability**: Visual indicators show which agent controls movement at each step
- **Flexibility**: Action windows allow strategic control allocation

## Training Process

1. Create zero-sum environment with protagonist vs adversary
2. Train single agent using DQN with adversarial opponent
3. Save checkpoints every 100k steps (20 checkpoints for 2M training steps)
4. Evaluate agent against each target position
5. Deploy multiple instances in cooperative evaluation

## Deployment

During cooperative evaluation:
- 3 instances of the same trained agent are deployed
- Each instance pursues a different target
- Agents bid for control of the shared body
- Highest bidder wins and controls movement
- Bid penalties only apply on first step of action window

## Configurable Parameters

Key parameters set at the beginning of `comprehensive_experiment.py`:

- `grid_size`: Size of the gridworld (default: 15)
- `num_agents`: Number of agent instances (default: 3)
- `target_reward`: Reward for reaching target (default: 100.0)
- `bid_penalty`: Penalty multiplier for bids (default: 0.1)
- `max_steps`: Maximum episode length (default: 100)
- `action_window`: Steps a winner maintains control (default: 1)
- `training_timesteps`: Total training steps (default: 2,000,000)

## Running Experiments

```bash
# Train and evaluate with static targets
python comprehensive_experiment.py

# Train and evaluate with moving targets
python comprehensive_experiment.py --moving-targets

# Evaluation only (load existing model)
python comprehensive_experiment.py --eval-only path/to/experiment
```

## Output Structure

```
logs/experiment_{name}_{timestamp}/
├── config.json                  # Experiment configuration
├── checkpoints/                 # Model checkpoints every 100k steps
├── models/                      # Final trained model
├── training/                    # Training logs and metrics
├── rollouts/                    # Individual agent rollout GIFs
├── competition/                 # Multi-agent cooperative episode GIFs
└── plots/                       # Training progress visualizations
```

## Visualization

GIF outputs show:
- **Gold border + lightning bolt (⚡)**: Agent currently controlling movement
- **Colored rings**: Which target the controller is pursuing
- **Bid information**: All agents' bids and chosen directions
- **Winner indication**: Which agent won the bid (or "No Movement" if all bid 0)
- **Target status**: Checkmarks for reached targets
- **Cumulative rewards**: Running reward totals for all agents

## Special Mechanisms

### Action Windows
When `action_window > 1`:
- Winner maintains control for multiple consecutive steps
- Bid penalty only charged on the first step
- Encourages strategic bidding and reduces overall penalties
- Allows smoother trajectories toward targets

### Zero Bids
If all agents bid 0:
- No movement occurs
- No agent is selected as winner
- Useful for end-of-episode or when all targets are reached

### Moving Targets
With `--moving-targets` flag:
- Targets move randomly with configurable probability
- Observations track actual target positions (not fixed positions)
- Tests agent adaptability to dynamic objectives
