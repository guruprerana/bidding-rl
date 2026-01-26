# Coding Agent Guide: bidding-rl

This file captures the working context for this codebase. It is derived from `CLAUDE.md` and updated to include the GPU-native batched gridworld.

## Project Summary

This repo implements multi-agent bidding RL in a gridworld. A shared policy learns to navigate to targets and, in multi-agent mode, agents bid for control of a shared body via action windows and bid penalties.

## Core Files

- `src/bidding_gridworld/bidding_gridworld.py` - CPU gridworld environment with bidding, action windows, and moving targets.
- `src/bidding_gridworld/bidding_ppo.py` - Multi-agent PPO trainer with shared actor-critic.
- `src/bidding_gridworld/single_agent_ppo.py` - Single-agent PPO trainer.
- `train_ppo_moving_targets.py` - Training entry point and configuration.
- `evaluate_trained_models.py` - Evaluation script for saved checkpoints.
- `src/bidding_gridworld/bidding_gridworld_torch.py` - GPU-native batched gridworld environment (torch tensors on CUDA).
- `src/assault/assault_torch.py` - OCAtari Assault env with object-state observations (single-agent + bidding).
- `src/assault/assault_bidding_ppo.py` - Assault multi-agent PPO trainer with bidding.
- `src/assault/assault_single_agent_ppo.py` - Assault single-agent PPO trainer.
- `src/assault/assault_experiment.py` - Assault experiment runner with checkpoints and eval.

## Key Mechanics

- **Bidding**: agents bid each step; highest bidder chooses the direction.
- **Action windows**: winner can control for multiple steps; bid penalty applies on first step.
- **Target expiry**: penalties if a target is not reached within configured steps.
- **Moving targets**: targets can move and respawn after being reached or expired.
- **Per-agent observations**: optional decentralized observation mode (`visible_targets`).

## Training Overview

- Multi-agent training uses PPO with a shared actor-critic network.
- Single-agent mode learns to reach all targets without bidding.
- Configure runs in `train_ppo_moving_targets.py` (iterations are set via `NUM_ITERATIONS`).
- Video rollouts can be throttled separately with `VIDEO_FREQ`.
- Optional masked attention pooling over target observations can be enabled via `USE_TARGET_ATTENTION_POOLING`
  (with `TARGET_EMBED_DIM` and `TARGET_ENCODER_HIDDEN_SIZES`) to support variable target counts.

## GPU Batched Environment (Torch)

The GPU-native env is implemented in `src/bidding_gridworld/bidding_gridworld_torch.py` as `BiddingGridworld`.

Capabilities:
- Batched, GPU-only step logic (bidding/windowing, movement, rewards, expiry).
- Moving targets and respawn logic on GPU.
- Centralized and per-agent observation layouts on GPU.
- Evaluation helpers and video rendering used by `train_ppo_moving_targets.py`.

Notes:
- The GPU env returns torch tensors directly; the PPO loops avoid CPU conversions when enabled.
- Logging info is minimal on the GPU env to avoid overhead.

## Common Entry Points

```
python train_ppo_moving_targets.py
python train_assault_ppo.py
python evaluate_trained_models.py
```

## Output Structure

Logs and checkpoints are saved under `logs/` with per-run subfolders containing configs, checkpoints, rollouts, and evaluation stats.

## Full CLAUDE.md Reference

# Multi-Agent Bidding RL via Zero-Sum Adversarial Training

This project implements a reinforcement learning algorithm that trains a single agent through zero-sum adversarial games, then deploys multiple instances of that agent to cooperatively navigate a shared environment via competitive bidding.

## Core Algorithm

The key innovation is learning a generalizable navigation policy through adversarial training:
- A single agent (protagonist) learns to navigate to targets while competing against an adversary
- Training uses zero-sum DQN where the protagonist and adversary have opposing rewards
- The trained agent is then deployed as multiple instances that compete for movement control via bidding
- Agents bid for control of a shared body, with the highest bidder determining the next action

## Project Structure

- `src/bidding_gridworld/bidding_gridworld.py` - Core gridworld environment with bidding mechanism and action windows
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

## Evaluating Trained Models

Use `evaluate_trained_models.py` to evaluate saved checkpoints:

```bash
source venv/bin/activate
python evaluate_trained_models.py
```

### Configuration

Edit the script's `main()` function to customize:

```python
# Experiment directories to evaluate
SINGLE_AGENT_EXP = "logs/ppo_moving_targets_single_agent_exp5_20251216_214906"
MULTI_AGENT_EXP = "logs/ppo_moving_targets_exp8_20251216_214612"

# Evaluation settings
NUM_EVAL_EPISODES = 20  # Number of episodes to run
NUM_GIF_EPISODES = 5    # Number of GIFs to create
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Output

Creates an `evaluation/` subdirectory in each experiment folder:

```
logs/[experiment_name]/evaluation/eval_[timestamp]/
├── episode_0.gif          # Episode visualization
├── episode_1.gif
├── ...
├── episode_N.gif
└── eval_stats.json        # Comprehensive statistics
```

### Key Statistics

The `eval_stats.json` includes:
- **avg_return**: Average total return per episode
- **avg_targets_reached**: Average unique targets reached per episode
- **avg_min_targets_reached**: Minimum reaches across all agents/targets
- **success_rate**: Percentage of episodes where all targets reached
- **avg_expired_targets**: Average target expiries per episode
- **per_episode**: Detailed per-episode breakdown

### Features

- Automatically loads latest checkpoint from experiment
- Uses deterministic (greedy) policy for reproducible evaluation
- Correctly tracks targets with moving targets that respawn
- Handles observation reordering for multi-agent models
- Supports window bidding and target expiry mechanisms
