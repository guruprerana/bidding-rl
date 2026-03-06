# Auction-Based Online Policy Adaptation for Evolving Objectives

Code for the paper: **"Auction-Based Online Policy Adaptation for Evolving Objectives"**

## Setup

### 1. Clone the repository with submodules

```bash
git clone --recurse-submodules <repo-url>
cd bidding-rl
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install OCAtari from the submodule

```bash
pip install -e OC_Atari/
```

## Codebase Overview

The PPO implementation is based on [CleanRL](https://github.com/vwxyzjn/cleanrl). The DWN baseline implementation is taken from [JuanK120/RL_EWS](https://github.com/JuanK120/RL_EWS/tree/master).

### Gridworld (Cat Feeder environment in the paper)

- `src/bidding_gridworld/bidding_gridworld_torch.py` - GPU-native batched gridworld environment (bidding, action windows, moving targets, target expiry).
- `src/bidding_gridworld/bidding_ppo.py` - Multi-agent PPO trainer with shared actor-critic.
- `src/bidding_gridworld/single_agent_ppo.py` - Single-agent PPO trainer.
- `train_ppo_moving_targets.py` - Training entry point; all configuration lives at the top of this file.

### Assault (OCAtari)

- `src/assault/assault_torch.py` - OCAtari Assault environment with object-state observations, supporting single-agent and bidding modes.
- `src/assault/assault_bidding_ppo.py` - Multi-agent PPO trainer for Assault with bidding.
- `src/assault/assault_single_agent_ppo.py` - Single-agent PPO trainer for Assault.
- `src/assault/assault_experiment.py` - Experiment runner with checkpointing and evaluation.
- `train_assault_ppo.py` - Assault training entry point.

### Evaluation

- `evaluate_trained_models.py` - Load saved checkpoints and run evaluation with optional video rollouts.

### Outputs

Training runs save logs, checkpoints, and rollouts under `logs/<run_name>/`.
