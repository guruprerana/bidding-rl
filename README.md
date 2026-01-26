# bidding-rl

Multi-agent bidding RL in a gridworld. A shared PPO policy learns to navigate toward targets while agents bid for control of a shared body. The project runs on a GPU-native batched environment and supports moving targets, action windows, bid penalties, target expiry, and decentralized observations.

Note: The legacy CPU environment in `src/bidding_gridworld/bidding_gridworld.py` is deprecated. The active environment is the GPU-native implementation in `src/bidding_gridworld/bidding_gridworld_torch.py`.

## Highlights

- Shared policy PPO for multi-agent bidding and single-agent navigation.
- GPU-native batched environment with moving targets and target expiry.
- Optional action windows with bid penalties on the first step.
- Centralized or per-agent observations (`visible_targets`).
- Evaluation and MP4 rollout generation integrated into the training script.

## Requirements

- Python 3.10+ recommended
- CUDA-capable GPU recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Train a PPO run (edit config in the script):

```bash
python train_ppo_moving_targets.py
```

Train Assault PPO (edit config in the script):

```bash
python train_assault_ppo.py
```

Evaluate checkpoints (edit paths in the script):

```bash
python evaluate_trained_models.py
```

## Configuration

All training configuration lives at the top of `train_ppo_moving_targets.py`.

Key sections:

- Mode selection: `SINGLE_AGENT_MODE` and `MOVING_TARGETS`.
- Environment params: grid size, targets, expiry, rewards, bidding, windows.
- Training params: `NUM_ITERATIONS`, rollout length, PPO epochs.
- Evaluation params: `EVAL_FREQ`, `NUM_EVAL_EPISODES`, `VIDEO_FREQ`.

Example:

```python
SINGLE_AGENT_MODE = False
MOVING_TARGETS = True

NUM_ITERATIONS = 1000
NUM_ENVS = 2048
NUM_STEPS = 256

EVAL_FREQ = 50
VIDEO_FREQ = 100
NUM_EVAL_EPISODES = 20
NUM_VIDEO_EPISODES = 1
```

## Environment

GPU-native environment (torch tensors on CUDA):

- File: `src/bidding_gridworld/bidding_gridworld_torch.py`
- OCAtari Assault (object-state): `src/assault/assault_torch.py`
- Assault PPO (multi-agent bidding): `src/assault/assault_bidding_ppo.py`
- Assault PPO (single-agent): `src/assault/assault_single_agent_ppo.py`
- Assault experiment runner: `src/assault/assault_experiment.py`
- Class: `BiddingGridworld`
- Config: `BiddingGridworldConfig`

Core mechanics:

- Bidding: agents bid each step, highest bidder controls movement.
- Action windows: winner can control for multiple steps; bid penalty applies on the first step.
- Target expiry: penalties if a target is not reached within configured steps.
- Moving targets: targets move and respawn after being reached or expired.
- Observations: centralized (all targets) or per-agent (visible nearest targets).

## Training

PPO trainers:

- Multi-agent: `src/bidding_gridworld/bidding_ppo.py`
- Single-agent: `src/bidding_gridworld/single_agent_ppo.py`

Both trainers use the GPU environment directly and avoid CPU conversions.

## Evaluation and Rollouts

`train_ppo_moving_targets.py` performs periodic evaluation and can save MP4 rollouts. Video frequency is independent from eval frequency via `VIDEO_FREQ`.

Outputs are saved under `logs/` per run:

```
logs/<run_name>/
  checkpoints/
  rollouts/
  config/
```

## Deprecation Notice

`src/bidding_gridworld/bidding_gridworld.py` is deprecated and only kept for reference. Use `src/bidding_gridworld/bidding_gridworld_torch.py` for new work.

## License

See `LICENSE`.
