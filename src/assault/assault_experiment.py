"""Experiment wrapper for Assault PPO training."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import wandb

from assault.assault_torch import AssaultConfig, AssaultEnv
from assault.assault_bidding_ppo import AssaultPPOTrainer, AssaultArgs
from assault.assault_single_agent_ppo import AssaultSingleAgentPPOTrainer, AssaultSingleAgentArgs


class AssaultExperiment:
    """Experiment wrapper for PPO training with periodic evaluation and checkpointing."""

    def __init__(
        self,
        base_log_dir: str = "logs",
        experiment_name: str = "",
        checkpoint_freq: int = 50,
        eval_freq: int = 25,
        num_eval_episodes: int = 5,
        single_agent_mode: bool = False,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not experiment_name:
            mode_prefix = "single_agent" if single_agent_mode else "multi_agent"
            experiment_name = f"assault_ppo_{mode_prefix}"

        self.log_dir = Path(base_log_dir) / f"{experiment_name}_{timestamp}"
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.single_agent_mode = single_agent_mode

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.log_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.config_dir = self.log_dir / "config"
        self.config_dir.mkdir(exist_ok=True)

        print(f"📁 Experiment directory: {self.log_dir}")

    def log_codebase_to_wandb(self, run):
        disabled = os.environ.get("WANDB_MODE", "").lower() == "disabled" or \
            os.environ.get("WANDB_DISABLED", "").lower() in {"true", "1"}
        if not run or disabled:
            return

        artifact = wandb.Artifact(
            name=f"codebase-{run.id}",
            type="code",
            description="Codebase snapshot (src folder + training script)"
        )
        project_root = self.log_dir.parent.parent
        src_dir = project_root / "src"
        if src_dir.exists():
            artifact.add_dir(str(src_dir), name="src")
        train_script = project_root / "train_ppo_moving_targets.py"
        if train_script.exists():
            artifact.add_file(str(train_script), name="train_ppo_moving_targets.py")
        run.log_artifact(artifact)

    def save_config(self, args):
        config = vars(args)
        config_file = self.config_dir / "training_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2, default=str)
        print(f"💾 Config saved to {config_file}")

    def save_checkpoint(self, trainer, iteration: int, global_step: int):
        checkpoint_dir = self.checkpoints_dir / f"iter_{iteration}"
        checkpoint_dir.mkdir(exist_ok=True)
        model_path = checkpoint_dir / "agent.pt"
        torch.save(trainer.agent.state_dict(), str(model_path))
        info_path = checkpoint_dir / "checkpoint_info.json"
        with open(info_path, "w") as f:
            json.dump(
                {
                    "iteration": iteration,
                    "global_step": global_step,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        if trainer.args.track:
            wandb.save(str(model_path))

    def _evaluate(self, trainer, args, single_agent_mode: bool) -> Dict[str, float]:
        env_config = AssaultConfig(
            num_agents=args.num_agents,
            max_enemies=args.max_enemies,
            bid_upper_bound=getattr(args, "bid_upper_bound", 0),
            bid_penalty=getattr(args, "bid_penalty", 0.0),
            action_window=getattr(args, "action_window", 1),
            window_bidding=getattr(args, "window_bidding", False),
            window_penalty=getattr(args, "window_penalty", 0.0),
            enemy_destroy_reward=args.enemy_destroy_reward,
            hit_penalty=args.hit_penalty,
            life_loss_penalty=args.life_loss_penalty,
            health_loss_penalty=args.health_loss_penalty,
            max_steps=args.max_steps,
            hud=args.hud,
            single_agent_mode=single_agent_mode,
            allow_variable_enemies=args.allow_variable_enemies,
        )
        env = AssaultEnv(env_config, num_envs=1, device=trainer.device, seed=args.seed)
        returns = []
        scores = []
        lengths = []

        for _ in range(self.num_eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            ep_len = 0
            last_score = 0.0
            while not done:
                with torch.no_grad():
                    if single_agent_mode:
                        action, _, _, _ = trainer.agent.get_action_and_value(obs)
                    else:
                        flat_obs = obs.reshape(-1, trainer.obs_dim)
                        action, _, _, _ = trainer.agent.get_action_and_value(flat_obs)
                        action = action.reshape(1, args.num_agents, -1)

                obs, reward, terminated, truncated, info = env.step(action)
                if isinstance(info, dict):
                    score_tensor = info.get("score", None)
                    if torch.is_tensor(score_tensor):
                        last_score = float(score_tensor.item())
                ep_return += reward.sum().item() if not single_agent_mode else reward.item()
                done = bool(terminated.item() or truncated.item())
                ep_len += 1

            returns.append(ep_return)
            scores.append(last_score)
            lengths.append(ep_len)

        env.close()
        return {
            "avg_score": float(np.mean(scores)),
            "avg_length": float(np.mean(lengths)),
            "std_score": float(np.std(scores)),
            "std_length": float(np.std(lengths)),
            "avg_return": float(np.mean(returns)),
        }

    def run(self, args):
        mode_str = "SINGLE-AGENT" if self.single_agent_mode else "MULTI-AGENT"
        print(f"\n{'='*80}")
        print(f"ASSAULT PPO TRAINING - {mode_str} MODE")
        print(f"{'='*80}\n")

        if hasattr(args, "num_iterations"):
            if self.single_agent_mode:
                args.total_timesteps = args.num_iterations * args.num_envs * args.num_steps
            else:
                args.total_timesteps = args.num_iterations * args.num_envs * args.num_steps * args.num_agents

        self.save_config(args)

        def on_iteration_end(trainer, iteration, global_step):
            if iteration % self.checkpoint_freq == 0:
                self.save_checkpoint(trainer, iteration, global_step)
            if iteration % self.eval_freq == 0:
                stats = self._evaluate(trainer, args, self.single_agent_mode)
                if trainer.args.track:
                    wandb.log({f"eval/{k}": v for k, v in stats.items()}, step=global_step)

        def on_training_end(trainer, global_step):
            stats = self._evaluate(trainer, args, self.single_agent_mode)
            if trainer.args.track:
                wandb.log({f"eval/{k}": v for k, v in stats.items()}, step=global_step)

        callbacks = {
            "on_iteration_end": on_iteration_end,
            "on_training_end": on_training_end,
        }

        if self.single_agent_mode:
            trainer = AssaultSingleAgentPPOTrainer(args, callbacks=callbacks)
        else:
            trainer = AssaultPPOTrainer(args, callbacks=callbacks)

        trainer.setup()
        self.log_codebase_to_wandb(wandb.run)
        trainer.train()
        trainer.save_model()
        trainer.cleanup()
