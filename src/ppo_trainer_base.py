"""Shared PPO trainer base classes."""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
import torch
import wandb

from ppo_utils import compute_gae, ppo_update_step, compute_explained_variance, format_duration


class PPOTrainerBase:
    def __init__(self, args, callbacks=None):
        self.args = args
        self.callbacks = callbacks or {}
        self.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

        if getattr(self.args, "track", False):
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=vars(args),
                name=self.run_name,
                save_code=True,
            )

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.envs = None
        self.agent = None
        self.optimizer = None
        self._last_rollout_stats = {}

    def setup(self):
        raise NotImplementedError

    def _maybe_log_iteration(self, global_step: int, metrics: dict, clipfracs: list[float], start_time: float):
        if not getattr(self.args, "track", False):
            return
        log_dict = {
            "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
            "losses/value_loss": metrics["v_loss"],
            "losses/policy_loss": metrics["pg_loss"],
            "losses/entropy": metrics["entropy_loss"],
            "losses/old_approx_kl": metrics["old_approx_kl"],
            "losses/approx_kl": metrics["approx_kl"],
            "losses/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "charts/SPS": int(global_step / (time.time() - start_time)),
        }
        extra = self._extra_log_dict(global_step)
        if extra:
            log_dict.update(extra)
        wandb.log(log_dict, step=global_step)

    def _extra_log_dict(self, global_step: int) -> dict:
        return {}

    def _on_rollout_step(self, infos: Any, global_step: int):
        return

    def _on_iteration_start(self, iteration: int):
        return

    def _on_iteration_end(self, iteration: int, global_step: int):
        return


class SingleAgentPPOTrainerBase(PPOTrainerBase):
    def __init__(self, args, callbacks=None):
        super().__init__(args, callbacks=callbacks)
        self.obs_dim = None

    def train(self):
        if self.envs is None:
            raise RuntimeError("Must call setup() before train()")

        args = self.args
        args.batch_size = args.num_envs * args.num_steps
        args.minibatch_size = args.batch_size // args.num_minibatches
        args.total_timesteps = args.num_iterations * args.num_envs * args.num_steps

        obs = torch.zeros((args.num_steps, args.num_envs, self.obs_dim)).to(self.device)
        actions = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(self.device)

        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.args.seed)
        next_done = torch.zeros((args.num_envs,), device=self.device)

        for iteration in range(1, args.num_iterations + 1):
            iteration_start = time.time()
            self._on_iteration_start(iteration)
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.view(-1)

                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminations, truncations, infos = self.envs.step(action)
                next_done = (terminations | truncations).to(self.device, dtype=torch.float32)
                rewards[step] = reward.view(-1)
                self._on_rollout_step(infos, global_step)

            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(args.num_envs)
                advantages, returns = compute_gae(
                    rewards, values, dones, next_value, next_done, args.gamma, args.gae_lambda
                )
            self._last_rollout_stats = {
                "rewards": rewards.detach(),
                "values": values.detach(),
                "advantages": advantages.detach(),
            }

            b_obs = obs.reshape(-1, self.obs_dim)
            b_actions = actions.reshape(-1)
            b_logprobs = logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            clipfracs: list[float] = []
            for epoch in range(args.update_epochs):
                b_inds = torch.randperm(args.batch_size, device=self.device)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    metrics = ppo_update_step(
                        self.agent,
                        self.optimizer,
                        b_obs[mb_inds],
                        b_actions.long()[mb_inds],
                        b_logprobs[mb_inds],
                        b_advantages[mb_inds],
                        b_returns[mb_inds],
                        b_values[mb_inds],
                        args.clip_coef,
                        args.ent_coef,
                        args.vf_coef,
                        args.max_grad_norm,
                        args.norm_adv,
                        args.clip_vloss,
                    )
                    clipfracs.append(metrics["clipfrac"])
                if args.target_kl is not None and metrics["approx_kl"] > args.target_kl:
                    break

            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            explained_var = compute_explained_variance(y_pred, y_true)

            sps = int(global_step / (time.time() - start_time))
            iter_time = time.time() - iteration_start
            remaining_iters = args.num_iterations - iteration
            eta = format_duration(remaining_iters * iter_time)
            print(
                f"Iteration {iteration}/{args.num_iterations} - SPS: {sps} - "
                f"Value Loss: {metrics['v_loss']:.4f} - Policy Loss: {metrics['pg_loss']:.4f} - "
                f"Iter Time: {format_duration(iter_time)} - ETA: {eta}"
            )

            self._maybe_log_iteration(global_step, metrics, clipfracs, start_time)
            if getattr(self.args, "track", False):
                wandb.log(
                    {
                        "losses/explained_variance": explained_var,
                        "charts/iteration": iteration,
                    },
                    step=global_step,
                )

            if self.callbacks.get("on_iteration_end"):
                self.callbacks["on_iteration_end"](self, iteration, global_step)
            self._on_iteration_end(iteration, global_step)

        if self.callbacks.get("on_training_end"):
            self.callbacks["on_training_end"](self, global_step)


class MultiAgentPPOTrainerBase(PPOTrainerBase):
    def __init__(self, args, callbacks=None):
        super().__init__(args, callbacks=callbacks)
        self.obs_dim = None
        self.num_action_components = None

    def train(self):
        if self.envs is None:
            raise RuntimeError("Must call setup() before train()")

        args = self.args
        args.batch_size = args.num_envs * args.num_steps * args.num_agents
        args.minibatch_size = args.batch_size // args.num_minibatches
        args.total_timesteps = args.num_iterations * args.num_envs * args.num_steps * args.num_agents

        obs = torch.zeros((args.num_steps, args.num_envs, args.num_agents, self.obs_dim)).to(self.device)
        actions = torch.zeros((args.num_steps, args.num_envs, args.num_agents, self.num_action_components)).to(self.device)
        logprobs = torch.zeros((args.num_steps, args.num_envs, args.num_agents)).to(self.device)
        rewards = torch.zeros((args.num_steps, args.num_envs, args.num_agents)).to(self.device)
        dones = torch.zeros((args.num_steps, args.num_envs, args.num_agents)).to(self.device)
        values = torch.zeros((args.num_steps, args.num_envs, args.num_agents)).to(self.device)

        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.args.seed)
        next_done = torch.zeros((args.num_envs, args.num_agents)).to(self.device)

        for iteration in range(1, args.num_iterations + 1):
            iteration_start = time.time()
            self._on_iteration_start(iteration)
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(args.num_steps):
                global_step += args.num_envs * args.num_agents
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    flat_obs = next_obs.reshape(-1, self.obs_dim)
                    action, logprob, _, value = self.agent.get_action_and_value(flat_obs)
                    action = action.reshape(args.num_envs, args.num_agents, self.num_action_components)
                    logprob = logprob.reshape(args.num_envs, args.num_agents)
                    value = value.reshape(args.num_envs, args.num_agents)
                    values[step] = value

                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminations, truncations, infos = self.envs.step(action)
                next_done_scalar = terminations | truncations
                next_done = next_done_scalar.unsqueeze(1).expand(-1, args.num_agents).to(self.device, dtype=torch.float32)
                rewards[step] = reward
                self._on_rollout_step(infos, global_step)

            with torch.no_grad():
                flat_next_obs = next_obs.reshape(-1, self.obs_dim)
                next_value = self.agent.get_value(flat_next_obs).reshape(args.num_envs, args.num_agents)
                advantages, returns = compute_gae(
                    rewards, values, dones, next_value, next_done, args.gamma, args.gae_lambda
                )
            self._last_rollout_stats = {
                "rewards": rewards.detach(),
                "values": values.detach(),
                "advantages": advantages.detach(),
            }

            b_obs = obs.reshape(-1, self.obs_dim)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1, self.num_action_components)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            clipfracs: list[float] = []
            for epoch in range(args.update_epochs):
                b_inds = torch.randperm(args.batch_size, device=self.device)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    metrics = ppo_update_step(
                        self.agent,
                        self.optimizer,
                        b_obs[mb_inds],
                        b_actions[mb_inds],
                        b_logprobs[mb_inds],
                        b_advantages[mb_inds],
                        b_returns[mb_inds],
                        b_values[mb_inds],
                        args.clip_coef,
                        args.ent_coef,
                        args.vf_coef,
                        args.max_grad_norm,
                        args.norm_adv,
                        args.clip_vloss,
                    )
                    clipfracs.append(metrics["clipfrac"])
                if args.target_kl is not None and metrics["approx_kl"] > args.target_kl:
                    break

            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            explained_var = compute_explained_variance(y_pred, y_true)

            sps = int(global_step / (time.time() - start_time))
            iter_time = time.time() - iteration_start
            remaining_iters = args.num_iterations - iteration
            eta = format_duration(remaining_iters * iter_time)
            print(
                f"Iteration {iteration}/{args.num_iterations} - SPS: {sps} - "
                f"Value Loss: {metrics['v_loss']:.4f} - Policy Loss: {metrics['pg_loss']:.4f} - "
                f"Iter Time: {format_duration(iter_time)} - ETA: {eta}"
            )

            self._maybe_log_iteration(global_step, metrics, clipfracs, start_time)
            if getattr(self.args, "track", False):
                wandb.log(
                    {
                        "losses/explained_variance": explained_var,
                        "charts/iteration": iteration,
                    },
                    step=global_step,
                )

            if self.callbacks.get("on_iteration_end"):
                self.callbacks["on_iteration_end"](self, iteration, global_step)
            self._on_iteration_end(iteration, global_step)

        if self.callbacks.get("on_training_end"):
            self.callbacks["on_training_end"](self, global_step)
