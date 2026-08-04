import math

import torch

from bidding_gridworld.bidding_ppo import SharedAgent
from ppo_utils import counterfactual_factorized_auction_ppo_update_step


def _counterfactual_agent():
    agent = SharedAgent(
        obs_dim=5,
        num_actions_per_agent=2,
        actor_hidden_sizes=(8,),
        critic_hidden_sizes=(8,),
        counterfactual_bid_critic=True,
    )
    agent.set_bid_head(2)
    return agent


def test_counterfactual_bid_baseline_marginalizes_current_policy():
    agent = _counterfactual_agent()
    with torch.no_grad():
        agent.bid_head.weight.zero_()
        agent.bid_head.bias.copy_(
            torch.tensor([0.0, math.log(2.0), math.log(3.0)])
        )
        for layer in agent.bid_critic:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        agent.bid_critic[-1].bias.copy_(torch.tensor([1.0, 2.0, 4.0]))

    obs = torch.randn(4, 5)
    selected, baseline, q_values = agent.get_counterfactual_bid_values(
        obs, torch.tensor([0, 1, 2, 2])
    )

    assert torch.allclose(q_values, torch.tensor([[1.0, 2.0, 4.0]]).expand(4, -1))
    assert torch.allclose(baseline, torch.full((4,), 17.0 / 6.0))
    assert torch.equal(selected, torch.tensor([1.0, 2.0, 4.0, 4.0]))


def test_counterfactual_factorized_update_is_finite_and_trains_both_critics():
    torch.manual_seed(7)
    agent = _counterfactual_agent()
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    obs = torch.randn(24, 5)
    with torch.no_grad():
        (
            actions,
            direction_logprobs,
            bid_logprobs,
            _,
            _,
            direction_values,
        ) = agent.get_factorized_action_and_value(obs)
        bid_values, bid_baselines, _ = agent.get_counterfactual_bid_values(
            obs, actions[:, 1]
        )

    old_direction_critic = agent.critic[-1].weight.detach().clone()
    old_bid_critic = agent.bid_critic[-1].weight.detach().clone()
    direction_mask = torch.zeros(24, dtype=torch.bool)
    direction_mask[::3] = True
    bid_mask = torch.ones(24, dtype=torch.bool)
    metrics = counterfactual_factorized_auction_ppo_update_step(
        agent=agent,
        optimizer=optimizer,
        obs=obs,
        actions=actions,
        old_direction_logprobs=direction_logprobs,
        old_bid_logprobs=bid_logprobs,
        direction_mask=direction_mask,
        bid_mask=bid_mask,
        direction_advantages=torch.randn(24),
        bid_advantages=bid_values - bid_baselines,
        direction_returns=torch.randn(24),
        bid_returns=torch.randn(24),
        direction_values=direction_values.flatten(),
        bid_values=bid_values,
        clip_coef=0.2,
        ent_coef=0.01,
        direction_vf_coef=0.1,
        bid_vf_coef=0.1,
        max_grad_norm=0.5,
        norm_adv=True,
        clip_vloss=False,
    )

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert not torch.equal(old_direction_critic, agent.critic[-1].weight)
    assert not torch.equal(old_bid_critic, agent.bid_critic[-1].weight)
