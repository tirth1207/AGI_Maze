import torch

from agent import PPO


def test_agent_samples_valid_actions_and_returns_scalar_values():
    ppo = PPO(obs_dim=9, n_actions=4)
    obs = torch.rand(1, 9)

    action, log_prob, value = ppo.net.act(obs)

    assert action.shape == (1,)
    assert 0 <= action.item() < 4
    assert log_prob.shape == (1,)
    assert value.shape == (1,)


def test_gae_returns_finite_advantages_and_returns():
    ppo = PPO(obs_dim=9, n_actions=4)
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.2, 0.3, 0.4])
    dones = torch.tensor([0.0, 0.0, 1.0])
    last_value = torch.tensor(0.0)

    advantages, returns = ppo.compute_gae(rewards, values, dones, last_value)

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()
    assert torch.allclose(returns, advantages + values)
