import torch

from icm import ICM


def test_icm_returns_finite_nonnegative_intrinsic_rewards():
    model = ICM(obs_dim=9, n_actions=4)
    obs = torch.rand(8, 9)
    next_obs = torch.rand(8, 9)
    actions = torch.randint(0, 4, (8,))

    intrinsic_reward, inverse_loss, forward_loss = model(obs, next_obs, actions)

    assert intrinsic_reward.shape == (8,)
    assert torch.all(intrinsic_reward >= 0)
    assert torch.isfinite(intrinsic_reward).all()
    assert torch.isfinite(inverse_loss)
    assert torch.isfinite(forward_loss)
