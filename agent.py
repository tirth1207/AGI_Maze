"""
A compact PPO (Proximal Policy Optimization) actor-critic agent.

This is the "acting" brain: given an observation, the actor outputs a
probability distribution over actions (the policy), and the critic estimates
how good the current state is (the value function). The two are trained
jointly on experience the agent collects itself.

PPO is used (rather than a naive policy gradient) because it's stable enough
that intrinsic + extrinsic rewards can be combined without the training
collapsing -- important once curiosity reward is added on top of a sparse
task reward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden, n_actions)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        z = self.shared(obs)
        logits = self.actor_head(z)
        value = self.critic_head(z).squeeze(-1)
        return logits, value

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value


class PPO:
    def __init__(self, obs_dim, n_actions, lr=3e-4, clip_eps=0.2,
                 gamma=0.99, gae_lambda=0.95, epochs=4, device="cpu"):
        self.device = device
        self.net = ActorCritic(obs_dim, n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs

    def compute_gae(self, rewards, values, dones, last_value):
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        values_ext = torch.cat([values, last_value.unsqueeze(0)])
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values_ext[t + 1] * mask - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self, obs, actions, old_log_probs, returns, advantages):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            logits, values = self.net(obs)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, returns)

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()
