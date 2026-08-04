"""
Intrinsic Curiosity Module (ICM), after Pathak et al. 2017.

The core idea: curiosity = surprise about the CONSEQUENCES of your own actions.

Three pieces:
1. Feature encoder  phi(s)         -- compresses raw observation into a
                                       compact representation.
2. Inverse model     a_hat = g(phi(s), phi(s'))
                                    -- predicts which action was taken, given
                                       before/after states. Forces phi() to
                                       encode only things the agent's OWN
                                       actions affect (ignores irrelevant
                                       noise in the environment -- this is
                                       what separates real curiosity from an
                                       agent that just gets addicted to
                                       random TV static).
3. Forward model     phi_hat(s') = f(phi(s), a)
                                    -- predicts what the next state's features
                                       will look like, given the current
                                       state and the action taken.

Intrinsic reward = how WRONG the forward model was.
Wrong prediction -> "I didn't expect that" -> curiosity reward -> agent is
pulled toward exploring that part of the world more.
As the agent visits a region repeatedly, the forward model gets better there,
prediction error drops, curiosity reward drops, and the agent naturally
moves on to explore somewhere less understood.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICM(nn.Module):
    def __init__(self, obs_dim, n_actions, feature_dim=32, hidden=64):
        super().__init__()
        self.n_actions = n_actions

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

        # inverse model: (phi(s), phi(s')) -> predicted action logits
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

        # forward model: (phi(s), one-hot action) -> predicted phi(s')
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

    def encode(self, obs):
        return self.encoder(obs)

    def forward(self, obs, next_obs, actions):
        """Returns intrinsic reward + the losses used to train the ICM itself."""
        phi = self.encode(obs)
        phi_next = self.encode(next_obs)

        # --- inverse model loss: can we recover the action that was taken? ---
        inv_input = torch.cat([phi, phi_next], dim=-1)
        pred_action_logits = self.inverse_model(inv_input)
        inverse_loss = F.cross_entropy(pred_action_logits, actions)

        # --- forward model loss: can we predict the resulting features? ---
        action_onehot = F.one_hot(actions, num_classes=self.n_actions).float()
        fwd_input = torch.cat([phi, action_onehot], dim=-1)
        pred_phi_next = self.forward_model(fwd_input)
        forward_loss_per_sample = 0.5 * ((pred_phi_next - phi_next.detach()) ** 2).sum(dim=-1)
        forward_loss = forward_loss_per_sample.mean()

        # intrinsic reward = per-sample prediction error (this is the curiosity signal)
        intrinsic_reward = forward_loss_per_sample.detach()

        return intrinsic_reward, inverse_loss, forward_loss
