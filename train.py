"""
Train a curiosity-driven agent on the sparse-reward GridWorld.

Total reward given to PPO each step = extrinsic reward (from the task,
almost always 0) + intrinsic reward (from the ICM's surprise, nonzero
whenever the agent sees something its forward model didn't expect).

Run:
    python3 train.py --episodes 400 --curiosity 1
    python3 train.py --episodes 400 --curiosity 0   # baseline, no curiosity

Comparing the two runs is the actual demonstration: in a sparse-reward
maze, a baseline PPO agent very often never finds the goal at all within
the episode budget, because it has no reason to explore systematically.
The curious agent finds it far more reliably because unexplored cells are
worth something to it even before the real goal is ever seen.
"""

import argparse
import numpy as np
import torch

from environment import GridWorld
from icm import ICM
from agent import PPO


def run_episode(env, ppo, icm, use_curiosity, icm_optimizer, intrinsic_scale=0.02):
    obs = env.reset()
    obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf, next_obs_buf = [], [], [], [], [], [], []
    done = False
    ext_return = 0.0
    int_return = 0.0

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, log_prob, value = ppo.net.act(obs_t)
        next_obs, ext_reward, done, _ = env.step(action.item())

        obs_buf.append(obs)
        next_obs_buf.append(next_obs)
        act_buf.append(action.item())
        logp_buf.append(log_prob.item())
        val_buf.append(value.item())
        rew_buf.append(ext_reward)
        done_buf.append(float(done))

        ext_return += ext_reward
        obs = next_obs

    # --- compute intrinsic rewards for the whole episode at once ---
    obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32)
    next_obs_t = torch.tensor(np.array(next_obs_buf), dtype=torch.float32)
    act_t = torch.tensor(act_buf, dtype=torch.long)

    intrinsic_reward, inverse_loss, forward_loss = icm(obs_t, next_obs_t, act_t)

    if use_curiosity:
        total_rewards = torch.tensor(rew_buf, dtype=torch.float32) + intrinsic_scale * intrinsic_reward
    else:
        total_rewards = torch.tensor(rew_buf, dtype=torch.float32)

    int_return = intrinsic_reward.sum().item()

    # --- train the ICM itself (it needs to keep learning too) ---
    icm_loss = inverse_loss + forward_loss
    icm_optimizer.zero_grad()
    icm_loss.backward()
    icm_optimizer.step()

    # --- train PPO on this episode's trajectory ---
    values_t = torch.tensor(val_buf, dtype=torch.float32)
    dones_t = torch.tensor(done_buf, dtype=torch.float32)
    with torch.no_grad():
        _, last_value = ppo.net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
    advantages, returns = ppo.compute_gae(total_rewards, values_t, dones_t, last_value.squeeze(0))

    logp_t = torch.tensor(logp_buf, dtype=torch.float32)
    ppo.update(obs_t, act_t, logp_t, returns, advantages)

    goal_reached = ext_return > 0
    return ext_return, int_return, goal_reached, len(obs_buf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--curiosity", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = GridWorld(size=args.size, view_radius=2, max_steps=args.max_steps, seed=args.seed)
    ppo = PPO(env.obs_dim, env.action_space_n)
    icm = ICM(env.obs_dim, env.action_space_n)
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=1e-3)

    history = {"ext_return": [], "int_return": [], "goal_reached": [], "steps": []}

    for ep in range(args.episodes):
        ext_r, int_r, reached, steps = run_episode(
            env, ppo, icm, use_curiosity=bool(args.curiosity), icm_optimizer=icm_optimizer
        )
        history["ext_return"].append(ext_r)
        history["int_return"].append(int_r)
        history["goal_reached"].append(reached)
        history["steps"].append(steps)

        if (ep + 1) % 20 == 0:
            recent_success = np.mean(history["goal_reached"][-20:]) * 100
            recent_int = np.mean(history["int_return"][-20:])
            print(f"[ep {ep+1:4d}] success rate (last 20): {recent_success:5.1f}%  "
                  f"avg curiosity reward: {recent_int:6.3f}  "
                  f"avg steps: {np.mean(history['steps'][-20:]):5.1f}")

    label = "curious" if args.curiosity else "baseline"
    np.savez(f"history_{label}.npz", **history)
    print(f"\nSaved history_{label}.npz")


if __name__ == "__main__":
    main()
