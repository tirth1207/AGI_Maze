import numpy as np

from environment import EMPTY, GOAL, WALL, GridWorld


def test_reset_returns_normalized_observation_and_start_position():
    env = GridWorld(size=7, view_radius=1, seed=7)
    obs = env.reset()

    assert obs.shape == (9,)
    assert obs.dtype == np.float32
    assert np.all((obs >= 0.0) & (obs <= 1.0))
    assert env.pos == [1, 1]
    assert env.steps == 0


def test_agent_cannot_walk_through_a_wall():
    env = GridWorld(size=6, view_radius=1, seed=1)
    env.reset()
    env.grid[1, 2] = WALL

    env.step(3)  # right, into the wall

    assert env.pos == [1, 1]


def test_reaching_goal_ends_episode_with_reward():
    env = GridWorld(size=5, view_radius=1, max_steps=10, seed=2)
    env.reset()
    env.grid[:] = EMPTY
    env.grid[0, :] = WALL
    env.grid[-1, :] = WALL
    env.grid[:, 0] = WALL
    env.grid[:, -1] = WALL
    env.pos = [3, 2]
    env.grid[3, 3] = GOAL
    env.goal_pos = (3, 3)

    _, reward, done, _ = env.step(3)

    assert reward == 1.0
    assert done is True


def test_episode_terminates_at_step_limit():
    env = GridWorld(size=6, view_radius=1, max_steps=1, seed=3)
    env.reset()

    _, reward, done, _ = env.step(0)

    assert reward == 0.0
    assert done is True
