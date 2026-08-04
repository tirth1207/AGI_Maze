"""
A sparse-reward GridWorld.

Why this environment: with a normal RL reward signal, an agent that hasn't
found the goal yet gets ZERO learning signal from the reward -- it's acting
almost randomly. This is exactly the setting where curiosity (an internal
reward for "I did not expect that") matters: it gives the agent something
to learn from *before* it ever sees the real goal.

The world contains walls, a start, and a goal. Observation is a small local
window around the agent (partial observability) so the agent genuinely has
to explore to build a picture of the world -- it can't see the goal from
across the map.
"""

import numpy as np

WALL, EMPTY, GOAL, AGENT = 0, 1, 2, 3

class GridWorld:
    def __init__(self, size=10, view_radius=2, max_steps=200, seed=None):
        self.size = size
        self.view_radius = view_radius
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self._build_map()
        self.action_space_n = 4  # up, down, left, right
        obs_side = 2 * view_radius + 1
        self.obs_dim = obs_side * obs_side  # flattened local view

    def _build_map(self):
        s = self.size
        grid = np.ones((s, s), dtype=np.int64) * EMPTY
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL

        # scatter some internal walls to force real navigation/exploration
        n_walls = int(s * s * 0.15)
        for _ in range(n_walls):
            x = self.rng.integers(1, s - 1)
            y = self.rng.integers(1, s - 1)
            grid[x, y] = WALL

        self.base_grid = grid
        self.start_pos = (1, 1)
        self.goal_pos = (s - 2, s - 2)
        self.base_grid[self.start_pos] = EMPTY
        self.base_grid[self.goal_pos] = EMPTY

    def reset(self):
        self.grid = self.base_grid.copy()
        self.pos = list(self.start_pos)
        self.steps = 0
        return self._obs()

    def _obs(self):
        r = self.view_radius
        x, y = self.pos
        padded = np.pad(self.grid, r, mode="constant", constant_values=WALL)
        window = padded[x:x + 2 * r + 1, y:y + 2 * r + 1]
        return window.flatten().astype(np.float32) / 3.0  # normalize

    def step(self, action):
        self.steps += 1
        dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy

        if self.grid[nx, ny] != WALL:
            self.pos = [nx, ny]

        done = False
        reward = 0.0
        if tuple(self.pos) == self.goal_pos:
            reward = 1.0
            done = True
        if self.steps >= self.max_steps:
            done = True

        return self._obs(), reward, done, {}
