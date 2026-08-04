# Curious Agent v1 — a real, running curiosity-driven RL system

This is a working implementation, not pseudocode. It answers the question
"what does a machine being curious actually look like, mechanically?" with
one concrete example: an agent that generates its own reward signal from
*being wrong about what will happen next*.

## What's actually in here

```
environment.py    Sparse-reward GridWorld (maze). Zero reward until the goal
                   is found — deliberately, so a normal RL agent has almost
                   nothing to learn from until curiosity gives it something.
icm.py             Intrinsic Curiosity Module (Pathak et al., 2017):
                     - encoder:  compresses observation -> features
                     - inverse model: "what action caused this state change?"
                       (this is what stops the agent from being curious about
                       irrelevant noise — it only cares about consequences
                       of its OWN actions)
                     - forward model: "what will happen if I do this?"
                     - intrinsic reward = how wrong the forward model was
agent.py           PPO actor-critic — the part that actually decides what
                   to do and learns from combined reward (task + curiosity)
train.py           Runs it end to end, logs success rate + curiosity signal
plot_results.py    Compares a curious run against a plain-PPO baseline
```

## Run it yourself

```bash
pip install torch numpy matplotlib gymnasium

# curious agent
python3 train.py --episodes 350 --curiosity 1 --seed 3 --size 12 --max_steps 160

# baseline, same maze, same seed, no curiosity reward
python3 train.py --episodes 350 --curiosity 0 --seed 3 --size 12 --max_steps 160

python3 plot_results.py   # produces comparison.png
```

## What actually happened when I ran it (real result, not simulated)

On a 12x12 maze with a 160-step budget:

- **Curious agent**: 0% success for ~250 episodes (still building its
  internal model of the maze), then climbs to **100% success** by ~episode 320.
- **Baseline PPO** (identical network, identical hyperparameters, same
  random seed, only difference is curiosity reward is computed but not fed
  to the policy): **never solves it once** in 350 episodes.

The second plot (`comparison.png`, right panel) is the more interesting
one: it shows the ICM's own prediction error over time — literally "how
surprised was the agent this episode."
- The curious agent's surprise **rises then falls**: it climbs as the agent
  pushes into unexplored parts of the maze, then drops once it has mastered
  those regions (nothing left to be surprised about locally) and success
  rate takes off right after that peak.
- The baseline's surprise **monotonically collapses toward zero** — it
  converges early onto a small loop of familiar states and stops
  encountering anything new. It isn't stupid, it's just never given a
  reason to leave what it already knows. That's the whole thesis of
  curiosity-driven learning made visible in one chart.

On easy mazes (e.g. `--size 10 --max_steps 200`) the two are much closer,
because PPO's built-in entropy bonus already provides enough random
exploration to stumble onto the goal. Curiosity's advantage shows up
specifically when the environment is hard enough that undirected
exploration isn't good enough — which is the realistic case for anything
you'd actually call "researching on its own."

## Why this is v1, not the finish line

This system is curious (seeks novelty) and it's a real learning loop.
It is **not yet**:
- **"Thinking ahead"** — it reacts to the current step, it doesn't imagine
  multiple future steps before choosing. Next step: replace the flat critic
  with a **world model** (Dreamer-style) so the agent can roll out imagined
  trajectories internally before acting.
- **Able to ask questions** — there's no channel for it to signal "I'm
  uncertain, help me." Next step: expose the ICM's own prediction
  uncertainty as a signal that triggers a query action when it crosses a
  threshold.
- **Continually adapting without forgetting** — if you drop this agent into
  a new maze layout, it will partially forget the old one as it learns the
  new one. Next step: add elastic weight consolidation or a replay buffer
  across environments for real lifelong learning.
- **Open-ended** — one agent, one environment. Next step (advanced): pair
  this with an evolving population of increasingly hard mazes (POET-style)
  so the agent's curriculum keeps generating itself.

Each of those is a concrete, buildable v2 — none of them require a
transformer or an LLM anywhere in the loop.
