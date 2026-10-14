# Reproducible experiment protocol

This project compares PPO in a sparse-reward GridWorld with and without the
Intrinsic Curiosity Module (ICM). Keep the environment and seed identical
between the two runs so curiosity is the main changed variable.

## Setup

Use Python 3.10–3.12 in a virtual environment:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Run a paired comparison

Run both commands from the repository root with the same seed and episode
budget:

```bash
python train.py --episodes 400 --curiosity 0 --seed 0
python train.py --episodes 400 --curiosity 1 --seed 0
```

Repeat with several seeds before drawing conclusions. Record the command,
Python/PyTorch versions, seed, episode budget, and final success rate. A
single run is not enough to establish that curiosity reliably improves
performance.

## Interpretation

- Compare goal-reaching success rates, not just intrinsic reward.
- Report results across seeds with mean and spread.
- Treat the checked-in history arrays and comparison plot as artifacts from
  a particular run unless their generating commands and metadata are recorded.
- Do not compare runs with different environment sizes, step limits, or seeds
  as if curiosity were the only changed variable.
