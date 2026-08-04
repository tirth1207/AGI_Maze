# Curious Agent v1 🧠

### Curiosity-Driven Reinforcement Learning using Intrinsic Curiosity Modules (ICM) and PPO

> *Can an AI learn to explore simply because it is curious?*

Curious Agent v1 is an experimental artificial intelligence research project that demonstrates how intrinsic motivation can improve learning in sparse-reward environments. Instead of relying only on external rewards, the agent generates its own internal reward by measuring how surprising the consequences of its actions are.

The project combines **Intrinsic Curiosity Modules (ICM)** with **Proximal Policy Optimization (PPO)** to build an autonomous agent capable of efficient exploration without prior knowledge of the environment.

---

# Overview

Traditional reinforcement learning agents learn only when the environment provides rewards.

```
State
   ↓
Action
   ↓
Reward
   ↓
Learning
```

However, in many real-world problems, rewards are extremely rare.

This project explores a different idea:

> **An intelligent agent should be able to learn simply because it wants to understand the world better.**

Instead of waiting for rewards, the agent creates an **intrinsic reward** based on its own prediction errors.

```
State
   ↓
Predict Future
   ↓
Reality
   ↓
Prediction Error
   ↓
Intrinsic Reward
   ↓
Exploration
```

This enables the agent to actively explore unfamiliar parts of an environment even before discovering the actual task objective.

---

# Features

* Intrinsic Curiosity Module (ICM)
* PPO Actor-Critic implementation
* Sparse-reward GridWorld environment
* Partial observability
* Forward and Inverse Dynamics Models
* Prediction-error-based intrinsic rewards
* Curiosity vs Baseline comparison
* Training visualization
* Clean and modular PyTorch implementation

---

# Project Structure

```
.
├── agent.py                # PPO Actor-Critic implementation
├── environment.py          # Sparse reward GridWorld
├── icm.py                  # Intrinsic Curiosity Module
├── train.py                # Training pipeline
├── plot_results.py         # Training visualization
├── history_curious.npz
├── history_baseline.npz
├── requirements.txt
└── README.md
```

---

# Architecture

```
                   GridWorld
                       │
                Current Observation
                       │
                       ▼
                PPO Actor-Critic
                       │
                 Selected Action
                       │
                       ▼
                 Environment Step
                       │
         Next State + External Reward
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
 Intrinsic Curiosity Module     PPO Update
          │
          ▼
Prediction Error
          │
          ▼
Intrinsic Reward
          │
          ▼
Combined Reward
```

---

# Components

## GridWorld Environment

The environment is a procedurally generated maze designed to provide extremely sparse rewards.

Characteristics:

* Random internal obstacles
* Partial observations
* Unknown goal location
* Zero reward until reaching the goal
* Episode termination by success or maximum steps

This setup forces meaningful exploration rather than memorization.

---

## PPO Agent

The decision-making component is implemented using **Proximal Policy Optimization (PPO)**.

The network consists of:

```
Observation
      │
 Shared Neural Network
      │
 ┌────┴────┐
 │         │
 ▼         ▼
Actor    Critic
```

The actor learns which action to take.

The critic estimates the expected future reward.

---

## Intrinsic Curiosity Module (ICM)

The ICM provides the internal motivation mechanism.

It consists of three neural networks.

### Encoder

Transforms raw observations into compact feature representations.

```
Observation
      │
      ▼
 Latent Features
```

---

### Inverse Model

Learns which action caused the observed transition.

```
State

+

Next State

↓

Predicted Action
```

This forces the learned representation to focus on aspects of the environment that the agent can actually control.

---

### Forward Model

Predicts what should happen after performing an action.

```
Current State

+

Action

↓

Predicted Next State
```

Prediction error becomes the intrinsic reward.

```
Prediction Error
        ↓
Curiosity Reward
```

The more surprising the outcome, the greater the motivation to explore.

---

# Training Process

Each episode follows the following pipeline:

```
Reset Environment
        ↓
Observe State
        ↓
Choose Action
        ↓
Interact
        ↓
Store Transition
        ↓
Compute Curiosity Reward
        ↓
Train ICM
        ↓
Compute PPO Advantages
        ↓
Update Policy
        ↓
Repeat
```

---

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install torch numpy matplotlib gymnasium
```

---

## Train Curious Agent

```bash
python train.py \
    --episodes 350 \
    --curiosity 1 \
    --seed 3 \
    --size 12 \
    --max_steps 160
```

---

## Train Baseline

```bash
python train.py \
    --episodes 350 \
    --curiosity 0 \
    --seed 3 \
    --size 12 \
    --max_steps 160
```

---

## Generate Comparison Graphs

```bash
python plot_results.py
```

This generates

```
comparison.png
```

containing

* Success rate comparison
* Curiosity signal comparison

---

# Experimental Results

The repository demonstrates a comparison between:

### Curious Agent

* PPO
* Intrinsic Curiosity Module
* Prediction-error reward

and

### Baseline Agent

* PPO only
* No intrinsic reward

In the included experiment, the curiosity-driven agent eventually learns to solve the sparse-reward maze, while the baseline agent fails to discover the goal under the same training conditions. Refer to the generated plots and training histories for the observed behavior in this project.

---

# Technologies

* Python
* PyTorch
* NumPy
* Matplotlib
* Gymnasium-inspired environment

---

# Future Roadmap

Curious Agent v1 demonstrates intrinsic curiosity.

Future versions aim to evolve into a broader cognitive architecture including:

* World Models
* Planning before acting
* Episodic Memory
* Semantic Memory
* Working Memory
* Knowledge Graphs
* Continual Learning
* Lifelong Adaptation
* Question Generation
* Belief Revision
* Goal Management
* Self Reflection
* Autonomous Research
* Multi-Agent Learning

The long-term objective is to explore cognitive AI systems that learn continuously through curiosity, reasoning, memory, and interaction rather than relying solely on predefined rewards or transformer-centric architectures.

---

# Inspiration

This project is inspired by research in:

* Intrinsic Curiosity Modules (ICM)
* Reinforcement Learning
* Cognitive Architectures
* Neuroscience-inspired AI
* Autonomous Learning
* Intrinsic Motivation

---

# License

This project is released under the MIT License.

---

# Author

**Tirth Alpeshbhai Rathod**

AI Research • Reinforcement Learning • Cognitive Architectures • Autonomous Intelligence

> *"The future of AI isn't just generating better answers—it's building systems that are driven to discover them."*
