"""
Plots curious vs baseline agent results side by side.
Run after both training runs have produced history_curious.npz and history_baseline.npz.
"""
import numpy as np
import matplotlib.pyplot as plt

def smooth(x, window=15):
    x = np.array(x, dtype=np.float32)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")

curious = np.load("history_curious.npz")
baseline = np.load("history_baseline.npz")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(smooth(np.array(curious["goal_reached"]) * 100), label="Curious agent (ICM + PPO)", color="#d1495b")
axes[0].plot(smooth(np.array(baseline["goal_reached"]) * 100), label="Baseline (PPO only)", color="#6c757d")
axes[0].set_title("Goal-reaching success rate (15-ep rolling avg)")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Success rate (%)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(smooth(curious["int_return"]), label="Curious agent surprise", color="#d1495b")
axes[1].plot(smooth(baseline["int_return"]), label="Baseline surprise (unused, measured only)", color="#6c757d")
axes[1].set_title("Forward-model prediction error (proxy for exploration)")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Total surprise per episode")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("comparison.png", dpi=140)
print("Saved comparison.png")
