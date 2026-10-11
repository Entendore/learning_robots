#!/usr/bin/env python3
"""Plot learning curves from the agents' progress CSV files.

Usage: python3 plot_progress.py controllers/qlearning_agent/progress_robotQ.csv \
                               controllers/qlearning_agent/progress_robotSARSA.csv
"""
import csv
import sys

import matplotlib.pyplot as plt


def moving_average(values, window=10):
    out, run = [], []
    for v in values:
        run.append(v)
        if len(run) > window:
            run.pop(0)
        out.append(sum(run) / len(run))
    return out


def load(path):
    eps, steps, rewards = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            eps.append(int(row["episode"]))
            steps.append(int(row["steps"]))
            if row.get("reward"):
                rewards.append(float(row["reward"]))
    return eps, steps, rewards


def main(paths):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for path in paths:
        eps, steps, rewards = load(path)
        label = path.replace("\\", "/").split("/")[-1]
        ax1.plot(eps, steps, alpha=0.25)                  # raw
        ax1.plot(eps, moving_average(steps), label=label) # smoothed
        if rewards:
            ax2.plot(eps, rewards, alpha=0.25)
            ax2.plot(eps, moving_average(rewards), label=label)
    ax1.set(title="Steps per episode (lower = better)",
            xlabel="episode", ylabel="steps")
    ax2.set(title="Reward per episode (higher = better)",
            xlabel="episode", ylabel="reward")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 plot_progress.py <progress.csv> [...]")
    main(sys.argv[1:])
