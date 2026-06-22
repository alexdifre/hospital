"""Robustness figures for Section 8."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._shared import ensure_runs, group_by_profile, save_fig


def fig_nr_noise_robustness():
    levels = ["0.05", "0.1", "0.2", "0.4"]
    xs, ys = [], []
    for level in levels:
        runs = ensure_runs(f"robustness/noise_{level}")
        if runs:
            xs.append(float(level))
            ys.append(np.mean([r.get("best_distance", np.nan) for r in runs]))
    if not xs:
        print("  no noise robustness data; skipping")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Rating noise sigma")
    ax.set_ylabel("Mean best distance")
    ax.set_title("Noise robustness")
    ax.grid(alpha=0.25)
    save_fig(fig, "NR_noise_robustness")


def fig_nr2_noise_conv_rates():
    levels = ["0.05", "0.1", "0.2", "0.4"]
    xs, ys = [], []
    for level in levels:
        runs = ensure_runs(f"robustness/noise_{level}")
        if runs:
            xs.append(level)
            ys.append(np.mean([1.0 if r.get("convergence_episode", -1) >= 0 else 0.0 for r in runs]))
    if not xs:
        print("  no noise convergence data; skipping")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(xs, ys, color="#4c78a8")
    ax.set_xlabel("Rating noise sigma")
    ax.set_ylabel("Convergence rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Noise convergence rate")
    save_fig(fig, "NR2_noise_conv_rates")


def fig_is_init_sensitivity():
    runs = ensure_runs("robustness/random_init")
    if not runs:
        return
    grouped = group_by_profile(runs)
    labels = list(grouped)
    data = [[r.get("best_distance", np.nan) for r in grouped[p]] for p in labels]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.boxplot(data, labels=labels)
    ax.set_ylabel("Best distance")
    ax.set_title("Initialisation sensitivity")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, "IS_init_sensitivity")
