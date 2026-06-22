"""Comparison figures for baselines and ablations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._shared import ensure_runs, save_fig


def _bar_from_subdirs(name: str, subdirs: list[tuple[str, str]], outfile: str):
    labels, means, rates = [], [], []
    for label, subdir in subdirs:
        runs = ensure_runs(subdir)
        if not runs:
            continue
        labels.append(label)
        means.append(np.mean([r.get("best_distance", np.nan) for r in runs]))
        rates.append(np.mean([r.get("success_rate", np.nan) for r in runs]))
    if not labels:
        print(f"  no comparison data for {name}; skipping")
        return
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax2 = ax1.twinx()
    ax1.bar(x - 0.18, means, width=0.36, label="best distance", color="#4c78a8")
    ax2.bar(x + 0.18, rates, width=0.36, label="success rate", color="#54a24b")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylabel("Best distance")
    ax2.set_ylabel("Success rate")
    ax1.set_title(name)
    ax1.grid(axis="y", alpha=0.25)
    save_fig(fig, outfile)


def fig_bl_baselines():
    _bar_from_subdirs(
        "Baseline comparison",
        [
            ("Full", "full"),
            ("Uniform", "baselines/uniform"),
            ("Random", "baselines/random"),
            ("Outer only", "baselines/outer_only"),
            ("Bandit", "baselines/bandit"),
        ],
        "BL_baselines",
    )


def fig_ab_ablations():
    _bar_from_subdirs(
        "Ablation comparison",
        [
            ("Full", "full"),
            ("Crisp", "ablations/crisp"),
            ("No decay", "ablations/no_decay"),
            ("Med only", "ablations/med_only"),
            ("Meal only", "ablations/meal_only"),
            ("Finite diff", "ablations/finite_diff"),
        ],
        "AB_ablations",
    )


def fig_ac_ablation_curves():
    fig_ab_ablations()
