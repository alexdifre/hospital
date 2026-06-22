"""Core Section 8 figures from full-system JSON results."""

from __future__ import annotations

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

from ._shared import (
    DIM_LABELS,
    DIMENSIONS,
    PROFILE_COLORS,
    PROFILE_LABELS,
    ensure_runs,
    episode_series,
    group_by_profile,
    mean_std,
    nested_episode_series,
    profile_order,
    save_fig,
)


def fig_b1_convergence():
    runs = ensure_runs("full")
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for profile, prs in group_by_profile(runs).items():
        mean, std = mean_std([episode_series(r, "distance_to_true") for r in prs])
        x = np.arange(len(mean))
        ax.plot(x, mean, label=PROFILE_LABELS.get(profile, profile), color=PROFILE_COLORS.get(profile))
        ax.fill_between(x, mean - std, mean + std, color=PROFILE_COLORS.get(profile), alpha=0.15)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Distance to true weights")
    ax.set_title("Preference convergence")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    save_fig(fig, "B1_convergence_curves")


def fig_b2_weight_evolution():
    runs = ensure_runs("full")
    if not runs:
        return
    targets = ["speed_oriented", "safety_first", "presentation_focused"]
    fig, axes = plt.subplots(len(targets), 1, figsize=(8.5, 8.0), sharex=True)
    by_profile = group_by_profile(runs)
    for ax, profile in zip(axes, targets):
        prs = by_profile.get(profile, [])
        if not prs:
            continue
        for dim_i, label in enumerate(DIM_LABELS):
            series = []
            for run in prs:
                values = []
                for ep in run.get("episodes", []):
                    w = ep.get("weights_after", [])
                    values.append(w[dim_i] if len(w) > dim_i else np.nan)
                series.append(np.array(values, dtype=float))
            mean, _ = mean_std(series)
            ax.plot(mean, label=label)
        ax.set_ylabel(PROFILE_LABELS.get(profile, profile))
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Episode")
    axes[0].set_title("Learned preference weights")
    axes[0].legend(ncol=5, fontsize=8)
    save_fig(fig, "B2_weight_evolution")


def fig_b3_final_weights():
    runs = ensure_runs("full")
    if not runs:
        return
    order = profile_order(runs)
    x = np.arange(len(DIMENSIONS))
    fig, axes = plt.subplots(1, len(order), figsize=(3.0 * len(order), 3.8), sharey=True)
    if len(order) == 1:
        axes = [axes]
    by_profile = group_by_profile(runs)
    for ax, profile in zip(axes, order):
        final_ws, true_ws = [], []
        for run in by_profile[profile]:
            last = run.get("episodes", [{}])[-1]
            final_ws.append(last.get("weights_after", [np.nan] * 5))
            true_ws.append(last.get("true_weights", [np.nan] * 5))
        final_mean = np.nanmean(np.array(final_ws, dtype=float), axis=0)
        true_mean = np.nanmean(np.array(true_ws, dtype=float), axis=0)
        ax.bar(x, final_mean, color=PROFILE_COLORS.get(profile), alpha=0.75, label="learned")
        ax.scatter(x, true_mean, marker="D", color="black", s=28, label="true")
        ax.set_xticks(x)
        ax.set_xticklabels(DIM_LABELS, rotation=45, ha="right")
        ax.set_title(PROFILE_LABELS.get(profile, profile))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Weight")
    axes[-1].legend(fontsize=8)
    save_fig(fig, "B3_final_weights")


def fig_b4_feature_space():
    runs = ensure_runs("full")
    if not runs:
        return
    order = profile_order(runs)
    task_types = ["medication", "meal"]
    fig, axes = plt.subplots(1, len(task_types), figsize=(10, 4.5), sharey=True)
    for ax, task in zip(axes, task_types):
        mat = []
        for profile in order:
            vals = []
            for run in group_by_profile(runs).get(profile, []):
                for ep in run.get("episodes", []):
                    if ep.get("task_type") == task:
                        f = ep.get("features", {})
                        vals.append([float(f.get(d, np.nan)) for d in DIMENSIONS])
            mat.append(np.nanmean(np.array(vals, dtype=float), axis=0) if vals else [np.nan] * 5)
        im = ax.imshow(np.array(mat, dtype=float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_title(task.capitalize())
        ax.set_xticks(np.arange(len(DIM_LABELS)))
        ax.set_xticklabels(DIM_LABELS, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([PROFILE_LABELS.get(p, p) for p in order])
    fig.colorbar(im, ax=axes, shrink=0.8, label="Feature value")
    save_fig(fig, "B4_feature_centroids")


def fig_b5_plan_diversity():
    runs = ensure_runs("full")
    if not runs:
        return
    order = profile_order(runs)
    categories = sorted(
        {
            ep.get("meal_type")
            for run in runs
            for ep in run.get("episodes", [])
            if ep.get("meal_type")
        }
    )
    if not categories:
        print("  no meal_type data; skipping B5")
        return
    counts = defaultdict(Counter)
    for run in runs:
        profile = run.get("profile")
        for ep in run.get("episodes", []):
            meal = ep.get("meal_type")
            if meal:
                counts[profile][meal] += 1
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bottom = np.zeros(len(order))
    for cat in categories:
        vals = np.array([counts[p][cat] for p in order], dtype=float)
        totals = np.array([sum(counts[p].values()) for p in order], dtype=float)
        frac = np.divide(vals, totals, out=np.zeros_like(vals), where=totals > 0)
        ax.bar(order, frac, bottom=bottom, label=cat)
        bottom += frac
    ax.set_xticklabels([PROFILE_LABELS.get(p, p) for p in order], rotation=25, ha="right")
    ax.set_ylabel("Meal episode proportion")
    ax.set_title("Meal plan diversity")
    ax.legend(fontsize=8)
    save_fig(fig, "B5_plan_diversity")


def fig_b6_mse_loss():
    runs = ensure_runs("full")
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for profile, prs in group_by_profile(runs).items():
        mean, std = mean_std([episode_series(r, "learner_mse") for r in prs])
        x = np.arange(len(mean))
        ax.plot(x, mean, label=PROFILE_LABELS.get(profile, profile), color=PROFILE_COLORS.get(profile))
        ax.fill_between(x, mean - std, mean + std, color=PROFILE_COLORS.get(profile), alpha=0.15)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Learner MSE")
    ax.set_title("Preference learner loss")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    save_fig(fig, "B6_mse_loss")


def fig_b7_translator_params():
    runs = ensure_runs("full")
    if not runs:
        return
    params = ["q_base", "r_base_ax", "_derived_Q_pos", "_derived_R"]
    fig, axes = plt.subplots(len(params), 1, figsize=(8.5, 8.5), sharex=True)
    for ax, param in zip(axes, params):
        for profile, prs in group_by_profile(runs).items():
            series = [nested_episode_series(r, "translator_params", param) for r in prs]
            mean, _ = mean_std(series)
            if np.isfinite(mean).any():
                ax.plot(mean, label=PROFILE_LABELS.get(profile, profile), color=PROFILE_COLORS.get(profile))
        ax.set_ylabel(param)
        ax.grid(alpha=0.25)
    axes[0].set_title("Translator parameter evolution")
    axes[0].legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Episode")
    save_fig(fig, "B7_translator_params")


def fig_b8_trajectories():
    runs = ensure_runs("full")
    if not runs:
        return
    order = profile_order(runs)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for profile in order:
        run = group_by_profile(runs)[profile][0]
        episode = next((ep for ep in run.get("episodes", []) if ep.get("trajectory_xy")), None)
        if not episode:
            continue
        xy = np.array(episode.get("trajectory_xy", []), dtype=float)
        if xy.ndim == 2 and xy.shape[1] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], label=PROFILE_LABELS.get(profile, profile), color=PROFILE_COLORS.get(profile))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Example MPC trajectories")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_fig(fig, "B8_trajectories")


def fig_b9_battery_efficiency():
    runs = ensure_runs("full")
    if not runs:
        return
    order = profile_order(runs)
    battery, efficiency = [], []
    for profile in order:
        eps = [ep for run in group_by_profile(runs)[profile] for ep in run.get("episodes", [])]
        battery.append(np.nanmean([ep.get("battery_used_pct", np.nan) for ep in eps]))
        efficiency.append(np.nanmean([ep.get("path_efficiency", np.nan) for ep in eps]))
    x = np.arange(len(order))
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax2 = ax1.twinx()
    ax1.bar(x - 0.18, battery, width=0.36, color="#4c78a8", label="battery used")
    ax2.bar(x + 0.18, efficiency, width=0.36, color="#f58518", label="path efficiency")
    ax1.set_xticks(x)
    ax1.set_xticklabels([PROFILE_LABELS.get(p, p) for p in order], rotation=25, ha="right")
    ax1.set_ylabel("Battery used [%]")
    ax2.set_ylabel("Path efficiency")
    ax1.set_title("Battery use and path quality")
    ax1.grid(axis="y", alpha=0.25)
    save_fig(fig, "B9_battery_efficiency")
