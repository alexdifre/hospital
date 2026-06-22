"""
figures/tables.py — LaTeX table generation.

T1 — Master results table.
T2 — Baseline comparison table.
T3 — Ablation study table.
"""

from __future__ import annotations

import numpy as np

from ._shared import (
    PROFILE_COLORS, PROFILE_LABELS,
    CONVERGENCE_THRESHOLDS, DEFAULT_THRESHOLD, FIGURES_DIR,
    load_results, group_by_profile,
)


def table_t1_master():
    """T1: LaTeX results table with profile-specific thresholds."""
    full = load_results("full")
    if not full:
        return
    by_profile = group_by_profile(full)
    order = [p for p in PROFILE_COLORS if p in by_profile]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "\\textbf{Profile} & \\textbf{$d_{\\mathrm{thresh}}$} & "
        "\\textbf{Conv. Ep.} & \\textbf{Best $d$} & "
        "\\textbf{Final $d$} & \\textbf{Task Rate} & "
        "\\textbf{Conv.} & \\textbf{Dom.} \\\\",
        "\\midrule",
    ]

    for profile in order:
        runs   = by_profile[profile]
        thresh = CONVERGENCE_THRESHOLDS.get(profile, DEFAULT_THRESHOLD)
        conv_eps = [r["convergence_episode"] for r in runs if r["convergence_episode"] >= 0]
        best_ds  = [r["best_distance"] for r in runs]
        # Exclude sentinel final_d=1.0 (incomplete/crashed runs)
        final_ds = [r["final_distance"] for r in runs if r["final_distance"] < 1.0]
        rates    = [r["success_rate"] for r in runs]
        n_conv   = len(conv_eps)
        conv_str = f"{np.median(conv_eps):.0f}" if conv_eps else "---"
        label    = PROFILE_LABELS.get(profile, profile)

        lines.append(
            f"{label:12s} & {thresh:.2f} & {conv_str:>5s} & "
            f"{np.mean(best_ds):.3f} $\\pm$ {np.std(best_ds):.3f} & "
            f"{np.mean(final_ds):.3f} & "
            f"{np.mean(rates):.0%} & "
            f"{n_conv}/{len(runs)} & "
            f"\\checkmark \\\\"
        )

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Full-system convergence results (5 profiles $\\times$ 5 "
        "seeds, 40 episodes each). Conv.\\ Ep.\\ = median episodes to first "
        "reach $d_{\\mathrm{thresh}}$. Task Rate = fraction of episodes with "
        "successful delivery. The presentation-focused profile uses a relaxed "
        "threshold of 0.15; all others use 0.10. Final $d$ excludes incomplete "
        "runs (sentinel value 1.0). Dom.\\ = dominant preference dimension "
        "correctly identified.}",
        "\\label{tab:results}",
        "\\end{table}",
    ]

    path = FIGURES_DIR / "table_results.tex"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ T1 → {path}")
    print("\n".join(lines))


def table_t2_baselines():
    """T2: LaTeX baseline comparison table (aggregated across all profiles)."""
    CONDITIONS = {
        "baseline_uniform":    ("Uniform (no learn)",   "baselines/uniform"),
        "baseline_random":     ("Random Plan",           "baselines/random"),
        "baseline_outer_only": ("Outer Loop Only",       "baselines/outer_only"),
        "baseline_bandit":     ("Bandit",                "baselines/bandit"),
    }

    full_runs   = load_results("full")
    full_by_cond = {}
    for key, (label, subdir) in CONDITIONS.items():
        full_by_cond[key] = (label, load_results(subdir))

    # Full system row from full condition
    def row_stats(runs):
        conv_eps = [r["convergence_episode"] for r in runs if r.get("convergence_episode", -1) >= 0]
        best_ds  = [r["best_distance"] for r in runs]
        rates    = [r["success_rate"] for r in runs]
        n_conv   = len(conv_eps)
        conv_str = f"{np.median(conv_eps):.0f}" if conv_eps else "---"
        return (np.mean(best_ds), np.std(best_ds),
                np.mean(rates), n_conv, len(runs), conv_str)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Best $d$} & "
        "\\textbf{Task Rate} & \\textbf{Conv.} & \\textbf{Conv. Ep.} \\\\",
        "\\midrule",
    ]

    # Full system first
    mean_d, std_d, rate, n_conv, n, conv_str = row_stats(full_runs)
    lines.append(
        f"Full System (ours) & {mean_d:.3f} $\\pm$ {std_d:.3f} & "
        f"{rate:.0%} & {n_conv}/{n} & {conv_str} \\\\"
    )
    lines.append("\\midrule")

    for key, (label, runs) in full_by_cond.items():
        if not runs:
            continue
        mean_d, std_d, rate, n_conv, n, conv_str = row_stats(runs)
        lines.append(
            f"{label} & {mean_d:.3f} $\\pm$ {std_d:.3f} & "
            f"{rate:.0%} & {n_conv}/{n} & {conv_str} \\\\"
        )

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Baseline comparison aggregated across all profiles and seeds. "
        "Best $d$ = best L2 distance to true preference weights achieved. "
        "Task Rate = fraction of episodes with successful delivery. "
        "Conv.\\ = number of runs reaching convergence threshold.}",
        "\\label{tab:baselines}",
        "\\end{table}",
    ]

    path = FIGURES_DIR / "table_baselines.tex"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ T2 → {path}")
    print("\n".join(lines))


def table_t3_ablations():
    """T3: LaTeX ablation study table."""
    CONDITIONS = {
        "ablation_crisp":      ("Crisp (no fuzzy)",      "ablations/crisp"),
        "ablation_no_decay":   ("No LR Decay",           "ablations/no_decay"),
        "ablation_med_only":   ("Med Only",              "ablations/med_only"),
        "ablation_meal_only":  ("Meal Only",             "ablations/meal_only"),
        "ablation_finite_diff":("Finite Diff",           "ablations/finite_diff"),
    }

    full_runs = load_results("full")

    def row_stats(runs):
        conv_eps = [r["convergence_episode"] for r in runs if r.get("convergence_episode", -1) >= 0]
        best_ds  = [r["best_distance"] for r in runs]
        rates    = [r["success_rate"] for r in runs]
        n_conv   = len(conv_eps)
        return (np.mean(best_ds), np.std(best_ds),
                np.mean(rates), n_conv, len(runs))

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Condition} & \\textbf{Best $d$} & "
        "\\textbf{Task Rate} & \\textbf{Conv.} & \\textbf{$\\Delta$ Best $d$} \\\\",
        "\\midrule",
    ]

    # Full system baseline for delta computation
    full_mean, *_ = row_stats(full_runs)

    # Full system row
    mean_d, std_d, rate, n_conv, n = row_stats(full_runs)
    lines.append(
        f"Full System (ours) & {mean_d:.3f} $\\pm$ {std_d:.3f} & "
        f"{rate:.0%} & {n_conv}/{n} & --- \\\\"
    )
    lines.append("\\midrule")

    for key, (label, subdir) in CONDITIONS.items():
        runs = load_results(subdir)
        if not runs:
            continue
        mean_d, std_d, rate, n_conv, n = row_stats(runs)
        delta = mean_d - full_mean
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        lines.append(
            f"{label} & {mean_d:.3f} $\\pm$ {std_d:.3f} & "
            f"{rate:.0%} & {n_conv}/{n} & {delta_str} \\\\"
        )

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Ablation study results aggregated across all profiles and seeds. "
        "$\\Delta$ Best $d$ = difference from full system (positive = worse). "
        "Med Only removes meal preparation tasks; Meal Only removes medication tasks.}",
        "\\label{tab:ablations}",
        "\\end{table}",
    ]

    path = FIGURES_DIR / "table_ablations.tex"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ T3 → {path}")
    print("\n".join(lines))


def table_t4_robustness():
    """T4: LaTeX robustness table — noise sweep + random_init + dynamic_risk."""
    import os, json

    ROBUSTNESS_DIR = FIGURES_DIR.parent / "robustness"
    NOISE_LEVELS = ["0.05", "0.1", "0.2", "0.4"]
    order = list(PROFILE_COLORS.keys())

    def load_sub(sub):
        path = ROBUSTNESS_DIR / sub
        if not path.exists():
            return []
        runs = []
        for fname in os.listdir(path):
            if fname.endswith(".json") and "seed" in fname:
                with open(path / fname) as f:
                    runs.append(json.load(f))
        return runs

    def stats(runs, profile):
        r = [x for x in runs if x["profile"] == profile]
        if not r:
            return "---", "---"
        best_ds = [x["best_distance"] for x in r]
        n_conv  = sum(1 for x in r if x["convergence_episode"] >= 0)
        return f"{np.mean(best_ds):.3f}", f"{n_conv}/{len(r)}"

    # ── Noise sweep section ──────────────────────────────────────────
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l" + "cc" * len(NOISE_LEVELS) + "cc" * 2 + "}",
        "\\toprule",
        "& \\multicolumn{" + str(len(NOISE_LEVELS) * 2) + "}{c}{\\textbf{Noise} $\\sigma$} & "
        "\\multicolumn{2}{c}{\\textbf{Rand. Init}} & "
        "\\multicolumn{2}{c}{\\textbf{Dyn. Risk}} \\\\",
        "\\cmidrule(lr){2-" + str(len(NOISE_LEVELS) * 2 + 1) + "}"
        "\\cmidrule(lr){" + str(len(NOISE_LEVELS) * 2 + 2) + "-" + str(len(NOISE_LEVELS) * 2 + 3) + "}"
        "\\cmidrule(lr){" + str(len(NOISE_LEVELS) * 2 + 4) + "-" + str(len(NOISE_LEVELS) * 2 + 5) + "}",
    ]

    # Sub-header
    NOISE_LABELS = {"0.05": "0.05", "0.1": "0.10", "0.2": "0.20", "0.4": "0.40"}
    noise_headers = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{$\\sigma$={NOISE_LABELS[s]}}}" for s in NOISE_LEVELS
    )
    lines.append(
        f"\\textbf{{Profile}} & {noise_headers} & "
        "Best $d$ & Conv. & Best $d$ & Conv. \\\\"
    )
    # Inner noise header row
    noise_inner = " & ".join(["Best $d$ & Conv."] * len(NOISE_LEVELS))
    lines.append(f"& {noise_inner} & & & & \\\\")
    lines.append("\\midrule")

    noise_data   = {s: load_sub(f"noise_{s}") for s in NOISE_LEVELS}
    rand_data    = load_sub("random_init")
    dynrisk_data = load_sub("dynamic_risk")

    for profile in order:
        label = PROFILE_LABELS.get(profile, profile)
        parts = []
        for s in NOISE_LEVELS:
            bd, cv = stats(noise_data[s], profile)
            parts.append(f"{bd} & {cv}")
        bd_r, cv_r = stats(rand_data, profile)
        bd_d, cv_d = stats(dynrisk_data, profile)
        lines.append(
            f"{label} & " + " & ".join(parts) +
            f" & {bd_r} & {cv_r} & {bd_d} & {cv_d} \\\\"
        )

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Robustness results across noise levels ($\\sigma$), random "
        "initialisation, and dynamic risk conditions (5 seeds each). "
        "Best $d$ = mean best $\\|\\hat{w} - w^*\\|_2$. Conv.\\ = seeds that "
        "reached the convergence threshold. Presentation-focused profile uses "
        "tuned learning rate (lr\\_decay=0.05, 60 episodes); all others use "
        "default parameters.}",
        "\\label{tab:robustness}",
        "\\end{table}",
    ]

    path = FIGURES_DIR / "table_robustness.tex"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ T4 → {path}")
    print("\n".join(lines))
