#!/usr/bin/env python3
"""
Parallel Section 8 runner — one subprocess per profile.

Usage:
    python tests/run_section8_parallel.py --condition full --episodes 40 --seeds 5
    python tests/run_section8_parallel.py --condition all  --episodes 40 --seeds 5

Runs all profiles in parallel (5 subprocesses), waits for all to finish,
then merges per-profile result files into a single summary.json.
Conditions run sequentially: full → baselines → ablations → robustness.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROFILES = [
    "speed_oriented",
    "safety_first",
    "energy_conscious",
    "comfort_focused",
    "presentation_focused",
]

# Per-profile overrides for robustness condition.
# presentation_focused needs a slower decay and more episodes to converge.
ROBUSTNESS_PROFILE_OVERRIDES = {
    "presentation_focused": {
        "episodes": 60,
        "lr_decay": 0.05,
        "ema_alpha": 0.70,
    },
}

CONDITIONS_ORDER = ["full", "baselines", "ablations", "robustness"]

RESULTS_DIR = Path("results/section8")

# Maps condition name → results subdirectory (mirrors runner internals)
CONDITION_SUBDIRS = {
    "full":       ["full"],
    "baselines":  ["baselines/uniform", "baselines/random",
                   "baselines/outer_only", "baselines/bandit"],
    "ablations":  ["ablations/crisp", "ablations/no_decay",
                   "ablations/med_only", "ablations/meal_only",
                   "ablations/finite_diff"],
    "robustness": ["robustness/noise_0.05", "robustness/noise_0.1",
                   "robustness/noise_0.2", "robustness/random_init",
                   "robustness/dynamic_risk", "robustness/ambiguous"],
}


def run_profiles_parallel(condition: str, episodes: int, seeds: int) -> bool:
    """Launch one subprocess per profile for a given condition, wait for all."""
    runner = Path(__file__).parent / "run_section8_experiments.py"
    procs = []
    logs  = []

    print(f"\n{'='*70}")
    print(f"  LAUNCHING: {condition.upper()} — {len(PROFILES)} profiles in parallel")
    print(f"  episodes={episodes}  seeds={seeds}")
    print(f"{'='*70}")

    for profile in PROFILES:
        log_path = RESULTS_DIR / f"_log_{condition}_{profile}.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w")
        logs.append((profile, log_path, log_file))

        overrides = ROBUSTNESS_PROFILE_OVERRIDES.get(profile, {}) if condition == "robustness" else {}
        profile_episodes = overrides.get("episodes", episodes)

        cmd = [
            sys.executable, str(runner),
            "--condition", condition,
            "--profile",   profile,
            "--episodes",  str(profile_episodes),
            "--seeds",     str(seeds),
        ]
        if "lr_decay" in overrides:
            cmd += ["--lr_decay", str(overrides["lr_decay"])]
        if "ema_alpha" in overrides:
            cmd += ["--ema_alpha", str(overrides["ema_alpha"])]
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((profile, p))
        print(f"  [{profile}] started (pid {p.pid}) → {log_path}")

    # Wait for all
    failed = []
    for profile, p in procs:
        p.wait()
        if p.returncode != 0:
            failed.append(profile)
            print(f"  [{profile}] FAILED (exit {p.returncode})")
        else:
            print(f"  [{profile}] done")

    for _, _, f in logs:
        f.close()

    if failed:
        print(f"\n  ⚠  {len(failed)} profile(s) failed: {failed}")
        print("  Check log files in results/section8/ for details.")
        return False
    return True


def merge_summaries(condition: str):
    """Collect all per-profile result JSONs and write a merged summary.json."""
    subdirs = CONDITION_SUBDIRS.get(condition, [condition])

    for subdir in subdirs:
        out_dir = RESULTS_DIR / subdir
        if not out_dir.exists():
            continue

        result_files = sorted(out_dir.glob("*_seed*.json"))
        if not result_files:
            continue

        summary = []
        for rf in result_files:
            try:
                with open(rf) as f:
                    d = json.load(f)
                summary.append({
                    "condition":           d.get("condition", condition),
                    "profile":             d.get("profile", ""),
                    "seed":                d.get("seed", 0),
                    "convergence_episode": d.get("convergence_episode", -1),
                    "final_distance":      d.get("final_distance", None),
                    "best_distance":       d.get("best_distance", None),
                    "success_rate":        d.get("success_rate", 0),
                    "unique_plans":        d.get("unique_plans", 0),
                    "wall_time":           d.get("wall_time", 0),
                })
            except Exception as e:
                print(f"  Warning: could not read {rf}: {e}")

        path = out_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  → Merged summary ({len(summary)} runs): {path}")


def print_condition_summary(condition: str):
    subdirs = CONDITION_SUBDIRS.get(condition, [condition])
    for subdir in subdirs:
        path = RESULTS_DIR / subdir / "summary.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        converged = sum(1 for r in data if (r.get("convergence_episode") or -1) >= 0)
        rates     = [r["success_rate"] for r in data if r.get("success_rate") is not None]
        dists     = [r["best_distance"] for r in data if r.get("best_distance") is not None]
        print(f"\n  {subdir.upper():20s}  "
              f"runs={len(data)}  converged={converged}  "
              f"avg_best_d={sum(dists)/len(dists):.4f}  "
              f"avg_rate={sum(rates)/len(rates)*100:.0f}%" if dists and rates else f"\n  {subdir}: no data")


def main():
    parser = argparse.ArgumentParser(description="Parallel Section 8 runner")
    parser.add_argument("--condition", default="all",
                        choices=["all", "full", "baselines", "ablations", "robustness"])
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seeds",    type=int, default=5)
    args = parser.parse_args()

    conditions = CONDITIONS_ORDER if args.condition == "all" else [args.condition]

    t0 = time.time()
    print(f"\n{'#'*70}")
    print(f"  SECTION 8 PARALLEL RUNNER")
    print(f"  Conditions: {conditions}")
    print(f"  Episodes: {args.episodes}  Seeds: {args.seeds}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    for condition in conditions:
        t_cond = time.time()
        ok = run_profiles_parallel(condition, args.episodes, args.seeds)
        merge_summaries(condition)
        print_condition_summary(condition)
        elapsed = time.time() - t_cond
        status = "✓" if ok else "⚠"
        print(f"\n  {status} {condition.upper()} done in {elapsed/60:.1f} min")

    total = time.time() - t0
    print(f"\n{'#'*70}")
    print(f"  ALL DONE in {total/60:.1f} min")
    print(f"  Results: {RESULTS_DIR.resolve()}/")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
