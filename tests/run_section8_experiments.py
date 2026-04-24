#!/usr/bin/env python3
"""
Section 8 Experiment Runner — MLC Stack Paper
==============================================

Runs all experimental conditions defined in Section 7.7:
  - Full system (5 profiles × N seeds)
  - Baselines (uniform, random, outer-only, bandit)
  - Ablations (crisp, no-decay, single-task, finite-diff)
  - Robustness (noise sweep, init sensitivity, dynamic risk, ambiguous profiles)

Results are saved as JSON for plotting by generate_section8_figures.py.

Usage:
    python run_section8_experiments.py                    # everything
    python run_section8_experiments.py --condition full    # just full system
    python run_section8_experiments.py --condition baselines
    python run_section8_experiments.py --condition ablations
    python run_section8_experiments.py --condition robustness
    python run_section8_experiments.py --profile speed_oriented  # single profile
    python run_section8_experiments.py --episodes 40 --seeds 5   # custom

Output:
    results/section8/
    ├── full/                     # Full system convergence
    │   ├── speed_oriented_seed0.json
    │   └── summary.json
    ├── baselines/                # Baseline comparisons
    │   ├── uniform/
    │   ├── random/
    │   ├── outer_only/
    │   └── bandit/
    ├── ablations/                # Ablation studies
    │   ├── crisp/
    │   ├── no_decay/
    │   ├── med_only/
    │   ├── meal_only/
    │   └── finite_diff/
    ├── robustness/               # Robustness conditions
    │   ├── noise_0.05/ ... noise_0.40/
    │   ├── random_init/
    │   ├── dynamic_risk/
    │   └── ambiguous/
    └── figures/                  # Generated plots

Episode JSON schema (per episode record):
    episode             int     Episode number
    task_type           str     "medication" | "meal"
    success             bool    Whether episode completed
    features            dict    {time, safety, battery, proximity, approach}
    weights_before      list    [5 floats] before update
    weights_after       list    [5 floats] after update
    distance_to_true    float   L2 distance to w*
    plan_signature      str     Action sequence hash
    meal_type           str     "sandwich" | "soup" | "full_meal" | ""
    learner_mse         float   Preference learner loss (for B6)
    translator_params   dict    Translator φ params snapshot (for B7)
    trajectory_xy       list    [[x,y], ...] robot path (for B8)
    battery_used_pct    float   Battery consumed as % (for B9)
    path_efficiency     float   Euclidean/actual path ratio (for B9)
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ── Project imports ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integration.integrator2 import FullMedicationDeliverySystem

# ── Constants ────────────────────────────────────────────────────────
PROFILES = [
    "speed_oriented",
    "safety_first",
    "energy_conscious",
    "comfort_focused",
    "presentation_focused",
]

AMBIGUOUS_PROFILES = {
    "mild_speed": [0.30, 0.20, 0.18, 0.17, 0.15],
    "mild_safety": [0.18, 0.30, 0.20, 0.17, 0.15],
}

DIM_NAMES = ["time", "safety", "battery", "proximity", "approach"]

RESULTS_DIR = Path("results/section8")

# Profile-specific convergence thresholds
CONVERGENCE_THRESHOLDS = {
    "presentation_focused": 0.15,
}
DEFAULT_THRESHOLD = 0.10


# =====================================================================
# DATA STRUCTURES
# =====================================================================


@dataclass
class ExperimentResult:
    condition: str
    profile: str
    seed: int
    episodes: List[Dict]
    convergence_episode: int = -1
    final_distance: float = 1.0
    best_distance: float = 1.0
    success_rate: float = 0.0
    unique_plans: int = 0
    wall_time: float = 0.0
    config: Dict = field(default_factory=dict)


# =====================================================================
# HELPERS: Extract extra fields from system internals
# =====================================================================


def extract_learner_mse(system) -> Optional[float]:
    """Get latest MSE/loss from the preference learner (for B6)."""
    # Try direct attribute
    for attr in ("last_mse", "last_loss", "mse"):
        val = getattr(system.preference_learner, attr, None)
        if val is not None:
            return float(val)
    # Try loss history
    hist = getattr(system.preference_learner, "loss_history", None)
    if hist and len(hist) > 0:
        return float(hist[-1])
    return None


def extract_translator_params(system) -> Optional[Dict]:
    """Snapshot current translator φ parameters (for B7)."""
    translator = getattr(system, "translator", None)
    if translator is None:
        return None

    # Try get_params() method first
    try:
        p = translator.get_params()
        if isinstance(p, dict):
            out = {}
            for k, v in p.items():
                if isinstance(v, np.ndarray):
                    out[k] = v.tolist()
                elif isinstance(v, (int, float, np.floating)):
                    out[k] = float(v)
                else:
                    out[k] = str(v)
            return out if out else None
    except (AttributeError, TypeError):
        pass

    # Fallback: probe common attribute names
    params = {}
    for attr in [
        "q_base",
        "q_time",
        "q_safety",
        "q_battery",
        "q_proximity",
        "r_base",
        "r_time",
        "r_safety",
        "weights",
        "bias",
    ]:
        val = getattr(translator, attr, None)
        if val is not None:
            if isinstance(val, np.ndarray):
                params[attr] = val.tolist()
            elif isinstance(val, (int, float, np.floating)):
                params[attr] = float(val)
    return params if params else None


def extract_trajectory_xy(result: Dict) -> Optional[List]:
    """Get robot xy path from episode result (for B8)."""
    # Direct field
    traj = result.get("trajectory_xy")
    if traj is not None:
        return traj

    # From states array
    states = result.get("states")
    if states and len(states) > 0:
        try:
            return [[float(s[0]), float(s[1])] for s in states]
        except (IndexError, TypeError):
            pass

    # From trajectory field (some integrators use this name)
    traj = result.get("trajectory")
    if traj and len(traj) > 0:
        try:
            return [[float(s[0]), float(s[1])] for s in traj]
        except (IndexError, TypeError):
            pass

    return None


def extract_battery_and_efficiency(result: Dict) -> tuple:
    """Get battery usage % and path efficiency ratio (for B9)."""
    battery_pct = result.get("battery_used_pct")
    path_eff = result.get("path_efficiency")

    # Derive battery from features if not directly available
    if battery_pct is None:
        feats = result.get("features", {})
        if isinstance(feats, dict):
            bat = feats.get("battery")
            if bat is not None:
                battery_pct = float(bat) * 100.0

    # Derive path efficiency from trajectory
    if path_eff is None:
        traj = extract_trajectory_xy(result)
        if traj and len(traj) >= 2:
            pts = np.array(traj)
            euclidean = np.linalg.norm(pts[-1] - pts[0])
            actual = sum(
                np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)
            )
            if actual > 1e-6:
                path_eff = float(euclidean / actual)

    return battery_pct, path_eff


# =====================================================================
# CORE EXPERIMENT RUNNER
# =====================================================================


def run_experiment(
    profile: str,
    num_episodes: int = 40,
    seed: int = 0,
    condition: str = "full",
    # System config
    learning_rate: float = 0.12,
    explore_sigma: float = 0.15,
    explore_decay: float = 0.2,
    use_fuzzy: bool = True,
    use_nav_stack: bool = True,
    # Baseline flags
    fix_weights: bool = False,
    random_planning: bool = False,
    fix_translator: bool = False,
    bandit_mode: bool = False,
    # Ablation flags
    use_finite_diff: bool = False,
    single_task: Optional[str] = None,
    # Robustness
    rating_noise: float = 0.1,
    dynamic_risk_perturbation: float = 0.0,
    random_init: bool = False,
    # Learning schedule
    lr_decay: float = 0.15,
    ema_alpha: float = 0.60,
) -> ExperimentResult:
    """Run a single experiment and collect episode-level data."""

    np.random.seed(seed)
    t_start = time.time()
    threshold = CONVERGENCE_THRESHOLDS.get(profile, DEFAULT_THRESHOLD)

    print(f"\n{'='*70}")
    print(
        f"  {condition.upper()} | {profile} | seed={seed} | "
        f"{num_episodes} ep | thresh={threshold}"
    )
    print(f"{'='*70}")

    # ── Build system ─────────────────────────────────────────────────
    system = FullMedicationDeliverySystem(
        patient_profile_name=profile,
        preference_learning_rate=learning_rate if not fix_weights else 0.0,
        render=False,
        verbose=False,
        save_summaries=False,
        use_nav_stack=use_nav_stack,
        use_fuzzy=use_fuzzy,
        explore_sigma=explore_sigma if not fix_weights else 0.0,
        explore_decay=explore_decay,
        fix_translator=fix_translator,
        use_finite_diff=use_finite_diff,
        rating_noise=rating_noise,
        dynamic_risk_perturbation=dynamic_risk_perturbation,
        lr_decay=lr_decay,
        ema_alpha=ema_alpha,
    )

    # Random init override
    if random_init:
        try:
            system.preference_learner.estimated_weights = np.random.dirichlet(
                np.ones(5)
            )
        except AttributeError:
            pass

    # Bandit: disable gradient, we'll do EMA manually
    if bandit_mode:
        try:
            system.preference_learner.learning_rate = 0.0
        except AttributeError:
            pass

    # ── Episode loop ─────────────────────────────────────────────────
    episode_records = []
    convergence_episode = -1
    best_distance = 1.0
    successes = 0
    plan_signatures = set()
    bandit_weights = np.ones(5) / 5.0
    bandit_alpha = 0.15

    for ep in range(num_episodes):
        # Task type
        if single_task:
            task_type = single_task
        else:
            task_type = "medication" if ep % 2 == 0 else "meal"

        start_loc = "home" if task_type == "medication" else "pantry"

        # Weights before
        try:
            w_before = system.preference_learner.get_current_weights().tolist()
        except Exception:
            w_before = [0.2] * 5

        # Baseline overrides
        if fix_weights:
            try:
                system.preference_learner.estimated_weights = np.array([0.2] * 5)
            except AttributeError:
                pass

        if random_planning:
            try:
                system.preference_learner.estimated_weights = np.random.dirichlet(
                    np.ones(5)
                )
            except AttributeError:
                pass

        # Run episode___________________________________________________________________________________
        is_meal = task_type == "meal"
        from tasks.medication_delivery.task_actions import TaskAction  # type: ignore
        from tasks.meal_preparation.task_actions import MealAction # type: ignore
        
        
        generated_classes = {}
        total_available_actions = []
        if is_meal:
            for action in MealAction:
                class_name = action.name   

                new_cls = type(
                    class_name,           
                    (FullMedicationDeliverySystem,),          
                    {
                        "task_action": action,
                        "action_name": action.value,
                    }
                )
                generated_classes[class_name] = new_cls

                obj = new_cls()
                total_available_actions.append(obj)
        else:
            for action in TaskAction:
                class_name = action.name   

                new_cls = type(
                    class_name,           
                    (FullMedicationDeliverySystem,),           
                    {
                        "task_action": action,
                        "action_name": action.value,
                    }
                )
                generated_classes[class_name] = new_cls

                obj = new_cls()
                total_available_actions.append(obj) 
            
            
            
            
                   
        try:
            result = system.run_episode(
                total_available_actions,
                start_location=start_loc,
                task_type=task_type
            )
            success = result.get("success", False)
        except Exception as e:
            print(f"  Ep {ep} CRASHED: {e}")
            result = {}
            success = False

        # Weights after
        try:
            w_after = system.preference_learner.get_current_weights().tolist()
        except Exception:
            w_after = w_before

        # Bandit update
        if bandit_mode and success and "features" in result:
            feats = result["features"]
            feat_vals = (
                [feats.get(d, 0.5) for d in DIM_NAMES]
                if isinstance(feats, dict)
                else list(feats)[:5]
            )
            signal = np.array([1.0 - f for f in feat_vals])
            signal /= signal.sum() + 1e-8
            bandit_weights = (1 - bandit_alpha) * bandit_weights + bandit_alpha * signal
            bandit_weights /= bandit_weights.sum()
            try:
                system.preference_learner.estimated_weights = bandit_weights.copy()
            except AttributeError:
                pass
            w_after = bandit_weights.tolist()

        # ── Extract all fields ───────────────────────────────────────
        dist = result.get("distance_to_true", 1.0)
        features = result.get("features", {})
        plan = result.get("plan_structure", {})
        plan_sig = str(plan.get("plan_signature", plan.get("action_sequence", "")))
        meal_type = plan.get("meal_type", "")

        # New fields (B6-B9)
        learner_mse = extract_learner_mse(system)
        translator_params = extract_translator_params(system)
        trajectory_xy = extract_trajectory_xy(result)
        battery_pct, path_eff = extract_battery_and_efficiency(result)

        if success:
            successes += 1
            plan_signatures.add(plan_sig)
        if dist < best_distance:
            best_distance = dist
        if convergence_episode < 0 and dist <= threshold:
            convergence_episode = ep

        record = {
            "episode": ep,
            "task_type": task_type,
            "success": success,
            "features": features if isinstance(features, dict) else {},
            "weights_before": w_before,
            "weights_after": w_after,
            "distance_to_true": dist,
            "plan_signature": plan_sig,
            "meal_type": meal_type,
            # New fields for blocked figures
            "learner_mse": learner_mse,
            "translator_params": translator_params,
            "trajectory_xy": trajectory_xy,
            "battery_used_pct": battery_pct,
            "path_efficiency": path_eff,
        }
        episode_records.append(record)

        sym = "✓" if success else "✗"
        extras = ""
        if learner_mse is not None:
            extras += f"  mse={learner_mse:.4f}"
        if battery_pct is not None:
            extras += f"  bat={battery_pct:.0f}%"
        print(
            f"  Ep {ep:2d} [{task_type[:4]}] {sym}  d={dist:.4f}"
            f"  w={[f'{x:.2f}' for x in w_after]}{extras}"
        )

    wall_time = time.time() - t_start

    exp_result = ExperimentResult(
        condition=condition,
        profile=profile,
        seed=seed,
        episodes=episode_records,
        convergence_episode=convergence_episode,
        final_distance=(
            episode_records[-1]["distance_to_true"] if episode_records else 1.0
        ),
        best_distance=best_distance,
        success_rate=successes / num_episodes if num_episodes > 0 else 0.0,
        unique_plans=len(plan_signatures),
        wall_time=wall_time,
        config={
            "num_episodes": num_episodes,
            "learning_rate": learning_rate,
            "explore_sigma": explore_sigma,
            "explore_decay": explore_decay,
            "use_fuzzy": use_fuzzy,
            "fix_weights": fix_weights,
            "random_planning": random_planning,
            "fix_translator": fix_translator,
            "bandit_mode": bandit_mode,
            "single_task": single_task,
            "rating_noise": rating_noise,
            "random_init": random_init,
            "seed": seed,
            "threshold": threshold,
        },
    )

    print(
        f"\n  Summary: conv={convergence_episode}, "
        f"best_d={best_distance:.4f}, "
        f"final_d={exp_result.final_distance:.4f}, "
        f"plans={len(plan_signatures)}, "
        f"rate={exp_result.success_rate:.0%}, "
        f"time={wall_time:.1f}s"
    )

    system.close()
    return exp_result


# =====================================================================
# SAVE / LOAD
# =====================================================================


def save_result(result: ExperimentResult, subdir: str = ""):
    out_dir = RESULTS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{result.profile}_seed{result.seed}.json"
    path = out_dir / fname

    data = {
        "condition": result.condition,
        "profile": result.profile,
        "seed": result.seed,
        "convergence_episode": result.convergence_episode,
        "final_distance": result.final_distance,
        "best_distance": result.best_distance,
        "success_rate": result.success_rate,
        "unique_plans": result.unique_plans,
        "wall_time": result.wall_time,
        "config": result.config,
        "episodes": result.episodes,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  → Saved: {path}")
    return path


def save_summary(results: List[ExperimentResult], subdir: str):
    out_dir = RESULTS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        {
            "condition": r.condition,
            "profile": r.profile,
            "seed": r.seed,
            "convergence_episode": r.convergence_episode,
            "final_distance": r.final_distance,
            "best_distance": r.best_distance,
            "success_rate": r.success_rate,
            "unique_plans": r.unique_plans,
            "wall_time": r.wall_time,
        }
        for r in results
    ]
    path = out_dir / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → Summary: {path}")


# =====================================================================
# EXPERIMENT CONDITIONS
# =====================================================================


def run_full_system(profiles, num_episodes, seeds, lr_decay=0.15, ema_alpha=0.60):
    """8.1: Full system convergence for all profiles."""
    print("\n" + "=" * 70)
    print("  CONDITION: FULL SYSTEM")
    print("=" * 70)
    results = []
    for profile in profiles:
        for seed in range(seeds):
            r = run_experiment(
                profile=profile, num_episodes=num_episodes, seed=seed, condition="full",
                lr_decay=lr_decay, ema_alpha=ema_alpha,
            )
            save_result(r, "full")
            results.append(r)
    save_summary(results, "full")
    return results


def run_baselines(profiles, num_episodes, seeds):
    """8.2: Baseline comparisons (4 baselines × profiles × seeds)."""
    print("\n" + "=" * 70)
    print("  CONDITION: BASELINES")
    print("=" * 70)
    results = []
    for profile in profiles:
        for seed in range(seeds):
            for cond, kwargs in [
                ("baseline_uniform", {"fix_weights": True}),
                ("baseline_random", {"random_planning": True}),
                ("baseline_outer_only", {"fix_translator": True}),
                ("baseline_bandit", {"bandit_mode": True}),
            ]:
                subdir = "baselines/" + cond.replace("baseline_", "")
                r = run_experiment(
                    profile=profile,
                    num_episodes=num_episodes,
                    seed=seed,
                    condition=cond,
                    **kwargs,
                )
                save_result(r, subdir)
                results.append(r)
    save_summary(results, "baselines")
    return results


def run_ablations(profiles, num_episodes, seeds):
    """8.3: Ablation studies (5 ablations × profiles × seeds)."""
    print("\n" + "=" * 70)
    print("  CONDITION: ABLATIONS")
    print("=" * 70)
    results = []
    for profile in profiles:
        for seed in range(seeds):
            for cond, kwargs in [
                ("ablation_crisp", {"use_fuzzy": False}),
                ("ablation_no_decay", {"explore_decay": 0.0}),
                ("ablation_med_only", {"single_task": "medication"}),
                ("ablation_meal_only", {"single_task": "meal"}),
                ("ablation_finite_diff", {"use_finite_diff": True}),
            ]:
                subdir = "ablations/" + cond.replace("ablation_", "")
                r = run_experiment(
                    profile=profile,
                    num_episodes=num_episodes,
                    seed=seed,
                    condition=cond,
                    **kwargs,
                )
                save_result(r, subdir)
                results.append(r)
    save_summary(results, "ablations")
    return results


def run_robustness(profiles, num_episodes, seeds, lr_decay=0.15, ema_alpha=0.60, sub=None):
    """8.7: Robustness conditions. sub= filters to a single sub-condition."""
    print("\n" + "=" * 70)
    print("  CONDITION: ROBUSTNESS" + (f" [{sub}]" if sub else ""))
    print("=" * 70)
    results = []

    # R1: Noise sweep
    if sub in (None, "noise"):
        for noise in [0.05, 0.10, 0.20, 0.40]:
            for profile in profiles:
                for seed in range(seeds):
                    r = run_experiment(
                        profile=profile,
                        num_episodes=num_episodes,
                        seed=seed,
                        condition=f"robust_noise_{noise}",
                        rating_noise=noise,
                        lr_decay=lr_decay,
                        ema_alpha=ema_alpha,
                    )
                    save_result(r, f"robustness/noise_{noise}")
                    results.append(r)

    # R2: Random init
    if sub in (None, "random_init"):
        for profile in profiles:
            for seed in range(seeds):
                r = run_experiment(
                    profile=profile,
                    num_episodes=num_episodes,
                    seed=seed,
                    condition="robust_random_init",
                    random_init=True,
                    lr_decay=lr_decay,
                    ema_alpha=ema_alpha,
                )
                save_result(r, "robustness/random_init")
                results.append(r)

    # R3: Dynamic risk
    if sub in (None, "dynamic_risk"):
        for profile in profiles:
            for seed in range(seeds):
                r = run_experiment(
                    profile=profile,
                    num_episodes=num_episodes,
                    seed=seed,
                    condition="robust_dynamic_risk",
                    dynamic_risk_perturbation=0.15,
                    lr_decay=lr_decay,
                    ema_alpha=ema_alpha,
                )
                save_result(r, "robustness/dynamic_risk")
                results.append(r)

    # R4: Ambiguous profiles
    if sub in (None, "ambiguous"):
        for amb_name in AMBIGUOUS_PROFILES:
            for seed in range(seeds):
                try:
                    r = run_experiment(
                        profile=amb_name,
                        num_episodes=num_episodes,
                        seed=seed,
                        condition=f"robust_ambiguous_{amb_name}",
                    )
                    save_result(r, "robustness/ambiguous")
                    results.append(r)
                except (KeyError, Exception) as e:
                    print(f"  SKIP {amb_name}: {e} (register profile first)")

    save_summary(results, "robustness")
    return results


# =====================================================================
# MAIN
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="Section 8 Experiment Runner")
    parser.add_argument(
        "--condition",
        type=str,
        default="all",
        choices=["all", "full", "baselines", "ablations", "robustness"],
    )
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--lr_decay", type=float, default=0.15)
    parser.add_argument("--ema_alpha", type=float, default=0.60)
    parser.add_argument("--robustness_sub", type=str, default=None,
                        choices=["noise", "random_init", "dynamic_risk", "ambiguous"],
                        help="Run only a specific robustness sub-condition")
    args = parser.parse_args()

    profiles = [args.profile] if args.profile else PROFILES

    print(f"\n{'#'*70}")
    print(f"  MLC STACK — SECTION 8 EXPERIMENTS")
    print(f"  Conditions: {args.condition}")
    print(f"  Profiles:   {profiles}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Output:     {RESULTS_DIR}/")
    print(f"  Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    dispatch = {
        "full": run_full_system,
        "baselines": run_baselines,
        "ablations": run_ablations,
        "robustness": run_robustness,
    }

    lr_kwargs = dict(lr_decay=args.lr_decay, ema_alpha=args.ema_alpha)

    if args.condition == "all":
        for name, fn in dispatch.items():
            if name in ("full", "robustness"):
                all_results[name] = fn(profiles, args.episodes, args.seeds, **lr_kwargs)
            else:
                all_results[name] = fn(profiles, args.episodes, args.seeds)
    else:
        fn = dispatch[args.condition]
        if args.condition in ("full", "robustness"):
            extra = dict(**lr_kwargs)
            if args.condition == "robustness" and args.robustness_sub:
                extra["sub"] = args.robustness_sub
            all_results[args.condition] = fn(profiles, args.episodes, args.seeds, **extra)
        else:
            all_results[args.condition] = fn(profiles, args.episodes, args.seeds)

    # Grand summary
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'#'*70}")
    for cond, results in all_results.items():
        n_conv = sum(1 for r in results if r.convergence_episode >= 0)
        avg_best = np.mean([r.best_distance for r in results]) if results else 0
        avg_rate = np.mean([r.success_rate for r in results]) if results else 0
        total_t = sum(r.wall_time for r in results)
        print(
            f"\n  {cond.upper():20s}  runs={len(results):3d}  "
            f"converged={n_conv:3d}  avg_best_d={avg_best:.4f}  "
            f"avg_rate={avg_rate:.0%}  time={total_t:.0f}s"
        )

    print(f"\n  Results: {RESULTS_DIR.resolve()}/")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
