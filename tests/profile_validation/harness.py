#!/usr/bin/env python3
"""
Profile Validation Test Harness
================================

Shared infrastructure for all profile integration tests.  Each test
script supplies only what is unique to its profile:

    - ProfileConfig  (profile name, dominant dim, task type, etc.)
    - route check    (weight profiles + assertions specific to the profile)

Everything else (single episode, mixed roster, argparse, summary printing) lives here.
"""

from __future__ import annotations

import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integration.integrator2 import FullMedicationDeliverySystem

DIMS = ["time", "safety", "battery", "proximity", "approach"]


# ── Config ────────────────────────────────────────────────────────────


@dataclass
class ProfileConfig:
    profile: str
    task_type: str           # "medication" | "meal"
    start_location: str      # "home" | "pantry"
    dominant_dim: int        # index in DIMS
    description: str         # argparse description line
    convergence_threshold: float = 0.05
    summary_dir: Optional[str] = None  # defaults to f"test_mixed_{profile}"

    @property
    def dominant_label(self) -> str:
        return DIMS[self.dominant_dim]

    @property
    def effective_summary_dir(self) -> str:
        return self.summary_dir or f"test_mixed_{self.profile}"


# ── System factory ────────────────────────────────────────────────────


def make_system(
    cfg: ProfileConfig,
    *,
    save_summaries: bool = False,
    summary_dir: Optional[str] = None,
    explore_sigma: float = 0.0,
    explore_decay: Optional[float] = None,
) -> FullMedicationDeliverySystem:
    kwargs: dict = dict(
        patient_profile_name=cfg.profile,
        preference_learning_rate=0.12,
        render=False,
        verbose=True,
        save_summaries=save_summaries,
        use_fuzzy=True,
        explore_sigma=explore_sigma,
    )
    if summary_dir:
        kwargs["summary_dir"] = summary_dir
    if explore_decay is not None:
        kwargs["explore_decay"] = explore_decay
    return FullMedicationDeliverySystem(**kwargs)


# ── Helpers ───────────────────────────────────────────────────────────


def _feature_vec(f: dict) -> List[float]:
    return [f.get(d, 0) for d in DIMS]


# ── Check 1: single episode plumbing check ────────────────────────────


def run_single_episode_check(cfg: ProfileConfig) -> bool:
    print("\n" + "=" * 80)
    print(f"TEST 1: SINGLE {cfg.task_type.upper()} EPISODE  [{cfg.profile}]")
    print("=" * 80)

    system = make_system(cfg)
    result = system.run_episode(
        start_location=cfg.start_location, task_type=cfg.task_type
    )

    success = result.get("success", False)
    has_features = "features" in result and len(result["features"]) > 0
    has_weights = "weights_after" in result
    has_distance = "distance_to_true" in result

    true_w = system.preference_learner.true_profile.weights
    dominant_correct = np.argmax(true_w) == cfg.dominant_dim

    print(f"\n{'='*60}")
    print("TEST 1 RESULTS:")
    print(f"  Episode success:             {'✓' if success else '✗'}")
    print(f"  Features computed:           {'✓' if has_features else '✗'}")
    print(f"  Weights updated:             {'✓' if has_weights else '✗'}")
    print(f"  Distance tracked:            {'✓' if has_distance else '✗'}")
    print(f"  {cfg.dominant_label.title()} dominant w*:       {'✓' if dominant_correct else '✗'}")
    if true_w is not None:
        print(f"  True weights (w*):           [{', '.join(f'{w:.2f}' for w in true_w)}]")

    if success:
        plan = result.get("plan_structure", {})
        if cfg.task_type == "medication":
            print(f"  Pharmacy:   {plan.get('pharmacy_choice', '?')}")
            print(f"  Supply:     {plan.get('supply_choice', '?')}")
            print(f"  Approach:   {plan.get('approach_choice', '?')}")
            print(f"  Recharge:   {'yes' if plan.get('recharge_added') else 'no'}")
        else:
            print(f"  Meal type:  {plan.get('meal_type', '?')}")
        print(f"  Features:   {result.get('features', {})}")

    passed = success and has_features and has_weights and dominant_correct
    print(f"\n  TEST 1: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"{'='*60}")
    system.close()
    return passed


# ── Check 2: weight-dependent choice (caller-supplied logic) ───────────
#
# Callers supply:
#   weight_profiles  – list of (label, weights[, start_location])
#   assert_fn(ctx)   – returns (passed: bool, notes: str)
#
# ctx keys available to assert_fn:
#   valid_plans, plan_keys, plan_by_label,
#   pharmacy_by_label, meal_type_by_label, distances_by_label, total


WeightProfileMed = Tuple[str, np.ndarray, str]       # label, w, start
WeightProfileMeal = Tuple[str, np.ndarray]            # label, w  (start fixed to cfg)
AssertFn = Callable[[dict], Tuple[bool, str]]


def run_route_choice_check(
    cfg: ProfileConfig,
    weight_profiles: list,
    assert_fn: AssertFn,
) -> bool:
    print("\n" + "=" * 80)
    print(f"TEST 2: WEIGHT-DEPENDENT CHOICE  [{cfg.profile}]")
    print("=" * 80)

    system = make_system(cfg)

    plans = []
    pharmacy_by_label: dict = {}
    meal_type_by_label: dict = {}
    distances_by_label: dict = {}
    plan_by_label: dict = {}

    for entry in weight_profiles:
        if len(entry) == 3:
            label, weights, start = entry
        else:
            label, weights = entry
            start = cfg.start_location

        print(f"\n--- Profile: {label} (start={start}) ---")
        print(f"  Weights: [{', '.join(f'{w:.2f}' for w in weights)}]")

        # Some callers use current_weights, others estimated_weights; set both
        system.preference_learner.estimated_weights = weights.copy()
        if hasattr(system.preference_learner, "current_weights"):
            system.preference_learner.current_weights = weights.copy()

        result = system.run_episode(start_location=start, task_type=cfg.task_type)

        if result.get("success"):
            plan = result["plan_structure"]
            plans.append(plan)
            plan_by_label[label] = plan
            pharmacy_by_label[label] = plan.get("pharmacy_choice")
            meal_type_by_label[label] = plan.get("meal_type")
            distances_by_label[label] = result.get("total_distance", 0)
            if cfg.task_type == "medication":
                print(
                    f"  → Route: {plan.get('pharmacy_choice')} → "
                    f"{plan.get('supply_choice')} → patient_{plan.get('approach_choice')}"
                    f"  ({distances_by_label[label]:.1f}m)"
                )
            else:
                print(f"  → Chose: {plan.get('meal_type')}  ({distances_by_label[label]:.1f}m)")
        else:
            plans.append(None)
            print(f"  → FAILED: {result.get('reason')}")

    valid_plans = [p for p in plans if p is not None]
    plan_keys: set = set()
    for p in valid_plans:
        if cfg.task_type == "medication":
            plan_keys.add(
                (p.get("pharmacy_choice"), p.get("supply_choice"), p.get("approach_choice"))
            )
        else:
            plan_keys.add(p.get("meal_type"))

    ctx = dict(
        valid_plans=valid_plans,
        plan_keys=plan_keys,
        plan_by_label=plan_by_label,
        pharmacy_by_label=pharmacy_by_label,
        meal_type_by_label=meal_type_by_label,
        distances_by_label=distances_by_label,
        total=len(weight_profiles),
    )

    print(f"\n{'='*60}")
    print("TEST 2 RESULTS:")
    print(f"  Successful episodes:     {len(valid_plans)}/{len(weight_profiles)}")
    print(f"  Unique route structures: {len(plan_keys)}")
    for k in plan_keys:
        print(f"    {k}")

    passed, notes = assert_fn(ctx)
    if notes:
        print(notes)
    print(f"\n  TEST 2: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"{'='*60}")
    system.close()
    return passed


# ── Check 3: mixed roster convergence ─────────────────────────────────


def run_mixed_roster_check(cfg: ProfileConfig, num_episodes: int = 10) -> bool:
    print("\n" + "=" * 80)
    print(f"TEST 3: MIXED TASK ROSTER ({num_episodes} episodes)  [{cfg.profile}]")
    print("=" * 80)

    system = make_system(
        cfg,
        save_summaries=True,
        summary_dir=cfg.effective_summary_dir,
        explore_sigma=0.15,
        explore_decay=0.2,
    )

    results = system.run_mixed_episodes(
        num_episodes=num_episodes,
        start_location="home",
        add_variability=True,
    )

    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]

    unique_plans: set = set()
    for p in system.plan_history:
        if p.get("task_type") == "meal":
            unique_plans.add(system._get_meal_plan_key(p))
        else:
            unique_plans.add(system._get_med_plan_key(p))

    med_results = [r for r in successful if r.get("task_type") == "medication"]
    meal_results = [r for r in successful if r.get("task_type") == "meal"]
    meal_types_seen = {r.get("plan_structure", {}).get("meal_type") for r in meal_results} - {None}

    med_features = [r["features"] for r in med_results if "features" in r]
    meal_features = [r["features"] for r in meal_results if "features" in r]

    feature_separation = med_centroid = meal_centroid = None
    if med_features and meal_features:
        med_centroid = np.array([_feature_vec(f) for f in med_features]).mean(axis=0)
        meal_centroid = np.array([_feature_vec(f) for f in meal_features]).mean(axis=0)
        feature_separation = float(np.linalg.norm(meal_centroid - med_centroid))

    distances = [r["distance_to_true"] for r in successful if "distance_to_true" in r]
    final_distance = distances[-1] if distances else None
    min_distance = min(distances) if distances else None
    improving = len(distances) >= 4 and np.mean(distances[-3:]) < np.mean(distances[:3])

    final_weights = (
        getattr(system.preference_learner, "current_weights", None)
        or system.preference_learner.estimated_weights
    )
    learned_dominant = int(np.argmax(final_weights)) if final_weights is not None else -1
    dominant_correct = learned_dominant == cfg.dominant_dim
    converged_eps = [i + 1 for i, d in enumerate(distances) if d < cfg.convergence_threshold]

    print(f"\n{'='*60}")
    print("TEST 3 RESULTS:")
    print(f"  Episodes: {len(successful)} successful / {len(results)} total")
    print(f"  Failed:   {len(failed)}")

    print(f"\n  PLAN DIVERSITY:")
    print(f"    Unique plans:        {len(unique_plans)}")
    print(f"    Medication episodes: {len(med_results)}")
    print(f"    Meal episodes:       {len(meal_results)}")
    print(f"    Meal types seen:     {meal_types_seen}")

    if feature_separation is not None:
        print(f"\n  FEATURE SPREAD:")
        print(f"    Med centroid:  [{', '.join(f'{v:.3f}' for v in med_centroid)}]")
        print(f"    Meal centroid: [{', '.join(f'{v:.3f}' for v in meal_centroid)}]")
        print(f"    Separation:    {feature_separation:.4f}")
        print(f"    Distinct:      {'✓' if feature_separation > 0.05 else '✗'}")

    if distances:
        print(f"\n  CONVERGENCE:")
        print(f"    Final distance to w*: {final_distance:.4f}")
        print(f"    Best distance to w*:  {min_distance:.4f}")
        print(f"    Improving trend:      {'✓' if improving else '✗'}")
        print(f"    Distance trajectory:  {[f'{d:.3f}' for d in distances]}")
        thresh_str = f"threshold: {cfg.convergence_threshold}"
        print(f"    Converged episodes:   {converged_eps if converged_eps else f'None ({thresh_str})'}")

    if final_weights is not None:
        print(f"\n  LEARNED WEIGHTS:")
        print(f"    [{', '.join(f'{w:.3f}' for w in final_weights)}]")
        print(f"    {cfg.dominant_label.title()} component:  {final_weights[cfg.dominant_dim]:.3f}")
        print(f"    Learned dominant:   {DIMS[learned_dominant]} ({final_weights[learned_dominant]:.1%})")
        print(f"    Dominant correct:   {'✓' if dominant_correct else '✗'}")

    diversity_ok = len(unique_plans) >= 3
    no_crashes = len(successful) >= num_episodes * 0.7
    spread_ok = (feature_separation or 0) > 0.05

    passed = diversity_ok and no_crashes
    print(f"\n  Diversity >= 3 plans:  {'✓' if diversity_ok else '✗'}")
    print(f"  Feature spread > 0.05: {'✓' if spread_ok else '✗'}")
    print(f"  Success rate >= 70%:   {'✓' if no_crashes else '✗'}")
    print(f"  Improving trend:       {'✓' if improving else '✗ (informational)'}")
    print(f"  Dominant correct:      {'✓' if dominant_correct else '✗ (informational)'}")
    print(f"\n  TEST 3: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"{'='*60}")
    system.close()
    return passed


# ── Suite runner & CLI ────────────────────────────────────────────────


def run_suite(
    cfg: ProfileConfig,
    test_2_fn: Callable[[], bool],
    *,
    test: Optional[int] = None,
    episodes: int = 10,
) -> int:
    """Run selected tests, print summary. Returns shell exit code."""
    results: dict = {}
    if test is None or test == 1:
        results[1] = run_single_episode_check(cfg)
    if test is None or test == 2:
        results[2] = test_2_fn()
    if test is None or test == 3:
        results[3] = run_mixed_roster_check(cfg, num_episodes=episodes)

    labels = {
        1: "Single episode",
        2: "Weight-dependent choice",
        3: "Mixed task roster",
    }
    print(f"\n{'='*80}")
    print(f"INTEGRATION TEST SUMMARY  [{cfg.profile}]")
    print(f"{'='*80}")
    for num, passed in results.items():
        print(f"  Test {num} ({labels[num]}): {'PASS ✓' if passed else 'FAIL ✗'}")
    all_passed = all(results.values())
    print(f"\n  Overall: {'ALL PASS ✓' if all_passed else 'SOME FAILED ✗'}")
    print(f"{'='*80}")
    return 0 if all_passed else 1


def make_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--test", type=int, choices=[1, 2, 3], default=None,
        help="Run specific test (1, 2, or 3). Default: all.",
    )
    p.add_argument(
        "--episodes", type=int, default=10,
        help="Number of episodes for test 3 (default: 10)",
    )
    return p
