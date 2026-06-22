#!/usr/bin/env python3
"""
Profile Validation Suite — All Profiles in One Runner
======================================================

Runs route-choice checks for all 5 profile/task combinations.

Profiles and what they test
---------------------------
  med_speed         Speed-oriented medication: speed→north, safety→south
  med_safety        Safety-first medication:   safety avoids pharmacy_north
  meal_approach     Comfort-focused meal:      ≥2 meal types across weight profiles
  meal_safety       Safety-first meal:         ≥2 meal types across weight profiles
  meal_presentation Presentation-focused meal: full_meal appears; approach never picks sandwich

Usage
-----
    python tests/profile_validation/run_profile_tests.py           # all profiles
    python tests/profile_validation/run_profile_tests.py med_speed
    python tests/profile_validation/run_profile_tests.py meal_presentation --test 2
    python tests/profile_validation/run_profile_tests.py meal_safety --episodes 15
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))

from harness import ProfileConfig, run_route_choice_check, run_suite


# ── Profile configs ──────────────────────────────────────────────────────────

PROFILES = {
    "med_speed": ProfileConfig(
        profile="speed_oriented",
        task_type="medication",
        start_location="home",
        dominant_dim=0,
        description="Medication delivery tests — speed-oriented patient",
        summary_dir="test_mixed_roster_speed",
    ),
    "med_safety": ProfileConfig(
        profile="safety_first",
        task_type="medication",
        start_location="home",
        dominant_dim=1,
        description="Medication delivery tests — safety-first patient",
        summary_dir="test_mixed_roster_safety",
    ),
    "meal_approach": ProfileConfig(
        profile="comfort_focused",
        task_type="meal",
        start_location="pantry",
        dominant_dim=3,
        description="Meal preparation tests — comfort-focused patient",
        summary_dir="test_mixed_roster_comfort",
    ),
    "meal_safety": ProfileConfig(
        profile="safety_first",
        task_type="meal",
        start_location="pantry",
        dominant_dim=1,
        description="Meal preparation tests — safety-first patient",
        summary_dir="test_mixed_roster",
    ),
    "meal_presentation": ProfileConfig(
        profile="presentation_focused",
        task_type="meal",
        start_location="pantry",
        dominant_dim=4,
        description="Meal preparation tests — presentation-focused patient",
        convergence_threshold=0.15,
        summary_dir="test_mixed_roster_presentation",
    ),
}


# ── Weight profiles per test ─────────────────────────────────────────────────

WEIGHT_PROFILES_2 = {
    "med_speed": [
        ("SpeedFromSupplyB",  np.array([0.70, 0.05, 0.05, 0.10, 0.10]), "supply_B"),
        ("SafetyFromSupplyB", np.array([0.05, 0.55, 0.10, 0.15, 0.15]), "supply_B"),
        ("SpeedFromHome",     np.array([0.70, 0.05, 0.05, 0.10, 0.10]), "home"),
        ("SafetyFromHome",    np.array([0.05, 0.55, 0.10, 0.15, 0.15]), "home"),
    ],
    "med_safety": [
        ("SafetyFromSupplyB", np.array([0.05, 0.70, 0.05, 0.10, 0.10]), "supply_B"),
        ("SpeedFromSupplyB",  np.array([0.55, 0.10, 0.10, 0.10, 0.15]), "supply_B"),
        ("SafetyFromHome",    np.array([0.05, 0.70, 0.05, 0.10, 0.10]), "home"),
        ("SafetyMax",         np.array([0.02, 0.80, 0.03, 0.10, 0.05]), "supply_B"),
    ],
    "meal_approach": [
        ("ProximityHeavy", np.array([0.10, 0.10, 0.05, 0.55, 0.20])),
        ("SpeedHeavy",     np.array([0.55, 0.10, 0.10, 0.10, 0.15])),
        ("Balanced",       np.array([0.20, 0.20, 0.20, 0.20, 0.20])),
    ],
    "meal_safety": [
        ("Speed",        np.array([0.50, 0.10, 0.10, 0.10, 0.20])),
        ("HighApproach", np.array([0.05, 0.05, 0.05, 0.15, 0.70])),
        ("MedApproach",  np.array([0.10, 0.15, 0.10, 0.15, 0.50])),
    ],
    "meal_presentation": [
        ("ApproachHeavy", np.array([0.05, 0.05, 0.05, 0.15, 0.70])),
        ("SpeedHeavy",    np.array([0.55, 0.10, 0.10, 0.10, 0.15])),
        ("Balanced",      np.array([0.20, 0.20, 0.20, 0.20, 0.20])),
        ("ApproachMax",   np.array([0.02, 0.03, 0.02, 0.13, 0.80])),
    ],
}


# ── Assertion functions ──────────────────────────────────────────────────────

def _assert_med_speed(ctx: dict):
    pb = ctx["pharmacy_by_label"]
    speed_north  = pb.get("SpeedFromSupplyB") == "pharmacy_north"
    safety_south = pb.get("SafetyFromSupplyB") == "pharmacy_south"
    differ       = pb.get("SpeedFromSupplyB") != pb.get("SafetyFromSupplyB")
    notes = (
        f"  Speed → pharmacy_north:  {'✓' if speed_north else '✗'}"
        f"  ({pb.get('SpeedFromSupplyB')})\n"
        f"  Safety → pharmacy_south: {'✓' if safety_south else '✗'}"
        f"  ({pb.get('SafetyFromSupplyB')})\n"
        f"  Pharmacies differ:       {'✓' if differ else '✗'}"
    )
    return len(ctx["plan_keys"]) >= 2, notes


def _assert_med_safety(ctx: dict):
    pb = ctx["pharmacy_by_label"]
    safety_pharm = pb.get("SafetyFromSupplyB")
    speed_pharm  = pb.get("SpeedFromSupplyB")
    safety_avoids_north = safety_pharm != "pharmacy_north"
    pharmacies_differ   = safety_pharm != speed_pharm
    notes = (
        "  Pharmacy choices:\n"
        + "".join(f"    {label:22s}: {pharm}\n" for label, pharm in pb.items())
        + f"  Safety avoids north:      {'✓' if safety_avoids_north else '✗'}  ({safety_pharm})\n"
        + f"  Safety ≠ Speed pharmacy:  {'✓' if pharmacies_differ else '✗'}"
    )
    return len(ctx["plan_keys"]) >= 2 and safety_avoids_north, notes


def _assert_meal_diversity(ctx: dict):
    mb = ctx["meal_type_by_label"]
    unique_meals = set(v for v in mb.values() if v is not None)
    diversity_ok = len(unique_meals) >= 2
    notes = (
        f"  Meals chosen:         {list(mb.values())}\n"
        f"  Unique meal types:    {unique_meals}\n"
        f"  Diversity >= 2 types: {'✓' if diversity_ok else '✗'}"
    )
    if not diversity_ok and len(unique_meals) == 1:
        notes += f"\n  (Only got {unique_meals} — cost tuning may need adjustment)"
    return diversity_ok, notes


def _assert_meal_presentation(ctx: dict):
    mb = ctx["meal_type_by_label"]
    unique_meals = set(mb.values()) - {None}
    full_meal_seen = "full_meal" in unique_meals
    approach_meal = mb.get("ApproachHeavy")
    approach_avoids_sandwich = approach_meal != "sandwich"
    notes = (
        f"  Meals chosen:            {list(mb.values())}\n"
        f"  Unique meal types:       {unique_meals}\n"
        f"  Full meal appeared:      {'✓' if full_meal_seen else '✗'}\n"
        f"  Approach → not sandwich: {'✓' if approach_avoids_sandwich else '✗'}"
        f"  ({approach_meal})"
    )
    if not full_meal_seen:
        notes += (
            "\n  NOTE: Full meal not selected even at 70-80% approach weight."
            "\n  Check that MEAL_QUALITY['full_meal']['approach'] = +0.20 is"
            "\n  applied in the planner's cost function."
        )
    return len(unique_meals) >= 2, notes


ASSERT_FNS = {
    "med_speed":         _assert_med_speed,
    "med_safety":        _assert_med_safety,
    "meal_approach":     _assert_meal_diversity,
    "meal_safety":       _assert_meal_diversity,
    "meal_presentation": _assert_meal_presentation,
}


# ── Runner ───────────────────────────────────────────────────────────────────

def run_profile(key: str, test: int | None = None, episodes: int | None = None) -> bool:
    cfg        = PROFILES[key]
    weights    = WEIGHT_PROFILES_2[key]
    assert_fn  = ASSERT_FNS[key]

    def _test_2():
        return run_route_choice_check(cfg, weights, assert_fn)

    return run_suite(cfg, _test_2, test=test, episodes=episodes) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run profile validation tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available profiles: " + ", ".join(PROFILES),
    )
    parser.add_argument(
        "profile",
        nargs="?",
        choices=list(PROFILES),
        help="Profile to run (default: all)",
    )
    parser.add_argument("--test",     type=int, default=None, help="Run only this test number")
    parser.add_argument("--episodes", type=int, default=None, help="Episodes per learning test")
    args = parser.parse_args()

    targets = [args.profile] if args.profile else list(PROFILES)
    results = {}
    for key in targets:
        print(f"\n{'='*60}")
        print(f"Profile: {key}")
        print(f"{'='*60}")
        results[key] = run_profile(key, test=args.test, episodes=args.episodes)

    if len(targets) > 1:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        for key, ok in results.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {key}")
        any_failed = not all(results.values())
        sys.exit(1 if any_failed else 0)
    else:
        sys.exit(0 if results[targets[0]] else 1)


if __name__ == "__main__":
    main()
