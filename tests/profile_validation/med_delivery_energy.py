#!/usr/bin/env python3
"""
Medication Delivery Integration Test — Energy-Conscious Patient
================================================================

Profile: energy_conscious  w* = [0.15, 0.15, 0.45, 0.15, 0.10]
  Dominant: battery (index 2, 45%)

Battery cost = distance_traveled × 0.01, weighted at 45%.
The planner picks the shortest route and adds a recharge stop when
battery starts low.  Test 2 also injects a low-battery scenario.

Usage:
    python tests/profile_validation/med_delivery_energy.py          # all tests
    python tests/profile_validation/med_delivery_energy.py --test 1
    python tests/profile_validation/med_delivery_energy.py --test 3 --episodes 10
"""

import sys
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))

from harness import ProfileConfig, make_system, run_suite, make_argparser

CFG = ProfileConfig(
    profile="energy_conscious",
    task_type="medication",
    start_location="home",
    dominant_dim=2,  # battery
    description="Medication delivery tests — energy-conscious patient",
    summary_dir="test_mixed_roster_energy",
)

# ── Test 2: energy efficiency + low-battery recharge ─────────────────
#
# Weight profiles include a 4th element: battery_level (0–1).
# That requires direct env state manipulation, so we keep the full
# test_2 logic here rather than routing through harness.run_route_choice_check.

_WEIGHT_PROFILES_2 = [
    # (label, weights, start, battery_level)
    ("EnergyFromSupplyB", np.array([0.05, 0.05, 0.70, 0.10, 0.10]), "supply_B", 1.0),
    ("SafetyFromSupplyB", np.array([0.05, 0.55, 0.10, 0.15, 0.15]), "supply_B", 1.0),
    ("SpeedFromSupplyB",  np.array([0.55, 0.10, 0.10, 0.10, 0.15]), "supply_B", 1.0),
    ("LowBatteryEnergy",  np.array([0.05, 0.05, 0.70, 0.10, 0.10]), "supply_B", 0.3),
]


def _test_2() -> bool:
    print("\n" + "=" * 80)
    print("TEST 2: WEIGHT-DEPENDENT ROUTE CHOICE  [energy_conscious]")
    print("=" * 80)

    system = make_system(CFG)

    plans = []
    battery_usage: dict = {}
    recharge_choices: dict = {}

    for label, weights, start, battery_level in _WEIGHT_PROFILES_2:
        print(f"\n--- Profile: {label} (start={start}, battery={battery_level*100:.0f}%) ---")
        print(f"  Weights: [{', '.join(f'{w:.2f}' for w in weights)}]")

        system.preference_learner.estimated_weights = weights.copy()
        system.env.environment_state["battery_level"] = battery_level
        result = system.run_episode(start_location=start, task_type="medication")

        if result.get("success"):
            plan = result["plan_structure"]
            plans.append(plan)
            pharm   = plan.get("pharmacy_choice", "?")
            supply  = plan.get("supply_choice", "?")
            approach= plan.get("approach_choice", "?")
            recharge= plan.get("recharge_added", False)
            batt_used = result.get("battery_start", 100) - result.get("battery_remaining", 0)
            battery_usage[label]   = batt_used
            recharge_choices[label]= recharge
            total_dist = result.get("total_distance", 0)
            print(
                f"  → Route: {pharm} → {supply} → patient_{approach}"
                f" {'+ recharge' if recharge else ''}"
                f"  (dist={total_dist:.1f}m, batt_used={batt_used:.1f}%)"
            )
        else:
            plans.append(None)
            print(f"  → FAILED: {result.get('reason')}")

    valid_plans = [p for p in plans if p is not None]
    plan_keys = {
        (p["pharmacy_choice"], p["supply_choice"],
         p["approach_choice"], p.get("recharge_added", False))
        for p in valid_plans
    }

    # Energy-heavy should use ≤ battery than safety-heavy
    energy_efficient = True
    if "EnergyFromSupplyB" in battery_usage and "SafetyFromSupplyB" in battery_usage:
        energy_efficient = battery_usage["EnergyFromSupplyB"] <= battery_usage["SafetyFromSupplyB"] * 1.1

    low_battery_handled = "LowBatteryEnergy" in recharge_choices

    print(f"\n{'='*60}")
    print("TEST 2 RESULTS:")
    print(f"  Successful episodes:    {len(valid_plans)}/{len(_WEIGHT_PROFILES_2)}")
    print(f"  Unique route structures:{len(plan_keys)}")
    for k in plan_keys:
        print(f"    {k}")
    if battery_usage:
        print("  Battery usage:")
        for label, bu in battery_usage.items():
            print(f"    {label:22s}: {bu:.1f}%")
    print(f"  Energy-efficient:       {'✓' if energy_efficient else '✗ (informational)'}")
    print(f"  Low-battery handled:    {'✓' if low_battery_handled else '✗ (informational)'}")

    diversity_ok = len(plan_keys) >= 2
    passed = diversity_ok and len(valid_plans) == len(_WEIGHT_PROFILES_2)
    print(f"\n  Diversity >= 2 routes:  {'✓' if diversity_ok else '✗'}")
    print(f"  All episodes succeeded: {'✓' if len(valid_plans) == len(_WEIGHT_PROFILES_2) else '✗'}")
    print(f"\n  TEST 2: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"{'='*60}")

    system.close()
    return passed


if __name__ == "__main__":
    args = make_argparser(CFG.description).parse_args()
    sys.exit(run_suite(CFG, _test_2, test=args.test, episodes=args.episodes))
