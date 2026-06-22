"""
Small meal-preparation smoke checks.

These checks intentionally avoid legacy assumptions about hard-coded meal paths
and stock names. The end-to-end runtime behavior is covered by pytest tests.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tasks.meal_preparation.task_actions import (
    MEAL_SANDWICH,
    MEAL_SOUP,
    MEAL_FULL,
)
from tasks.meal_preparation.task_state import MealTaskState
from tasks.meal_preparation.task_state_manager import MealTaskStateManager
from tasks.meal_preparation.task_planner import MealTaskPlanner
from tasks.meal_preparation.meal_profiles import compute_meal_features


def run_planner_smoke_check():
    """Check that the current meal planner finds a valid delivered plan."""
    print("=" * 60)
    print("MEAL CHECK 1: Planner Finds A Delivered Plan")
    print("=" * 60)

    mgr = MealTaskStateManager()

    # Uniform weights → let's see what the planner picks
    uniform = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    planner = MealTaskPlanner(task_state_manager=mgr, preference_weights=uniform)

    state = MealTaskState(location="pantry")
    actions, states, info = planner.plan(state, verbose=True)

    assert info["success"], "Planner should find a solution"
    assert states[-1].delivered, "Final state should be delivered"
    meal = states[-1].meal_type
    print(f"\n  Planner chose: {meal}")
    print(f"  Steps: {len(actions)}")
    print(f"  Actions: {[a.value for a in actions]}")
    planner.print_plan(actions, states)

    # Try a few representative weights. This is a smoke check, not a guarantee
    # that the planner will pick different meal labels for every cost profile.
    speed_weights = np.array([0.5, 0.1, 0.1, 0.1, 0.2])
    planner_speed = MealTaskPlanner(
        task_state_manager=mgr, preference_weights=speed_weights
    )
    state = MealTaskState(location="pantry")
    actions_s, states_s, info_s = planner_speed.plan(state, verbose=False)
    assert info_s["success"]
    meal_s = states_s[-1].meal_type
    print(
        f"\n  Speed weights → {meal_s} (steps={len(actions_s)}, "
        f"time={states_s[-1].time_elapsed:.1f}s)"
    )

    # Approach-oriented weights → should prefer full meal (plating bonus)
    approach_weights = np.array([0.05, 0.1, 0.05, 0.1, 0.7])
    planner_approach = MealTaskPlanner(
        task_state_manager=mgr, preference_weights=approach_weights
    )
    state = MealTaskState(location="pantry")
    actions_a, states_a, info_a = planner_approach.plan(state, verbose=False)
    assert info_a["success"]
    meal_a = states_a[-1].meal_type
    print(
        f"  Approach weights → {meal_a} (steps={len(actions_a)}, "
        f"time={states_a[-1].time_elapsed:.1f}s)"
    )

    # Safety-oriented weights → should prefer sandwich (safest) or soup
    safety_weights = np.array([0.1, 0.6, 0.1, 0.1, 0.1])
    planner_safety = MealTaskPlanner(
        task_state_manager=mgr, preference_weights=safety_weights
    )
    state = MealTaskState(location="pantry")
    actions_sf, states_sf, info_sf = planner_safety.plan(state, verbose=False)
    assert info_sf["success"]
    meal_sf = states_sf[-1].meal_type
    print(
        f"  Safety weights → {meal_sf} (steps={len(actions_sf)}, "
        f"time={states_sf[-1].time_elapsed:.1f}s)"
    )

    print("\n  ✓✓ PLANNER SMOKE CHECK PASSED")
    print()


def run_feature_generation_check():
    """Test feature generation for each meal type."""
    print("=" * 60)
    print("MEAL CHECK 2: Feature Generation")
    print("=" * 60)

    for meal in [MEAL_SANDWICH, MEAL_SOUP, MEAL_FULL]:
        features = compute_meal_features(
            total_time=(
                40.0 if meal == MEAL_SANDWICH else 60.0 if meal == MEAL_SOUP else 80.0
            ),
            total_distance=(
                20.0 if meal == MEAL_SANDWICH else 30.0 if meal == MEAL_SOUP else 40.0
            ),
            battery_start=1.0,
            battery_end=(
                0.7 if meal == MEAL_SANDWICH else 0.6 if meal == MEAL_SOUP else 0.5
            ),
            delivery_error=0.7,
            approach_quality=(
                0.6 if meal == MEAL_SANDWICH else 0.7 if meal == MEAL_SOUP else 0.85
            ),
            meal_type=meal,
        )
        print(f"\n  {meal}:")
        for k in ["time", "safety", "battery", "proximity", "approach"]:
            print(f"    {k:12s}: {features[k]:.3f}")

    print("\n  ✓✓ FEATURES GENERATED FOR ALL MEAL TYPES\n")


if __name__ == "__main__":
    run_planner_smoke_check()
    run_feature_generation_check()

    print("=" * 60)
    print("MEAL PREPARATION CHECKS PASSED")
    print("=" * 60)
