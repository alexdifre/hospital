"""
Small meal-preparation smoke checks.

These checks use the PDDL/ENHSP planning path. The end-to-end runtime behavior
is covered by pytest tests.
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
from tasks.meal_preparation.meal_profiles import compute_meal_features
from core.task_planning.pddl_engine import make_pddl_oneshot_planner


def _pddl_action_name(action_instance) -> str:
    action = getattr(action_instance, "action", None)
    name = getattr(action, "name", None)
    if name is not None:
        return str(name)
    return str(action_instance).split("(", 1)[0].strip()


def _solve_meal_pddl(weights):
    from unified_planning.io import PDDLReader

    repo = Path(__file__).resolve().parents[2]
    domain = repo / "unified_planning" / "domain_meal.pddl"
    problem_path = repo / "unified_planning" / "problem_meal.pddl"
    problem = PDDLReader().parse_problem(str(domain), str(problem_path))
    robot = problem.object("robot1")
    for idx, weight in enumerate(np.asarray(weights, dtype=float).reshape(5)):
        problem.set_initial_value(problem.fluent(f"w{idx}")(robot), float(weight))

    with make_pddl_oneshot_planner("enhsp-opt") as planner:
        result = planner.solve(problem)

    actions = list(getattr(getattr(result, "plan", None), "actions", []) or [])
    return str(getattr(result, "status", "")), [_pddl_action_name(a) for a in actions]


def run_planner_smoke_check():
    """Check that ENHSP-opt finds valid meal plans from the PDDL files."""
    print("=" * 60)
    print("MEAL CHECK 1: ENHSP-opt Solves Meal PDDL")
    print("=" * 60)

    uniform = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    status, actions = _solve_meal_pddl(uniform)
    assert "SOLVED" in status, f"ENHSP should solve meal PDDL, got {status}"
    assert actions, "ENHSP should return a non-empty plan"
    assert actions[-1].startswith("deliver_on_bedside_table")
    print(f"\n  Status: {status}")
    print(f"  Steps: {len(actions)}")
    print(f"  Actions: {actions}")

    profiles = {
        "Speed": np.array([0.5, 0.1, 0.1, 0.1, 0.2]),
        "Approach": np.array([0.05, 0.1, 0.05, 0.1, 0.7]),
        "Safety": np.array([0.1, 0.6, 0.1, 0.1, 0.1]),
    }
    for label, weights in profiles.items():
        status, actions = _solve_meal_pddl(weights)
        assert "SOLVED" in status, f"{label} weights should solve, got {status}"
        print(f"  {label} weights -> {actions[0]} ... {actions[-1]} ({len(actions)} steps)")

    print("\n  ✓✓ PDDL/ENHSP SMOKE CHECK PASSED")
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
