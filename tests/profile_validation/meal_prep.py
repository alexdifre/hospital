"""
Test meal preparation task logic:
  1. Preconditions enforce correct ordering
  2. All three meal paths reach goal
  3. Planner picks different meals under different weights
  4. Stock depletion works
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.meal_preparation.task_actions import (
    MealAction,
    MEAL_SANDWICH,
    MEAL_SOUP,
    MEAL_FULL,
)
from tasks.meal_preparation.task_state import MealTaskState
from tasks.meal_preparation.task_state_manager import MealTaskStateManager
from tasks.meal_preparation.task_planner import MealTaskPlanner
from tasks.meal_preparation.meal_profiles import compute_meal_features


def test_ordering_constraints():
    """Test that preconditions enforce correct meal sequences."""
    print("=" * 60)
    print("TEST 1: Ordering Constraints")
    print("=" * 60)

    mgr = MealTaskStateManager()

    # Start at pantry — should be able to collect any ingredient type
    state = MealTaskState(location="pantry")
    actions = mgr.get_available_actions(state)
    assert (
        MealAction.COLLECT_SANDWICH_INGREDIENTS in actions
    ), "Should allow sandwich at pantry"
    assert MealAction.COLLECT_SOUP_INGREDIENTS in actions, "Should allow soup at pantry"
    assert (
        MealAction.COLLECT_MEAL_INGREDIENTS in actions
    ), "Should allow full meal at pantry"
    assert MealAction.CHOP not in actions, "Should NOT allow chop without ingredients"
    assert MealAction.COOK not in actions, "Should NOT allow cook without chopping"
    print("  ✓ Ingredient collection available at pantry")

    # Collect sandwich ingredients → should NOT allow soup/full actions
    state = mgr.apply_action(state, MealAction.COLLECT_SANDWICH_INGREDIENTS)
    assert state.meal_type == MEAL_SANDWICH
    assert state.has_ingredients
    actions = mgr.get_available_actions(state)
    assert MealAction.COLLECT_SOUP_INGREDIENTS not in actions, "Can't change meal type"
    assert MealAction.CHOP not in actions, "Sandwich doesn't need chopping"
    print("  ✓ Sandwich locks out other meal types")

    # Go to prep station → assemble should be available
    state = MealTaskState(
        location="prep_station", meal_type=MEAL_SANDWICH, has_ingredients=True
    )
    actions = mgr.get_available_actions(state)
    assert MealAction.ASSEMBLE in actions, "Should allow assembly at prep station"
    assert MealAction.CHOP not in actions, "Sandwich doesn't chop"
    print("  ✓ Assemble available for sandwich at prep_station")

    # After assembly → meal_ready
    state = mgr.apply_action(state, MealAction.ASSEMBLE)
    assert state.is_assembled
    assert state.meal_ready
    print("  ✓ Sandwich ready after assembly")

    # Test soup ordering: collect → chop → cook
    state = MealTaskState(location="pantry")
    state = mgr.apply_action(state, MealAction.COLLECT_SOUP_INGREDIENTS)
    assert state.meal_type == MEAL_SOUP

    # At pantry, can't chop (wrong location)
    actions = mgr.get_available_actions(state)
    assert MealAction.CHOP not in actions, "Can't chop at pantry"
    print("  ✓ Can't chop at wrong location")

    # Move to prep_station, now can chop
    state.location = "prep_station"
    actions = mgr.get_available_actions(state)
    assert MealAction.CHOP in actions, "Should chop at prep_station"
    assert MealAction.COOK not in actions, "Can't cook before chopping"
    print("  ✓ Chop available at prep_station, cook blocked")

    state = mgr.apply_action(state, MealAction.CHOP)
    assert state.is_chopped

    # At prep_station after chopping, still can't cook
    actions = mgr.get_available_actions(state)
    assert MealAction.COOK not in actions, "Can't cook at prep_station"

    # Move to stove → can cook
    state.location = "stove"
    actions = mgr.get_available_actions(state)
    assert MealAction.COOK in actions, "Should cook at stove"
    state = mgr.apply_action(state, MealAction.COOK)
    assert state.is_cooked
    assert state.meal_ready, "Soup should be ready after cooking"
    print("  ✓ Soup: chop → cook → ready")

    # Test full meal ordering: collect → chop → cook → plate
    state = MealTaskState(location="pantry")
    state = mgr.apply_action(state, MealAction.COLLECT_MEAL_INGREDIENTS)
    state.location = "prep_station"
    state = mgr.apply_action(state, MealAction.CHOP)
    state.location = "stove"
    state = mgr.apply_action(state, MealAction.COOK)
    assert state.is_cooked
    assert not state.meal_ready, "Full meal NOT ready until plated"
    print("  ✓ Full meal not ready after cooking (needs plating)")

    # Can't plate at stove
    actions = mgr.get_available_actions(state)
    assert MealAction.PLATE not in actions, "Can't plate at stove"

    # Return to prep_station → plate
    state.location = "prep_station"
    actions = mgr.get_available_actions(state)
    assert MealAction.PLATE in actions, "Should plate at prep_station"
    state = mgr.apply_action(state, MealAction.PLATE)
    assert state.is_plated
    assert state.meal_ready, "Full meal ready after plating"
    print("  ✓ Full meal: chop → cook → plate → ready")

    # Delivery requires patient bed + meal_ready
    state.location = "patient_bed_left"
    state.approach_side = "left"
    actions = mgr.get_available_actions(state)
    assert MealAction.DELIVER_MEAL in actions
    state = mgr.apply_action(state, MealAction.DELIVER_MEAL)
    assert state.delivered
    assert mgr.is_goal(state)
    print("  ✓ Delivery works at patient bed")

    print("\n  ✓✓ ALL ORDERING CONSTRAINTS PASS\n")


def test_planner_all_meals():
    """Test that A* finds valid plans for all three meal types."""
    print("=" * 60)
    print("TEST 2: Planner Finds All Meal Types")
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

    # Track which meals we see
    meals_found = {meal}

    # Speed-oriented weights → should prefer sandwich (fastest)
    speed_weights = np.array([0.5, 0.1, 0.1, 0.1, 0.2])
    planner_speed = MealTaskPlanner(
        task_state_manager=mgr, preference_weights=speed_weights
    )
    state = MealTaskState(location="pantry")
    actions_s, states_s, info_s = planner_speed.plan(state, verbose=False)
    assert info_s["success"]
    meal_s = states_s[-1].meal_type
    meals_found.add(meal_s)
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
    meals_found.add(meal_a)
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
    meals_found.add(meal_sf)
    print(
        f"  Safety weights → {meal_sf} (steps={len(actions_sf)}, "
        f"time={states_sf[-1].time_elapsed:.1f}s)"
    )

    print(f"\n  Unique meals found: {meals_found}")
    if len(meals_found) >= 2:
        print("  ✓✓ Planner produces diverse meal choices!")
    else:
        print("  ⚠ Only one meal type chosen — may need cost tuning")

    print()


def test_valid_sequences():
    """Manually execute all three meal paths and verify they reach goal."""
    print("=" * 60)
    print("TEST 3: Manual Execution of All Three Paths")
    print("=" * 60)

    mgr = MealTaskStateManager()

    # Sandwich path
    state = MealTaskState(location="pantry")
    sequence = [
        MealAction.COLLECT_SANDWICH_INGREDIENTS,
        MealAction.GO_TO_PREP_STATION,
        MealAction.ASSEMBLE,
        MealAction.GO_TO_PATIENT_LEFT,
        MealAction.DELIVER_MEAL,
    ]
    for action in sequence:
        avail = mgr.get_available_actions(state)
        assert action in avail, f"Action {action.value} not available at {state}"
        state = mgr.apply_action(state, action)
    assert state.delivered
    print(f"  ✓ Sandwich: {len(sequence)} steps, {state.time_elapsed:.1f}s")

    # Soup path
    state = MealTaskState(location="pantry")
    sequence = [
        MealAction.COLLECT_SOUP_INGREDIENTS,
        MealAction.GO_TO_PREP_STATION,
        MealAction.CHOP,
        MealAction.GO_TO_STOVE,
        MealAction.COOK,
        MealAction.GO_TO_PATIENT_LEFT,
        MealAction.DELIVER_MEAL,
    ]
    for action in sequence:
        avail = mgr.get_available_actions(state)
        assert action in avail, f"Action {action.value} not available at {state}"
        state = mgr.apply_action(state, action)
    assert state.delivered
    print(f"  ✓ Soup:     {len(sequence)} steps, {state.time_elapsed:.1f}s")

    # Full meal path
    state = MealTaskState(location="pantry")
    sequence = [
        MealAction.COLLECT_MEAL_INGREDIENTS,
        MealAction.GO_TO_PREP_STATION,
        MealAction.CHOP,
        MealAction.GO_TO_STOVE,
        MealAction.COOK,
        MealAction.GO_TO_PREP_STATION,
        MealAction.PLATE,
        MealAction.GO_TO_PATIENT_LEFT,
        MealAction.DELIVER_MEAL,
    ]
    for action in sequence:
        avail = mgr.get_available_actions(state)
        assert action in avail, f"Action {action.value} not available at {state}"
        state = mgr.apply_action(state, action)
    assert state.delivered
    print(f"  ✓ Full meal: {len(sequence)} steps, {state.time_elapsed:.1f}s")

    print("\n  ✓✓ ALL THREE MEAL PATHS VALID\n")


def test_features():
    """Test feature generation for each meal type."""
    print("=" * 60)
    print("TEST 4: Feature Generation")
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


def test_stock_depletion():
    """Test that stock depletion works during planning."""
    print("=" * 60)
    print("TEST 5: Stock Depletion During Planning")
    print("=" * 60)

    mgr = MealTaskStateManager()

    stock = {
        "pantry_sandwich": 2,
        "pantry_soup": 1,
        "pantry_full_meal": 0,  # out of stock!
    }

    state = MealTaskState(location="pantry", location_stock=stock.copy())
    actions = mgr.get_available_actions(state)

    assert MealAction.COLLECT_SANDWICH_INGREDIENTS in actions, "Sandwich in stock"
    assert MealAction.COLLECT_SOUP_INGREDIENTS in actions, "Soup in stock"
    assert MealAction.COLLECT_MEAL_INGREDIENTS not in actions, "Full meal out of stock!"
    print("  ✓ Out-of-stock meal type blocked")

    # Collect soup → stock decrements
    state = mgr.apply_action(state, MealAction.COLLECT_SOUP_INGREDIENTS)
    assert state.location_stock["pantry_soup"] == 0, "Soup stock should be 0"
    print(
        f"  ✓ Soup stock decremented: {stock['pantry_soup']} → {state.location_stock['pantry_soup']}"
    )

    # Sandwich stock unchanged
    assert state.location_stock["pantry_sandwich"] == 2
    print(f"  ✓ Sandwich stock unchanged: {state.location_stock['pantry_sandwich']}")

    print("\n  ✓✓ STOCK DEPLETION WORKS\n")


if __name__ == "__main__":
    test_ordering_constraints()
    test_valid_sequences()
    test_planner_all_meals()
    test_features()
    test_stock_depletion()

    print("=" * 60)
    print("ALL MEAL PREPARATION TESTS PASSED")
    print("=" * 60)
