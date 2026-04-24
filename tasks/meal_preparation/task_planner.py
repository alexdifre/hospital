"""
Meal Preparation Task — A* planner.

Searches over the meal preparation state space to find optimal plans.
The cost function weights time, safety, battery, proximity, and approach
using the current preference weights — same 5 dimensions as medication delivery.

Key: the planner explores all three meal branches (sandwich, soup, full_meal)
and picks the lowest-cost valid sequence. Different preference weights lead
to different meal choices, generating diverse feature vectors for the
preference learner.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from core.task_planning.base_planner import BaseTaskPlanner

from .task_actions import (
    MealAction,
    NAVIGATION_ACTIONS,
    ACTION_TARGET_LOCATIONS,
    MEAL_SANDWICH,
    MEAL_SOUP,
    MEAL_FULL,
)
from .task_state import MealTaskState
from .task_state_manager import MealTaskStateManager, ACTION_DURATIONS


# ── Meal-specific safety costs ──────────────────────────────────
MEAL_SAFETY_COSTS = {
    MEAL_SANDWICH: 0.0,
    MEAL_SOUP: 0.15,
    MEAL_FULL: 0.25,
}

# ── Meal quality at delivery (positive costs only) ─────────────
MEAL_DELIVERY_PENALTIES = {
    MEAL_SANDWICH: {"approach": 1.20, "proximity": 0.45},
    MEAL_SOUP: {"approach": 0.65, "proximity": 0.12},
    MEAL_FULL: {"approach": 0.00, "proximity": 0.00},
}


class MealTaskPlanner(BaseTaskPlanner):
    """
    A* search planner for meal preparation.
    """

    def __init__(
        self,
        task_state_manager: MealTaskStateManager,
        preference_weights: Optional[np.ndarray] = None,
        fuzzy_estimator=None,
        max_expansions: int = 2000,
    ):
        super().__init__(preference_weights=preference_weights, fuzzy_estimator=fuzzy_estimator)
        self.manager = task_state_manager
        self.max_expansions = max_expansions

        self.w_time = self.weights[0]
        self.w_safety = self.weights[1]
        self.w_battery = self.weights[2]
        self.w_proximity = self.weights[3]
        self.w_approach = self.weights[4]

        print(f"MealTaskPlanner initialized")
        print(f"  Preference weights: {self.weights}")
        print(
            f"  w_time={self.w_time:.2f}, w_safety={self.w_safety:.2f}, "
            f"w_battery={self.w_battery:.2f}, w_proximity={self.w_proximity:.2f}, "
            f"w_approach={self.w_approach:.2f}"
        )

    def _expand(self, state: MealTaskState) -> List[Tuple]:
        """Generate (action, next_state, edge_cost) successors."""
        successors = []
        for action in self.manager.get_available_actions(state):
            next_state = self.manager.apply_action(state, action)
            edge_cost = self._calculate_action_cost(state, next_state, action)
            successors.append((action, next_state, edge_cost))
        return successors

    def _calculate_action_cost(
        self,
        state: MealTaskState,
        next_state: MealTaskState,
        action: MealAction,
    ) -> float:
        cost = 0.0

        # ── Time cost ───────────────────────────────────────────
        dt = next_state.time_elapsed - state.time_elapsed
        time_cost = dt / 60.0
        cost += self.w_time * time_cost

        # ── Safety cost ─────────────────────────────────────────
        safety_cost = 0.0

        if next_state.battery_soc < 0.15:
            safety_cost += 0.5
        elif next_state.battery_soc < 0.25:
            safety_cost += 0.2

        if next_state.meal_type is not None:
            safety_cost += MEAL_SAFETY_COSTS.get(next_state.meal_type, 0.0)

            if action == MealAction.COOK:
                safety_cost += 0.1
            if next_state.is_cooked and action in NAVIGATION_ACTIONS:
                safety_cost += 0.05

        if action in NAVIGATION_ACTIONS:
            target = ACTION_TARGET_LOCATIONS.get(action, "")
            if target == "nurse_station":
                safety_cost += 0.3
            elif target == "equipment_storage":
                safety_cost += 0.2

        if action in (
            MealAction.COLLECT_SANDWICH_INGREDIENTS,
            MealAction.COLLECT_SOUP_INGREDIENTS,
            MealAction.COLLECT_MEAL_INGREDIENTS,
        ):
            meal_key = {
                MealAction.COLLECT_SANDWICH_INGREDIENTS: MEAL_SANDWICH,
                MealAction.COLLECT_SOUP_INGREDIENTS: MEAL_SOUP,
                MealAction.COLLECT_MEAL_INGREDIENTS: MEAL_FULL,
            }[action]
            stock = self._get_ingredient_stock(state, meal_key)
            if stock is not None and stock >= 0:
                safety_cost += 0.3 / (1 + stock)

        cost += self.w_safety * safety_cost

        # ── Battery cost ────────────────────────────────────────
        battery_used = state.battery_soc - next_state.battery_soc
        cost += self.w_battery * battery_used

        # ── Proximity cost ──────────────────────────────────────
        if next_state.meal_ready and action in NAVIGATION_ACTIONS:
            target = ACTION_TARGET_LOCATIONS.get(action, "")
            if target not in ("patient_bed_left", "patient_bed_right"):
                cost += self.w_proximity * 0.3
        elif action in NAVIGATION_ACTIONS:
            dist = next_state.distance_traveled - state.distance_traveled
            cost += self.w_proximity * (dist / 30.0)

        # ── Approach cost ───────────────────────────────────────
        if action == MealAction.DELIVER_MEAL:
            base_deliver = 0.1
            cost += self.w_approach * base_deliver

            if next_state.meal_type in MEAL_DELIVERY_PENALTIES:
                penalties = MEAL_DELIVERY_PENALTIES[next_state.meal_type]
                cost += self.w_approach * penalties.get("approach", 0.0)
                cost += self.w_proximity * penalties.get("proximity", 0.0)

        return max(0.0, cost)

    def _heuristic(self, state: MealTaskState) -> float:
        if state.delivered:
            return 0.0

        h = 0.0

        remaining_steps = 0
        if not state.has_ingredients:
            remaining_steps += 2
        if state.meal_type in (MEAL_SOUP, MEAL_FULL) and not state.is_chopped:
            remaining_steps += 1
        if state.meal_type in (MEAL_SOUP, MEAL_FULL) and not state.is_cooked:
            remaining_steps += 1
        if state.meal_type == MEAL_FULL and not state.is_plated:
            remaining_steps += 1
        if state.meal_type == MEAL_SANDWICH and not state.is_assembled:
            remaining_steps += 1
        remaining_steps += 1  # deliver

        min_time = remaining_steps * 3.0
        h += self.w_time * (min_time / 60.0)

        if state.location not in ("patient_bed_left", "patient_bed_right"):
            dist_to_patient = self.manager.get_distance(
                state.location, "patient_bed_left"
            )
            h += self.w_battery * (dist_to_patient * 0.01 * 0.5)

        return h

    def _get_ingredient_stock(
        self, state: MealTaskState, meal_type: str
    ) -> Optional[int]:
        if state.location_stock is None:
            return None
        stock_key = f"pantry_{meal_type}"
        return state.location_stock.get(stock_key, None)

    def print_plan(
        self,
        actions: List[MealAction],
        states: List[MealTaskState],
    ):
        print(f"\n{'='*80}")
        print("PLANNED MEAL PREPARATION SEQUENCE")
        print(f"{'='*80}")
        print(f"Total steps: {len(actions)}")

        # states[0] is initial_state, states[1:] are post-action states
        post_action_states = states[1:] if len(states) > len(actions) else states

        if post_action_states:
            final = post_action_states[-1]
            print(f"Meal type: {final.meal_type or '?'}")
            print(f"Expected time: {final.time_elapsed:.1f}s")
            print(f"Expected distance: {final.distance_traveled:.1f}m")
            print(f"Final battery: {final.battery_soc:.1%}")

        print(f"\nSequence:\n")
        for i, (action, state) in enumerate(zip(actions, post_action_states)):
            print(f"  Step {i+1}: {action.value}")
            print(f"    → State: {state}")
            print(f"    → Battery: {state.battery_soc:.1%}")
            print(f"    → Time: {state.time_elapsed:.1f}s")
            print()
