"""
Meal Preparation Task - State manager aligned with domain_meal.pddl.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .task_actions import (
    ACTION_BATTERY_COSTS,
    ACTION_DURATIONS,
    ACTION_TARGET_LOCATIONS,
    DELIVERY_ACTIONS,
    INGREDIENT_LOCATIONS,
    MEAL_CHOICE_ACTIONS,
    MEAL_HOT,
    MEAL_REQUIRED_INGREDIENTS,
    NAVIGATION_ACTIONS,
    PDDL_NAV_EDGES,
    MealAction,
)
from .task_state import MealTaskState

SPEED = 1.5
BATTERY_PER_METER = 0.01

__all__ = [
    "MealAction",
    "MealTaskState",
    "MealTaskStateManager",
    "ACTION_DURATIONS",
]


class MealTaskStateManager:
    """Manage PDDL-style preconditions and transitions for meal preparation."""

    def __init__(
        self,
        env=None,
        locations: Optional[List[str]] = None,
        fuzzy_estimator=None,
    ):
        del locations
        self.env = env
        self.fuzzy_estimator = fuzzy_estimator
        self._distances: Dict[Tuple[str, str], float] = {}
        if env is not None:
            self._build_distance_table(env)

        self.pantry_locations = {"pantry"}
        self.fridge_locations = {"fridge"}
        self.prep_locations = {"prep_station"}
        self.stove_locations = {"stove"}
        self.quality_locations = {"quality_check"}
        self.patient_locations = {"patient_bed_left", "patient_bed_right"}
        self.kitchen_locations = (
            self.pantry_locations
            | self.fridge_locations
            | self.prep_locations
            | self.stove_locations
            | self.quality_locations
        )

    def _build_distance_table(self, env):
        """Pre-compute pairwise Euclidean distances between all locations."""
        for name_a, pos_a in env.locations.items():
            for name_b, pos_b in env.locations.items():
                self._distances[(name_a, name_b)] = float(np.linalg.norm(pos_a - pos_b))

    def get_distance(self, loc_a: str, loc_b: str) -> float:
        """Get distance between two locations."""
        if loc_a == loc_b:
            return 0.0
        return self._distances.get((loc_a, loc_b), 15.0)

    def get_initial_state(self, start_location: str) -> MealTaskState:
        """Create a fresh task state at the given location."""
        return MealTaskState(location=start_location)

    def _meal_requires_cooking(self, state: MealTaskState) -> bool:
        return state.meal_to_prepare == MEAL_HOT

    def _is_at(self, state: MealTaskState, location: str) -> bool:
        """Check if robot is at a specific location."""
        if state.location_memberships is not None:
            return state.location_memberships.get(location, 0.0) >= 0.8
        return state.location == location

    def _is_at_any(self, state: MealTaskState, locations: set) -> bool:
        """Check if robot is at any location in a set."""
        if state.location_memberships is not None:
            return any(
                state.location_memberships.get(loc, 0.0) >= 0.8 for loc in locations
            )
        return state.location in locations

    def _stock_keys(self, location: str, item: str) -> Tuple[str, ...]:
        return (f"{location}_{item}", f"{location}_{item}_stock", item, location)

    def _has_stock(self, state: MealTaskState, location: str, item: str) -> bool:
        """Check planning/execution stock for a PDDL ingredient."""
        if state.location_stock is not None:
            for key in self._stock_keys(location, item):
                if key in state.location_stock:
                    return state.location_stock[key] > 0

        if self.env is not None:
            for key in self._stock_keys(location, item):
                stock_key = key if key.endswith("_stock") else f"{key}_stock"
                if stock_key in self.env.environment_state:
                    return self.env.environment_state[stock_key] > 0
        return True

    def _next_collectable_ingredient(
        self, state: MealTaskState
    ) -> Optional[str]:
        """Select the next required ingredient available at the current location."""
        collected = set(state.collected_ingredients)
        for ingredient in state.required_ingredients:
            if ingredient in collected:
                continue
            location = INGREDIENT_LOCATIONS.get(ingredient)
            if location == state.location and self._has_stock(state, location, ingredient):
                return ingredient
        return None

    def _decrement_stock(self, state: MealTaskState, location: str, item: str) -> None:
        """Decrement planning stock for an ingredient."""
        if state.location_stock is None:
            return
        for key in self._stock_keys(location, item):
            if key in state.location_stock:
                state.location_stock[key] = max(0, state.location_stock[key] - 1)
                return

    def _add_collected(self, state: MealTaskState, ingredient: str) -> None:
        collected = set(state.collected_ingredients)
        collected.add(ingredient)
        state.collected_ingredients = tuple(
            ing for ing in state.required_ingredients if ing in collected
        )

    def _sync_legacy_flags(self, state: MealTaskState) -> None:
        """Keep old fields consistent with the PDDL predicate fields."""
        state.has_ingredients = set(state.required_ingredients).issubset(
            set(state.collected_ingredients)
        )
        state.is_chopped = state.ingredients_chopped
        state.is_cooked = state.meal_cooked
        state.is_assembled = state.meal_assembled
        state.is_plated = state.meal_assembled
        state.meal_ready = state.can_be_deliverable
        state.meal_type = state.meal_to_prepare

    def _update_deliverability(self, state: MealTaskState) -> None:
        state.can_be_deliverable = (
            state.meal_assembled
            and state.meal_palatable
            and state.location in self.patient_locations
        )
        self._sync_legacy_flags(state)

    def get_available_actions(self, state: MealTaskState) -> List[MealAction]:
        """Return PDDL actions whose preconditions are currently true."""
        if state.delivered:
            return []

        actions: List[MealAction] = []
        if state.meal_to_prepare is None:
            return list(MEAL_CHOICE_ACTIONS)

        collected_all = set(state.required_ingredients).issubset(
            set(state.collected_ingredients)
        )

        nav_targets: Dict[MealAction, str] = {}
        if not collected_all:
            for ingredient in state.required_ingredients:
                if ingredient not in state.collected_ingredients:
                    loc = INGREDIENT_LOCATIONS[ingredient]
                    nav_targets[
                        MealAction.GO_TO_PANTRY
                        if loc == "pantry"
                        else MealAction.GO_TO_FRIDGE
                    ] = loc
        elif not state.ingredients_checked:
            pass
        elif not state.workspace_clean or not state.ingredients_washed:
            nav_targets[MealAction.GO_TO_PREP_STATION] = "prep_station"
        elif not state.ingredients_chopped:
            nav_targets[MealAction.GO_TO_PREP_STATION] = "prep_station"
        elif not state.meal_cooked and self._meal_requires_cooking(state):
            nav_targets[MealAction.GO_TO_COOKING_STATION] = "stove"
        elif not state.cooking_level_checked or not state.meal_palatable:
            nav_targets[MealAction.GO_TO_QUALITY_CHECK] = "quality_check"
        elif not state.meal_assembled:
            nav_targets[MealAction.GO_TO_QUALITY_CHECK] = "quality_check"
        elif state.meal_assembled and state.meal_palatable:
            nav_targets[MealAction.APPROACH_TO_LEFT_SIDE] = "patient_bed_left"
            nav_targets[MealAction.APPROACH_TO_RIGHT_SIDE] = "patient_bed_right"

        for action, target in nav_targets.items():
            if state.location != target and (state.location, target) in PDDL_NAV_EDGES:
                actions.append(action)

        if self._next_collectable_ingredient(state) is not None:
            actions.append(MealAction.COLLECT_INGREDIENT)

        if collected_all and not state.ingredients_checked:
            actions.append(MealAction.CHECK_INGREDIENTS)

        if (
            state.ingredients_checked
            and state.ingredients_safe
            and not state.workspace_clean
            and self._is_at_any(state, self.prep_locations)
        ):
            actions.append(MealAction.SANITIZE_WORKSPACE)

        if (
            state.workspace_clean
            and not state.ingredients_washed
            and self._is_at_any(state, self.prep_locations)
        ):
            actions.append(MealAction.WASH_INGREDIENTS)

        if (
            state.ingredients_washed
            and not state.ingredients_chopped
            and self._is_at_any(state, self.prep_locations)
        ):
            actions.append(MealAction.CHOP_INGREDIENTS)

        if (
            state.ingredients_chopped
            and not state.meal_cooked
            and self._meal_requires_cooking(state)
            and self._is_at_any(state, self.stove_locations)
        ):
            actions.append(MealAction.COOK_MEAL)

        if (
            (state.meal_cooked or not self._meal_requires_cooking(state))
            and not state.cooking_level_checked
            and self._is_at_any(state, self.quality_locations)
        ):
            actions.append(MealAction.CHECK_COOKING_LEVEL)

        if (
            state.cooking_level_checked
            and not state.meal_palatable
            and self._is_at_any(state, self.quality_locations)
        ):
            actions.append(MealAction.CHECK_PALATABILITY)

        if (
            state.meal_palatable
            and not state.meal_assembled
            and self._is_at_any(state, self.quality_locations)
        ):
            actions.append(MealAction.ASSEMBLA)

        if state.can_be_deliverable and self._is_at_any(state, self.patient_locations):
            if state.location == "patient_bed_left":
                actions.append(MealAction.DELIVER_ON_BEDSIDE_TABLE_LEFT)
            elif state.location == "patient_bed_right":
                actions.append(MealAction.DELIVER_ON_BEDSIDE_TABLE_RIGHT)

        return actions

    def apply_action(self, state: MealTaskState, action: MealAction) -> MealTaskState:
        """Apply an action and return the successor state."""
        next_state = state.copy()
        next_state.step_count += 1

        if action in NAVIGATION_ACTIONS:
            target = ACTION_TARGET_LOCATIONS[action]
            dist = self.get_distance(state.location, target)
            travel_time = dist / SPEED
            battery_cost = dist * BATTERY_PER_METER

            next_state.location = target
            next_state.time_elapsed += travel_time
            next_state.distance_traveled += dist
            next_state.battery_soc = max(0.0, next_state.battery_soc - battery_cost)
            next_state.location_memberships = {target: 1.0}

            if target == "patient_bed_left":
                next_state.approach_side = "left"
            elif target == "patient_bed_right":
                next_state.approach_side = "right"

            self._update_deliverability(next_state)
            return next_state

        if action in MEAL_CHOICE_ACTIONS:
            meal_type = MEAL_CHOICE_ACTIONS[action]
            next_state.meal_to_prepare = meal_type
            next_state.meal_type = meal_type
            next_state.required_ingredients = tuple(MEAL_REQUIRED_INGREDIENTS[meal_type])
            next_state.missing_ingredients = tuple(next_state.required_ingredients)

        elif action == MealAction.COLLECT_INGREDIENT:
            ingredient = self._next_collectable_ingredient(next_state)
            if ingredient is None:
                raise ValueError("No collectable ingredient at current location")
            self._add_collected(next_state, ingredient)
            next_state.missing_ingredients = tuple(
                ing for ing in next_state.missing_ingredients if ing != ingredient
            )
            self._decrement_stock(
                next_state, INGREDIENT_LOCATIONS[ingredient], ingredient
            )

        elif action == MealAction.CHECK_INGREDIENTS:
            next_state.ingredients_checked = True
            next_state.ingredients_safe = not (
                set(next_state.missing_ingredients) & set(next_state.required_ingredients)
                or set(next_state.expired_ingredients) & set(next_state.required_ingredients)
                or set(next_state.wrong_ingredients) & set(next_state.required_ingredients)
                or set(next_state.allergen_ingredients) & set(next_state.required_ingredients)
            )

        elif action == MealAction.SANITIZE_WORKSPACE:
            next_state.workspace_clean = True
            next_state.robot_hands_clean = True
            next_state.cross_contamination_risk = False

        elif action == MealAction.WASH_INGREDIENTS:
            next_state.ingredients_washed = True

        elif action == MealAction.CHOP_INGREDIENTS:
            next_state.ingredients_chopped = True

        elif action == MealAction.COOK_MEAL:
            next_state.meal_cooked = True

        elif action == MealAction.CHECK_COOKING_LEVEL:
            next_state.cooking_level_checked = True

        elif action == MealAction.CHECK_PALATABILITY:
            next_state.meal_palatable = True

        elif action == MealAction.ASSEMBLA:
            next_state.meal_assembled = True

        elif action in DELIVERY_ACTIONS:
            if next_state.can_be_deliverable:
                next_state.delivered = True

        elif action == MealAction.RECHARGE:
            next_state.battery_soc = 1.0

        else:
            raise ValueError(f"Unknown action: {action}")

        next_state.time_elapsed += ACTION_DURATIONS.get(action, 0.0)
        next_state.battery_soc = max(
            0.0, next_state.battery_soc - ACTION_BATTERY_COSTS.get(action, 0.0)
        )
        self._update_deliverability(next_state)
        return next_state

    def is_goal(self, state: MealTaskState) -> bool:
        """Check if the meal has been delivered."""
        return state.delivered
