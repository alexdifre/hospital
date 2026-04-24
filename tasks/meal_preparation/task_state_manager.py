"""
Meal Preparation Task — State manager.

Defines preconditions (which actions are available) and transitions
(how actions change the state). The A* planner uses this to expand
the search graph.

Valid meal sequences:
  Sandwich:  pantry → collect_sandwich → prep → assemble → patient → deliver
  Soup:      pantry → collect_soup → prep → chop → stove → cook → patient → deliver
  Full meal: pantry → collect_meal → prep → chop → stove → cook → prep → plate → patient → deliver
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .task_actions import (
    MealAction,
    ACTION_TARGET_LOCATIONS,
    NAVIGATION_ACTIONS,
    MEAL_SANDWICH,
    MEAL_SOUP,
    MEAL_FULL,
)
from .task_state import MealTaskState


# ── Distance lookup (Euclidean, filled from environment) ────────
# Overridden at runtime with actual env distances.
_DEFAULT_DISTANCES: Dict[Tuple[str, str], float] = {}

# ── Time/battery cost per unit distance ─────────────────────────
SPEED = 1.5  # m/s robot speed
BATTERY_PER_METER = 0.01  # 1% per meter

# ── In-place action durations (seconds) ────────────────────────
ACTION_DURATIONS = {
    MealAction.COLLECT_SANDWICH_INGREDIENTS: 5.0,
    MealAction.COLLECT_SOUP_INGREDIENTS: 5.0,
    MealAction.COLLECT_MEAL_INGREDIENTS: 5.0,
    MealAction.ASSEMBLE: 5.0,
    MealAction.CHOP: 8.0,
    MealAction.COOK: 12.0,
    MealAction.PLATE: 5.0,
    MealAction.DELIVER_MEAL: 10.0,
    MealAction.RECHARGE: 30.0,
}


class MealTaskStateManager:
    """
    Manages preconditions and state transitions for meal preparation.

    The environment reference is optional — without it, distance estimates
    use straight-line defaults. With it, actual location coordinates are used.
    """

    def __init__(
        self,
        env=None,
        locations: Optional[List[str]] = None,
        fuzzy_estimator=None,
    ):
        self.env = env
        self.fuzzy_estimator = fuzzy_estimator

        # Build distance table from environment
        self._distances: Dict[Tuple[str, str], float] = {}
        if env is not None:
            self._build_distance_table(env)

        # Locations relevant to meal prep
        self.pantry_locations = {"pantry"}
        self.prep_locations = {"prep_station"}
        self.stove_locations = {"stove"}
        self.patient_locations = {"patient_bed_left", "patient_bed_right"}
        self.charger_locations = {"charge_main", "charge_backup"}

        # All kitchen-area locations
        self.kitchen_locations = (
            self.pantry_locations | self.prep_locations | self.stove_locations
        )

    # ─────────────────────────────────────────────────────────────
    # Distance table
    # ─────────────────────────────────────────────────────────────

    def _build_distance_table(self, env):
        """Pre-compute pairwise Euclidean distances between all locations."""
        for name_a, pos_a in env.locations.items():
            for name_b, pos_b in env.locations.items():
                d = float(np.linalg.norm(pos_a - pos_b))
                self._distances[(name_a, name_b)] = d

    def get_distance(self, loc_a: str, loc_b: str) -> float:
        """Get distance between two locations."""
        if loc_a == loc_b:
            return 0.0
        return self._distances.get((loc_a, loc_b), 15.0)  # default 15m

    # ─────────────────────────────────────────────────────────────
    # Initial state
    # ─────────────────────────────────────────────────────────────

    def get_initial_state(self, start_location: str) -> MealTaskState:
        """Create a fresh task state at the given location."""
        return MealTaskState(location=start_location)

    # ─────────────────────────────────────────────────────────────
    # Location checks (with fuzzy support)
    # ─────────────────────────────────────────────────────────────

    def _is_at(self, state: MealTaskState, location: str) -> bool:
        """Check if robot is at a specific location (crisp or fuzzy)."""
        if state.location_memberships is not None:
            return state.location_memberships.get(location, 0.0) >= 0.8
        return state.location == location

    def _is_at_any(self, state: MealTaskState, locations: set) -> bool:
        """Check if robot is at any of the given locations."""
        if state.location_memberships is not None:
            return any(
                state.location_memberships.get(loc, 0.0) >= 0.8 for loc in locations
            )
        return state.location in locations

    def _has_stock(
        self, state: MealTaskState, location: str, item: str = "any"
    ) -> bool:
        """
        Check if location has stock for planning.
        For pantry, check per-ingredient-type stock.
        """
        if state.location_stock is not None:
            stock_key = f"{location}_{item}" if item != "any" else location
            stock = state.location_stock.get(stock_key, None)
            if stock is not None:
                return stock > 0
            # Fallback: check location-level stock
            stock = state.location_stock.get(location, None)
            if stock is not None:
                return stock > 0
        # Default: assume available
        return True

    # ─────────────────────────────────────────────────────────────
    # Available actions (precondition checks)
    # ─────────────────────────────────────────────────────────────

    def get_available_actions(self, state: MealTaskState) -> List[MealAction]:
        """
        Return all actions whose preconditions are satisfied in the current state.
        This is the core of the ordering constraint system.
        """
        if state.delivered:
            return []  # Goal reached, no more actions

        actions = []

        # ── Navigation actions ──────────────────────────────────
        # Only offer locations relevant to current task phase.
        # This keeps the branching factor manageable for A*.
        nav_targets = {}

        # Always allow going to charger if battery is low
        if state.battery_soc < 0.3:
            nav_targets[MealAction.GO_TO_CHARGE_MAIN] = "charge_main"

        # Phase-dependent navigation
        if not state.has_ingredients:
            # Need ingredients → go to pantry
            nav_targets[MealAction.GO_TO_PANTRY] = "pantry"
        elif state.meal_type == MEAL_SANDWICH and not state.is_assembled:
            # Sandwich needs prep station
            nav_targets[MealAction.GO_TO_PREP_STATION] = "prep_station"
        elif state.meal_type in (MEAL_SOUP, MEAL_FULL) and not state.is_chopped:
            # Need to chop → prep station
            nav_targets[MealAction.GO_TO_PREP_STATION] = "prep_station"
        elif state.meal_type in (MEAL_SOUP, MEAL_FULL) and not state.is_cooked:
            # Need to cook → stove
            nav_targets[MealAction.GO_TO_STOVE] = "stove"
        elif state.meal_type == MEAL_FULL and not state.is_plated:
            # Need to plate → prep station
            nav_targets[MealAction.GO_TO_PREP_STATION] = "prep_station"

        # Meal ready → go to patient
        if state.meal_ready and not state.delivered:
            nav_targets[MealAction.GO_TO_PATIENT_LEFT] = "patient_bed_left"
            nav_targets[MealAction.GO_TO_PATIENT_RIGHT] = "patient_bed_right"

        # Before meal type is chosen, allow exploring kitchen
        if state.meal_type is None and state.has_ingredients is False:
            nav_targets[MealAction.GO_TO_PANTRY] = "pantry"
            nav_targets[MealAction.GO_TO_PREP_STATION] = "prep_station"

        for action, target in nav_targets.items():
            if state.location != target:
                actions.append(action)

        # ── Ingredient collection (at pantry, no meal chosen yet) ──
        if self._is_at_any(state, self.pantry_locations) and state.meal_type is None:
            if self._has_stock(state, "pantry", MEAL_SANDWICH):
                actions.append(MealAction.COLLECT_SANDWICH_INGREDIENTS)
            if self._has_stock(state, "pantry", MEAL_SOUP):
                actions.append(MealAction.COLLECT_SOUP_INGREDIENTS)
            if self._has_stock(state, "pantry", MEAL_FULL):
                actions.append(MealAction.COLLECT_MEAL_INGREDIENTS)

        # ── Assemble (sandwich, at prep_station) ────────────────
        if (
            self._is_at_any(state, self.prep_locations)
            and state.meal_type == MEAL_SANDWICH
            and state.has_ingredients
            and not state.is_assembled
        ):
            actions.append(MealAction.ASSEMBLE)

        # ── Chop (soup/full_meal, at prep_station, not yet chopped) ──
        if (
            self._is_at_any(state, self.prep_locations)
            and state.meal_type in (MEAL_SOUP, MEAL_FULL)
            and state.has_ingredients
            and not state.is_chopped
        ):
            actions.append(MealAction.CHOP)

        # ── Cook (soup/full_meal, at stove, after chopping) ─────
        if (
            self._is_at_any(state, self.stove_locations)
            and state.meal_type in (MEAL_SOUP, MEAL_FULL)
            and state.is_chopped
            and not state.is_cooked
        ):
            actions.append(MealAction.COOK)

        # ── Plate (full_meal only, at prep_station, after cooking) ──
        if (
            self._is_at_any(state, self.prep_locations)
            and state.meal_type == MEAL_FULL
            and state.is_cooked
            and not state.is_plated
        ):
            actions.append(MealAction.PLATE)

        # ── Deliver (at patient bed, meal is ready) ─────────────
        if self._is_at_any(state, self.patient_locations) and state.meal_ready:
            actions.append(MealAction.DELIVER_MEAL)

        # ── Recharge (at charger) ───────────────────────────────
        if self._is_at_any(state, self.charger_locations) and state.battery_soc < 0.9:
            actions.append(MealAction.RECHARGE)

        return actions

    # ─────────────────────────────────────────────────────────────
    # State transitions (apply action → new state)
    # ─────────────────────────────────────────────────────────────

    def apply_action(
        self,
        state: MealTaskState,
        action: MealAction,
    ) -> MealTaskState:
        """
        Apply an action to produce a successor state.
        Used by A* during planning (no real environment side effects).
        """
        next_state = state.copy()
        next_state.step_count += 1

        # ── Navigation ──────────────────────────────────────────
        if action in NAVIGATION_ACTIONS:
            target = ACTION_TARGET_LOCATIONS[action]
            dist = self.get_distance(state.location, target)
            travel_time = dist / SPEED
            battery_cost = dist * BATTERY_PER_METER

            next_state.location = target
            next_state.time_elapsed += travel_time
            next_state.distance_traveled += dist
            next_state.battery_soc = max(0.0, next_state.battery_soc - battery_cost)

            # Synthesize crisp membership for planning
            # (overwritten by real fuzzy estimate during execution)
            next_state.location_memberships = {target: 1.0}

            # Set approach side for patient beds
            if target == "patient_bed_left":
                next_state.approach_side = "left"
            elif target == "patient_bed_right":
                next_state.approach_side = "right"

            return next_state

        # ── Ingredient collection ───────────────────────────────
        if action == MealAction.COLLECT_SANDWICH_INGREDIENTS:
            next_state.meal_type = MEAL_SANDWICH
            next_state.has_ingredients = True
            next_state.time_elapsed += ACTION_DURATIONS[action]
            self._decrement_stock(next_state, "pantry", MEAL_SANDWICH)
            return next_state

        if action == MealAction.COLLECT_SOUP_INGREDIENTS:
            next_state.meal_type = MEAL_SOUP
            next_state.has_ingredients = True
            next_state.time_elapsed += ACTION_DURATIONS[action]
            self._decrement_stock(next_state, "pantry", MEAL_SOUP)
            return next_state

        if action == MealAction.COLLECT_MEAL_INGREDIENTS:
            next_state.meal_type = MEAL_FULL
            next_state.has_ingredients = True
            next_state.time_elapsed += ACTION_DURATIONS[action]
            self._decrement_stock(next_state, "pantry", MEAL_FULL)
            return next_state

        # ── Preparation steps ───────────────────────────────────
        if action == MealAction.ASSEMBLE:
            next_state.is_assembled = True
            next_state.meal_ready = True
            next_state.time_elapsed += ACTION_DURATIONS[action]
            return next_state

        if action == MealAction.CHOP:
            next_state.is_chopped = True
            next_state.time_elapsed += ACTION_DURATIONS[action]
            return next_state

        if action == MealAction.COOK:
            next_state.is_cooked = True
            next_state.time_elapsed += ACTION_DURATIONS[action]
            # Soup is ready after cooking (served from pot)
            if next_state.meal_type == MEAL_SOUP:
                next_state.meal_ready = True
            return next_state

        if action == MealAction.PLATE:
            next_state.is_plated = True
            next_state.meal_ready = True
            next_state.time_elapsed += ACTION_DURATIONS[action]
            return next_state

        # ── Delivery ────────────────────────────────────────────
        if action == MealAction.DELIVER_MEAL:
            next_state.delivered = True
            next_state.time_elapsed += ACTION_DURATIONS[action]
            return next_state

        # ── Recharge ────────────────────────────────────────────
        if action == MealAction.RECHARGE:
            next_state.battery_soc = min(1.0, next_state.battery_soc + 0.4)
            next_state.time_elapsed += ACTION_DURATIONS[action]
            return next_state

        raise ValueError(f"Unknown action: {action}")

    # ─────────────────────────────────────────────────────────────
    # Stock management
    # ─────────────────────────────────────────────────────────────

    def _decrement_stock(
        self,
        state: MealTaskState,
        location: str,
        item: str,
    ):
        """Decrement stock during planning (no real env side effects)."""
        if state.location_stock is None:
            return
        stock_key = f"{location}_{item}"
        if stock_key in state.location_stock:
            state.location_stock[stock_key] = max(
                0, state.location_stock[stock_key] - 1
            )

    # ─────────────────────────────────────────────────────────────
    # Goal check
    # ─────────────────────────────────────────────────────────────

    def is_goal(self, state: MealTaskState) -> bool:
        """Check if the meal has been delivered."""
        return state.delivered
