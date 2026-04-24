#!/usr/bin/env python3
"""
Task State Manager for Medication Delivery
============================================

Manages state transitions, precondition checking, and action cost estimation
for medication delivery task planning.

Refactored to import from:
  - task_actions.py: TaskAction enum + constants
  - task_state.py: TaskState dataclass

Backwards-compatible: TaskAction and TaskState are re-exported from this
module so existing `from task_state_manager import TaskAction, TaskState`
still works.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List

# Import and re-export for backwards compatibility
from .task_actions import (
    TaskAction,
    ACTION_TARGET_LOCATIONS,
    NAVIGATION_ACTIONS,
    IN_PLACE_ACTIONS,
    ACTION_DURATIONS,
    PHARMACY_LOCATIONS,
    SUPPLY_LOCATIONS,
    CHARGE_LOCATIONS,
    PATIENT_LOCATIONS,
)
from .task_state import TaskState

# Re-export so `from task_state_manager import TaskAction, TaskState` still works
__all__ = [
    "TaskAction",
    "TaskState",
    "TaskStateManager",
    "ACTION_TARGET_LOCATIONS",
    "NAVIGATION_ACTIONS",
    "IN_PLACE_ACTIONS",
    "ACTION_DURATIONS",
]


class TaskStateManager:
    """
    Manages task state transitions and validity checking.

    This class handles the discrete task-level state space,
    coordinating with the environment's continuous state.
    """

    def __init__(self, environment, locations: List[str], fuzzy_estimator=None):
        """
        Args:
            environment: MuJoCo environment with locations and state
            locations: List of valid location names
            fuzzy_estimator: Optional FuzzyStateEstimator for soft preconditions.
                             If None, uses crisp location == checks (original behavior).
        """
        self.env = environment
        self.locations = locations
        self.fuzzy_estimator = fuzzy_estimator

        # Critical thresholds
        self.battery_critical = 0.15
        self.battery_low = 0.25

        # Backwards-compatible instance attribute (integrator uses this)
        self.action_locations = dict(ACTION_TARGET_LOCATIONS)

        print("TaskStateManager initialized")
        print(f"  Locations: {len(locations)}")
        print(f"  Battery critical threshold: {self.battery_critical*100:.0f}%")
        print(f"  Battery low threshold: {self.battery_low*100:.0f}%")
        print(
            f"  Fuzzy preconditions: {'enabled' if self.fuzzy_estimator else 'crisp'}"
        )

    def get_initial_state(self, start_location: str = "home") -> TaskState:
        """Create initial task state."""
        return TaskState(
            location=start_location,
            has_medication=False,
            has_supplement=False,
            delivered=False,
            battery_soc=1.0,
            approach_side=None,
        )

    def _is_at_any(
        self, state: TaskState, location_group: set, action_type: str = "default"
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if robot is at any location in a group.

        Fuzzy mode: checks if any location in group has μ >= threshold.
        Crisp mode: checks state.location in group.

        Returns:
            (is_at, matched_location)
        """
        if state.location_memberships is not None:
            threshold = 0.7
            if self.fuzzy_estimator is not None:
                threshold = self.fuzzy_estimator.get_action_threshold(action_type)

            for loc in location_group:
                mu = state.location_memberships.get(loc, 0.0)
                if mu >= threshold:
                    return True, loc
            return False, None
        else:
            if state.location in location_group:
                return True, state.location
            return False, None

    def _has_stock(self, state: TaskState, location: str) -> bool:
        """
        Check if a location has stock available.

        Priority: state.location_stock (planning) > env state (execution).
        """
        if state.location_stock is not None:
            return state.location_stock.get(location, 0) > 0

        stock_key = f"{location}_stock"
        if stock_key in self.env.environment_state:
            return self.env.environment_state[stock_key] > 0

        bool_key = f"{location}_in_stock"
        if bool_key in self.env.environment_state:
            return self.env.environment_state[bool_key]

        return True

    def get_available_actions(self, state: TaskState) -> List[TaskAction]:
        """
        Get list of valid actions from current state.

        Rules:
        - Can only collect medication at pharmacies with stock > 0
        - Can only collect supplement at supply rooms with stock > 0
        - Can only recharge at charging stations
        - Must have medication + supplement before delivery
        - Can only deliver at patient bed
        """
        actions = []

        # Movement actions (always available if not delivered)
        if not state.delivered:
            for action, location in ACTION_TARGET_LOCATIONS.items():
                if location != state.location:
                    actions.append(action)

        # Collect medication (pharmacy + stock check)
        if not state.has_medication:
            at_pharmacy, matched = self._is_at_any(
                state, PHARMACY_LOCATIONS, "collect_medication"
            )
            if at_pharmacy and matched and self._has_stock(state, matched):
                actions.append(TaskAction.COLLECT_MEDICATION)

        # Collect supplement (supply + stock check)
        if not state.has_supplement:
            at_supply, matched = self._is_at_any(
                state, SUPPLY_LOCATIONS, "collect_supplement"
            )
            if at_supply and matched and self._has_stock(state, matched):
                actions.append(TaskAction.COLLECT_SUPPLEMENT)

        # Recharge
        at_charger, _ = self._is_at_any(state, CHARGE_LOCATIONS, "recharge")
        if at_charger and state.battery_soc < 1.0:
            actions.append(TaskAction.RECHARGE)

        # Deliver
        if not state.delivered:
            at_patient, _ = self._is_at_any(state, PATIENT_LOCATIONS, "deliver")
            if at_patient and state.has_medication and state.has_supplement:
                actions.append(TaskAction.DELIVER)

        return actions

    def apply_action(
        self,
        state: TaskState,
        action: TaskAction,
        distance_cost: float = 0.0,
        time_cost: float = 0.0,
    ) -> TaskState:
        """
        Apply action to state, returning new state.

        Args:
            state: Current task state
            action: Action to apply
            distance_cost: Distance traveled (for battery depletion)
            time_cost: Time elapsed (seconds)

        Returns:
            New task state after applying action
        """
        new_state = state.copy()

        # Navigation actions
        if action in NAVIGATION_ACTIONS:
            new_state.location = ACTION_TARGET_LOCATIONS[action]

            # Synthesize location memberships for planning
            if new_state.location_memberships is not None:
                new_state.location_memberships = {new_state.location: 1.0}

            # Track approach side
            if action == TaskAction.GO_TO_PATIENT_LEFT:
                new_state.approach_side = "left"
            elif action == TaskAction.GO_TO_PATIENT_RIGHT:
                new_state.approach_side = "right"

            # Battery depletion from movement (1% per meter)
            battery_cost = distance_cost * 0.01
            new_state.battery_soc = max(0.0, new_state.battery_soc - battery_cost)
            new_state.distance_traveled += distance_cost

        # Collection actions (decrement stock in planning state)
        elif action == TaskAction.COLLECT_MEDICATION:
            new_state.has_medication = True
            if (
                new_state.location_stock is not None
                and new_state.location in new_state.location_stock
            ):
                new_state.location_stock[new_state.location] = max(
                    0, new_state.location_stock[new_state.location] - 1
                )

        elif action == TaskAction.COLLECT_SUPPLEMENT:
            new_state.has_supplement = True
            if (
                new_state.location_stock is not None
                and new_state.location in new_state.location_stock
            ):
                new_state.location_stock[new_state.location] = max(
                    0, new_state.location_stock[new_state.location] - 1
                )

        elif action == TaskAction.RECHARGE:
            new_state.battery_soc = 1.0

        elif action == TaskAction.DELIVER:
            if new_state.has_medication and new_state.has_supplement:
                new_state.delivered = True

        # Update time
        new_state.time_elapsed += time_cost
        new_state.step_count += 1

        return new_state

    def estimate_action_cost(
        self, state: TaskState, action: TaskAction, spatial_planner
    ) -> Tuple[float, float]:
        """
        Estimate distance and time cost for an action.

        Returns:
            (distance_meters, time_seconds)
        """
        # In-place actions: zero distance, fixed duration
        if action in IN_PLACE_ACTIONS:
            return (0.0, ACTION_DURATIONS.get(action, 5.0))

        # Navigation actions: compute from spatial layout
        target_location = ACTION_TARGET_LOCATIONS[action]
        start_pos = self.env.locations[state.location]
        goal_pos = self.env.locations[target_location]

        distance = np.linalg.norm(goal_pos - start_pos)
        base_time = distance / 1.5  # ~1.5 m/s average

        # Congestion penalties
        congestion_multiplier = 1.0
        if target_location == "nurse_station":
            congestion_multiplier = 1.5
        elif target_location == "equipment_storage":
            congestion_multiplier = 1.3

        time = base_time * congestion_multiplier
        return (distance, time)

    # =====================================================================
    # Environment bridge methods
    # =====================================================================

    def get_state_from_environment(self, task_flags: Dict) -> TaskState:
        """Construct TaskState from environment and task flags."""
        robot_pos = self.env.robot_state_6d[:2]
        current_location = self._find_nearest_location(robot_pos)

        return TaskState(
            location=current_location,
            has_medication=task_flags.get("has_medication", False),
            has_supplement=task_flags.get("has_supplement", False),
            delivered=task_flags.get("delivered", False),
            battery_soc=self.env.environment_state["battery_level"],
            approach_side=task_flags.get("approach_side", None),
            step_count=task_flags.get("step_count", 0),
            time_elapsed=task_flags.get("time_elapsed", 0.0),
            distance_traveled=task_flags.get("distance_traveled", 0.0),
        )

    def _find_nearest_location(
        self, position: np.ndarray, tolerance: float = 1.5
    ) -> str:
        """Find the nearest location to a position (crisp)."""
        min_dist = float("inf")
        nearest = "traveling"

        for loc_name, loc_pos in self.env.locations.items():
            dist = np.linalg.norm(position - loc_pos)
            if dist < tolerance and dist < min_dist:
                min_dist = dist
                nearest = loc_name

        return nearest

    def get_fuzzy_location(
        self, position: np.ndarray, battery_soc: float = 1.0
    ) -> Tuple[str, Optional[Dict[str, float]]]:
        """
        Get location from position using fuzzy estimation if available.

        Returns:
            (dominant_location, location_memberships)
        """
        if self.fuzzy_estimator is not None:
            fm = self.fuzzy_estimator.estimate(position, battery_soc)
            return fm.dominant_location, dict(fm.location_memberships)
        else:
            crisp_loc = self._find_nearest_location(position)
            return crisp_loc, None

    # =====================================================================
    # Display
    # =====================================================================

    def print_state(self, state: TaskState):
        """Pretty-print task state."""
        print(f"\n{'='*60}")
        print(f"TASK STATE")
        print(f"{'='*60}")
        print(f"Location: {state.location}")
        print(f"Medication: {'✓' if state.has_medication else '✗'}")
        print(f"Supplement: {'✓' if state.has_supplement else '✗'}")
        print(f"Delivered: {'✓' if state.delivered else '✗'}")
        print(f"Battery: {state.battery_soc*100:.1f}%", end="")
        if state.needs_recharge():
            print(" ⚠ CRITICAL")
        elif state.battery_soc < self.battery_low:
            print(" ⚡ LOW")
        else:
            print()
        print(f"Approach side: {state.approach_side or 'None'}")
        print(f"Time: {state.time_elapsed:.1f}s")
        print(f"Distance: {state.distance_traveled:.1f}m")
        print(f"{'='*60}\n")
