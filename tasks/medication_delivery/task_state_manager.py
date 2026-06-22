#!/usr/bin/env python3
"""
Task State Manager for Medication Delivery.

State transitions follow the action/predicate structure in
unified_planning/domain_med.pddl while preserving the public Python API used by
the planner and integration layer.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .task_actions import (
    ACTION_BATTERY_COSTS,
    ACTION_DURATIONS,
    ACTION_TARGET_LOCATIONS,
    DELIVERY_ACTIONS,
    IN_PLACE_ACTIONS,
    MEDICINE_BY_ACTION,
    MEDICATION_COLLECTION_ACTIONS,
    NAVIGATION_ACTIONS,
    REQUESTED_MEDICINE,
    REQUESTED_SUPPLEMENT,
    SUPPLEMENT_BY_ACTION,
    SUPPLEMENT_COLLECTION_ACTIONS,
    TaskAction,
)
from .task_state import TaskState

PHARMACY_LOCATIONS = {"pharmacy_north", "pharmacy_south"}
SUPPLY_LOCATIONS = {"supply_A", "supply_B"}
CHARGE_LOCATIONS = {"charge_main", "charge_backup"}
PATIENT_LOCATIONS = {"patient_bed_left", "patient_bed_right"}

__all__ = [
    "TaskAction",
    "TaskState",
    "TaskStateManager",
    "ACTION_TARGET_LOCATIONS",
    "NAVIGATION_ACTIONS",
    "IN_PLACE_ACTIONS",
    "ACTION_DURATIONS",
    "PHARMACY_LOCATIONS",
    "SUPPLY_LOCATIONS",
    "CHARGE_LOCATIONS",
    "PATIENT_LOCATIONS",
]


class TaskStateManager:
    """Manage symbolic medication-delivery state transitions."""

    def __init__(self, environment, locations: List[str], fuzzy_estimator=None):
        self.env = environment
        self.locations = locations
        self.fuzzy_estimator = fuzzy_estimator

        self.battery_critical = 0.15
        self.battery_low = 0.25
        self.action_locations = {
            action: location
            for action, location in ACTION_TARGET_LOCATIONS.items()
            if location in locations
        }

        print("TaskStateManager initialized")
        print(f"  Locations: {len(locations)}")
        print(f"  Battery critical threshold: {self.battery_critical*100:.0f}%")
        print(f"  Battery low threshold: {self.battery_low*100:.0f}%")
        print(
            f"  Fuzzy preconditions: {'enabled' if self.fuzzy_estimator else 'crisp'}"
        )

    def get_initial_state(self, start_location: str = "home") -> TaskState:
        """Create initial state matching problem_med.pddl."""
        return TaskState(
            location=start_location,
            has_medication=False,
            has_supplement=False,
            requested_medicine=REQUESTED_MEDICINE,
            requested_supplement=REQUESTED_SUPPLEMENT,
            delivered=False,
            battery_soc=1.0,
            approach_side=None,
        )

    def _is_at_any(
        self, state: TaskState, location_group: set, action_type: str = "default"
    ) -> Tuple[bool, Optional[str]]:
        """Check crisp/fuzzy membership in a location group."""
        if state.location_memberships is not None:
            threshold = 1.0
            if self.fuzzy_estimator is not None:
                threshold = self.fuzzy_estimator.get_action_threshold(action_type)

            for loc in location_group:
                if state.location_memberships.get(loc, 0.0) >= threshold:
                    return True, loc
            return False, None

        if state.location in location_group:
            return True, state.location
        return False, None

    def _has_stock(self, state: TaskState, location: str) -> bool:
        """Check planning/execution stock for a location."""
        if state.location_stock is not None:
            if state.location_stock.get(location, 0) > 0:
                return True
            stock_key = f"{location}_stock"
            if state.location_stock.get(stock_key, 0) > 0:
                return True

        stock_key = f"{location}_stock"
        bool_key = f"{location}_in_stock"
        if stock_key in self.env.environment_state:
            return self.env.environment_state[stock_key] > 0
        if bool_key in self.env.environment_state:
            return bool(self.env.environment_state[bool_key])
        return True

    def _decrement_planning_stock(self, state: TaskState) -> None:
        """Consume one unit from the current location in a planning state."""
        if state.location_stock is None:
            return

        for key in (state.location, f"{state.location}_stock"):
            if key in state.location_stock:
                state.location_stock[key] = max(0, state.location_stock[key] - 1)
                return

    def _update_deliverability(self, state: TaskState) -> None:
        """Update can_be_deliverable from the PDDL delivery preconditions."""
        state.can_be_deliverable = (
            state.location in PATIENT_LOCATIONS
            and state.correct_medication
            and state.correct_supplement
            and state.has_medication
            and state.has_supplement
        )

    def get_available_actions(self, state: TaskState) -> List[TaskAction]:
        """Return actions whose PDDL-style preconditions are currently true."""
        actions: List[TaskAction] = []
        if state.delivered:
            return actions

        has_unchecked = state.unchecked_medication or state.unchecked_supplement
        recollect_required = (
            state.medicine_recollect_required or state.supplement_recollect_required
        )

        # PDDL navigation actions are blocked while unchecked/recollect states are
        # active, except for going to charge.
        for action, location in self.action_locations.items():
            if location == state.location:
                continue
            is_charge = location in CHARGE_LOCATIONS
            is_patient_approach = location in PATIENT_LOCATIONS
            if is_charge and state.battery_soc > self.battery_low:
                continue
            if is_patient_approach and not (
                state.correct_medication and state.correct_supplement
            ):
                continue
            if not is_charge and (has_unchecked or recollect_required):
                continue
            actions.append(action)

        at_pharmacy, pharmacy = self._is_at_any(
            state, PHARMACY_LOCATIONS, "collect_medication"
        )
        if at_pharmacy and pharmacy and self._has_stock(state, pharmacy):
            if not state.has_medication:
                actions.extend(MEDICATION_COLLECTION_ACTIONS)

        at_supply, supply = self._is_at_any(
            state, SUPPLY_LOCATIONS, "collect_supplement"
        )
        if at_supply and supply and self._has_stock(state, supply):
            if not state.has_supplement:
                actions.extend(SUPPLEMENT_COLLECTION_ACTIONS)

        if state.unchecked_medication and state.carrying_medicine:
            if state.carrying_medicine == state.requested_medicine:
                actions.append(TaskAction.CHECK_MEDICATION_CORRECT)
            else:
                actions.append(TaskAction.CHECK_MEDICATION_WRONG)

        if state.unchecked_supplement and state.carrying_supplement:
            if state.carrying_supplement == state.requested_supplement:
                actions.append(TaskAction.CHECK_SUPPLEMENT_CORRECT)
            else:
                actions.append(TaskAction.CHECK_SUPPLEMENT_WRONG)

        if state.medicine_recollect_required:
            actions.append(TaskAction.PUT_DOWN_MEDICINE)
        if state.supplement_recollect_required:
            actions.append(TaskAction.PUT_DOWN_SUPPLEMENT)

        at_charger, _ = self._is_at_any(state, CHARGE_LOCATIONS, "recharge")
        if at_charger and state.battery_soc < 1.0:
            actions.append(TaskAction.RECHARGE)

        self._update_deliverability(state)
        if state.can_complete_delivery():
            if state.location == "patient_bed_left":
                actions.append(TaskAction.DELIVER_ON_BEDSIDE_TABLE_LEFT)
            elif state.location == "patient_bed_right":
                actions.append(TaskAction.DELIVER_ON_BEDSIDE_TABLE_RIGHT)

        return actions

    def apply_action(
        self,
        state: TaskState,
        action: TaskAction,
        distance_cost: float = 0.0,
        time_cost: float = 0.0,
    ) -> TaskState:
        """Apply an action and return the successor state."""
        new_state = state.copy()

        if action in NAVIGATION_ACTIONS:
            new_state.location = ACTION_TARGET_LOCATIONS[action]
            if new_state.location_memberships is not None:
                new_state.location_memberships = {new_state.location: 1.0}

            if action == TaskAction.APPROACH_LEFT_TO_BED:
                new_state.approach_side = "left"
            elif action == TaskAction.APPROACH_RIGHT_TO_BED:
                new_state.approach_side = "right"

            battery_cost = distance_cost * 0.01
            new_state.battery_soc = max(0.0, new_state.battery_soc - battery_cost)
            new_state.distance_traveled += distance_cost

        elif action in MEDICATION_COLLECTION_ACTIONS:
            new_state.has_medication = True
            new_state.carrying_medicine = MEDICINE_BY_ACTION[action]
            new_state.unchecked_medication = True
            new_state.correct_medication = False
            new_state.medicine_recollect_required = False
            self._decrement_planning_stock(new_state)

        elif action in SUPPLEMENT_COLLECTION_ACTIONS:
            new_state.has_supplement = True
            new_state.carrying_supplement = SUPPLEMENT_BY_ACTION[action]
            new_state.unchecked_supplement = True
            new_state.correct_supplement = False
            new_state.supplement_recollect_required = False
            self._decrement_planning_stock(new_state)

        elif action == TaskAction.CHECK_MEDICATION_CORRECT:
            new_state.unchecked_medication = False
            new_state.correct_medication = True
            new_state.medicine_recollect_required = False

        elif action == TaskAction.CHECK_MEDICATION_WRONG:
            new_state.unchecked_medication = False
            new_state.correct_medication = False
            new_state.medicine_recollect_required = True

        elif action == TaskAction.CHECK_SUPPLEMENT_CORRECT:
            new_state.unchecked_supplement = False
            new_state.correct_supplement = True
            new_state.supplement_recollect_required = False

        elif action == TaskAction.CHECK_SUPPLEMENT_WRONG:
            new_state.unchecked_supplement = False
            new_state.correct_supplement = False
            new_state.supplement_recollect_required = True

        elif action == TaskAction.PUT_DOWN_MEDICINE:
            new_state.has_medication = False
            new_state.carrying_medicine = None
            new_state.unchecked_medication = False
            new_state.correct_medication = False
            new_state.medicine_recollect_required = False

        elif action == TaskAction.PUT_DOWN_SUPPLEMENT:
            new_state.has_supplement = False
            new_state.carrying_supplement = None
            new_state.unchecked_supplement = False
            new_state.correct_supplement = False
            new_state.supplement_recollect_required = False

        elif action == TaskAction.RECHARGE:
            new_state.battery_soc = 1.0

        elif action in DELIVERY_ACTIONS:
            if new_state.can_complete_delivery():
                new_state.delivered = True

        fixed_time = ACTION_DURATIONS.get(action, 0.0)
        new_state.time_elapsed += time_cost if time_cost > 0.0 else fixed_time
        if action in IN_PLACE_ACTIONS:
            new_state.battery_soc = max(
                0.0, new_state.battery_soc - ACTION_BATTERY_COSTS.get(action, 0.0)
            )
        new_state.step_count += 1
        new_state.actions_taken += 1
        self._update_deliverability(new_state)
        return new_state

    def estimate_action_cost(self, state: TaskState, action: TaskAction) -> Tuple[float, float]:
        """Estimate distance/time cost for an action."""
        if action in IN_PLACE_ACTIONS:
            return (0.0, ACTION_DURATIONS.get(action, 5.0))

        target_location = ACTION_TARGET_LOCATIONS[action]
        start_pos = self.env.locations[state.location]
        goal_pos = self.env.locations[target_location]
        distance = float(np.linalg.norm(goal_pos - start_pos))
        base_time = distance / 1.5

        congestion_multiplier = 1.0
        meta = getattr(self.env, "location_metadata", {}).get(target_location, {})
        congestion_multiplier += float(meta.get("congestion", 0.0))

        return (distance, base_time * congestion_multiplier)

    def get_state_from_environment(self, task_flags: Dict) -> TaskState:
        """Construct TaskState from environment and task flags."""
        robot_pos = self.env.robot_state_6d[:2]
        current_location = self._find_nearest_location(robot_pos)

        return TaskState(
            location=current_location,
            has_medication=task_flags.get("has_medication", False),
            has_supplement=task_flags.get("has_supplement", False),
            carrying_medicine=task_flags.get("carrying_medicine"),
            carrying_supplement=task_flags.get("carrying_supplement"),
            unchecked_medication=task_flags.get("unchecked_medication", False),
            unchecked_supplement=task_flags.get("unchecked_supplement", False),
            correct_medication=task_flags.get("correct_medication", False),
            correct_supplement=task_flags.get("correct_supplement", False),
            medicine_recollect_required=task_flags.get(
                "medicine_recollect_required", False
            ),
            supplement_recollect_required=task_flags.get(
                "supplement_recollect_required", False
            ),
            can_be_deliverable=task_flags.get("can_be_deliverable", False),
            delivered=task_flags.get("delivered", False),
            battery_soc=self.env.environment_state["battery_level"],
            approach_side=task_flags.get("approach_side", None),
            step_count=task_flags.get("step_count", 0),
            actions_taken=task_flags.get("actions_taken", 0),
            time_elapsed=task_flags.get("time_elapsed", 0.0),
            distance_traveled=task_flags.get("distance_traveled", 0.0),
        )

    def _find_nearest_location(
        self, position: np.ndarray, tolerance: float = 1.5
    ) -> str:
        """Find nearest named location to a position."""
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
        """Return dominant location and optional fuzzy memberships."""
        if self.fuzzy_estimator is not None:
            fm = self.fuzzy_estimator.estimate(position, battery_soc)
            return fm.dominant_location, dict(fm.location_memberships)
        crisp_loc = self._find_nearest_location(position)
        return crisp_loc, None

    def print_state(self, state: TaskState):
        """Pretty-print task state."""
        print(f"\n{'='*60}")
        print("TASK STATE")
        print(f"{'='*60}")
        print(f"Location: {state.location}")
        print(f"Medication: {state.carrying_medicine or 'None'}")
        print(f"Supplement: {state.carrying_supplement or 'None'}")
        print(f"Medication checked: {'yes' if state.correct_medication else 'no'}")
        print(f"Supplement checked: {'yes' if state.correct_supplement else 'no'}")
        print(f"Deliverable: {'yes' if state.can_be_deliverable else 'no'}")
        print(f"Delivered: {'yes' if state.delivered else 'no'}")
        print(f"Battery: {state.battery_soc*100:.1f}%", end="")
        if state.needs_recharge():
            print(" CRITICAL")
        elif state.battery_soc < self.battery_low:
            print(" LOW")
        else:
            print()
        print(f"Approach side: {state.approach_side or 'None'}")
        print(f"Time: {state.time_elapsed:.1f}s")
        print(f"Distance: {state.distance_traveled:.1f}m")
        print(f"{'='*60}\n")
