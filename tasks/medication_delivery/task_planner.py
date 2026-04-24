#!/usr/bin/env python3
"""
High-Level Task Planner for Medication Delivery
================================================

A* search over task sequences to find optimal plans for:
- Collecting medication from pharmacy
- Collecting supplement from supply room
- (Optional) Recharging at charging station
- Delivering to patient via preferred approach

Uses the spatial A* planner as a subroutine to estimate movement costs.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict

from core.task_planning.base_planner import BaseTaskPlanner

from .task_actions import TaskAction, ACTION_TARGET_LOCATIONS
from .task_state import TaskState
from .task_state_manager import TaskStateManager


class HighLevelTaskPlanner(BaseTaskPlanner):
    """
    A* planner over task sequences.

    Searches through the discrete task state space to find optimal
    sequences of actions (collect medication, collect supplement, deliver).
    """

    def __init__(
        self,
        task_state_manager: TaskStateManager,
        spatial_planner,
        preference_weights: Optional[np.ndarray] = None,
        fuzzy_estimator=None,
    ):
        """
        Args:
            task_state_manager: TaskStateManager instance
            spatial_planner: SpatialAStarPlanner for movement cost estimation
            preference_weights: [w_time, w_safety, w_battery, w_proximity, w_approach]
            fuzzy_estimator: FuzzyStateEstimator for smooth cost computation.
        """
        super().__init__(preference_weights=preference_weights, fuzzy_estimator=fuzzy_estimator)
        self.task_manager = task_state_manager
        self.spatial_planner = spatial_planner
        self.env = task_state_manager.env

        print("HighLevelTaskPlanner initialized")
        print(f"  Preference weights: {self.weights}")
        print(
            f"  w_time={self.weights[0]:.2f}, w_safety={self.weights[1]:.2f}, "
            f"w_battery={self.weights[2]:.2f}, w_proximity={self.weights[3]:.2f}, "
            f"w_approach={self.weights[4]:.2f}"
        )
        print(
            f"  Fuzzy cost: {'enabled' if self.fuzzy_estimator else 'crisp fallback'}"
        )

    def _expand(self, state: TaskState) -> List[Tuple]:
        """Generate (action, next_state, edge_cost) successors."""
        successors = []
        for action in self.task_manager.get_available_actions(state):
            distance_cost, time_cost = self.task_manager.estimate_action_cost(
                state, action, self.spatial_planner
            )
            next_state = self.task_manager.apply_action(
                state, action, distance_cost, time_cost
            )
            edge_cost = self._calculate_action_cost(
                state, next_state, action, distance_cost, time_cost
            )
            successors.append((action, next_state, edge_cost))
        return successors

    def _heuristic(self, state: TaskState) -> float:
        """Admissible heuristic for remaining cost to goal."""
        h = 0.0
        current_pos = self.env.locations[state.location]

        if not state.has_medication:
            pharmacy_north_pos = self.env.locations["pharmacy_north"]
            pharmacy_south_pos = self.env.locations["pharmacy_south"]
            dist_north = np.linalg.norm(current_pos - pharmacy_north_pos)
            dist_south = np.linalg.norm(current_pos - pharmacy_south_pos)
            h += min(dist_north, dist_south) + 5.0

        if not state.has_supplement:
            supply_a_pos = self.env.locations["supply_A"]
            supply_b_pos = self.env.locations["supply_B"]
            dist_a = np.linalg.norm(current_pos - supply_a_pos)
            dist_b = np.linalg.norm(current_pos - supply_b_pos)
            h += min(dist_a, dist_b) + 5.0

        if not state.delivered:
            patient_left_pos = self.env.locations["patient_bed_left"]
            patient_right_pos = self.env.locations["patient_bed_right"]
            dist_left = np.linalg.norm(current_pos - patient_left_pos)
            dist_right = np.linalg.norm(current_pos - patient_right_pos)
            h += min(dist_left, dist_right) + 10.0

        h *= self.weights[0]
        return h

    def _calculate_action_cost(
        self,
        state: TaskState,
        next_state: TaskState,
        action: TaskAction,
        distance: float,
        time: float,
    ) -> float:
        """
        Calculate multi-objective cost for an action.

        Components:
        - Time cost (delivery speed)
        - Safety cost (congestion + battery risk + scarcity risk)
        - Battery cost (energy efficiency)
        - Proximity cost (approach distance)
        - Approach cost (preferred side)
        """
        costs = np.zeros(5)

        # 1. Time cost (normalized by typical mission time ~60s)
        costs[0] = time / 60.0

        # 2. Safety cost (congestion + battery risk + scarcity)
        if self.fuzzy_estimator is not None:
            next_pos = self.env.locations.get(next_state.location, np.zeros(2))
            fm = self.fuzzy_estimator.estimate(next_pos, next_state.battery_soc)

            battery_risk = fm.battery_penalty(
                penalty_low=0.5, penalty_med=0.1, penalty_high=0.0
            )
            congestion = fm.congestion_penalty(
                penalty_safe=0.0, penalty_mod=0.15, penalty_haz=0.3
            )
            costs[1] = battery_risk + congestion
        else:
            safety_cost = 0.0
            if next_state.location == "nurse_station":
                safety_cost += 0.3
            elif next_state.location == "equipment_storage":
                safety_cost += 0.2
            if next_state.battery_soc < 0.15:
                safety_cost += 0.5
            elif next_state.battery_soc < 0.25:
                safety_cost += 0.2
            costs[1] = safety_cost

        # Scarcity risk
        if action in (TaskAction.COLLECT_MEDICATION, TaskAction.COLLECT_SUPPLEMENT):
            stock = self._get_location_stock(state)
            if stock is not None:
                scarcity_penalty = 0.3 / (1.0 + stock)
                costs[1] += scarcity_penalty

        # 3. Battery cost
        battery_used = state.battery_soc - next_state.battery_soc
        costs[2] = battery_used

        # 4. Proximity cost
        if action == TaskAction.DELIVER:
            if next_state.location == "patient_bed_left":
                costs[3] = 0.0
            elif next_state.location == "patient_bed_right":
                costs[3] = 0.1

        # 5. Approach side cost
        if action in [TaskAction.GO_TO_PATIENT_LEFT, TaskAction.GO_TO_PATIENT_RIGHT]:
            if action == TaskAction.GO_TO_PATIENT_LEFT:
                costs[4] = 0.0
            else:
                costs[4] = 0.05

        total_cost = np.dot(self.weights, costs)
        return total_cost

    def _get_location_stock(self, state: TaskState) -> Optional[int]:
        """Get stock count at the robot's current location."""
        loc = state.location

        if state.location_stock is not None:
            if loc in state.location_stock:
                return state.location_stock[loc]

        stock_key = f"{loc}_stock"
        if hasattr(self, "env") and hasattr(self.env, "environment_state"):
            if stock_key in self.env.environment_state:
                return self.env.environment_state[stock_key]

        return None

    def print_plan(self, actions: List[TaskAction], states: List[TaskState]):
        """Pretty-print the planned task sequence."""
        print(f"\n{'='*80}")
        print(f"PLANNED TASK SEQUENCE")
        print(f"{'='*80}")
        print(f"Total steps: {len(actions)}")
        print(f"Expected time: {states[-1].time_elapsed:.1f}s")
        print(f"Expected distance: {states[-1].distance_traveled:.1f}m")
        print(f"Final battery: {states[-1].battery_soc*100:.1f}%")
        print(f"\nSequence:")

        for i, (action, state) in enumerate(zip(actions, states[1:]), 1):
            print(f"\n  Step {i}: {action.value}")
            print(f"    → State: {state}")
            print(f"    → Battery: {state.battery_soc*100:.1f}%")
            print(f"    → Time: {state.time_elapsed:.1f}s")
