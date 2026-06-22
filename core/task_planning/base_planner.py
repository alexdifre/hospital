"""
Abstract A* task planner base class.

Both HighLevelTaskPlanner (medication delivery) and MealTaskPlanner (meal
preparation) share the same A* skeleton. This module extracts that skeleton
so each task planner only needs to implement three methods:

    _expand(state)   → List[(action, next_state, edge_cost)]
    _heuristic(state) → float
    print_plan(actions, states)   (optional override)

The A* loop uses parent-pointer nodes (PlanNode) for memory-efficient path
reconstruction, matching the medication delivery planner's original design.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(order=True)
class PlanNode:
    """Node in the A* search tree."""

    f_cost: float
    g_cost: float = field(compare=False)
    h_cost: float = field(compare=False)
    state: Any = field(compare=False)
    parent: Optional["PlanNode"] = field(default=None, compare=False)
    action: Any = field(default=None, compare=False)

    def __hash__(self):
        return hash(self.state)


class BaseTaskPlanner(ABC):
    """
    Abstract A* planner over discrete task state spaces.

    Subclasses implement:
        _expand(state)    — successor generation + edge costs
        _heuristic(state) — admissible cost-to-go estimate

    The shared A* loop handles: open/closed sets, node expansion,
    goal detection, path reconstruction, and search stats.
    """

    def __init__(
        self,
        preference_weights: Optional[np.ndarray] = None,
        fuzzy_estimator=None,
    ):
        if preference_weights is None:
            preference_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        self.weights = preference_weights
        self.fuzzy_estimator = fuzzy_estimator

        self.nodes_expanded = 0
        self.nodes_generated = 0

    def plan(
        self,
        initial_state,
        max_nodes: int = 20000,
        verbose: bool = False,
    ) -> Tuple[List, List, Dict]:
        """
        Run A* search from initial_state to goal.

        Returns:
            actions: ordered list of actions
            states:  [initial_state, s1, s2, ...] — initial state prepended
            info:    {"success", "nodes_expanded", "nodes_generated",
                      "total_cost", "plan_length"}
        """
        self.nodes_expanded = 0
        self.nodes_generated = 0

        h0 = self._heuristic(initial_state)
        root = PlanNode(f_cost=h0, g_cost=0.0, h_cost=h0, state=initial_state)

        open_set: List[PlanNode] = [root]
        closed_set: set = set()

        while open_set and self.nodes_expanded < max_nodes:
            node = heapq.heappop(open_set)

            if node.state in closed_set:
                continue

            self.nodes_expanded += 1
            closed_set.add(node.state)

            if node.state.is_goal():
                if verbose:
                    print(f"[OK] Goal reached!")
                    print(f"   Nodes expanded: {self.nodes_expanded}")
                    print(f"   Nodes generated: {self.nodes_generated}")
                    print(f"   Total cost: {node.g_cost:.2f}")

                actions, states = self._reconstruct_path(node)
                return actions, states, {
                    "success": True,
                    "nodes_expanded": self.nodes_expanded,
                    "nodes_generated": self.nodes_generated,
                    "total_cost": node.g_cost,
                    "plan_length": len(actions),
                }

            for action, next_state, edge_cost in self._expand(node.state):
                if not next_state.is_goal() and hasattr(next_state, "is_valid"):
                    if not next_state.is_valid():
                        continue
                if next_state in closed_set:
                    continue

                g = node.g_cost + edge_cost
                h = self._heuristic(next_state)
                child = PlanNode(
                    f_cost=g + h,
                    g_cost=g,
                    h_cost=h,
                    state=next_state,
                    parent=node,
                    action=action,
                )
                heapq.heappush(open_set, child)
                self.nodes_generated += 1

            if verbose and self.nodes_expanded % 50 == 0:
                print(
                    f"   Expanded {self.nodes_expanded} nodes, "
                    f"open set size: {len(open_set)}"
                )

        if verbose:
            print(f"[FAIL] Planning failed after {self.nodes_expanded} expansions")

        return [], [initial_state], {
            "success": False,
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "total_cost": float("inf"),
            "plan_length": 0,
        }

    @abstractmethod
    def _expand(self, state) -> List[Tuple[Any, Any, float]]:
        """
        Generate successors from state.

        Returns list of (action, next_state, edge_cost).
        Each subclass handles its own manager calls and cost calculation.
        """
        ...

    @abstractmethod
    def _heuristic(self, state) -> float:
        """Admissible cost-to-go estimate from state."""
        ...

    def _reconstruct_path(self, goal_node: PlanNode) -> Tuple[List, List]:
        """Reconstruct action and state sequences via parent pointers."""
        actions, states = [], []
        node = goal_node
        while node is not None:
            if node.action is not None:
                actions.append(node.action)
            states.append(node.state)
            node = node.parent
        actions.reverse()
        states.reverse()
        return actions, states

    def print_plan(self, actions: List, states: List):
        """Pretty-print planned sequence. Override for task-specific formatting."""
        print(f"\n{'='*80}")
        print("PLANNED TASK SEQUENCE")
        print(f"{'='*80}")
        print(f"Total steps: {len(actions)}")
        if states:
            final = states[-1]
            print(f"Expected time:     {final.time_elapsed:.1f}s")
            print(f"Expected distance: {final.distance_traveled:.1f}m")
            print(f"Final battery:     {final.battery_soc*100:.1f}%")
        print(f"\nSequence:")
        post = states[1:] if len(states) > len(actions) else states
        for i, (action, state) in enumerate(zip(actions, post), 1):
            print(f"\n  Step {i}: {action.value}")
            print(f"    → State:   {state}")
            print(f"    → Battery: {state.battery_soc*100:.1f}%")
            print(f"    → Time:    {state.time_elapsed:.1f}s")
