#!/usr/bin/env python3
"""
Task State for Medication Delivery
====================================

Discrete state representation for high-level medication delivery planning.

Mirrors the meal_preparation/task_state.py pattern:
  - Immutable-ish dataclass with copy()
  - Hash/equality for A* closed-set membership
  - Goal check and validity constraints
  - Serialization (to_dict, __repr__)
"""

from typing import Dict, Optional
from dataclasses import dataclass

from core.task_planning.base_state import TaskStateMixin


@dataclass
class TaskState(TaskStateMixin):
    """
    High-level task state for medication delivery.

    This represents the discrete state space for task planning,
    separate from the continuous 6D robot state used by MPC.
    """

    # Location
    location: str  # Current location name (e.g., 'pharmacy_north', 'home')

    # Task flags
    has_medication: bool = False  # Collected primary medication?
    has_supplement: bool = False  # Collected supplementary item?
    delivered: bool = False  # Delivered to patient?

    # Battery state (discretized for planning)
    battery_soc: float = 1.0  # State of charge [0.0, 1.0]

    # Approach preference
    approach_side: Optional[str] = None  # 'left', 'right', or None

    # Fuzzy location memberships (optional, populated by FuzzyStateEstimator)
    # When set, maps location names to membership degrees in [0, 1]
    # e.g. {'pharmacy_north': 0.85, 'supply_A': 0.02}
    location_memberships: Optional[Dict[str, float]] = None

    # Stock levels at each stocked location (optional)
    # Tracks depletion during planning so A* can reason about stockouts.
    # e.g. {'pharmacy_north': 5, 'supply_A': 7, 'supply_B': 1}
    location_stock: Optional[Dict[str, int]] = None

    # Episode tracking
    step_count: int = 0
    time_elapsed: float = 0.0  # seconds

    # Performance metrics
    distance_traveled: float = 0.0
    num_replans: int = 0

    def __hash__(self):
        """Hash for use in search algorithms."""
        # Discretize battery for hashing (8 levels: 0%, 12.5%, 25%, ..., 100%)
        battery_discrete = int(self.battery_soc * 8)
        return hash(
            (
                self.location,
                self.has_medication,
                self.has_supplement,
                self.delivered,
                battery_discrete,
                self.approach_side,
            )
        )

    def __eq__(self, other):
        """Equality check for use in search."""
        if not isinstance(other, TaskState):
            return False
        return (
            self.location == other.location
            and self.has_medication == other.has_medication
            and self.has_supplement == other.has_supplement
            and self.delivered == other.delivered
            and abs(self.battery_soc - other.battery_soc) < 0.13  # Same discrete level
            and self.approach_side == other.approach_side
        )

    def copy(self) -> "TaskState":
        """Create a deep copy of this state."""
        return TaskState(
            location=self.location,
            has_medication=self.has_medication,
            has_supplement=self.has_supplement,
            delivered=self.delivered,
            **self._shared_copy_kwargs(),
        )

    def is_goal(self) -> bool:
        """Check if this is a goal state (medication delivered)."""
        return self.delivered

    def is_valid(self) -> bool:
        """Check if this state is physically valid."""
        if self.battery_soc < 0.0:
            return False
        if self.delivered and not self.has_medication:
            return False
        return True

    def can_complete_task(self) -> bool:
        """Check if robot has all items needed for delivery."""
        return self.has_medication and self.has_supplement

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "location": self.location,
            "has_medication": self.has_medication,
            "has_supplement": self.has_supplement,
            "delivered": self.delivered,
            **self._shared_to_dict(),
        }

    def __repr__(self) -> str:
        """String representation."""
        flags = []
        if self.has_medication:
            flags.append("MED")
        if self.has_supplement:
            flags.append("SUP")
        if self.delivered:
            flags.append("DELIV")

        flags_str = ",".join(flags) if flags else "NONE"

        return (
            f"TaskState({self.location}, [{flags_str}], "
            f"battery={self.battery_soc:.2f}, "
            f"side={self.approach_side})"
        )
