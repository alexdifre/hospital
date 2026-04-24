"""
Meal Preparation Task — State representation.

Tracks the robot's progress through the meal preparation pipeline:
  ingredients → chop/assemble → cook → plate → deliver
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from core.task_planning.base_state import TaskStateMixin
from .task_actions import MEAL_SANDWICH, MEAL_SOUP, MEAL_FULL


@dataclass
class MealTaskState(TaskStateMixin):
    """
    High-level task state for meal preparation.

    Separate from the continuous 6D MuJoCo state.
    The progression flags enforce ordering constraints via preconditions
    in the state manager.
    """

    # ── Location ────────────────────────────────────────────────
    location: str  # e.g. 'pantry', 'prep_station', 'stove', 'patient_bed_left'

    # ── Meal progression ────────────────────────────────────────
    meal_type: Optional[str] = None  # 'sandwich', 'soup', 'full_meal'
    has_ingredients: bool = False
    is_chopped: bool = False  # soup / full_meal
    is_cooked: bool = False  # soup / full_meal
    is_assembled: bool = False  # sandwich only
    is_plated: bool = False  # full_meal only
    meal_ready: bool = False  # ready for delivery
    delivered: bool = False  # GOAL

    # ── Shared state ────────────────────────────────────────────
    battery_soc: float = 1.0
    approach_side: Optional[str] = None  # 'left' or 'right'

    # Fuzzy memberships (optional, from FuzzyStateEstimator)
    location_memberships: Optional[Dict[str, float]] = None

    # Stock levels (optional, for planning-time scarcity reasoning)
    location_stock: Optional[Dict[str, int]] = None

    # ── Tracking ────────────────────────────────────────────────
    step_count: int = 0
    time_elapsed: float = 0.0
    distance_traveled: float = 0.0
    num_replans: int = 0

    # ── Hash / equality ─────────────────────────────────────────

    def __hash__(self):
        """Hash for A* search. Includes meal progression flags."""
        battery_discrete = int(self.battery_soc * 8)
        return hash(
            (
                self.location,
                self.meal_type,
                self.has_ingredients,
                self.is_chopped,
                self.is_cooked,
                self.is_assembled,
                self.is_plated,
                self.meal_ready,
                self.delivered,
                battery_discrete,
                self.approach_side,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, MealTaskState):
            return False
        return (
            self.location == other.location
            and self.meal_type == other.meal_type
            and self.has_ingredients == other.has_ingredients
            and self.is_chopped == other.is_chopped
            and self.is_cooked == other.is_cooked
            and self.is_assembled == other.is_assembled
            and self.is_plated == other.is_plated
            and self.meal_ready == other.meal_ready
            and self.delivered == other.delivered
            and int(self.battery_soc * 8) == int(other.battery_soc * 8)
            and self.approach_side == other.approach_side
        )

    # ── Utilities ───────────────────────────────────────────────

    def copy(self):
        """Deep copy for A* successor generation."""
        return MealTaskState(
            location=self.location,
            meal_type=self.meal_type,
            has_ingredients=self.has_ingredients,
            is_chopped=self.is_chopped,
            is_cooked=self.is_cooked,
            is_assembled=self.is_assembled,
            is_plated=self.is_plated,
            meal_ready=self.meal_ready,
            delivered=self.delivered,
            **self._shared_copy_kwargs(),
        )

    def is_goal(self) -> bool:
        """Check if meal has been successfully delivered."""
        return self.delivered

    def to_dict(self) -> Dict:
        """Serialize for logging."""
        return {
            "location": self.location,
            "meal_type": self.meal_type,
            "has_ingredients": self.has_ingredients,
            "is_chopped": self.is_chopped,
            "is_cooked": self.is_cooked,
            "is_assembled": self.is_assembled,
            "is_plated": self.is_plated,
            "meal_ready": self.meal_ready,
            "delivered": self.delivered,
            **self._shared_to_dict(),
        }

    def progress_str(self) -> str:
        """Compact string showing meal progress flags."""
        flags = []
        if self.meal_type:
            flags.append(self.meal_type.upper()[:4])
        if self.has_ingredients:
            flags.append("INGR")
        if self.is_chopped:
            flags.append("CHOP")
        if self.is_cooked:
            flags.append("COOK")
        if self.is_assembled:
            flags.append("ASSY")
        if self.is_plated:
            flags.append("PLAT")
        if self.meal_ready:
            flags.append("READY")
        if self.delivered:
            flags.append("DELIV")
        return ",".join(flags) if flags else "NONE"

    def __repr__(self):
        return (
            f"MealTaskState({self.location}, [{self.progress_str()}], "
            f"battery={self.battery_soc:.2f}, side={self.approach_side})"
        )
