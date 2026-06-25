"""
Meal-preparation state aligned with unified_planning/domain_meal.pddl.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from core.task_planning.base_state import TaskStateMixin


@dataclass
class MealTaskState(TaskStateMixin):
    """
    Symbolic state for meal preparation.

    PDDL predicates are represented directly. Legacy fields are kept as derived
    convenience flags for older planners/loggers.
    """

    # Location predicate: (at ?l)
    location: str

    # PDDL meal/ingredient predicates.
    meal_to_prepare: Optional[str] = None
    required_ingredients: Tuple[str, ...] = field(default_factory=tuple)
    collected_ingredients: Tuple[str, ...] = field(default_factory=tuple)
    missing_ingredients: Tuple[str, ...] = field(default_factory=tuple)
    expired_ingredients: Tuple[str, ...] = field(default_factory=tuple)
    wrong_ingredients: Tuple[str, ...] = field(default_factory=tuple)
    allergen_ingredients: Tuple[str, ...] = field(default_factory=tuple)

    ingredients_checked: bool = False
    ingredients_safe: bool = False
    workspace_clean: bool = False
    robot_hands_clean: bool = False
    cross_contamination_risk: bool = False
    ingredients_washed: bool = False
    ingredients_chopped: bool = False
    meal_cooked: bool = False
    cooking_level_checked: bool = False
    meal_palatable: bool = False
    meal_assembled: bool = False
    can_be_deliverable: bool = False
    delivered: bool = False

    # Legacy compatibility fields.
    meal_type: Optional[str] = None
    has_ingredients: bool = False
    is_chopped: bool = False
    is_cooked: bool = False
    is_assembled: bool = False
    is_plated: bool = False
    meal_ready: bool = False

    # Shared state.
    battery_soc: float = 1.0
    approach_side: Optional[str] = None
    location_memberships: Optional[Dict[str, float]] = None
    location_stock: Optional[Dict[str, int]] = None
    step_count: int = 0
    time_elapsed: float = 0.0
    distance_traveled: float = 0.0
    num_replans: int = 0

    def __hash__(self):
        """Hash for symbolic state comparison."""
        return hash(
            (
                self.location,
                self.meal_to_prepare,
                self.required_ingredients,
                self.collected_ingredients,
                self.missing_ingredients,
                self.expired_ingredients,
                self.wrong_ingredients,
                self.allergen_ingredients,
                self.ingredients_checked,
                self.ingredients_safe,
                self.workspace_clean,
                self.robot_hands_clean,
                self.cross_contamination_risk,
                self.ingredients_washed,
                self.ingredients_chopped,
                self.meal_cooked,
                self.cooking_level_checked,
                self.meal_palatable,
                self.meal_assembled,
                self.can_be_deliverable,
                self.delivered,
                self.get_discrete_battery_level(),
                self.approach_side,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, MealTaskState):
            return False
        return (
            self.location == other.location
            and self.meal_to_prepare == other.meal_to_prepare
            and self.required_ingredients == other.required_ingredients
            and self.collected_ingredients == other.collected_ingredients
            and self.missing_ingredients == other.missing_ingredients
            and self.expired_ingredients == other.expired_ingredients
            and self.wrong_ingredients == other.wrong_ingredients
            and self.allergen_ingredients == other.allergen_ingredients
            and self.ingredients_checked == other.ingredients_checked
            and self.ingredients_safe == other.ingredients_safe
            and self.workspace_clean == other.workspace_clean
            and self.robot_hands_clean == other.robot_hands_clean
            and self.cross_contamination_risk == other.cross_contamination_risk
            and self.ingredients_washed == other.ingredients_washed
            and self.ingredients_chopped == other.ingredients_chopped
            and self.meal_cooked == other.meal_cooked
            and self.cooking_level_checked == other.cooking_level_checked
            and self.meal_palatable == other.meal_palatable
            and self.meal_assembled == other.meal_assembled
            and self.can_be_deliverable == other.can_be_deliverable
            and self.delivered == other.delivered
            and self.meal_type == other.meal_type
            and self.has_ingredients == other.has_ingredients
            and self.is_chopped == other.is_chopped
            and self.is_cooked == other.is_cooked
            and self.is_assembled == other.is_assembled
            and self.is_plated == other.is_plated
            and self.meal_ready == other.meal_ready
            and self.battery_soc == other.battery_soc
            and self.approach_side == other.approach_side
        )

    def copy(self):
        """Deep copy for symbolic state transitions."""
        return MealTaskState(
            location=self.location,
            meal_to_prepare=self.meal_to_prepare,
            required_ingredients=tuple(self.required_ingredients),
            collected_ingredients=tuple(self.collected_ingredients),
            missing_ingredients=tuple(self.missing_ingredients),
            expired_ingredients=tuple(self.expired_ingredients),
            wrong_ingredients=tuple(self.wrong_ingredients),
            allergen_ingredients=tuple(self.allergen_ingredients),
            ingredients_checked=self.ingredients_checked,
            ingredients_safe=self.ingredients_safe,
            workspace_clean=self.workspace_clean,
            robot_hands_clean=self.robot_hands_clean,
            cross_contamination_risk=self.cross_contamination_risk,
            ingredients_washed=self.ingredients_washed,
            ingredients_chopped=self.ingredients_chopped,
            meal_cooked=self.meal_cooked,
            cooking_level_checked=self.cooking_level_checked,
            meal_palatable=self.meal_palatable,
            meal_assembled=self.meal_assembled,
            can_be_deliverable=self.can_be_deliverable,
            delivered=self.delivered,
            meal_type=self.meal_type,
            has_ingredients=self.has_ingredients,
            is_chopped=self.is_chopped,
            is_cooked=self.is_cooked,
            is_assembled=self.is_assembled,
            is_plated=self.is_plated,
            meal_ready=self.meal_ready,
            **self._shared_copy_kwargs(),
        )

    def is_goal(self) -> bool:
        """Goal predicate."""
        return self.delivered

    def to_dict(self) -> Dict:
        """Serialize for logging."""
        return {
            "location": self.location,
            "meal_to_prepare": self.meal_to_prepare,
            "required_ingredients": list(self.required_ingredients),
            "collected_ingredients": list(self.collected_ingredients),
            "missing_ingredients": list(self.missing_ingredients),
            "expired_ingredients": list(self.expired_ingredients),
            "wrong_ingredients": list(self.wrong_ingredients),
            "allergen_ingredients": list(self.allergen_ingredients),
            "ingredients_checked": self.ingredients_checked,
            "ingredients_safe": self.ingredients_safe,
            "workspace_clean": self.workspace_clean,
            "robot_hands_clean": self.robot_hands_clean,
            "cross_contamination_risk": self.cross_contamination_risk,
            "ingredients_washed": self.ingredients_washed,
            "ingredients_chopped": self.ingredients_chopped,
            "meal_cooked": self.meal_cooked,
            "cooking_level_checked": self.cooking_level_checked,
            "meal_palatable": self.meal_palatable,
            "meal_assembled": self.meal_assembled,
            "can_be_deliverable": self.can_be_deliverable,
            "delivered": self.delivered,
            "meal_type": self.meal_type,
            "has_ingredients": self.has_ingredients,
            "is_chopped": self.is_chopped,
            "is_cooked": self.is_cooked,
            "is_assembled": self.is_assembled,
            "is_plated": self.is_plated,
            "meal_ready": self.meal_ready,
            **self._shared_to_dict(),
        }

    def progress_str(self) -> str:
        """Compact string showing progress flags."""
        flags = [self.meal_to_prepare or "meal_unselected"]
        if self.collected_ingredients:
            flags.append(f"COLL={len(self.collected_ingredients)}")
        if self.ingredients_checked:
            flags.append("CHECK")
        if self.ingredients_safe:
            flags.append("SAFE")
        if self.workspace_clean:
            flags.append("CLEAN")
        if self.ingredients_washed:
            flags.append("WASH")
        if self.ingredients_chopped:
            flags.append("CHOP")
        if self.meal_cooked:
            flags.append("COOK")
        if self.cooking_level_checked:
            flags.append("LEVEL")
        if self.meal_palatable:
            flags.append("PAL")
        if self.meal_assembled:
            flags.append("ASSY")
        if self.can_be_deliverable:
            flags.append("READY")
        if self.delivered:
            flags.append("DELIV")
        return ",".join(flags)

    def __repr__(self):
        return (
            f"MealTaskState({self.location}, [{self.progress_str()}], "
            f"battery={self.battery_soc:.2f}, side={self.approach_side})"
        )
