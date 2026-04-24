"""
Meal Preparation Task — Action definitions.

Three meal types with distinct ordering constraints:
  Sandwich:  pantry → prep_station → patient
  Soup:      pantry → prep_station → stove → patient
  Full meal: pantry → prep_station → stove → prep_station → patient
"""

from enum import Enum


class MealAction(Enum):
    """High-level actions for meal preparation task."""

    # ── Navigation ──────────────────────────────────────────────
    GO_TO_PANTRY = "go_to_pantry"
    GO_TO_PREP_STATION = "go_to_prep_station"
    GO_TO_STOVE = "go_to_stove"
    GO_TO_PATIENT_LEFT = "go_to_patient_left"
    GO_TO_PATIENT_RIGHT = "go_to_patient_right"
    GO_TO_CHARGE_MAIN = "go_to_charge_main"
    GO_TO_CHARGE_BACKUP = "go_to_charge_backup"

    # ── Ingredient collection (determines meal path) ────────────
    COLLECT_SANDWICH_INGREDIENTS = "collect_sandwich_ingredients"
    COLLECT_SOUP_INGREDIENTS = "collect_soup_ingredients"
    COLLECT_MEAL_INGREDIENTS = "collect_meal_ingredients"

    # ── Preparation (location-gated) ────────────────────────────
    ASSEMBLE = "assemble"  # sandwich only, at prep_station
    CHOP = "chop"  # soup/full meal, at prep_station
    COOK = "cook"  # soup/full meal, at stove
    PLATE = "plate"  # full meal only, at prep_station (after cooking)

    # ── Delivery ────────────────────────────────────────────────
    DELIVER_MEAL = "deliver_meal"

    # ── Utility ─────────────────────────────────────────────────
    RECHARGE = "recharge"


# ── Action → target location mapping ───────────────────────────
# Used by the integrator to know where the robot should navigate.
ACTION_TARGET_LOCATIONS = {
    MealAction.GO_TO_PANTRY: "pantry",
    MealAction.GO_TO_PREP_STATION: "prep_station",
    MealAction.GO_TO_STOVE: "stove",
    MealAction.GO_TO_PATIENT_LEFT: "patient_bed_left",
    MealAction.GO_TO_PATIENT_RIGHT: "patient_bed_right",
    MealAction.GO_TO_CHARGE_MAIN: "charge_main",
    MealAction.GO_TO_CHARGE_BACKUP: "charge_backup",
}

# Navigation actions (involve MPC movement)
NAVIGATION_ACTIONS = {
    MealAction.GO_TO_PANTRY,
    MealAction.GO_TO_PREP_STATION,
    MealAction.GO_TO_STOVE,
    MealAction.GO_TO_PATIENT_LEFT,
    MealAction.GO_TO_PATIENT_RIGHT,
    MealAction.GO_TO_CHARGE_MAIN,
    MealAction.GO_TO_CHARGE_BACKUP,
}

# In-place actions (no movement, just state transition)
IN_PLACE_ACTIONS = {
    MealAction.COLLECT_SANDWICH_INGREDIENTS,
    MealAction.COLLECT_SOUP_INGREDIENTS,
    MealAction.COLLECT_MEAL_INGREDIENTS,
    MealAction.ASSEMBLE,
    MealAction.CHOP,
    MealAction.COOK,
    MealAction.PLATE,
    MealAction.DELIVER_MEAL,
    MealAction.RECHARGE,
}

# Meal type strings
MEAL_SANDWICH = "sandwich"
MEAL_SOUP = "soup"
MEAL_FULL = "full_meal"
ALL_MEAL_TYPES = [MEAL_SANDWICH, MEAL_SOUP, MEAL_FULL]
