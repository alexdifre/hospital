"""
Meal-preparation actions aligned with unified_planning/domain_meal.pddl.
"""

from enum import Enum


class MealAction(Enum):
    """Exact PDDL action names for meal preparation."""

    # Meal choice
    CHOOSE_SANDWICH = "choose_sandwich"
    CHOOSE_SALAD = "choose_salad"
    CHOOSE_HOT_MEAL = "choose_hot_meal"

    # Navigation
    GO_TO_PANTRY = "go_to_pantry"
    GO_TO_FRIDGE = "go_to_fridge"
    GO_TO_PREP_STATION = "go_to_prep_station"
    GO_TO_COOKING_STATION = "go_to_cooking_station"
    GO_TO_QUALITY_CHECK = "go_to_quality_check"
    APPROACH_TO_LEFT_SIDE = "approach_to_left_side"
    APPROACH_TO_RIGHT_SIDE = "approach_to_right_side"

    # Ingredient and preparation predicates
    COLLECT_INGREDIENT = "collect_ingredient"
    CHECK_INGREDIENTS = "check_ingredients"
    SANITIZE_WORKSPACE = "sanitize_workspace"
    WASH_INGREDIENTS = "wash_ingredients"
    CHOP_INGREDIENTS = "chop_ingredients"
    COOK_MEAL = "cook_meal"
    CHECK_COOKING_LEVEL = "check_cooking_level"
    CHECK_PALATABILITY = "check_palatability"
    ASSEMBLA = "assembla"

    # Delivery
    DELIVER_ON_BEDSIDE_TABLE_LEFT = "deliver_on_bedside_table_left"
    DELIVER_ON_BEDSIDE_TABLE_RIGHT = "deliver_on_bedside_table_right"

    # Legacy compatibility aliases.
    GO_TO_STOVE = "go_to_cooking_station"
    GO_TO_PATIENT_LEFT = "approach_to_left_side"
    GO_TO_PATIENT_RIGHT = "approach_to_right_side"
    COLLECT_SANDWICH_INGREDIENTS = "collect_ingredient"
    COLLECT_SOUP_INGREDIENTS = "collect_ingredient"
    COLLECT_MEAL_INGREDIENTS = "collect_ingredient"
    ASSEMBLE = "assembla"
    CHOP = "chop_ingredients"
    COOK = "cook_meal"
    PLATE = "assembla"
    DELIVER_MEAL = "deliver_on_bedside_table_left"

    # Charging
    GO_TO_CHARGE_MAIN = "go_to_charge_main"
    GO_TO_CHARGE_BACKUP = "go_to_charge_backup"
    RECHARGE = "recharge"


ACTION_TARGET_LOCATIONS = {
    MealAction.GO_TO_PANTRY: "pantry",
    MealAction.GO_TO_FRIDGE: "fridge",
    MealAction.GO_TO_PREP_STATION: "prep_station",
    MealAction.GO_TO_COOKING_STATION: "stove",
    MealAction.GO_TO_QUALITY_CHECK: "quality_check",
    MealAction.APPROACH_TO_LEFT_SIDE: "patient_bed_left",
    MealAction.APPROACH_TO_RIGHT_SIDE: "patient_bed_right",
    MealAction.GO_TO_CHARGE_MAIN: "charge_main",
    MealAction.GO_TO_CHARGE_BACKUP: "charge_backup",
}

PDDL_NAV_EDGES = {
    ("home", "pantry"),
    ("home", "fridge"),
    ("pantry", "fridge"),
    ("pantry", "prep_station"),
    ("fridge", "prep_station"),
    ("prep_station", "quality_check"),
    ("prep_station", "stove"),
    ("stove", "quality_check"),
    ("quality_check", "patient_bed_left"),
    ("quality_check", "patient_bed_right"),
    ("charge_main", "pantry"),
    ("charge_main", "fridge"),
    ("charge_main", "prep_station"),
    ("charge_main", "stove"),
    ("charge_main", "quality_check"),
    ("charge_main", "patient_bed_left"),
    ("charge_main", "patient_bed_right"),
    ("charge_backup", "pantry"),
    ("charge_backup", "fridge"),
    ("charge_backup", "prep_station"),
    ("charge_backup", "stove"),
    ("charge_backup", "quality_check"),
    ("charge_backup", "patient_bed_left"),
    ("charge_backup", "patient_bed_right"),
}

NAVIGATION_ACTIONS = {
    MealAction.GO_TO_PANTRY,
    MealAction.GO_TO_FRIDGE,
    MealAction.GO_TO_PREP_STATION,
    MealAction.GO_TO_COOKING_STATION,
    MealAction.GO_TO_QUALITY_CHECK,
    MealAction.APPROACH_TO_LEFT_SIDE,
    MealAction.APPROACH_TO_RIGHT_SIDE,
    MealAction.GO_TO_CHARGE_MAIN,
    MealAction.GO_TO_CHARGE_BACKUP,
}

DELIVERY_ACTIONS = {
    MealAction.DELIVER_ON_BEDSIDE_TABLE_LEFT,
    MealAction.DELIVER_ON_BEDSIDE_TABLE_RIGHT,
}

IN_PLACE_ACTIONS = {
    MealAction.CHOOSE_SANDWICH,
    MealAction.CHOOSE_SALAD,
    MealAction.CHOOSE_HOT_MEAL,
    MealAction.COLLECT_INGREDIENT,
    MealAction.CHECK_INGREDIENTS,
    MealAction.SANITIZE_WORKSPACE,
    MealAction.WASH_INGREDIENTS,
    MealAction.CHOP_INGREDIENTS,
    MealAction.COOK_MEAL,
    MealAction.CHECK_COOKING_LEVEL,
    MealAction.CHECK_PALATABILITY,
    MealAction.ASSEMBLA,
    *DELIVERY_ACTIONS,
    MealAction.RECHARGE,
}

ACTION_DURATIONS = {
    MealAction.CHOOSE_SANDWICH: 0.5000,
    MealAction.CHOOSE_SALAD: 0.5000,
    MealAction.CHOOSE_HOT_MEAL: 0.0,
    MealAction.COLLECT_INGREDIENT: 5.0,
    MealAction.CHECK_INGREDIENTS: 4.0,
    MealAction.SANITIZE_WORKSPACE: 5.0,
    MealAction.WASH_INGREDIENTS: 6.0,
    MealAction.CHOP_INGREDIENTS: 8.0,
    MealAction.COOK_MEAL: 12.0,
    MealAction.CHECK_COOKING_LEVEL: 4.0,
    MealAction.CHECK_PALATABILITY: 4.0,
    MealAction.ASSEMBLA: 5.0,
    MealAction.DELIVER_ON_BEDSIDE_TABLE_LEFT: 10.0,
    MealAction.DELIVER_ON_BEDSIDE_TABLE_RIGHT: 10.0,
    MealAction.RECHARGE: 30.0,
}

ACTION_BATTERY_COSTS = {
    MealAction.CHOOSE_SANDWICH: 0.0,
    MealAction.CHOOSE_SALAD: 0.0,
    MealAction.CHOOSE_HOT_MEAL: 0.0,
    MealAction.COLLECT_INGREDIENT: 0.0200,
    MealAction.CHECK_INGREDIENTS: 0.0100,
    MealAction.SANITIZE_WORKSPACE: 0.0200,
    MealAction.WASH_INGREDIENTS: 0.0200,
    MealAction.CHOP_INGREDIENTS: 0.0300,
    MealAction.COOK_MEAL: 0.5000,
    MealAction.CHECK_COOKING_LEVEL: 0.0100,
    MealAction.CHECK_PALATABILITY: 0.0100,
    MealAction.ASSEMBLA: 0.0200,
    MealAction.DELIVER_ON_BEDSIDE_TABLE_LEFT: 0.0100,
    MealAction.DELIVER_ON_BEDSIDE_TABLE_RIGHT: 0.0100,
    MealAction.RECHARGE: 0.0,
}

MEAL_SANDWICH = "sandwich"
MEAL_SALAD = "salad"
MEAL_HOT = "hot_meal"

MEAL_REQUIRED_INGREDIENTS = {
    MEAL_SANDWICH: ("bread", "vegetables"),
    MEAL_SALAD: ("nuts", "vegetables"),
    MEAL_HOT: ("chicken", "vegetables"),
}
DEFAULT_MEAL_TYPE = MEAL_SANDWICH
REQUIRED_INGREDIENTS = MEAL_REQUIRED_INGREDIENTS[DEFAULT_MEAL_TYPE]
INGREDIENT_LOCATIONS = {
    "bread": "pantry",
    "nuts": "pantry",
    "chicken": "fridge",
    "vegetables": "pantry",
}

# Legacy meal type strings retained for compatibility with older experiments.
MEAL_DIABETIC = MEAL_HOT
MEAL_SOUP = MEAL_SALAD
MEAL_FULL = MEAL_HOT
ALL_MEAL_TYPES = [MEAL_SANDWICH, MEAL_SALAD, MEAL_HOT]

MEAL_CHOICE_ACTIONS = {
    MealAction.CHOOSE_SANDWICH: MEAL_SANDWICH,
    MealAction.CHOOSE_SALAD: MEAL_SALAD,
    MealAction.CHOOSE_HOT_MEAL: MEAL_HOT,
}
