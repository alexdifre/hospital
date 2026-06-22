"""
Medication-delivery actions aligned with unified_planning/domain_med.pddl.

The PDDL files are the source of truth for action names. A few legacy aliases are
kept so older call sites continue to resolve, but planner/execution code should
prefer the exact PDDL action names below.
"""

from enum import Enum


class TaskAction(Enum):
    """Exact PDDL action names for the medication-delivery task."""

    # Navigation to resource/charge locations
    GO_TO_PHARMACY_NORTH = "go_to_pharmacy_north"
    GO_TO_PHARMACY_SOUTH = "go_to_pharmacy_south"
    GO_TO_SUPPLY_A = "go_to_supply_a"
    GO_TO_SUPPLY_B = "go_to_supply_b"
    GO_TO_CHARGE_MAIN = "go_to_charge_main"
    GO_TO_CHARGE_BACKUP = "go_to_charge_backup"

    # Navigation/approach to patient side
    APPROACH_LEFT_TO_BED = "approach_left_to_bed"
    APPROACH_RIGHT_TO_BED = "approach_right_to_bed"

    # Medication collection variants
    COLLECT_MED_ANTIBIOTIC = "collect_med_antibiotic"
    COLLECT_MED_ANALGESIC = "collect_med_analgesic"
    COLLECT_MED_INSULIN = "collect_med_insulin"

    # Supplement collection variants
    COLLECT_SUPP_VITAMIN_D = "collect_supp_vitamin_d"
    COLLECT_SUPP_ELECTROLYTE = "collect_supp_electrolyte"
    COLLECT_SUPP_OMEGA3 = "collect_supp_omega3"

    # Validation / recovery
    CHECK_MEDICATION_CORRECT = "check_medication_correct"
    CHECK_MEDICATION_WRONG = "check_medication_wrong"
    CHECK_SUPPLEMENT_CORRECT = "check_supplement_correct"
    CHECK_SUPPLEMENT_WRONG = "check_supplement_wrong"
    PUT_DOWN_MEDICINE = "put_down_medicine"
    PUT_DOWN_SUPPLEMENT = "put_down_supplement"

    # Delivery and charge
    DELIVER_ON_BEDSIDE_TABLE_LEFT = "deliver_on_bedside_table_left"
    DELIVER_ON_BEDSIDE_TABLE_RIGHT = "deliver_on_bedside_table_right"
    RECHARGE = "recharge"

    # Legacy aliases used by older Python code. They resolve to PDDL actions.
    GO_TO_PATIENT_LEFT = "approach_left_to_bed"
    GO_TO_PATIENT_RIGHT = "approach_right_to_bed"
    COLLECT_MEDICATION = "collect_med_antibiotic"
    COLLECT_SUPPLEMENT = "collect_supp_vitamin_d"
    DELIVER = "deliver_on_bedside_table_left"


MEDICATION_COLLECTION_ACTIONS = {
    TaskAction.COLLECT_MED_ANTIBIOTIC,
    TaskAction.COLLECT_MED_ANALGESIC,
    TaskAction.COLLECT_MED_INSULIN,
}

SUPPLEMENT_COLLECTION_ACTIONS = {
    TaskAction.COLLECT_SUPP_VITAMIN_D,
    TaskAction.COLLECT_SUPP_ELECTROLYTE,
    TaskAction.COLLECT_SUPP_OMEGA3,
}

MEDICATION_CHECK_ACTIONS = {
    TaskAction.CHECK_MEDICATION_CORRECT,
    TaskAction.CHECK_MEDICATION_WRONG,
}

SUPPLEMENT_CHECK_ACTIONS = {
    TaskAction.CHECK_SUPPLEMENT_CORRECT,
    TaskAction.CHECK_SUPPLEMENT_WRONG,
}

PUT_DOWN_ACTIONS = {
    TaskAction.PUT_DOWN_MEDICINE,
    TaskAction.PUT_DOWN_SUPPLEMENT,
}

DELIVERY_ACTIONS = {
    TaskAction.DELIVER_ON_BEDSIDE_TABLE_LEFT,
    TaskAction.DELIVER_ON_BEDSIDE_TABLE_RIGHT,
}

NAVIGATION_ACTIONS = {
    TaskAction.GO_TO_PHARMACY_NORTH,
    TaskAction.GO_TO_PHARMACY_SOUTH,
    TaskAction.GO_TO_SUPPLY_A,
    TaskAction.GO_TO_SUPPLY_B,
    TaskAction.GO_TO_CHARGE_MAIN,
    TaskAction.GO_TO_CHARGE_BACKUP,
    TaskAction.APPROACH_LEFT_TO_BED,
    TaskAction.APPROACH_RIGHT_TO_BED,
}

IN_PLACE_ACTIONS = (
    MEDICATION_COLLECTION_ACTIONS
    | SUPPLEMENT_COLLECTION_ACTIONS
    | MEDICATION_CHECK_ACTIONS
    | SUPPLEMENT_CHECK_ACTIONS
    | PUT_DOWN_ACTIONS
    | DELIVERY_ACTIONS
    | {TaskAction.RECHARGE}
)

ACTION_TARGET_LOCATIONS = {
    TaskAction.GO_TO_PHARMACY_NORTH: "pharmacy_north",
    TaskAction.GO_TO_PHARMACY_SOUTH: "pharmacy_south",
    TaskAction.GO_TO_SUPPLY_A: "supply_A",
    TaskAction.GO_TO_SUPPLY_B: "supply_B",
    TaskAction.GO_TO_CHARGE_MAIN: "charge_main",
    TaskAction.GO_TO_CHARGE_BACKUP: "charge_backup",
    TaskAction.APPROACH_LEFT_TO_BED: "patient_bed_left",
    TaskAction.APPROACH_RIGHT_TO_BED: "patient_bed_right",
}

MEDICINE_BY_ACTION = {
    TaskAction.COLLECT_MED_ANTIBIOTIC: "med_antibiotic",
    TaskAction.COLLECT_MED_ANALGESIC: "med_analgesic",
    TaskAction.COLLECT_MED_INSULIN: "med_insulin",
}

SUPPLEMENT_BY_ACTION = {
    TaskAction.COLLECT_SUPP_VITAMIN_D: "supp_vitamin_d",
    TaskAction.COLLECT_SUPP_ELECTROLYTE: "supp_electrolyte",
    TaskAction.COLLECT_SUPP_OMEGA3: "supp_omega3",
}

REQUESTED_MEDICINE = "med_antibiotic"
REQUESTED_SUPPLEMENT = "supp_vitamin_d"

ACTION_DURATIONS = {
    **{action: 5.0 for action in MEDICATION_COLLECTION_ACTIONS},
    **{action: 5.0 for action in SUPPLEMENT_COLLECTION_ACTIONS},
    **{action: 2.0 for action in MEDICATION_CHECK_ACTIONS},
    **{action: 2.0 for action in SUPPLEMENT_CHECK_ACTIONS},
    **{action: 3.0 for action in PUT_DOWN_ACTIONS},
    **{action: 167.0 for action in DELIVERY_ACTIONS},
    TaskAction.RECHARGE: 30.0,
}

ACTION_BATTERY_COSTS = {
    **{action: 0.05 for action in MEDICATION_COLLECTION_ACTIONS},
    **{action: 0.05 for action in SUPPLEMENT_COLLECTION_ACTIONS},
    **{action: 0.02 for action in MEDICATION_CHECK_ACTIONS},
    **{action: 0.02 for action in SUPPLEMENT_CHECK_ACTIONS},
    **{action: 0.05 for action in PUT_DOWN_ACTIONS},
    **{action: 0.02 for action in DELIVERY_ACTIONS},
    TaskAction.RECHARGE: 0.0,
}
