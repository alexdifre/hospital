#!/usr/bin/env python3
"""
Task Actions for Medication Delivery
=====================================

High-level discrete actions and constants for medication delivery planning.

Mirrors the meal_preparation/task_actions.py pattern:
  - Enum of all actions
  - ACTION_TARGET_LOCATIONS: action → location name (for navigation actions)
  - NAVIGATION_ACTIONS: set of actions that move the robot
  - IN_PLACE_ACTIONS: set of actions performed at current location
  - ACTION_DURATIONS: action → expected duration in seconds
"""

from enum import Enum
from typing import Dict, Set


class TaskAction(Enum):
    """High-level actions the robot can take for medication delivery."""

    # Navigation actions
    GO_TO_PHARMACY_NORTH = "go_to_pharmacy_north"
    GO_TO_PHARMACY_SOUTH = "go_to_pharmacy_south"
    GO_TO_SUPPLY_A = "go_to_supply_a"
    GO_TO_SUPPLY_B = "go_to_supply_b"
    GO_TO_CHARGE_MAIN = "go_to_charge_main"
    GO_TO_CHARGE_BACKUP = "go_to_charge_backup"
    GO_TO_PATIENT_LEFT = "go_to_patient_left"
    GO_TO_PATIENT_RIGHT = "go_to_patient_right"

    # In-place actions
    COLLECT_MEDICATION = "collect_medication"
    COLLECT_SUPPLEMENT = "collect_supplement"
    RECHARGE = "recharge"
    DELIVER = "deliver"


# =====================================================================
# MODULE-LEVEL CONSTANTS (match meal_preparation/task_actions.py pattern)
# =====================================================================

ACTION_TARGET_LOCATIONS: Dict[TaskAction, str] = {
    TaskAction.GO_TO_PHARMACY_NORTH: "pharmacy_north",
    TaskAction.GO_TO_PHARMACY_SOUTH: "pharmacy_south",
    TaskAction.GO_TO_SUPPLY_A: "supply_A",
    TaskAction.GO_TO_SUPPLY_B: "supply_B",
    TaskAction.GO_TO_CHARGE_MAIN: "charge_main",
    TaskAction.GO_TO_CHARGE_BACKUP: "charge_backup",
    TaskAction.GO_TO_PATIENT_LEFT: "patient_bed_left",
    TaskAction.GO_TO_PATIENT_RIGHT: "patient_bed_right",
}

NAVIGATION_ACTIONS: Set[TaskAction] = set(ACTION_TARGET_LOCATIONS.keys())

IN_PLACE_ACTIONS: Set[TaskAction] = {
    TaskAction.COLLECT_MEDICATION,
    TaskAction.COLLECT_SUPPLEMENT,
    TaskAction.RECHARGE,
    TaskAction.DELIVER,
}

ACTION_DURATIONS: Dict[TaskAction, float] = {
    # Navigation durations are computed dynamically by the planner/executor
    # In-place action durations (seconds)
    TaskAction.COLLECT_MEDICATION: 5.0,
    TaskAction.COLLECT_SUPPLEMENT: 5.0,
    TaskAction.RECHARGE: 30.0,
    TaskAction.DELIVER: 10.0,
}

# Location groups for precondition checks
PHARMACY_LOCATIONS: Set[str] = {"pharmacy_north", "pharmacy_south"}
SUPPLY_LOCATIONS: Set[str] = {"supply_A", "supply_B"}
CHARGE_LOCATIONS: Set[str] = {"charge_main", "charge_backup"}
PATIENT_LOCATIONS: Set[str] = {"patient_bed_left", "patient_bed_right"}
