"""
Medication task state aligned with unified_planning/domain_med.pddl predicates.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from .task_actions import REQUESTED_MEDICINE, REQUESTED_SUPPLEMENT


@dataclass
class MedicationDeliveryState:
    """
    Symbolic state for medication delivery.

    The field names mirror PDDL predicates where possible, while preserving a few
    legacy booleans used by the Python execution and logging code.
    """

    # Location predicate: (at ?l)
    location: str = "home"

    # Legacy carry flags plus PDDL carrying object names.
    has_medication: bool = False
    has_supplement: bool = False
    carrying_medicine: Optional[str] = None
    carrying_supplement: Optional[str] = None

    # Requested objects from problem_med.pddl.
    requested_medicine: str = REQUESTED_MEDICINE
    requested_supplement: str = REQUESTED_SUPPLEMENT

    # PDDL validation predicates.
    unchecked_medication: bool = False
    unchecked_supplement: bool = False
    correct_medication: bool = False
    correct_supplement: bool = False
    medicine_recollect_required: bool = False
    supplement_recollect_required: bool = False
    can_be_deliverable: bool = False

    # Terminal and resource state.
    delivered: bool = False
    battery_soc: float = 1.0

    # Approach side retained for metrics/features.
    approach_side: Optional[str] = None

    # Shared execution tracking.
    time_elapsed: float = 0.0
    step_count: int = 0
    actions_taken: int = 0
    distance_traveled: float = 0.0
    num_replans: int = 0
    location_memberships: Optional[Dict[str, float]] = None
    location_stock: Optional[Dict[str, int]] = None

    def is_valid(self) -> bool:
        """Check if state satisfies basic invariants."""
        if not 0.0 <= self.battery_soc <= 1.0:
            return False
        if self.delivered:
            return (
                self.has_medication
                and self.has_supplement
                and self.correct_medication
                and self.correct_supplement
            )
        if self.has_medication and self.carrying_medicine is None:
            return False
        if self.has_supplement and self.carrying_supplement is None:
            return False
        return True

    def is_goal(self) -> bool:
        """Goal predicate: delivered medication/supplement to the patient."""
        return self.delivered

    def can_complete_delivery(self) -> bool:
        """Whether PDDL delivery preconditions are satisfied."""
        return (
            self.has_medication
            and self.has_supplement
            and self.correct_medication
            and self.correct_supplement
            and self.can_be_deliverable
            and self.location in {"patient_bed_left", "patient_bed_right"}
        )

    def needs_recharge(self, threshold: float = 0.15) -> bool:
        """Return whether battery is below the critical threshold."""
        return self.battery_soc <= threshold

    def copy(self) -> "MedicationDeliveryState":
        """Create a copy of the state."""
        return MedicationDeliveryState(
            location=self.location,
            has_medication=self.has_medication,
            has_supplement=self.has_supplement,
            carrying_medicine=self.carrying_medicine,
            carrying_supplement=self.carrying_supplement,
            requested_medicine=self.requested_medicine,
            requested_supplement=self.requested_supplement,
            unchecked_medication=self.unchecked_medication,
            unchecked_supplement=self.unchecked_supplement,
            correct_medication=self.correct_medication,
            correct_supplement=self.correct_supplement,
            medicine_recollect_required=self.medicine_recollect_required,
            supplement_recollect_required=self.supplement_recollect_required,
            can_be_deliverable=self.can_be_deliverable,
            delivered=self.delivered,
            battery_soc=self.battery_soc,
            approach_side=self.approach_side,
            time_elapsed=self.time_elapsed,
            step_count=self.step_count,
            actions_taken=self.actions_taken,
            distance_traveled=self.distance_traveled,
            num_replans=self.num_replans,
            location_memberships=(
                None
                if self.location_memberships is None
                else dict(self.location_memberships)
            ),
            location_stock=None if self.location_stock is None else dict(self.location_stock),
        )

    def to_dict(self) -> dict:
        """Convert state to dictionary for logging."""
        return {
            "location": self.location,
            "has_medication": self.has_medication,
            "has_supplement": self.has_supplement,
            "carrying_medicine": self.carrying_medicine,
            "carrying_supplement": self.carrying_supplement,
            "requested_medicine": self.requested_medicine,
            "requested_supplement": self.requested_supplement,
            "unchecked_medication": self.unchecked_medication,
            "unchecked_supplement": self.unchecked_supplement,
            "correct_medication": self.correct_medication,
            "correct_supplement": self.correct_supplement,
            "medicine_recollect_required": self.medicine_recollect_required,
            "supplement_recollect_required": self.supplement_recollect_required,
            "can_be_deliverable": self.can_be_deliverable,
            "delivered": self.delivered,
            "battery_soc": self.battery_soc,
            "approach_side": self.approach_side,
            "time_elapsed": self.time_elapsed,
            "step_count": self.step_count,
            "actions_taken": self.actions_taken,
            "distance_traveled": self.distance_traveled,
            "num_replans": self.num_replans,
            "location_memberships": self.location_memberships,
            "location_stock": self.location_stock,
        }

    def __hash__(self) -> int:
        """Hash for use in search algorithms."""
        return hash(
            (
                self.location,
                self.has_medication,
                self.has_supplement,
                self.carrying_medicine,
                self.carrying_supplement,
                self.unchecked_medication,
                self.unchecked_supplement,
                self.correct_medication,
                self.correct_supplement,
                self.medicine_recollect_required,
                self.supplement_recollect_required,
                self.can_be_deliverable,
                self.delivered,
                round(self.battery_soc, 2),
                self.approach_side,
            )
        )

    def __eq__(self, other) -> bool:
        """Equality for state comparison."""
        if not isinstance(other, MedicationDeliveryState):
            return False
        return (
            self.location == other.location
            and self.has_medication == other.has_medication
            and self.has_supplement == other.has_supplement
            and self.carrying_medicine == other.carrying_medicine
            and self.carrying_supplement == other.carrying_supplement
            and self.unchecked_medication == other.unchecked_medication
            and self.unchecked_supplement == other.unchecked_supplement
            and self.correct_medication == other.correct_medication
            and self.correct_supplement == other.correct_supplement
            and self.medicine_recollect_required == other.medicine_recollect_required
            and self.supplement_recollect_required == other.supplement_recollect_required
            and self.can_be_deliverable == other.can_be_deliverable
            and self.delivered == other.delivered
            and abs(self.battery_soc - other.battery_soc) < 0.01
            and self.approach_side == other.approach_side
        )

    def __repr__(self) -> str:
        """String representation."""
        flags = []
        if self.has_medication:
            flags.append(f"med={self.carrying_medicine}")
        if self.has_supplement:
            flags.append(f"supp={self.carrying_supplement}")
        if self.correct_medication:
            flags.append("med_ok")
        if self.correct_supplement:
            flags.append("supp_ok")
        if self.can_be_deliverable:
            flags.append("deliverable")
        if self.delivered:
            flags.append("delivered")
        return (
            f"MedicationState(loc={self.location}, "
            f"battery={self.battery_soc:.1%}, {', '.join(flags)})"
        )


# Backwards-compatible name used throughout the integration/planner code.
TaskState = MedicationDeliveryState
