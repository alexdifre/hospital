"""
Shared task state utilities.

TaskStateMixin provides battery helpers and shared-field utilities for
both TaskState (medication delivery) and MealTaskState (meal preparation).
Both are dataclasses — this mixin contributes only methods, not fields.

Shared fields expected on the subclass:
    battery_soc: float
    location_memberships: Optional[Dict[str, float]]
    location_stock: Optional[Dict[str, int]]
    step_count: int
    time_elapsed: float
    distance_traveled: float
    num_replans: int
"""

from __future__ import annotations

from typing import Dict


class TaskStateMixin:
    """
    Mixin providing shared utilities for discrete task state dataclasses.

    Subclass alongside a @dataclass that defines the actual fields.
    """

    # ── Battery helpers ──────────────────────────────────────────

    def get_discrete_battery_level(self) -> int:
        """Discretize battery to 8 levels (0–7) for hashing."""
        return int(self.battery_soc * 8)

    def needs_recharge(self, threshold: float = 0.2) -> bool:
        """Check if battery is critically low."""
        return self.battery_soc < threshold

    # ── Shared-field utilities ───────────────────────────────────

    def _shared_copy_kwargs(self) -> Dict:
        """
        Return shared field values as kwargs for use in copy().

        Usage in subclass copy():
            return MyState(
                my_specific_field=self.my_specific_field,
                **self._shared_copy_kwargs(),
            )
        """
        return {
            "battery_soc": self.battery_soc,
            "approach_side": self.approach_side,
            "location_memberships": (
                dict(self.location_memberships) if self.location_memberships else None
            ),
            "location_stock": (
                dict(self.location_stock) if self.location_stock else None
            ),
            "step_count": self.step_count,
            "time_elapsed": self.time_elapsed,
            "distance_traveled": self.distance_traveled,
            "num_replans": self.num_replans,
        }

    def _shared_to_dict(self) -> Dict:
        """
        Return shared fields as a dict for use in to_dict().

        Usage in subclass to_dict():
            return {"my_specific_field": ..., **self._shared_to_dict()}
        """
        d = {
            "battery_soc": self.battery_soc,
            "approach_side": self.approach_side,
            "step_count": self.step_count,
            "time_elapsed": self.time_elapsed,
            "distance_traveled": self.distance_traveled,
            "num_replans": self.num_replans,
        }
        if self.location_memberships:
            d["location_memberships"] = dict(self.location_memberships)
        if self.location_stock:
            d["location_stock"] = dict(self.location_stock)
        return d
