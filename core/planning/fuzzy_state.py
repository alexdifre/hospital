#!/usr/bin/env python3
"""
Fuzzy State Estimation for Task Planning
=========================================

Bridges the continuous MuJoCo state space and the discrete task planner
via fuzzy set memberships.

Three layers of fuzzification:

1. **Position → Location memberships**
   Gaussian membership: μ_L(x,y) = exp(-d² / 2σ_L²)
   Each location has a characteristic radius σ_L.
   Robot can have partial membership in multiple locations simultaneously.

2. **Battery → {low, medium, high}**
   Sigmoid-based smooth membership functions — continuously differentiable
   everywhere, enabling gradient flow through the battery cost terms.
   Replaces crisp thresholds (battery < 0.15 → critical) with smooth costs.

3. **Risk/Congestion → {safe, moderate, hazardous}**
   Fuzzy risk based on proximity to high-traffic areas.
   Congestion penalty scales continuously rather than if/else.

Usage:
    estimator = FuzzyStateEstimator(environment)
    memberships = estimator.estimate(robot_position, battery_soc)

    # memberships.location_memberships  → {'pharmacy_north': 0.85, 'supply_A': 0.02, ...}
    # memberships.battery_memberships   → {'low': 0.0, 'medium': 0.3, 'high': 0.7}
    # memberships.risk_level            → {'safe': 0.6, 'moderate': 0.4, 'hazardous': 0.0}
    # memberships.dominant_location     → 'pharmacy_north'
    # memberships.is_at(location, threshold=0.8) → True/False
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# =====================================================================
# FUZZY MEMBERSHIP RESULT
# =====================================================================


@dataclass
class FuzzyMemberships:
    """
    Complete fuzzy state estimate at a given instant.

    Attributes:
        location_memberships: μ per location, NOT normalized (robot can be
                              partially at multiple places or nowhere strongly)
        battery_memberships:  μ for {low, medium, high}
        risk_memberships:     μ for {safe, moderate, hazardous} based on
                              position proximity to high-risk zones
        dominant_location:    argmax of location_memberships, or "in_transit"
                              when the best membership is below threshold
        dominant_membership:  max membership value
        position:             raw (x, y) that produced these memberships
        battery_soc:          raw battery that produced these memberships
    """

    location_memberships: Dict[str, float] = field(default_factory=dict)
    battery_memberships: Union[Dict[str, float], float] = field(default_factory=dict)
    risk_memberships: Dict[str, float] = field(default_factory=dict)

    dominant_location: str = "unknown"
    dominant_membership: float = 0.0

    position: Optional[np.ndarray] = None
    battery_soc: float = 1.0

    def is_at(self, location: str, threshold: float = 0.8) -> bool:
        """Check if robot has sufficient membership at a location."""
        return self.location_memberships.get(location, 0.0) >= threshold

    def membership_at(self, location: str) -> float:
        """Get membership degree at a specific location."""
        return self.location_memberships.get(location, 0.0)

    def battery_penalty(
        self,
        penalty_low: float = 0.5,
        penalty_med: float = 0.1,
        penalty_high: float = 0.0,
    ) -> float:
        """
        Compute smooth battery risk penalty via fuzzy weighted sum.

        Replaces:
            if battery < 0.15: penalty = 0.5
            elif battery < 0.25: penalty = 0.2

        With:
            penalty = μ_low * penalty_low + μ_med * penalty_med + μ_high * penalty_high
        """
        if isinstance(self.battery_memberships, (int, float)):
            return 1.0 - float(self.battery_memberships)
        return (
            self.battery_memberships.get("low", 0.0) * penalty_low
            + self.battery_memberships.get("medium", 0.0) * penalty_med
            + self.battery_memberships.get("high", 0.0) * penalty_high
        )

    def congestion_penalty(
        self,
        penalty_safe: float = 0.0,
        penalty_mod: float = 0.15,
        penalty_haz: float = 0.3,
    ) -> float:
        """
        Compute smooth congestion/risk penalty via fuzzy weighted sum.

        Uses continuous proximity-based penalty.
        """
        return (
            self.risk_memberships.get("safe", 0.0) * penalty_safe
            + self.risk_memberships.get("moderate", 0.0) * penalty_mod
            + self.risk_memberships.get("hazardous", 0.0) * penalty_haz
        )

    def to_dict(self) -> Dict:
        """Serialize for logging/JSON."""
        return {
            "location_memberships": dict(self.location_memberships),
            "battery_memberships": (
                dict(self.battery_memberships)
                if isinstance(self.battery_memberships, dict)
                else self.battery_memberships
            ),
            "risk_memberships": dict(self.risk_memberships),
            "dominant_location": self.dominant_location,
            "dominant_membership": self.dominant_membership,
            "battery_soc": self.battery_soc,
        }

    def summary(self) -> str:
        """One-line summary for logging."""
        top3 = sorted(
            self.location_memberships.items(), key=lambda kv: kv[1], reverse=True
        )[:3]
        loc_str = ", ".join(f"{k}={v:.2f}" for k, v in top3 if v > 0.01)
        if isinstance(self.battery_memberships, dict):
            batt_str = "/".join(
                f"{k[0].upper()}={v:.2f}" for k, v in self.battery_memberships.items()
            )
        else:
            batt_str = f"SoC={self.battery_memberships:.2f}"
        return f"[Fuzzy] loc=[{loc_str}] batt=[{batt_str}] risk={self.dominant_risk()}"

    def dominant_risk(self) -> str:
        """Return the dominant risk level."""
        if not self.risk_memberships:
            return "safe"
        return max(self.risk_memberships, key=self.risk_memberships.get)


# =====================================================================
# FUZZY STATE ESTIMATOR
# =====================================================================


# Default fallback location characteristic radius (σ in meters).
# The Gaussian variance is σ². Known locations use per-location σ below:
# tighter where nearby locations overlap, wider where the location is isolated.
DEFAULT_LOCATION_SIGMA = 4.5
DEFAULT_LOCATION_SIGMAS = {
    # Isolated / broad areas
    "home": 4.5,
    "pharmacy_south": 5.0,
    "supply_A": 4.6,
    "supply_B": 5.0,
    "charge_main": 4.6,
    "charge_backup": 5.0,
    # North medication / quality cluster
    "pharmacy_north": 3.3,
    "quality_check": 3.4,
    # Patient bed approaches are close to each other
    "patient_bed_left": 3.4,
    "patient_bed_right": 3.4,
    # Kitchen cluster: pantry/fridge/prep/stove are close together
    "pantry": 3.7,
    "fridge": 3.5,
    "prep_station": 3.4,
    "stove": 3.2,
}

# Risk levels for locations (used for congestion fuzzification)
LOCATION_RISK = {
    "pharmacy_north": 0.3,
    "pharmacy_south": 0.3,
    "supply_A": 0.2,
    "supply_B": 0.2,
    "patient_bed_left": 0.15,
    "patient_bed_right": 0.15,
    "charge_main": 0.1,
    "charge_backup": 0.1,
    "home": 0.05,
    # Kitchen area
    "pantry": 0.15,  # Low traffic storage
    "fridge": 0.17,  # Low traffic refrigerated storage
    "prep_station": 0.3,  # Active workspace — moderate risk
    "stove": 0.7,  # Heat hazard — high risk
    "quality_check": 0.1,  # Low traffic check station
}

# Battery sigmoid parameters
# μ_Low(SoC)  = sigmoid(-10 * (SoC - 0.3))   → 1 when SoC ≪ 0.3, 0 when SoC ≫ 0.3
# μ_High(SoC) = sigmoid(+10 * (SoC - 0.7))   → 0 when SoC ≪ 0.7, 1 when SoC ≫ 0.7
# μ_Med(SoC)  = 1 - μ_Low - μ_High           → peaks around SoC = 0.5
# Steepness=10 preserves the qualitative Low/Medium/High structure while
# being continuously differentiable everywhere (unlike trapezoidal).
BATTERY_SIGMOID = {
    "low_center":  0.3,
    "high_center": 0.7,
    "steepness":   10.0,
}

# Action precondition thresholds (how "at" a location must you be)
ACTION_MEMBERSHIP_THRESHOLDS = {
    "collect_medication": 0.7,  # Must be well within pharmacy zone
    "collect_supplement": 0.7,  # Must be well within supply zone
    "recharge": 0.7,  # Must be at charging station
    "deliver": 0.8,  # Must be precisely at patient bed
    # Meal preparation thresholds
    "collect_ingredients": 0.7,  # Must be within pantry zone
    "assemble": 0.7,  # Must be at prep station
    "chop": 0.7,  # Must be at prep station
    "cook": 0.8,  # Must be precisely at stove (heat hazard)
    "plate": 0.7,  # Must be at prep station
    "deliver_meal": 0.8,  # Must be precisely at patient bed
}

LOCATION_DEFUZZIFICATION_THRESHOLD = 0.7
LOCATION_MEMBERSHIP_CORE_RADIUS = 1.0
LOCATION_MEMBERSHIP_CUTOFF_DISTANCE = 10.0


class FuzzyStateEstimator:
    """
    Estimates fuzzy state memberships from continuous robot state.

    Bridges continuous MuJoCo space → discrete task planner via soft memberships.

    Usage:
        estimator = FuzzyStateEstimator(environment)
        fm = estimator.estimate(position, battery_soc)

        # Use in planner:
        if fm.is_at('pharmacy_north', threshold=0.7):
            allow_collect_medication()

        safety_cost = fm.battery_penalty() + fm.congestion_penalty()
    """

    def __init__(
        self,
        environment,
        location_sigmas: Optional[Dict[str, float]] = None,
        location_risk: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            environment: MuJoCo environment with .locations dict
            location_sigmas: Optional σ override per location.
            location_risk: Override risk per location (default uses LOCATION_RISK)
        """
        self.env = environment
        self.locations: Dict[str, np.ndarray] = {
            name: np.array(pos, dtype=float)
            for name, pos in environment.locations.items()
        }
        self.sigmas = dict(DEFAULT_LOCATION_SIGMAS)
        if location_sigmas:
            self.sigmas.update(location_sigmas)
        self.risk_map = location_risk or dict(LOCATION_RISK)

        # Precompute risk zone positions (only locations with risk > 0.3)
        self.risk_zones: List[Tuple[np.ndarray, float, float]] = []
        for name, pos in self.locations.items():
            risk = self.risk_map.get(name, 0.0)
            if risk > 0.3:
                sigma = self.sigmas.get(name, DEFAULT_LOCATION_SIGMA)
                self.risk_zones.append((np.array(pos, dtype=float), risk, sigma))

    def estimate(self, position: np.ndarray, battery_soc: float) -> FuzzyMemberships:
        """
        Compute full fuzzy state estimate.

        Args:
            position: Robot [x, y] (or 6D state, first 2 elements used)
            battery_soc: Battery state of charge [0.0, 1.0]

        Returns:
            FuzzyMemberships with all three layers populated
        """
        pos = np.array(position[:2], dtype=float)
        fm = FuzzyMemberships(position=pos.copy(), battery_soc=battery_soc)

        # --- Layer 1: Position → location memberships ---
        fm.location_memberships = self._compute_location_memberships(pos)

        # Find dominant
        if fm.location_memberships:
            dom_loc = max(fm.location_memberships, key=fm.location_memberships.get)
            dom_membership = float(fm.location_memberships[dom_loc])
            fm.dominant_membership = dom_membership
            if dom_membership < LOCATION_DEFUZZIFICATION_THRESHOLD:
                fm.dominant_location = "in_transit"
            else:
                fm.dominant_location = dom_loc
        else:
            fm.dominant_location = "in_transit"
            fm.dominant_membership = 0.0

        # --- Layer 2: Battery ---
        fm.battery_memberships = self._compute_battery_memberships(battery_soc)

        return fm

    def _compute_location_memberships(self, pos: np.ndarray) -> Dict[str, float]:
        """Gaussian membership for each location."""
        memberships = {}
        for name, center in self.locations.items():
            sigma = self.sigmas.get(name, DEFAULT_LOCATION_SIGMA)
            distance = float(np.linalg.norm(pos[:2] - center[:2]))
            if distance > LOCATION_MEMBERSHIP_CUTOFF_DISTANCE:
                mu = 0.0
            elif distance < LOCATION_MEMBERSHIP_CORE_RADIUS:
                mu = 1.0
            else:
                mu = float(np.exp(-(distance**2) / (2.0 * sigma**2)))
            memberships[name] = mu
        return memberships

    def _compute_battery_memberships(self, battery_soc: float) -> float:
        """Return raw battery state of charge."""
        return float(battery_soc)

    def get_action_threshold(self, action_type: str) -> float:
        """Get the membership threshold for a specific action type."""
        return ACTION_MEMBERSHIP_THRESHOLDS.get(action_type, 0.7)

    def can_perform_action(
        self, action_type: str, location: str, memberships: FuzzyMemberships
    ) -> bool:
        """
        Check if membership is sufficient for an action.

        Replaces crisp: location == 'pharmacy_north'
        With fuzzy:     μ_pharmacy_north >= threshold
        """
        threshold = self.get_action_threshold(action_type)
        return memberships.membership_at(location) >= threshold

    def print_estimate(self, fm: FuzzyMemberships) -> None:
        """Pretty-print fuzzy state estimate."""
        print(f"\n  [Fuzzy State Estimate]")
        print(f"    Position: ({fm.position[0]:.1f}, {fm.position[1]:.1f})")
        print(f"    Battery: {fm.battery_soc:.1%}")

        # Location memberships (sorted, top 5)
        sorted_locs = sorted(
            fm.location_memberships.items(), key=lambda kv: kv[1], reverse=True
        )
        print(f"    Location memberships:")
        for name, mu in sorted_locs[:5]:
            bar = "█" * int(mu * 20)
            marker = " ← dominant" if name == fm.dominant_location else ""
            print(f"      {name:<22} μ={mu:.3f} {bar}{marker}")
        if len(sorted_locs) > 5:
            print(
                f"      ... and {len(sorted_locs) - 5} more (μ < {sorted_locs[4][1]:.3f})"
            )

        # Battery
        print(f"    Battery:")
        if isinstance(fm.battery_memberships, dict):
            for name in ["low", "medium", "high"]:
                mu = fm.battery_memberships.get(name, 0.0)
                bar = "█" * int(mu * 20)
                print(f"      {name:<10} μ={mu:.3f} {bar}")
        else:
            print(f"      state_of_charge={fm.battery_memberships:.3f}")

        # Risk
        print(f"    Risk memberships:")
        for name in ["safe", "moderate", "hazardous"]:
            mu = fm.risk_memberships.get(name, 0.0)
            bar = "█" * int(mu * 20)
            print(f"      {name:<12} μ={mu:.3f} {bar}")

        # Derived penalties
        print(f"    Derived penalties:")
        print(f"      Battery penalty: {fm.battery_penalty():.3f}")
        print(f"      Congestion penalty: {fm.congestion_penalty():.3f}")


# =====================================================================
# TEST
# =====================================================================


def test_fuzzy_state():
    """Test fuzzy state estimation with various positions and battery levels."""
    print("=" * 80)
    print("FUZZY STATE ESTIMATOR TEST")
    print("=" * 80)

    # Mock environment
    class MockEnv:
        def __init__(self):
            self.locations = {
                "home": np.array([0.0, 0.0]),
                "pharmacy_north": np.array([5.0, 18.0]),
                "pharmacy_south": np.array([6.0, -15.0]),
                "supply_A": np.array([14.0, 10.0]),
                "supply_B": np.array([15.0, -12.0]),
                "charge_main": np.array([3.0, 5.0]),
                "charge_backup": np.array([17.0, -18.0]),
                "patient_bed_left": np.array([20.5, 12.0]),
                "patient_bed_right": np.array([23.5, 10.0]),
                "pantry": np.array([-3.0, 15.0]),
                "fridge": np.array([-6.0, 17.5]),
                "prep_station": np.array([0.0, 20.0]),
                "stove": np.array([-3.0, 20.5]),
                "quality_check": np.array([3.0, 21.0]),
            }

    env = MockEnv()
    estimator = FuzzyStateEstimator(env)

    # Test cases: (position, battery, description)
    test_cases = [
        (np.array([5.0, 18.0]), 0.90, "Exactly at pharmacy_north, high battery"),
        (np.array([5.8, 17.2]), 0.90, "Near pharmacy_north (0.8m away)"),
        (np.array([7.0, 16.0]), 0.90, "Drifting from pharmacy_north (~2.8m)"),
        (np.array([10.0, 10.0]), 0.50, "Open space (in transit), medium battery"),
        (np.array([20.5, 12.0]), 0.10, "At patient_bed_left, critical battery"),
        (np.array([3.0, 5.0]), 0.20, "At charge_main, low battery"),
        (np.array([13.5, 10.0]), 0.60, "Near supply_A, medium battery"),
    ]

    for pos, batt, desc in test_cases:
        print(f"\n{'─' * 60}")
        print(f"  TEST: {desc}")
        print(f"{'─' * 60}")
        fm = estimator.estimate(pos, batt)
        estimator.print_estimate(fm)

        # Test action preconditions
        if "pharmacy" in desc.lower():
            can_collect = estimator.can_perform_action(
                "collect_medication", "pharmacy_north", fm
            )
            print(f"    Can collect medication? {can_collect}")

        if "patient" in desc.lower():
            can_deliver = estimator.can_perform_action(
                "deliver", "patient_bed_left", fm
            )
            print(f"    Can deliver? {can_deliver}")

        if "charge" in desc.lower():
            can_recharge = estimator.can_perform_action("recharge", "charge_main", fm)
            print(f"    Can recharge? {can_recharge}")

    # Test: Show how battery penalty varies smoothly
    print(f"\n{'=' * 60}")
    print("BATTERY PENALTY CURVE (smooth vs crisp)")
    print(f"{'=' * 60}")
    print(f"{'Battery':>8} {'SoC':>6} {'Fuzzy':>8} {'Crisp':>8}")
    print(f"{'─' * 35}")

    for batt_pct in range(0, 105, 5):
        batt = batt_pct / 100.0
        fm = estimator.estimate(np.array([0.0, 0.0]), batt)

        fuzzy_penalty = fm.battery_penalty()

        # Original crisp penalty
        if batt < 0.15:
            crisp_penalty = 0.5
        elif batt < 0.25:
            crisp_penalty = 0.2
        else:
            crisp_penalty = 0.0

        print(
            f"  {batt:>5.0%}   "
            f"{fm.battery_memberships:>5.3f} "
            f"{fuzzy_penalty:>7.3f}  "
            f"{crisp_penalty:>7.3f}"
        )

    print(f"\n{'=' * 60}")
    print("Fuzzy state estimator test complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_fuzzy_state()
