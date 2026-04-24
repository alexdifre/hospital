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
from typing import Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# MEMBERSHIP FUNCTIONS
# =====================================================================


def gaussian_membership(x: float, center: float, sigma: float) -> float:
    """
    Gaussian membership function.

    μ(x) = exp(-(x - center)² / (2σ²))

    Returns 1.0 at center, decays smoothly to 0.
    """
    return float(np.exp(-((x - center) ** 2) / (2.0 * sigma**2)))


def gaussian_2d(pos: np.ndarray, center: np.ndarray, sigma: float) -> float:
    """
    2D Gaussian membership for position → location.

    μ_L(x, y) = exp(-||pos - center||² / (2σ²))

    Args:
        pos: Robot position [x, y]
        center: Location center [x, y]
        sigma: Characteristic radius of the location

    Returns:
        Membership degree in [0, 1]
    """
    d_sq = float(np.sum((pos[:2] - center[:2]) ** 2))
    return float(np.exp(-d_sq / (2.0 * sigma**2)))


def trapezoidal_membership(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership function.
    
         b_____c
        /       \\
       /         \\
      a           d
    
    Returns:
        0.0 outside [a, d]
        1.0 inside [b, c]
        Linear ramp between [a, b] and [c, d]
    """
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    elif c < x < d:
        return (d - x) / (d - c)
    return 0.0


def left_shoulder(x: float, b: float, c: float) -> float:
    """
    Left shoulder membership (1.0 for x <= b, ramps down to 0 at c).
    
    1.0 ____b
              \\
               \\
                c  0.0
    """
    if x <= b:
        return 1.0
    elif x >= c:
        return 0.0
    else:
        return (c - x) / (c - b)


def _sigmoid(x: float) -> float:
    """Sigmoid function — 1 / (1 + exp(-x)).  Used for smooth battery memberships."""
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0))))


def battery_defuzzify(battery_soc: float) -> float:
    """
    Defuzzified battery cost feature from SoC using sigmoid memberships.

    Uses the same sigmoid parameters as the fuzzy state estimator (BATTERY_SIGMOID).
    Maps SoC → [0, 1] where 0 = full battery (no cost) and 1 = empty (high cost).

      μ_low  = sigmoid(-10 * (SoC - 0.3))   # high when SoC is low
      μ_high = sigmoid(+10 * (SoC - 0.7))   # high when SoC is high
      μ_med  = max(0, 1 - μ_low - μ_high)

    Weighted defuzzification:
      feature = μ_low * 1.0 + μ_med * 0.5 + μ_high * 0.0
    """
    k = BATTERY_SIGMOID["steepness"]
    mu_low  = _sigmoid(-k * (battery_soc - BATTERY_SIGMOID["low_center"]))
    mu_high = _sigmoid( k * (battery_soc - BATTERY_SIGMOID["high_center"]))
    mu_med  = max(0.0, 1.0 - mu_low - mu_high)
    return float(mu_low * 1.0 + mu_med * 0.5 + mu_high * 0.0)


def right_shoulder(x: float, a: float, b: float) -> float:
    """
    Right shoulder membership (0.0 for x <= a, ramps up to 1.0 at b).

              b____ 1.0
             /
            /
    0.0  a
    """
    if x <= a:
        return 0.0
    elif x >= b:
        return 1.0
    else:
        return (x - a) / (b - a)


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
        dominant_location:    argmax of location_memberships
        dominant_membership:  max membership value
        position:             raw (x, y) that produced these memberships
        battery_soc:          raw battery that produced these memberships
    """

    location_memberships: Dict[str, float] = field(default_factory=dict)
    battery_memberships: Dict[str, float] = field(default_factory=dict)
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

        Replaces:
            if location == 'nurse_station': penalty = 0.3
            elif location == 'equipment_storage': penalty = 0.2

        With continuous proximity-based penalty.
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
            "battery_memberships": dict(self.battery_memberships),
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
        batt_str = "/".join(
            f"{k[0].upper()}={v:.2f}" for k, v in self.battery_memberships.items()
        )
        return f"[Fuzzy] loc=[{loc_str}] batt=[{batt_str}] risk={self.dominant_risk()}"

    def dominant_risk(self) -> str:
        """Return the dominant risk level."""
        if not self.risk_memberships:
            return "safe"
        return max(self.risk_memberships, key=self.risk_memberships.get)


# =====================================================================
# FUZZY STATE ESTIMATOR
# =====================================================================


# Default location characteristic radii (σ in meters)
# Smaller σ = tighter zone (must be closer to count as "at" that location)
# Larger σ = broader zone (e.g., corridors, open areas)
DEFAULT_LOCATION_SIGMAS = {
    "home": 1.5,
    "pharmacy_north": 1.5,
    "pharmacy_south": 1.5,
    "supply_A": 1.5,
    "supply_B": 1.5,
    "charge_main": 1.2,
    "charge_backup": 1.2,
    "nurse_station": 2.0,  # Larger zone — open area
    "equipment_storage": 1.8,  # Moderate zone
    "patient_bed_left": 1.2,  # Tight — precision matters near patient
    "patient_bed_right": 1.2,
    # Kitchen area
    "pantry": 1.5,  # Open storage area
    "prep_station": 1.2,  # Compact workspace — need precision
    "stove": 1.0,  # Tight zone — hot surface, careful positioning
}

# Risk levels for locations (used for congestion fuzzification)
LOCATION_RISK = {
    "nurse_station": 0.8,  # High traffic
    "equipment_storage": 0.6,  # Moderate congestion
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
    "prep_station": 0.3,  # Active workspace — moderate risk
    "stove": 0.7,  # Heat hazard — high risk
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
        membership_threshold: float = 0.01,
    ):
        """
        Args:
            environment: MuJoCo environment with .locations dict
            location_sigmas: Override σ per location (default uses DEFAULT_LOCATION_SIGMAS)
            location_risk: Override risk per location (default uses LOCATION_RISK)
            membership_threshold: Below this, membership is clamped to 0 (noise floor)
        """
        self.env = environment
        self.locations: Dict[str, np.ndarray] = {
            name: np.array(pos, dtype=float)
            for name, pos in environment.locations.items()
        }
        self.sigmas = location_sigmas or dict(DEFAULT_LOCATION_SIGMAS)
        self.risk_map = location_risk or dict(LOCATION_RISK)
        self.threshold = membership_threshold

        # Precompute risk zone positions (only locations with risk > 0.3)
        self.risk_zones: List[Tuple[np.ndarray, float, float]] = []
        for name, pos in self.locations.items():
            risk = self.risk_map.get(name, 0.0)
            if risk > 0.3:
                sigma = self.sigmas.get(name, 1.5)
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
            fm.dominant_location = dom_loc
            fm.dominant_membership = fm.location_memberships[dom_loc]
        else:
            fm.dominant_location = "in_transit"
            fm.dominant_membership = 0.0

        # --- Layer 2: Battery → {low, medium, high} ---
        fm.battery_memberships = self._compute_battery_memberships(battery_soc)

        # --- Layer 3: Risk/congestion from position ---
        fm.risk_memberships = self._compute_risk_memberships(pos)

        return fm

    def _compute_location_memberships(self, pos: np.ndarray) -> Dict[str, float]:
        """Gaussian membership for each location."""
        memberships = {}
        for name, center in self.locations.items():
            sigma = self.sigmas.get(name, 1.5)
            mu = gaussian_2d(pos, center, sigma)
            if mu >= self.threshold:
                memberships[name] = mu
        return memberships

    def _compute_battery_memberships(self, battery_soc: float) -> Dict[str, float]:
        """Sigmoid-based smooth membership functions for battery level.

        μ_Low(SoC)  = sigmoid(-k * (SoC - 0.3))
        μ_High(SoC) = sigmoid(+k * (SoC - 0.7))
        μ_Med(SoC)  = 1 - μ_Low - μ_High         (clamped to [0, 1])

        k=10 gives a transition width of ~0.4 SoC units — comparable to the
        old trapezoidal overlap but continuously differentiable everywhere.
        """
        k = BATTERY_SIGMOID["steepness"]
        mu_low  = _sigmoid(-k * (battery_soc - BATTERY_SIGMOID["low_center"]))
        mu_high = _sigmoid( k * (battery_soc - BATTERY_SIGMOID["high_center"]))
        mu_med  = max(0.0, 1.0 - mu_low - mu_high)
        return {"low": mu_low, "medium": mu_med, "high": mu_high}

    def _compute_risk_memberships(self, pos: np.ndarray) -> Dict[str, float]:
        """
        Compute risk memberships based on proximity to high-risk zones.

        Aggregates risk from nearby high-traffic locations using
        fuzzy OR (max) over proximity-weighted risk values.
        """
        # Compute aggregate risk from proximity to risk zones
        max_risk = 0.0
        for zone_pos, zone_risk, zone_sigma in self.risk_zones:
            proximity = gaussian_2d(pos, zone_pos, zone_sigma * 1.5)  # wider influence
            contribution = proximity * zone_risk
            max_risk = max(max_risk, contribution)

        # Map aggregate risk to fuzzy {safe, moderate, hazardous}
        memberships = {
            "safe": left_shoulder(max_risk, 0.15, 0.35),
            "moderate": trapezoidal_membership(max_risk, 0.15, 0.30, 0.50, 0.65),
            "hazardous": right_shoulder(max_risk, 0.45, 0.65),
        }
        return memberships

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
        print(f"    Battery memberships:")
        for name in ["low", "medium", "high"]:
            mu = fm.battery_memberships.get(name, 0.0)
            bar = "█" * int(mu * 20)
            print(f"      {name:<10} μ={mu:.3f} {bar}")

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
                "nurse_station": np.array([12.0, 0.0]),
                "equipment_storage": np.array([22.0, 6.0]),
                "patient_bed_left": np.array([20.5, 12.0]),
                "patient_bed_right": np.array([23.5, 10.0]),
            }

    env = MockEnv()
    estimator = FuzzyStateEstimator(env)

    # Test cases: (position, battery, description)
    test_cases = [
        (np.array([5.0, 18.0]), 0.90, "Exactly at pharmacy_north, high battery"),
        (np.array([5.8, 17.2]), 0.90, "Near pharmacy_north (0.8m away)"),
        (np.array([7.0, 16.0]), 0.90, "Drifting from pharmacy_north (~2.8m)"),
        (np.array([10.0, 10.0]), 0.50, "Open space (in transit), medium battery"),
        (np.array([12.5, 0.5]), 0.30, "Near nurse_station, low-ish battery"),
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
    print(
        f"{'Battery':>8} {'μ_low':>6} {'μ_med':>6} {'μ_high':>6} {'Fuzzy':>8} {'Crisp':>8}"
    )
    print(f"{'─' * 50}")

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
            f"{fm.battery_memberships['low']:>5.3f} "
            f"{fm.battery_memberships['medium']:>5.3f} "
            f"{fm.battery_memberships['high']:>5.3f}  "
            f"{fuzzy_penalty:>7.3f}  "
            f"{crisp_penalty:>7.3f}"
        )

    print(f"\n{'=' * 60}")
    print("Fuzzy state estimator test complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_fuzzy_state()
