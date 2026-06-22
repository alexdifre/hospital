"""
Meal Preparation Task — Feature generation and meal quality profiles.

Maps execution metrics into the same 5 preference dimensions used by
medication delivery, so both task types feed the same preference learner.

Different meal types produce structurally different feature vectors,
giving the preference learner diverse training signal.
"""

import numpy as np
from typing import Dict, Optional

from .task_actions import MEAL_SANDWICH, MEAL_SOUP, MEAL_FULL


# ── Meal quality bonuses ────────────────────────────────────────
# Added to the base features to reflect inherent meal properties.
# These are the "different distribution of enjoyment" from Sebastien's email.
MEAL_QUALITY_BONUS = {
    MEAL_SANDWICH: {
        "time": -0.05,  # quick prep -> slight time bonus (lower feature = better)
        "safety": 0.0,  # cold, safe
        "battery": 0.0,  # neutral
        "proximity": +0.05,  # less fresh concern (cold food)
        "approach": +0.15,  # no plating → worse presentation
    },
    MEAL_SOUP: {
        "time": 0.0,  # neutral
        "safety": 0.1,  # hot liquid → higher safety feature (worse)
        "battery": 0.0,  # neutral
        "proximity": 0.05,  # warm food → freshness matters more
        "approach": 0.0,  # neutral presentation
    },
    MEAL_FULL: {
        "time": +0.05,  # slow prep -> slight time penalty (higher feature = worse)
        "safety": 0.12,  # hot, complex → highest safety concern
        "battery": 0.02,  # extra travel for plating return trip
        "proximity": -0.10,  # hot plated food → freshness very important
        "approach": -0.20,  # beautiful plating → major presentation bonus
    },
}

# Feature keys (same order as medication delivery)
FEATURE_KEYS = ["time", "safety", "battery", "proximity", "approach"]


def compute_meal_features(
    total_time: float,
    total_distance: float,
    battery_start: float,
    battery_end: float,
    delivery_error: float,
    approach_quality: float,
    meal_type: Optional[str] = None,
    max_time: float = 120.0,
    max_distance: float = 60.0,
) -> Dict[str, float]:
    """
    Compute normalized features from meal delivery execution metrics.

    Same 5 dimensions as medication delivery:
      time:      total time / max_time (lower = faster)
      safety:    composite safety score (congestion, handling risk)
      battery:   energy used (battery_start - battery_end)
      proximity: delivery_error / max_error (lower = closer to patient)
      approach:  1 - approach_quality (lower = better approach)

    Meal quality bonuses are added to shift feature distributions
    per meal type.

    Args:
        total_time: Seconds from start to delivery.
        total_distance: Meters traveled.
        battery_start: Battery at episode start.
        battery_end: Battery after delivery.
        delivery_error: Position error at delivery (meters).
        approach_quality: Approach quality score [0, 1].
        meal_type: 'sandwich', 'soup', or 'full_meal'.
        max_time: Normalization constant for time.
        max_distance: Normalization constant for distance.

    Returns:
        Dict with keys: time, safety, battery, proximity, approach
    """
    # Base features (same computation as medication delivery)
    f_time = np.clip(total_time / max_time, 0.0, 1.0)
    f_battery = np.clip(battery_start - battery_end, 0.0, 1.0)
    f_proximity = np.clip(delivery_error / 3.0, 0.0, 1.0)
    # Approach quality naturally lives in a tight band in the simulated episodes.
    # Expanding the badness keeps presentation-focused profiles identifiable.
    f_approach = np.clip(2.0 * (1.0 - approach_quality), 0.0, 1.0)

    # Safety feature: distance-based + handling risk
    dist_component = np.clip(total_distance / max_distance, 0.0, 1.0)
    handling_risk = 0.0
    if meal_type == MEAL_SOUP:
        handling_risk = 0.15  # hot liquid
    elif meal_type == MEAL_FULL:
        handling_risk = 0.20  # hot, complex
    f_safety = np.clip(dist_component * 0.5 + handling_risk, 0.0, 1.0)

    features = {
        "time": float(f_time),
        "safety": float(f_safety),
        "battery": float(f_battery),
        "proximity": float(f_proximity),
        "approach": float(f_approach),
    }

    # Apply meal quality bonuses
    if meal_type is not None and meal_type in MEAL_QUALITY_BONUS:
        bonuses = MEAL_QUALITY_BONUS[meal_type]
        for key in FEATURE_KEYS:
            features[key] = np.clip(features[key] + bonuses.get(key, 0.0), 0.0, 1.0)

    return features
