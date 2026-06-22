import types

import numpy as np

from core.planning.fuzzy_state import (
    DEFAULT_LOCATION_SIGMA,
    DEFAULT_LOCATION_SIGMAS,
    FuzzyStateEstimator,
    LOCATION_MEMBERSHIP_CUTOFF_DISTANCE,
    LOCATION_DEFUZZIFICATION_THRESHOLD,
)


def test_location_membership_keeps_gradient_at_offset_scale_distance():
    env = types.SimpleNamespace(
        locations={"patient_bed_left": np.array([20.5, 12.0], dtype=float)}
    )
    estimator = FuzzyStateEstimator(env)

    fm = estimator.estimate(np.array([16.83, 15.49], dtype=float), battery_soc=1.0)
    membership = fm.location_memberships["patient_bed_left"]

    assert 0.0 < membership < LOCATION_DEFUZZIFICATION_THRESHOLD
    assert fm.dominant_location == "in_transit"


def test_default_location_membership_uses_location_sigmas_and_fixed_cutoff():
    env = types.SimpleNamespace(
        locations={
            "home": np.array([0.0, 0.0], dtype=float),
            "stove": np.array([0.0, 0.0], dtype=float),
            "patient_bed_left": np.array([20.0, 0.0], dtype=float),
        }
    )
    estimator = FuzzyStateEstimator(env)

    assert DEFAULT_LOCATION_SIGMA == 4.5
    assert estimator.sigmas["home"] > estimator.sigmas["stove"]
    assert estimator.sigmas["stove"] == DEFAULT_LOCATION_SIGMAS["stove"]
    assert LOCATION_MEMBERSHIP_CUTOFF_DISTANCE == 10.0

    fm_near = estimator.estimate(np.array([5.0, 0.0], dtype=float), battery_soc=1.0)
    expected_home = np.exp(-(5.0**2) / (2.0 * estimator.sigmas["home"] ** 2))
    expected_stove = np.exp(-(5.0**2) / (2.0 * estimator.sigmas["stove"] ** 2))
    np.testing.assert_allclose(fm_near.location_memberships["home"], expected_home)
    np.testing.assert_allclose(fm_near.location_memberships["stove"], expected_stove)
    assert fm_near.location_memberships["home"] > fm_near.location_memberships["stove"]

    fm_far = estimator.estimate(np.array([10.1, 0.0], dtype=float), battery_soc=1.0)
    assert fm_far.location_memberships["home"] == 0.0


def test_location_sigma_overrides_defaults():
    env = types.SimpleNamespace(
        locations={"stove": np.array([0.0, 0.0], dtype=float)}
    )
    estimator = FuzzyStateEstimator(env, location_sigmas={"stove": 4.0})

    fm = estimator.estimate(np.array([5.0, 0.0], dtype=float), battery_soc=1.0)
    expected = np.exp(-(5.0**2) / (2.0 * 4.0**2))
    np.testing.assert_allclose(
        fm.location_memberships["stove"],
        expected,
    )
