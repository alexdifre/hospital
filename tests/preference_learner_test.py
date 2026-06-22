import numpy as np

from core.learning.preference_learner import (
    FEATURE_KEYS,
    PATIENT_PROFILES,
    PreferenceLearningEngine,
)


def test_update_weights_preserves_simplex_and_records_gradient():
    learner = PreferenceLearningEngine(
        true_patient_profile=PATIENT_PROFILES["safety_first"],
        initial_weights=np.array([0.4, 0.1, 0.2, 0.2, 0.1]),
        learning_rate=0.05,
        rating_noise=0.0,
        lr_decay=0.0,
    )
    features = {
        "time": 0.8,
        "safety": 0.6,
        "battery": 0.4,
        "proximity": 0.3,
        "approach": 0.2,
    }
    ratings = np.array([4.2, 4.0, 4.7, 4.6, 4.8])

    old_weights = learner.estimated_weights.copy()
    f = np.array([features[k] for k in FEATURE_KEYS], dtype=float)
    expected_r_hat = 5.0 - 4.0 * f * old_weights - learner.bias_vec
    expected_gradient = 2.0 * (expected_r_hat - ratings) * (-4.0) * f

    update = learner.update_weights(ratings, features)

    np.testing.assert_allclose(update["gradient"], expected_gradient)
    assert np.all(update["new_weights"] >= 0.0)
    assert np.isclose(np.sum(update["new_weights"]), 1.0)
    assert len(learner.weight_history) == 2
    assert len(learner.loss_history) == 1
    assert len(learner.gradient_norm_history) == 1
    assert update["gradient_norm"] > 0.0


def test_noiseless_preference_update_moves_toward_true_profile():
    learner = PreferenceLearningEngine(
        true_patient_profile=PATIENT_PROFILES["safety_first"],
        learning_rate=0.02,
        rating_noise=0.0,
        lr_decay=0.0,
    )
    features = dict.fromkeys(FEATURE_KEYS, 1.0)
    ratings = 5.0 - 4.0 * PATIENT_PROFILES["safety_first"].weights

    before = np.linalg.norm(
        learner.estimated_weights - learner.true_profile.weights
    )
    update = learner.update_weights(ratings, features)
    after = np.linalg.norm(update["new_weights"] - learner.true_profile.weights)

    assert after < before
    assert update["new_weights"][1] > learner.weight_history[0][1]

