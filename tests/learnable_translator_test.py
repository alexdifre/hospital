import numpy as np

from core.execution.formulation import SharedMPCFormulation
from core.learning.learnable_translator import LearnableTranslator


class MockEnvironment:
    locations = {
        "home": np.array([0.0, 0.0]),
        "patient_bed": np.array([10.0, 4.0]),
    }
    location_metadata = {}


def test_external_weights_do_not_affect_mpc_params_or_gradients():
    translator = LearnableTranslator(MockEnvironment(), learning_rate=0.01)

    translator.update_preference_weights(np.array([0.8, 0.05, 0.05, 0.05, 0.05]))
    q_time, r_time, q_term_time = translator.get_mpc_params(near_patient=True)
    grads_time = translator.compute_parameter_gradients(near_patient=True)

    translator.update_preference_weights(np.array([0.05, 0.8, 0.05, 0.05, 0.05]))
    q_safety, r_safety, q_term_safety = translator.get_mpc_params(near_patient=True)
    grads_safety = translator.compute_parameter_gradients(near_patient=True)

    assert np.allclose(q_time, q_safety)
    assert np.allclose(r_time, r_safety)
    assert np.allclose(q_term_time, q_term_safety)
    assert np.allclose(grads_time.dQ_dphi, grads_safety.dQ_dphi)


def test_terminal_cost_defaults_use_shared_terminal_multiplier():
    translator = LearnableTranslator(MockEnvironment(), learning_rate=0.01)

    _, _, q_terminal = translator.get_mpc_params(near_patient=True)

    np.testing.assert_allclose(
        q_terminal,
        SharedMPCFormulation.Q_default
        * SharedMPCFormulation.TERMINAL_COST_MULTIPLIER,
    )


def test_update_parameters_applies_chain_rule_step():
    translator = LearnableTranslator(
        MockEnvironment(), learning_rate=0.001, max_grad_norm=1e9
    )
    translator.update_preference_weights(np.array([0.3, 0.25, 0.2, 0.15, 0.1]))
    old_params = translator.params.to_vector().copy()
    grads = translator.compute_parameter_gradients(near_patient=True)

    dJ_dQ = np.array([0.2, 0.1, 0.05, 0.03, 0.02, 0.01])
    dJ_dR = np.array([0.04, 0.03, 0.02])
    expected_gradient = (
        grads.dQ_dphi.T @ dJ_dQ
        + grads.dR_dphi.T @ dJ_dR
    )
    expected_params = translator._apply_param_bounds(
        old_params - translator.learning_rate * expected_gradient
    )

    update = translator.update_parameters(
        dJ_dQ=dJ_dQ,
        dJ_dR=dJ_dR,
        near_patient=True,
        cost=12.0,
    )

    np.testing.assert_allclose(update["gradient"], expected_gradient)
    np.testing.assert_allclose(translator.params.to_vector(), expected_params)
    assert update["param_change"] > 0.0
    assert translator.update_count == 1
    assert len(translator.param_history) == 2


def test_stage_cost_update_changes_q_and_r_without_touching_terminal_cost():
    translator = LearnableTranslator(
        MockEnvironment(), learning_rate=0.002, max_grad_norm=1e9
    )
    q_before, r_before, q_term_before = translator.get_mpc_params(near_patient=True)
    old_params = translator.params.to_vector().copy()
    grads = translator.compute_parameter_gradients(near_patient=True)

    dJ_dQ = np.array([0.3, 0.2, 0.1, 0.05, 0.04, 0.03])
    dJ_dR = np.array([0.2, 0.1, 0.05])
    expected_stage_gradient = (
        grads.dQ_dphi.T @ dJ_dQ
        + grads.dR_dphi.T @ dJ_dR
    )
    expected_params = translator._apply_param_bounds(
        old_params - translator.learning_rate * expected_stage_gradient
    )

    update = translator.update_parameters(
        dJ_dQ=dJ_dQ,
        dJ_dR=dJ_dR,
        near_patient=True,
        E=None,
    )
    q_after, r_after, q_term_after = translator.get_mpc_params(near_patient=True)

    np.testing.assert_allclose(update["gradient"], expected_stage_gradient)
    np.testing.assert_allclose(update["new_params"], expected_params)
    assert update["param_change"] > 0.0
    assert update["terminal_change"] == 0.0
    assert not np.allclose(q_before, q_after)
    assert not np.allclose(r_before, r_after)
    np.testing.assert_allclose(q_term_after, q_term_before)


def test_terminal_cost_update_uses_membership_error_vector():
    translator = LearnableTranslator(
        MockEnvironment(), learning_rate=0.001, max_grad_norm=1e9
    )
    translator.terminal_learning_rate = 0.25
    old_params = translator.params.to_vector().copy()
    old_terminal = translator.terminal_cost_diag.copy()

    E = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    aligned_E = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.0])
    expected_terminal = old_terminal - translator.terminal_learning_rate * aligned_E

    update = translator.update_parameters(
        dJ_dQ=np.zeros(6),
        dJ_dR=np.zeros(3),
        near_patient=True,
        E=E,
    )

    np.testing.assert_allclose(update["terminal_error"], aligned_E)
    np.testing.assert_allclose(update["old_terminal_cost"], old_terminal)
    np.testing.assert_allclose(update["new_terminal_cost"], expected_terminal)
    np.testing.assert_allclose(translator.terminal_cost_diag, expected_terminal)
    np.testing.assert_allclose(translator.params.to_vector(), old_params)
    assert update["terminal_change"] > 0.0
    assert update["param_change"] == 0.0
