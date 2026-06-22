import io
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tasks.meal_preparation.task_actions import MealAction
from tasks.medication_delivery.task_actions import TaskAction


def _pddl_action_names(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"\(:action\s+([^\s]+)", text))


def _action_value(action):
    return action.value if hasattr(action, "value") else str(action)


def _state_payload(state):
    if hasattr(state, "to_dict"):
        return state.to_dict()
    return state


def _mpc_snapshot(system):
    q, r, q_terminal = system.translator.get_mpc_params(near_patient=True)
    return {
        "Q": np.array(q, dtype=float).copy(),
        "R": np.array(r, dtype=float).copy(),
        "Q_terminal": np.array(q_terminal, dtype=float).copy(),
    }


def _events(run, name, episode=None):
    events = [event for event in run.events if event["event"] == name]
    if episode is not None:
        events = [event for event in events if event.get("episode") == episode]
    return events


@pytest.fixture(scope="module")
def real_two_episode_run():
    from integration.system import FullMedicationDeliverySystem

    events = []

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        system = FullMedicationDeliverySystem(
            save_summaries=False,
            verbose=False,
            render=False,
            use_fuzzy=False,
            rating_noise=0.0,
            explore_sigma=0.0,
        )
        def hook(event, **payload):
            row = dict(payload)
            row["event"] = event
            row.setdefault("episode", getattr(system, "episode_count", None))

            if "action" in row:
                row["action"] = _action_value(row["action"])
            if "actions" in row:
                row["actions"] = [_action_value(action) for action in row["actions"]]
            if "state" in row:
                row["state"] = _state_payload(row["state"])
            if "start_state" in row:
                row["start_state"] = _state_payload(row["start_state"])

            if event == "episode_start":
                row["weights_patient"] = (
                    system.preference_learner.get_current_weights().copy()
                )
                row["battery"] = float(
                    system.env.environment_state.get("battery_level", 1.0)
                )
                row["mpc_params"] = _mpc_snapshot(system)

            events.append(row)

        system._episode_property_hook = hook

        first = system.run_episode(start_location="home", task_type="medication")
        after_first = {
            "battery": float(system.env.environment_state["battery_level"]),
            "weights_patient": system.preference_learner.get_current_weights().copy(),
            "mpc_params": _mpc_snapshot(system),
        }
        second = system.run_episode(start_location="home", task_type="medication")

    return SimpleNamespace(
        system=system,
        events=events,
        first=first,
        second=second,
        after_first=after_first,
    )


def test_pddl_actions_are_declared_in_python_action_enums():
    repo = Path(__file__).resolve().parents[1]

    med_pddl_actions = _pddl_action_names(repo / "unified_planning" / "domain_med.pddl")
    meal_pddl_actions = _pddl_action_names(repo / "unified_planning" / "domain_meal.pddl")

    med_enum_actions = {action.value for action in TaskAction}
    meal_enum_actions = {action.value for action in MealAction}

    assert med_pddl_actions <= med_enum_actions
    assert meal_pddl_actions <= meal_enum_actions


def test_real_planner_output_translates_without_dropping_actions(real_two_episode_run):
    plan_events = _events(real_two_episode_run, "plan_ready", episode=1)
    assert plan_events

    for plan_event in plan_events:
        assert plan_event["pddl_action_names"]
        assert plan_event["actions"] == plan_event["pddl_action_names"]
        assert plan_event["plan_length"] == len(plan_event["pddl_action_names"])


def test_real_medication_episode_smoke_reaches_delivery_and_updates_weights(
    real_two_episode_run,
):
    result = real_two_episode_run.first

    assert result["success"] is True
    assert result["task_state"]["delivered"] is True
    assert result["executed_actions"][-1].startswith("deliver_on_bedside_table")
    assert result["total_steps"] == len(result["executed_actions"])

    features = result["features"]
    assert set(features) == {"time", "safety", "battery", "proximity", "approach"}
    assert all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in features.values())

    ratings = np.array(result["ratings"], dtype=float)
    assert ratings.shape == (5,)
    assert np.all(np.isfinite(ratings))
    assert np.all((1.0 <= ratings) & (ratings <= 5.0))

    weights_before = np.array(result["weights_before"], dtype=float)
    weights_after = np.array(result["weights_after"], dtype=float)
    np.testing.assert_allclose(weights_after.sum(), 1.0, atol=1e-8)
    assert np.all(weights_after >= 0.0)
    assert not np.allclose(weights_before, weights_after)


def test_real_episode_trace_preserves_action_and_state_invariants(real_two_episode_run):
    run = real_two_episode_run
    action_starts = _events(run, "action_start", episode=1)
    action_ends = _events(run, "action_end", episode=1)
    execute_legs = _events(run, "execute_leg", episode=1)
    weight_updates = _events(run, "action_weight_update", episode=1)
    mpc_parameter_updates = _events(run, "mpc_parameter_update", episode=1)
    terminal_target_updates = _events(run, "terminal_target_update", episode=1)

    assert len(action_starts) == len(action_ends) == run.first["total_steps"]
    assert weight_updates == []

    location_changes = [
        end
        for start, end in zip(action_starts, action_ends)
        if start["state"]["location"] != end["state"]["location"]
    ]
    assert len(execute_legs) == len(location_changes)
    assert mpc_parameter_updates == []
    assert len(terminal_target_updates) <= len(execute_legs)

    for leg in execute_legs:
        assert leg["start_location"] != leg["goal_location"]

    assert getattr(run.system.translator, "update_count", 0) == 0
    assert len(run.system.terminal_target_update_history) >= len(terminal_target_updates)

    for update in terminal_target_updates:
        assert update["fuzzy_mismatch"] is True
        assert update["update_applied"] is True
        assert update["loss"] >= 0.0
        assert np.array(update["z_target_before"], dtype=float).shape == (2,)
        assert np.array(update["z_final_reached"], dtype=float).shape == (2,)
        assert np.array(update["z_true_reached"], dtype=float).shape == (2,)
        assert np.array(update["observation_offset"], dtype=float).shape == (2,)
        assert np.array(update["E"], dtype=float).shape == (1,)
        assert np.array(update["grad_z_target"], dtype=float).shape == (2,)
        assert np.array(update["z_target_after"], dtype=float).shape == (2,)

    for previous_end, next_start in zip(action_ends, action_starts[1:]):
        assert previous_end["state"] == next_start["state"]
        assert previous_end["battery"] == next_start["battery"]


def test_real_second_episode_starts_from_previous_episode_outputs(
    real_two_episode_run,
):
    run = real_two_episode_run
    episode_two_start = _events(run, "episode_start", episode=2)[0]

    assert episode_two_start["battery"] == run.after_first["battery"]
    np.testing.assert_allclose(
        episode_two_start["weights_patient"],
        run.after_first["weights_patient"],
    )
    np.testing.assert_allclose(
        episode_two_start["mpc_params"]["Q"],
        run.after_first["mpc_params"]["Q"],
    )
    np.testing.assert_allclose(
        episode_two_start["mpc_params"]["R"],
        run.after_first["mpc_params"]["R"],
    )
    np.testing.assert_allclose(
        episode_two_start["mpc_params"]["Q_terminal"],
        run.after_first["mpc_params"]["Q_terminal"],
    )
