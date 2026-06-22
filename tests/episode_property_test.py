import sys
import types
from contextlib import contextmanager

import numpy as np

from integration.episode_runner import EpisodeRunnerMixin, update_terminal_target
from core.execution.formulation import SharedMPCFormulation
from tasks.medication_delivery.task_actions import TaskAction
from tasks.medication_delivery.task_state_manager import TaskStateManager


PDDL_ACTION_NAMES = [
    "go_to_supply_a",
    "collect_supp_vitamin_d",
    "check_supplement_correct",
    "go_to_pharmacy_north",
    "collect_med_antibiotic",
    "check_medication_correct",
    "approach_left_to_bed",
    "deliver_on_bedside_table_left",
]

PDDL_FIRST_ACTION_SEQUENCE = PDDL_ACTION_NAMES


def test_terminal_target_update_uses_sensitivity_chain_rule():
    z_target = np.array([8.0, 0.0])
    E = np.array([-3.0, 0.0])
    sensitivity = np.array([[0.875, 0.0], [0.0, 0.875]])

    z_after, info = update_terminal_target(
        z_target,
        E,
        sensitivity=sensitivity,
        lr=1.0,
        max_grad_norm=100.0,
    )

    np.testing.assert_allclose(info["grad_z_target"], np.array([-2.625, 0.0]))
    np.testing.assert_allclose(z_after, np.array([10.625, 0.0]))
    assert info["alpha"] == 1.0
    assert info["alpha_source"] == "lr"
    assert info["loss"] == 4.5


def test_terminal_target_update_uses_dynamic_alpha_when_provided():
    z_target = np.array([8.0, 0.0])
    E = np.array([-3.0, 0.0])
    sensitivity = np.array([[0.875, 0.0], [0.0, 0.875]])

    z_after, info = update_terminal_target(
        z_target,
        E,
        sensitivity=sensitivity,
        lr=1.0,
        alpha=0.5,
        alpha_source="test_alpha",
        max_grad_norm=100.0,
    )

    np.testing.assert_allclose(info["grad_z_target"], np.array([-2.625, 0.0]))
    np.testing.assert_allclose(z_after, np.array([9.3125, 0.0]))
    assert info["alpha"] == 0.5
    assert info["alpha_source"] == "test_alpha"
    assert info["change_norm"] == 3.0


def test_terminal_target_waypoints_stop_at_learned_target():
    waypoints = EpisodeRunnerMixin._terminal_target_waypoints(
        np.array([0.0, 0.0]),
        np.array([5.0, 5.0]),
        np.array([8.0, 9.0]),
    )

    np.testing.assert_allclose(waypoints[-1], np.array([8.0, 9.0]))
    assert len(waypoints) == 20


def _state_snapshot(state):
    return {
        "location": state.location,
        "has_medication": bool(state.has_medication),
        "has_supplement": bool(state.has_supplement),
        "delivered": bool(state.delivered),
        "approach_side": state.approach_side,
        "battery_soc": float(state.battery_soc),
        "step_count": int(state.step_count),
    }


def _mpc_snapshot(translator):
    q, r, q_term = translator.get_mpc_params(near_patient=True)
    return {
        "Q": np.array(q, dtype=float).copy(),
        "R": np.array(r, dtype=float).copy(),
        "Q_terminal": np.array(q_term, dtype=float).copy(),
    }


class FakePlanAction:
    def __init__(self, name):
        self.action = types.SimpleNamespace(name=name)


class FakePlan:
    def __init__(self, names):
        self.actions = [FakePlanAction(name) for name in names]


class FakePlannerResult:
    def __init__(self, names=None):
        self.plan = FakePlan(names or PDDL_FIRST_ACTION_SEQUENCE)
        self.status = "SOLVED"


class FakeProblem:
    locations = [
        "home",
        "pharmacy_north",
        "supply_A",
        "patient_bed_left",
        "patient_bed_right",
    ]

    def __init__(self):
        self.initial_values = {}

    def fluent(self, name):
        def _call(*args):
            return (name, args)

        return _call

    def object(self, name):
        return name

    def user_type(self, name):
        return name

    def objects(self, user_type):
        if user_type == "location":
            return list(self.locations)
        return []

    def set_initial_value(self, fluent_key, value):
        self.initial_values[fluent_key] = value


class FakePDDLReader:
    def parse_problem(self, *_args, **_kwargs):
        return FakeProblem()


class FakeOneshotPlanner:
    solve_count = 0

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def solve(self, _problem):
        index = min(self.__class__.solve_count, len(PDDL_FIRST_ACTION_SEQUENCE) - 1)
        self.__class__.solve_count += 1
        return FakePlannerResult(PDDL_FIRST_ACTION_SEQUENCE[index:])


class FakeReplanner(FakeOneshotPlanner):
    def __init__(self, problem=None, *args, **kwargs):
        self.problem = problem
        self.initial_values = {}

    def update_initial_value(self, fluent_key, value):
        self.initial_values[fluent_key] = value


@contextmanager
def fake_unified_planning_modules():
    old_modules = {
        name: sys.modules.get(name)
        for name in [
            "unified_planning",
            "unified_planning.io",
            "unified_planning.shortcuts",
        ]
    }

    fake_root = types.ModuleType("unified_planning")
    fake_io = types.ModuleType("unified_planning.io")
    fake_shortcuts = types.ModuleType("unified_planning.shortcuts")
    fake_env = types.SimpleNamespace(credits_stream=None)
    FakeOneshotPlanner.solve_count = 0

    fake_io.PDDLReader = FakePDDLReader
    fake_shortcuts.OneshotPlanner = FakeOneshotPlanner
    fake_shortcuts.Replanner = FakeReplanner
    fake_shortcuts.get_environment = lambda: fake_env

    sys.modules["unified_planning"] = fake_root
    sys.modules["unified_planning.io"] = fake_io
    sys.modules["unified_planning.shortcuts"] = fake_shortcuts

    try:
        yield
    finally:
        for name, old in old_modules.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


class FakeEnvironment:
    def __init__(self):
        self.locations = {
            "home": np.array([0.0, 0.0]),
            "pharmacy_north": np.array([4.0, 0.0]),
            "supply_A": np.array([8.0, 0.0]),
            "patient_bed_left": np.array([12.0, 0.0]),
            "patient_bed_right": np.array([12.0, 2.0]),
        }
        self.environment_state = {
            "battery_level": 1.0,
            "pharmacy_north_stock": 5,
            "supply_A_stock": 5,
        }
        self.robot_state_6d = np.zeros(6)
        self.previous_position = None

    def reset(self, initial_position):
        self.robot_state_6d = np.array(
            [initial_position[0], initial_position[1], 0.0, 0.0, 0.0, 0.0],
            dtype=float,
        )
        return self.robot_state_6d.copy()

    def get_all_stock_levels(self):
        return {"pharmacy_north": 5, "supply_A": 5}

    def consume_stock(self, loc):
        key = f"{loc}_stock"
        self.environment_state[key] = max(0, self.environment_state.get(key, 1) - 1)

    def get_stock(self, loc):
        return self.environment_state.get(f"{loc}_stock", 0)


class FakeParams:
    def __init__(self, owner):
        self.owner = owner

    def to_vector(self):
        return np.concatenate(
            [
                self.owner.q,
                self.owner.r,
                self.owner.q_terminal,
                np.array([self.owner.update_count], dtype=float),
            ]
        )


class FakeTranslator:
    def __init__(self):
        self.q = np.array([20.0, 20.0, 3.0, 8.0, 8.0, 8.0])
        self.r = np.array([2.0, 2.0, 1.5])
        self.q_terminal = self.q * SharedMPCFormulation.TERMINAL_COST_MULTIPLIER
        self.params = FakeParams(self)
        self.update_count = 0
        self.weight_syncs = []

    def update_preference_weights(self, weights):
        self.weight_syncs.append(np.array(weights, dtype=float).copy())

    def get_mpc_params(self, near_patient=False):
        return self.q.copy(), self.r.copy(), self.q_terminal.copy()

    def update_parameters(self, dJ_dQ, dJ_dR, cost=None):
        self.update_count += 1
        self.q = self.q + 0.01 * np.array(dJ_dQ, dtype=float)
        self.r = self.r + 0.01 * np.array(dJ_dR, dtype=float)
        return {
            "param_change": 1.0,
            "gradient_norm": float(np.linalg.norm(np.concatenate([dJ_dQ, dJ_dR]))),
        }


class FakePreferenceLearner:
    def __init__(self):
        self.weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float)
        self.true_profile = types.SimpleNamespace(
            weights=np.array([0.1, 0.4, 0.05, 0.25, 0.2], dtype=float)
        )
        self.process_calls = 0

    def get_current_weights(self):
        return self.weights.copy()

    def process_episode(self, _features):
        self.process_calls += 1
        old = self.weights.copy()
        self.weights = np.array([0.18, 0.24, 0.18, 0.2, 0.2], dtype=float)
        return np.ones(5) * 4.0, {
            "old_weights": old,
            "new_weights": self.weights.copy(),
            "distance_to_true": float(
                np.linalg.norm(self.weights - self.true_profile.weights)
            ),
            "converged": False,
        }


class FakeMPC:
    def __init__(self):
        self.stats = {
            "total_solves": 0,
            "acados_solves": 0,
            "sensitivity_computes": 0,
            "total_solve_time": 0.0,
            "total_sens_time": 0.0,
        }
        self.update_calls = []

    def update_parameters(self, Q_diag, R_diag, obstacles, z_target=None):
        self.update_calls.append(
            {
                "Q": np.array(Q_diag, dtype=float).copy(),
                "R": np.array(R_diag, dtype=float).copy(),
                "obstacles": list(obstacles),
                "z_target": None
                if z_target is None
                else np.array(z_target, dtype=float).copy(),
            }
        )


class FakeLearningTracker:
    def __init__(self):
        self.records = []

    def record(self, metrics):
        self.records.append(metrics)


class PropertyProbeSystem(EpisodeRunnerMixin):
    def __init__(self):
        self.events = []
        self.verbose = False
        self.save_summaries = False
        self.summary_dir = None
        self.episode_count = 0
        self.fix_translator = False
        self.sensitivity_interval = 1
        self.env = FakeEnvironment()
        self.task_manager = TaskStateManager(
            self.env, list(self.env.locations), fuzzy_estimator=None
        )
        self.fuzzy_estimator = None
        self.translator = FakeTranslator()
        self.preference_learner = FakePreferenceLearner()
        self.mpc = FakeMPC()
        self.learning_tracker = FakeLearningTracker()
        self.episode_history = []
        self.plan_history = []
        self._current_risk_map = {"home": 0.02, "pharmacy_north": 0.3, "supply_A": 0.05}

    def _episode_property_hook(self, event, **payload):
        if event == "episode_start":
            payload["mpc_params"] = _mpc_snapshot(self.translator)
            payload["weights_patient"] = self.preference_learner.get_current_weights()
            payload["battery"] = float(self.env.environment_state["battery_level"])
        if "state" in payload:
            payload["state"] = _state_snapshot(payload["state"])
        if "start_state" in payload:
            payload["start_state"] = _state_snapshot(payload["start_state"])
        self.events.append({"event": event, **payload})

    def _perturb_risk_map(self):
        return None

    def _perturb_weights_for_exploration(self, weights, _episode):
        return weights.copy(), {"explored": False, "sigma": 0.0}

    def _simulate_first_up_step(self, result, task_state, is_meal=False):
        first = result.plan.actions[0].action.name
        if first == "go_to_pharmacy_north":
            next_state = task_state.copy()
            next_state.location = "pharmacy_north"
            return next_state
        return task_state.copy()

    def _execute_leg(
        self,
        start_state_6d,
        goal_location,
        start_location,
        action=None,
        Action_taken=None,
        near_patient=False,
        max_leg_steps=200,
    ):
        self._emit_episode_property_event(
            "execute_leg",
            action=action,
            start_location=start_location,
            goal_location=goal_location,
            near_patient=near_patient,
        )
        q, r, _ = self.translator.get_mpc_params(near_patient)
        self.mpc.update_parameters(q, r, [])

        final_state = np.array(start_state_6d, dtype=float).copy()
        final_state[:2] = self.env.locations[goal_location]
        self.env.robot_state_6d = final_state.copy()
        distance = float(np.linalg.norm(self.env.locations[goal_location] - self.env.locations[start_location]))
        return {
            "success": True,
            "final_state_6d": final_state,
            "trajectory": np.array([start_state_6d[:2], final_state[:2]]),
            "total_distance": distance,
            "straight_line_distance": distance,
            "path_efficiency": 1.0,
            "final_error": 0.0,
            "steps": 1,
            "execution_time": 1.0,
            "avg_mpc_cost": 10.0,
            "dJ_dQ_avg": np.zeros(6),
            "dJ_dR_avg": np.zeros(3),
            "num_sensitivities": 0,
            "nav_stack_used": False,
            "mpc_stats": dict(self.mpc.stats),
        }

    def _update_task_state_with_fuzzy(self, task_state, _current_6d_state, goal_location):
        task_state.location = goal_location
        task_state.location_memberships = {goal_location: 1.0}
        return task_state

    def _get_risk_value(self, location):
        return self._current_risk_map.get(location, 0.1)

    def _extract_plan_structure(self, actions):
        structure = {
            "task_type": "medication",
            "pharmacy_choice": None,
            "supply_choice": None,
            "approach_choice": None,
            "recharge_added": False,
            "plan_length": len(actions),
        }
        for action in actions:
            if action == TaskAction.GO_TO_PHARMACY_NORTH:
                structure["pharmacy_choice"] = "pharmacy_north"
            elif action == TaskAction.GO_TO_PHARMACY_SOUTH:
                structure["pharmacy_choice"] = "pharmacy_south"
            elif action == TaskAction.GO_TO_SUPPLY_A:
                structure["supply_choice"] = "supply_A"
            elif action == TaskAction.GO_TO_SUPPLY_B:
                structure["supply_choice"] = "supply_B"
            elif action == TaskAction.GO_TO_PATIENT_LEFT:
                structure["approach_choice"] = "left"
            elif action == TaskAction.GO_TO_PATIENT_RIGHT:
                structure["approach_choice"] = "right"
            elif action == TaskAction.RECHARGE:
                structure["recharge_added"] = True
        return structure

    @staticmethod
    def _wrap_angle(rad):
        return float(np.arctan2(np.sin(rad), np.cos(rad)))

    @staticmethod
    def _pos_score_from_error(pos_err, max_ok=2.0):
        return float(1.0 - min(max(pos_err, 0.0) / max_ok, 1.0))

    @staticmethod
    def _yaw_score(yaw_err):
        return float(1.0 - min(abs(float(yaw_err)) / np.pi, 1.0))

    def _print_episode_summary(self, _summary):
        return None


def _events(system, name):
    return [event for event in system.events if event["event"] == name]


def test_single_episode_action_properties_hold():
    system = PropertyProbeSystem()

    with fake_unified_planning_modules():
        result = system.run_episode({}, start_location="home", task_type="medication")

    assert result["success"] is True

    plan = _events(system, "plan_ready")[0]
    action_starts = _events(system, "action_start")
    action_ends = _events(system, "action_end")
    execute_legs = _events(system, "execute_leg")
    weight_updates = _events(system, "action_weight_update")
    mpc_parameter_updates = _events(system, "mpc_parameter_update")

    location_changes = [
        end
        for start, end in zip(action_starts, action_ends)
        if start["state"]["location"] != end["state"]["location"]
    ]

    assert len(execute_legs) == len(location_changes)
    assert weight_updates == []
    assert system.translator.weight_syncs == []
    assert mpc_parameter_updates == []
    assert len(system.mpc.update_calls) == len(execute_legs)
    assert len(system.mpc.update_calls) < plan["plan_length"]
    assert system.translator.update_count == 0

    for previous_end, next_start in zip(action_ends, action_starts[1:]):
        assert previous_end["state"] == next_start["state"]
        assert previous_end["battery"] == next_start["battery"]


def test_next_episode_starts_from_previous_episode_state():
    system = PropertyProbeSystem()

    with fake_unified_planning_modules():
        first = system.run_episode({}, start_location="home", task_type="medication")

    expected_mpc = _mpc_snapshot(system.translator)
    expected_weights = system.preference_learner.get_current_weights()
    expected_battery = float(system.env.environment_state["battery_level"])
    assert np.isclose(expected_battery, first["battery_remaining"] / 100.0)

    with fake_unified_planning_modules():
        system.run_episode({}, start_location="home", task_type="medication")

    episode_two_start = [
        event
        for event in _events(system, "episode_start")
        if event["episode"] == 2
    ][0]

    np.testing.assert_allclose(episode_two_start["mpc_params"]["Q"], expected_mpc["Q"])
    np.testing.assert_allclose(episode_two_start["mpc_params"]["R"], expected_mpc["R"])
    np.testing.assert_allclose(
        episode_two_start["mpc_params"]["Q_terminal"],
        expected_mpc["Q_terminal"],
    )
    np.testing.assert_allclose(episode_two_start["weights_patient"], expected_weights)
    assert episode_two_start["battery"] == expected_battery
    assert system.translator.update_count == 0


def test_physical_alignment_moves_robot_to_predicted_location():
    system = PropertyProbeSystem()
    current_state = np.array([1.0, 2.0, 0.7, 0.3, 0.2, 0.1], dtype=float)

    aligned_state, aligned_xy = system._align_physical_state_to_location(
        current_state,
        "supply_A",
    )

    np.testing.assert_allclose(aligned_xy, system.env.locations["supply_A"])
    np.testing.assert_allclose(aligned_state[:2], system.env.locations["supply_A"])
    assert aligned_state[2] == current_state[2]
    np.testing.assert_allclose(aligned_state[3:], np.zeros(3))
    np.testing.assert_allclose(system.env.robot_state_6d, aligned_state)
    np.testing.assert_allclose(system.env.previous_position, aligned_xy)


def test_fuzzy_location_classification_decides_symbolic_arrival():
    system = PropertyProbeSystem()

    class FakeFuzzyEstimator:
        def __init__(self, memberships):
            self.memberships = memberships

        def estimate(self, _position, _battery_soc):
            return types.SimpleNamespace(location_memberships=dict(self.memberships))

    system.fuzzy_estimator = FakeFuzzyEstimator(
        {"home": 0.0, "supply_A": 0.0, "pharmacy_north": 0.0}
    )
    classified = system._fuzzy_location_classification(
        np.array([100.0, 100.0, 0.0, 0.0, 0.0, 0.0]),
        1.0,
        "supply_A",
    )
    assert classified["arrived"] is False
    assert classified["location"] == "in_transit"

    system.fuzzy_estimator = FakeFuzzyEstimator(
        {"home": 0.2, "supply_A": 0.69, "pharmacy_north": 0.1}
    )
    classified = system._fuzzy_location_classification(
        np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        1.0,
        "supply_A",
    )
    assert classified["arrived"] is False
    assert classified["location"] == "in_transit"
    assert classified["goal_membership"] == 0.69

    system.fuzzy_estimator = FakeFuzzyEstimator(
        {"home": 0.2, "supply_A": 0.7, "pharmacy_north": 0.1}
    )
    classified = system._fuzzy_location_classification(
        np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        1.0,
        "supply_A",
    )
    assert classified["arrived"] is True
    assert classified["location"] == "supply_A"
    assert classified["goal_membership"] == 0.7


def test_fuzzy_location_filter_matches_pddl_names_case_insensitively():
    system = PropertyProbeSystem()
    system._pddl_location_names = lambda: {"home", "supply_a", "pharmacy_north"}

    class FakeFuzzyEstimator:
        def estimate(self, _position, _battery_soc):
            return types.SimpleNamespace(
                location_memberships={
                    "home": 0.0,
                    "supply_A": 1.0,
                    "pharmacy_north": 0.0,
                }
            )

    system.fuzzy_estimator = FakeFuzzyEstimator()
    classified = system._fuzzy_location_classification(
        np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        1.0,
        "supply_A",
    )

    assert classified["arrived"] is True
    assert classified["location"] == "supply_A"
    assert classified["goal_membership"] == 1.0
