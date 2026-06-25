"""
integration/episode_runner.py — EpisodeRunnerMixin for FullMedicationDeliverySystem.

Contains the two hot-path methods:
    _execute_leg   — one navigation leg: direct waypoints → HybridMPC
    run_episode    — full 5-phase episode (plan → execute → skip translator update → outer loop → metrics)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.execution.formulation import SharedMPCFormulation
from core.execution.mpc_solver import AcadosRuntimeError
from core.planning.fuzzy_state import LOCATION_DEFUZZIFICATION_THRESHOLD
from core.task_planning.pddl_engine import make_pddl_oneshot_planner
from .metrics import EpisodeMetrics
from tasks.medication_delivery.task_actions import (
    DELIVERY_ACTIONS as MED_DELIVERY_ACTIONS,
    MEDICATION_COLLECTION_ACTIONS,
    SUPPLEMENT_COLLECTION_ACTIONS,
    TaskAction,
)

# ── Meal-prep task imports (optional) ────────────────────────────────
try:
    from tasks.meal_preparation.task_actions import (
        ACTION_TARGET_LOCATIONS,
        ACTION_DURATIONS as MEAL_ACTION_DURATIONS,
        DELIVERY_ACTIONS as MEAL_DELIVERY_ACTIONS,
        MEAL_HOT,
        MEAL_REQUIRED_INGREDIENTS,
        NAVIGATION_ACTIONS as MEAL_NAV_ACTIONS,
        MealAction,
    )
    from tasks.meal_preparation.meal_profiles import compute_meal_features

    _HAS_MEAL_PREP = True
except ImportError:
    _HAS_MEAL_PREP = False


def update_terminal_target(
    z_target: np.ndarray,
    E: np.ndarray,
    sensitivity: Optional[np.ndarray] = None,
    lr: float = 0.05,
    alpha: Optional[float] = None,
    alpha_source: str = "lr",
    max_grad_norm: float = 5.0,
    lower_bounds: Optional[np.ndarray] = None,
    upper_bounds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict]:
    """Gradient step for the learnable terminal target using J_E_z_target.T @ E."""
    z_before = np.array(z_target, dtype=float).reshape(2)
    E_vec = np.array(E, dtype=float).reshape(-1)
    change_norm = float(np.linalg.norm(E_vec))
    loss = 0.5 * float(E_vec @ E_vec)
    effective_alpha = float(lr if alpha is None else alpha)

    if sensitivity is None:
        grad = np.zeros_like(z_before)
        update_applied = False
    else:
        J = np.array(sensitivity, dtype=float)
        if E_vec.size == 1:
            grad = E_vec[0] * J.reshape(-1)[:2]
        else:
            J = J.reshape(E_vec.size, 2)
            grad = J.T @ E_vec
        update_applied = True

    grad_norm = float(np.linalg.norm(grad))
    clipped = False
    if update_applied and grad_norm > max_grad_norm > 0.0:
        grad = grad * (max_grad_norm / grad_norm)
        grad_norm = float(np.linalg.norm(grad))
        clipped = True

    z_after = z_before - effective_alpha * grad if update_applied else z_before.copy()
    if lower_bounds is not None or upper_bounds is not None:
        lo = (
            np.full_like(z_after, -np.inf)
            if lower_bounds is None
            else np.array(lower_bounds, dtype=float).reshape(2)
        )
        hi = (
            np.full_like(z_after, np.inf)
            if upper_bounds is None
            else np.array(upper_bounds, dtype=float).reshape(2)
        )
        z_after = np.clip(z_after, lo, hi)

    info = {
        "z_target_before": z_before.copy(),
        "E": E_vec.copy(),
        "change_norm": change_norm,
        "loss": loss,
        "alpha": effective_alpha,
        "alpha_source": alpha_source,
        "sensitivity": None if sensitivity is None else np.array(sensitivity, dtype=float).copy(),
        "grad_z_target": grad.copy(),
        "grad_norm": grad_norm,
        "grad_clipped": clipped,
        "z_target_after": z_after.copy(),
        "target_delta_norm": float(np.linalg.norm(z_after - z_before)),
        "update_applied": update_applied,
    }
    return z_after, info


class EpisodeRunnerMixin:
    """Core execution methods for FullMedicationDeliverySystem."""

    def _emit_episode_property_event(self, event: str, **payload) -> None:
        """Forward structured episode trace events when a hook is installed."""
        hook = getattr(self, "_episode_property_hook", None)
        if callable(hook):
            payload.setdefault("episode", getattr(self, "episode_count", None))
            hook(event, **payload)

    @staticmethod
    def _state_copy(state):
        return state.copy() if hasattr(state, "copy") else state

    @staticmethod
    def _pddl_action_name(action_instance) -> str:
        action = getattr(action_instance, "action", None)
        name = getattr(action, "name", None)
        if name is not None:
            return str(name)
        return str(action_instance).split("(", 1)[0].strip()

    @staticmethod
    def _pddl_object_name(obj) -> str:
        return str(getattr(obj, "name", obj))

    def _pddl_paths_for_task(self, is_meal: bool) -> Tuple[Path, Path]:
        root = Path(__file__).resolve().parent.parent
        if is_meal:
            return root / "unified_planning" / "domain_meal.pddl", root / "unified_planning" / "problem_meal.pddl"
        return root / "unified_planning" / "domain_med.pddl", root / "unified_planning" / "problem_med.pddl"

    def _pddl_problem_objects(self, problem, type_name: str) -> List[str]:
        try:
            user_type = problem.user_type(type_name)
            return [
                self._pddl_object_name(obj)
                for obj in problem.objects(user_type)
            ]
        except Exception:
            return []

    def _pddl_object(self, problem, name: Optional[str]):
        if name is None:
            return None
        object_name = str(name)
        try:
            return problem.object(object_name)
        except Exception:
            pass

        try:
            return problem.object(object_name.lower())
        except Exception:
            pass

        wanted = object_name.lower()
        for type_name in ("location", "medicine", "supplement", "meal", "ingredient", "delivery_target"):
            for obj_name in self._pddl_problem_objects(problem, type_name):
                if obj_name.lower() == wanted:
                    try:
                        return problem.object(obj_name)
                    except Exception:
                        return obj_name
        return object_name

    def _pddl_set_bool(self, problem, fluent_name: str, args: Tuple, value: bool) -> None:
        try:
            fluent = problem.fluent(fluent_name)
            pddl_args = tuple(self._pddl_object(problem, arg) for arg in args)
            if any(arg is None for arg in pddl_args):
                return
            problem.set_initial_value(fluent(*pddl_args), bool(value))
        except Exception:
            return

    def _pddl_set_num(self, problem, fluent_name: str, args: Tuple, value: float) -> None:
        try:
            fluent = problem.fluent(fluent_name)
            pddl_args = tuple(self._pddl_object(problem, arg) for arg in args)
            if any(arg is None for arg in pddl_args):
                return
            problem.set_initial_value(fluent(*pddl_args), float(value))
        except Exception:
            return

    def _sync_common_pddl_initial_state(self, problem, task_state, planning_weights) -> None:
        robot = "robot1"
        locations = self._pddl_problem_objects(problem, "location")
        for location in locations:
            self._pddl_set_bool(problem, "at", (robot, location), False)
        self._pddl_set_bool(problem, "at", (robot, task_state.location), True)
        self._pddl_set_num(
            problem,
            "battery-soc",
            (robot,),
            float(getattr(task_state, "battery_soc", 1.0)),
        )
        for idx, weight in enumerate(np.asarray(planning_weights, dtype=float).reshape(-1)[:5]):
            self._pddl_set_num(problem, f"w{idx}", (robot,), float(weight))

    def _sync_medication_pddl_initial_state(self, problem, task_state, planning_weights) -> None:
        robot = "robot1"
        self._sync_common_pddl_initial_state(problem, task_state, planning_weights)
        for fluent_name, attr in (
            ("has-medication", "has_medication"),
            ("has-supplement", "has_supplement"),
            ("unchecked-medication", "unchecked_medication"),
            ("unchecked-supplement", "unchecked_supplement"),
            ("correct-medication", "correct_medication"),
            ("correct-supplement", "correct_supplement"),
            ("medicine-recollect-required", "medicine_recollect_required"),
            ("supplement-recollect-required", "supplement_recollect_required"),
            ("can_be_deliverable", "can_be_deliverable"),
            ("delivered", "delivered"),
        ):
            self._pddl_set_bool(problem, fluent_name, (robot,), bool(getattr(task_state, attr, False)))

        for medicine in self._pddl_problem_objects(problem, "medicine"):
            self._pddl_set_bool(problem, "carrying-medicine", (robot, medicine), False)
        for supplement in self._pddl_problem_objects(problem, "supplement"):
            self._pddl_set_bool(problem, "carrying-supplement", (robot, supplement), False)
        if getattr(task_state, "carrying_medicine", None):
            self._pddl_set_bool(
                problem,
                "carrying-medicine",
                (robot, task_state.carrying_medicine),
                True,
            )
        if getattr(task_state, "carrying_supplement", None):
            self._pddl_set_bool(
                problem,
                "carrying-supplement",
                (robot, task_state.carrying_supplement),
                True,
            )

    def _sync_meal_pddl_initial_state(self, problem, task_state, planning_weights) -> None:
        robot = "robot1"
        self._sync_common_pddl_initial_state(problem, task_state, planning_weights)
        meal_names = list(MEAL_REQUIRED_INGREDIENTS.keys()) if _HAS_MEAL_PREP else []
        ingredient_names = sorted(
            {ingredient for ingredients in MEAL_REQUIRED_INGREDIENTS.values() for ingredient in ingredients}
        ) if _HAS_MEAL_PREP else []

        meal_to_prepare = getattr(task_state, "meal_to_prepare", None)
        self._pddl_set_bool(problem, "meal_chosen", (robot,), meal_to_prepare is not None)
        for meal_name in meal_names:
            self._pddl_set_bool(problem, "meal_to_prepare", (meal_name,), meal_name == meal_to_prepare)
            self._pddl_set_bool(
                problem,
                "ingredients_safe",
                (robot, meal_name),
                bool(getattr(task_state, "ingredients_safe", False) and meal_name == meal_to_prepare),
            )

        if meal_to_prepare is not None:
            collected = set(getattr(task_state, "collected_ingredients", ()))
            missing = set(getattr(task_state, "missing_ingredients", ()))
            expired = set(getattr(task_state, "expired_ingredients", ()))
            wrong = set(getattr(task_state, "wrong_ingredients", ()))
            allergens = set(getattr(task_state, "allergen_ingredients", ()))
            for ingredient in ingredient_names:
                self._pddl_set_bool(problem, "collected_ingredient", (robot, ingredient), ingredient in collected)
                self._pddl_set_bool(problem, "missing_ingredient", (robot, ingredient), ingredient in missing)
                self._pddl_set_bool(problem, "expired_ingredient", (robot, ingredient), ingredient in expired)
                self._pddl_set_bool(problem, "wrong_ingredient", (robot, ingredient), ingredient in wrong)
                self._pddl_set_bool(problem, "allergen_present", (robot, ingredient), ingredient in allergens)

        ready_for_quality = bool(
            getattr(task_state, "meal_cooked", False)
            or getattr(task_state, "cooking_level_checked", False)
            or getattr(task_state, "meal_palatable", False)
            or getattr(task_state, "meal_assembled", False)
            or (
                meal_to_prepare is not None
                and meal_to_prepare != MEAL_HOT
                and getattr(task_state, "ingredients_chopped", False)
            )
        )
        for fluent_name, attr, value in (
            ("ingredients_checked", "ingredients_checked", None),
            ("workspace-clean", "workspace_clean", None),
            ("robot-hands-clean", "robot_hands_clean", None),
            ("cross-contamination-risk", "cross_contamination_risk", None),
            ("ingredients_washed", "ingredients_washed", None),
            ("ingredients_chopped", "ingredients_chopped", None),
            ("meal_cooked", "meal_cooked", None),
            ("ready_for_quality", None, ready_for_quality),
            ("cooking_level_checked", "cooking_level_checked", None),
            ("meal_palatable", "meal_palatable", None),
            ("meal_assembled", "meal_assembled", None),
            ("can_be_deliverable", "can_be_deliverable", None),
            ("delivered", "delivered", None),
        ):
            self._pddl_set_bool(
                problem,
                fluent_name,
                (robot,),
                bool(value if attr is None else getattr(task_state, attr, False)),
            )

    def _plan_with_pddl(self, task_state, planning_weights, is_meal: bool):
        from unified_planning.io import PDDLReader

        domain_path, problem_path = self._pddl_paths_for_task(is_meal)
        problem = PDDLReader().parse_problem(str(domain_path), str(problem_path))
        if is_meal:
            self._sync_meal_pddl_initial_state(problem, task_state, planning_weights)
            enum_cls = MealAction
        else:
            self._sync_medication_pddl_initial_state(problem, task_state, planning_weights)
            enum_cls = TaskAction

        engine = getattr(self, "planning_engine", None)
        with make_pddl_oneshot_planner(engine) as planner:
            result = planner.solve(problem)

        raw_actions = list(getattr(getattr(result, "plan", None), "actions", []) or [])
        pddl_action_names = [self._pddl_action_name(action) for action in raw_actions]
        planned_actions = []
        unknown_actions = []
        for action_name in pddl_action_names:
            try:
                planned_actions.append(enum_cls(action_name))
            except ValueError:
                unknown_actions.append(action_name)

        plan_info = {
            "success": bool(planned_actions) and not unknown_actions,
            "mode": "pddl_enhsp_replan_first_action",
            "engine": engine,
            "status": str(getattr(result, "status", "")),
            "domain_path": str(domain_path),
            "problem_path": str(problem_path),
            "plan_length": len(planned_actions),
            "pddl_action_names": pddl_action_names,
            "unknown_actions": unknown_actions,
        }
        return planned_actions, [self._state_copy(task_state)], plan_info

    def _align_physical_state_to_location(
        self, current_state_6d: np.ndarray, location: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        target_xy = np.array(self.env.locations[location], dtype=float)
        aligned_state = np.array(current_state_6d, dtype=float).copy()
        aligned_state[:2] = target_xy
        aligned_state[3:] = 0.0

        if hasattr(self.env, "set_robot_pose"):
            aligned_state = self.env.set_robot_pose(target_xy, float(aligned_state[2]))
        else:
            self.env.robot_state_6d = aligned_state.copy()

        if hasattr(self.env, "previous_position"):
            self.env.previous_position = target_xy.copy()

        return aligned_state.copy(), target_xy.copy()

    def _filtered_fuzzy_memberships(self, memberships: Dict[str, float]) -> Dict[str, float]:
        if not hasattr(self, "_pddl_location_names"):
            return memberships

        pddl_locations = self._pddl_location_names()
        if pddl_locations is None:
            return memberships

        pddl_locations_lower = {loc.lower() for loc in pddl_locations}
        return {
            loc: val
            for loc, val in memberships.items()
            if loc in pddl_locations or loc.lower() in pddl_locations_lower
        }

    def _fuzzy_location_classification(
        self,
        current_6d_state: np.ndarray,
        battery_soc: float,
        goal_location: str,
    ) -> Dict:
        if self.fuzzy_estimator is None:
            return {
                "arrived": True,
                "location": goal_location,
                "goal_membership": 1.0,
                "dominant_membership": 1.0,
                "memberships": {goal_location: 1.0},
            }

        fm = self.fuzzy_estimator.estimate(current_6d_state[:2], battery_soc)
        memberships = self._filtered_fuzzy_memberships(dict(fm.location_memberships))
        if not memberships:
            return {
                "arrived": False,
                "location": "in_transit",
                "goal_membership": 0.0,
                "dominant_membership": 0.0,
                "memberships": {},
            }

        dominant_location, dominant_membership = max(
            memberships.items(), key=lambda item: item[1]
        )
        if float(dominant_membership) < LOCATION_DEFUZZIFICATION_THRESHOLD:
            dominant_location = "in_transit"

        goal_membership = float(memberships.get(goal_location, 0.0))
        arrived = (
            dominant_location.lower() == goal_location.lower()
            and goal_membership > 0.0
        )

        return {
            "arrived": arrived,
            "location": dominant_location,
            "goal_membership": goal_membership,
            "dominant_membership": float(dominant_membership),
            "memberships": memberships,
        }

    def _get_med_plan_key(self, structure: Dict) -> Tuple:
        return (
            "med",
            structure.get("pharmacy_choice"),
            structure.get("supply_choice"),
            structure.get("approach_choice"),
        )

    def _get_meal_plan_key(self, structure: Dict) -> Tuple:
        return ("meal", structure.get("meal_type"), structure.get("approach_choice"))

    @staticmethod
    def _terminal_target_key(action, goal_location: str) -> str:
        action_name = getattr(action, "value", action)
        return f"{action_name or 'unknown'}->{goal_location}"

    def _get_terminal_target(self, key: str, default_target: np.ndarray) -> np.ndarray:
        targets = getattr(self, "terminal_targets", None)
        if targets is None:
            targets = {}
            self.terminal_targets = targets
        if key not in targets:
            targets[key] = np.array(default_target, dtype=float).copy()
        return np.array(targets[key], dtype=float).copy()

    def _set_terminal_target(self, key: str, target: np.ndarray) -> None:
        targets = getattr(self, "terminal_targets", None)
        if targets is None:
            targets = {}
            self.terminal_targets = targets
        targets[key] = np.array(target, dtype=float).copy()

    def _terminal_target_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            SharedMPCFormulation.x_min[:2].astype(float),
            SharedMPCFormulation.x_max[:2].astype(float),
        )

    def _observation_offset_for_location(self, location: str) -> np.ndarray:
        offsets = getattr(self, "terminal_observation_offsets", {})
        if location in offsets:
            return np.array(offsets[location], dtype=float).reshape(2)
        return np.zeros(2, dtype=float)

    def _observed_terminal_xy(self, true_xy: np.ndarray, location: str) -> Tuple[np.ndarray, np.ndarray]:
        offset = self._observation_offset_for_location(location)
        return np.array(true_xy, dtype=float).reshape(2) + offset, offset

    @staticmethod
    def _terminal_target_waypoints(
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        z_target: np.ndarray,
        n_waypoints: int = 21,
    ) -> List[np.ndarray]:
        """Direct waypoints with the final arrival target set to z_target."""
        waypoints = [
            np.array(wp, dtype=float)
            for wp in np.linspace(start_pos, goal_pos, n_waypoints)[1:]
        ]
        final_target = np.array(z_target, dtype=float).reshape(2)
        if waypoints:
            waypoints[-1] = final_target
        else:
            waypoints = [final_target]
        return waypoints

    def _membership_from_observed_xy(
        self,
        observed_xy: np.ndarray,
        battery_level: float,
        goal_location: str,
    ) -> Dict:
        if self.fuzzy_estimator is None:
            return {
                "location": goal_location,
                "goal_membership": 1.0,
                "dominant_membership": 1.0,
                "memberships": {goal_location: 1.0},
                "mismatch": False,
            }

        fm = self.fuzzy_estimator.estimate(observed_xy, battery_level)
        memberships = dict(fm.location_memberships)
        if hasattr(self, "_pddl_location_names"):
            pddl_locations = self._pddl_location_names()
            if pddl_locations is not None:
                pddl_locations_lower = {loc.lower() for loc in pddl_locations}
                memberships = {
                    loc: val
                    for loc, val in memberships.items()
                    if loc in pddl_locations or loc.lower() in pddl_locations_lower
                }

        if memberships:
            fuzzy_location, fuzzy_dominant_membership = max(
                memberships.items(), key=lambda item: item[1]
            )
            if float(fuzzy_dominant_membership) < LOCATION_DEFUZZIFICATION_THRESHOLD:
                fuzzy_location = "in_transit"
            fuzzy_goal_membership = float(memberships.get(goal_location, 0.0))
        else:
            fuzzy_location = "in_transit"
            fuzzy_dominant_membership = 0.0
            fuzzy_goal_membership = 0.0

        return {
            "location": fuzzy_location,
            "goal_membership": fuzzy_goal_membership,
            "dominant_membership": float(fuzzy_dominant_membership),
            "memberships": memberships,
            "mismatch": fuzzy_location != goal_location,
        }

    def _simulate_terminal_position_for_target(
        self,
        start_state_6d: np.ndarray,
        goal_pos: np.ndarray,
        goal_location: str,
        waypoints: List[np.ndarray],
        Q_diag: np.ndarray,
        R_diag: np.ndarray,
        exclude: List[str],
        near_patient: bool,
        z_target: np.ndarray,
        max_leg_steps: int,
    ) -> Optional[np.ndarray]:
        """Re-run the same leg in controller space and return the final x/y."""
        if not hasattr(self.mpc, "solve"):
            return None

        current_state = np.array(start_state_6d, dtype=float).copy()
        sim_waypoints = self._terminal_target_waypoints(
            current_state[:2],
            goal_pos,
            z_target,
            n_waypoints=len(waypoints) + 1 if waypoints else 21,
        )
        first_wp = sim_waypoints[0] if sim_waypoints else goal_pos
        first_ref = np.array(
            [
                first_wp[0],
                first_wp[1],
                float(np.arctan2(first_wp[1] - current_state[1], first_wp[0] - current_state[0])),
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        if hasattr(self.mpc, "reset_episode"):
            self.mpc.reset_episode(x_init=current_state, x_ref=first_ref)

        obstacles = self._get_obstacles_for_leg(current_state[:2], goal_pos, exclude)
        self.mpc.update_parameters(Q_diag, R_diag, obstacles, z_target=z_target)

        step = 0
        dt = float(getattr(self.mpc, "dt", 0.2))
        for wp_idx, wp_target in enumerate(sim_waypoints):
            wp_target = np.array(wp_target, dtype=float)
            dx = wp_target[0] - current_state[0]
            dy = wp_target[1] - current_state[1]
            desired_yaw = float(np.arctan2(dy, dx))
            if wp_idx == len(sim_waypoints) - 1 and near_patient:
                desired_yaw = float(
                    getattr(self.translator, "location_orientations", {}).get(
                        goal_location, desired_yaw
                    )
                )

            x_ref = np.array([wp_target[0], wp_target[1], desired_yaw, 0.0, 0.0, 0.0])
            wp_tolerance = 1.5 if wp_idx < len(sim_waypoints) - 1 else 0.8

            for _ in range(max_leg_steps // max(len(sim_waypoints), 1)):
                if step >= max_leg_steps:
                    break
                if np.linalg.norm(current_state[:2] - wp_target) < wp_tolerance:
                    break
                if step > 0 and step % 20 == 0:
                    obstacles = self._get_obstacles_for_leg(
                        current_state[:2], wp_target, exclude
                    )
                    self.mpc.update_parameters(
                        Q_diag, R_diag, obstacles, z_target=z_target
                    )

                try:
                    sol = self.mpc.solve(current_state, x_ref)
                except Exception:
                    return None
                if not sol.success:
                    return None
                current_state = SharedMPCFormulation.discrete_dynamics_numpy(
                    current_state, sol.control, dt
                )
                step += 1

        return current_state[:2].copy()

    def _finite_difference_terminal_target_sensitivity(
        self,
        start_state_6d: np.ndarray,
        goal_pos: np.ndarray,
        goal_location: str,
        waypoints: List[np.ndarray],
        Q_diag: np.ndarray,
        R_diag: np.ndarray,
        exclude: List[str],
        near_patient: bool,
        z_target: np.ndarray,
        E0: np.ndarray,
        observation_offset: np.ndarray,
        battery_level: float,
        max_leg_steps: int,
    ) -> Optional[np.ndarray]:
        eps = float(getattr(self, "terminal_target_fd_eps", 1e-2))
        if eps <= 0.0:
            return None

        stats_before = dict(getattr(self.mpc, "stats", {}))
        z_before = (
            None
            if getattr(self.mpc, "z_target", None) is None
            else np.array(self.mpc.z_target, dtype=float).copy()
        )
        J = np.zeros((np.array(E0, dtype=float).reshape(-1).size, 2), dtype=float)

        try:
            for i in range(2):
                z_perturbed = np.array(z_target, dtype=float).copy()
                z_perturbed[i] += eps
                final_xy = self._simulate_terminal_position_for_target(
                    start_state_6d=start_state_6d,
                    goal_pos=goal_pos,
                    goal_location=goal_location,
                    waypoints=waypoints,
                    Q_diag=Q_diag,
                    R_diag=R_diag,
                    exclude=exclude,
                    near_patient=near_patient,
                    z_target=z_perturbed,
                    max_leg_steps=max_leg_steps,
                )
                if final_xy is None:
                    return None
                observed_perturbed = final_xy + np.array(
                    observation_offset, dtype=float
                ).reshape(2)
                E_perturbed = np.array(
                    [
                        1.0
                        - self._membership_from_observed_xy(
                            observed_perturbed, battery_level, goal_location
                        )["goal_membership"]
                    ],
                    dtype=float,
                ).reshape(-1)
                J[:, i] = (E_perturbed - np.array(E0, dtype=float).reshape(-1)) / eps
        finally:
            if stats_before:
                self.mpc.stats = stats_before
            obstacles = self._get_obstacles_for_leg(start_state_6d[:2], goal_pos, exclude)
            self.mpc.update_parameters(Q_diag, R_diag, obstacles, z_target=z_before)

        return J

    # -----------------------------------------------------------------
    # Leg execution: direct waypoints → HybridMPC
    # -----------------------------------------------------------------

    def _execute_leg(
        self,
        start_state_6d: np.ndarray,
        goal_location: str,
        start_location: str,
        action,
        near_patient: bool = False,
        max_leg_steps: int = 200,
    ) -> Dict:
        goal_pos  = np.array(self.env.locations[goal_location], dtype=float)
        start_pos = start_state_6d[:2].copy()

        # Step 1: Get MPC params from translator
        if hasattr(self.translator, "get_mpc_params"):
            Q_diag, R_diag, _ = self.translator.get_mpc_params(near_patient)
            horizon = 20
            default_z_target = goal_pos.copy()
            
            
        else:
            translation = self.translator.translate(
                start_location=start_location,
                goal_location=goal_location,
                current_state=start_state_6d,
            )
            mpc_cfg = translation.get("mpc_config", {})
            from core.execution.hybrid import SharedMPCFormulation
            Q_diag  = np.array(mpc_cfg.get("Q_diag", SharedMPCFormulation.Q_default))
            R_diag  = np.array(mpc_cfg.get("R_diag", SharedMPCFormulation.R_default))
            horizon = int(mpc_cfg.get("horizon", 40))
            default_z_target = np.array(
                mpc_cfg.get("z_target", goal_pos), dtype=float
            )

        terminal_target_key = self._terminal_target_key(action, goal_location)
        z_target = self._get_terminal_target(terminal_target_key, default_z_target)

        self._emit_episode_property_event(
            "execute_leg",
            action=action,
            start_location=start_location,
            goal_location=goal_location,
            near_patient=near_patient,
        )

        if self.verbose:
            print(
                f"    [Translator] Q_pos={Q_diag[0]:.1f}, Q_vel={Q_diag[3]:.1f}, "
                f"R={R_diag[0]:.2f}, horizon={horizon}"
            )

        # Step 2: Direct waypoint reference. The symbolic layer still targets
        # the planned location, while the final controller stop target is the
        # learned terminal target.
        waypoints = self._terminal_target_waypoints(start_pos, goal_pos, z_target)
        nav_used = False

        # Step 3: Follow waypoints with HybridMPC
        # Pass start/goal so reset pre-populates a straight-line warm-start
        # (Fix 3: eliminates the zero-init cold-start penalty on first solve).
        first_wp = waypoints[0] if waypoints else goal_pos
        _dx = first_wp[0] - start_state_6d[0]
        _dy = first_wp[1] - start_state_6d[1]
        first_ref_6d = np.array([first_wp[0], first_wp[1], float(np.arctan2(_dy, _dx)), 0.0, 0.0, 0.0])
        self.mpc.reset_episode(x_init=start_state_6d, x_ref=first_ref_6d)

        LOCATION_ZONES = {
            "pantry": "kitchen", "prep_station": "kitchen", "stove": "kitchen",
            "patient_bed": "bed", "patient_bed_left": "bed", "patient_bed_right": "bed",
        }
        exclude = [start_location, goal_location]
        for loc in [start_location, goal_location]:
            zone = LOCATION_ZONES.get(loc)
            if zone:
                for name, z in LOCATION_ZONES.items():
                    if z == zone and name not in exclude:
                        exclude.append(name)

        obstacles = self._get_obstacles_for_leg(start_pos, goal_pos, exclude)
        self.mpc.update_parameters(Q_diag, R_diag, obstacles, z_target=z_target)

        if hasattr(self.mpc, "warm_start_trajectory"):
            first_wp = waypoints[0] if waypoints else goal_pos
            self.mpc.warm_start_trajectory(start_state_6d, first_wp)

        current_state  = start_state_6d.copy()
        trajectory     = [current_state[:2].copy()]
        total_cost     = 0.0
        cost_count     = 0
        step           = 0
        solver_failure = None

        for wp_idx, wp_target in enumerate(waypoints):
            if solver_failure is not None:
                break
            dx = wp_target[0] - current_state[0]
            dy = wp_target[1] - current_state[1]
            desired_yaw = float(np.arctan2(dy, dx))

            if wp_idx == len(waypoints) - 1 and near_patient:
                desired_yaw = float(
                    getattr(self.translator, "location_orientations", {}).get(
                        goal_location, desired_yaw
                    )
                )

            x_ref        = np.array([wp_target[0], wp_target[1], desired_yaw, 0.0, 0.0, 0.0])
            wp_tolerance = 1.5 if wp_idx < len(waypoints) - 1 else 0.8

            for _ in range(max_leg_steps // max(len(waypoints), 1)):
                if step >= max_leg_steps:
                    break

                dist_to_wp = np.linalg.norm(current_state[:2] - wp_target)
                if dist_to_wp < wp_tolerance:
                    if self.verbose and wp_idx < len(waypoints) - 1:
                        print(f"    [WP] Reached waypoint {wp_idx+1}/{len(waypoints)}")
                    break

                if step > 0 and step % 20 == 0:
                    obstacles = self._get_obstacles_for_leg(
                        current_state[:2], wp_target, exclude
                    )
                    self.mpc.update_parameters(Q_diag, R_diag, obstacles, z_target=z_target)

                try:
                    sol = self.mpc.solve(current_state, x_ref)
                except AcadosRuntimeError as exc:
                    solver_failure = str(exc)
                    if self.verbose:
                        print(
                            f"    [Acados] solve failed at step {step} "
                            f"for action={getattr(action, 'value', action)}; "
                            "recording mismatch and continuing"
                        )
                    break

                if not sol.success:
                    solver_failure = (
                        f"MPC solve failed at step {step} "
                        f"for action={action}, goal={goal_location}, "
                        f"solver={sol.solver_used}"
                    )
                    if self.verbose:
                        print(f"    [Acados] {solver_failure}")
                    break

                control = sol.control
                total_cost += sol.cost
                cost_count += 1

                self.env.step(control)
                current_state = self.env.robot_state_6d.copy()
                trajectory.append(current_state[:2].copy())
                step += 1

        # Step 4: Q/R translator updates stay disabled; only z_target is learned.
        dJ_dQ_avg = np.zeros_like(Q_diag, dtype=float)
        dJ_dR_avg = np.zeros_like(R_diag, dtype=float)
        dJ_dz_target_avg = np.zeros(2)
        n_sens = 0

        traj_array = np.array(trajectory)
        if len(traj_array) > 1:
            diffs          = np.linalg.norm(traj_array[1:] - traj_array[:-1], axis=1)
            total_distance = float(np.sum(diffs))
        else:
            total_distance = 0.0

        straight_line = float(np.linalg.norm(goal_pos - start_pos))
        true_final_xy = current_state[:2].copy()
        observed_final_xy, observation_offset = self._observed_terminal_xy(
            true_final_xy, goal_location
        )
        final_error   = float(np.linalg.norm(observed_final_xy - goal_pos))
        true_final_error = float(np.linalg.norm(true_final_xy - goal_pos))
        avg_cost      = total_cost / max(cost_count, 1)
        battery_level = float(
            getattr(self.env, "environment_state", {}).get("battery_level", 1.0)
        )
        membership_info = self._membership_from_observed_xy(
            observed_final_xy, battery_level, goal_location
        )
        fuzzy_location = membership_info["location"]
        fuzzy_goal_membership = float(membership_info["goal_membership"])
        fuzzy_dominant_membership = float(membership_info["dominant_membership"])
        fuzzy_mismatch = bool(membership_info["mismatch"])
        E = np.array([1.0 - fuzzy_goal_membership], dtype=float)
        terminal_target_update = None
        if (
            bool(getattr(self, "terminal_target_learning_enabled", True))
            and fuzzy_mismatch
        ):
            sensitivity = None
            if solver_failure is None:
                sensitivity = self._finite_difference_terminal_target_sensitivity(
                    start_state_6d=start_state_6d,
                    goal_pos=goal_pos,
                    goal_location=goal_location,
                    waypoints=waypoints,
                    Q_diag=Q_diag,
                    R_diag=R_diag,
                    exclude=exclude,
                    near_patient=near_patient,
                    z_target=z_target,
                    E0=E,
                    observation_offset=observation_offset,
                    battery_level=battery_level,
                    max_leg_steps=max_leg_steps,
                )
            lower_bounds, upper_bounds = self._terminal_target_bounds()
            terminal_target_change_norm = float(np.linalg.norm(E))
            terminal_target_alpha_gain = float(
                getattr(
                    self,
                    "terminal_target_alpha_gain",
                    getattr(self, "terminal_target_learning_rate", 0.05),
                )
            )
            terminal_target_alpha = (
                terminal_target_alpha_gain * terminal_target_change_norm
            )
            z_target_after, terminal_target_update = update_terminal_target(
                z_target=z_target,
                E=E,
                sensitivity=sensitivity,
                lr=float(getattr(self, "terminal_target_learning_rate", 0.05)),
                alpha=terminal_target_alpha,
                alpha_source="alpha_gain*membership_deficit",
                max_grad_norm=float(
                    getattr(self, "terminal_target_max_grad_norm", 5.0)
                ),
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            terminal_target_update.update(
                {
                    "episode_or_action_id": terminal_target_key,
                    "action": getattr(action, "value", action),
                    "goal_location": goal_location,
                    "z_final_reached": observed_final_xy.copy(),
                    "z_true_reached": true_final_xy.copy(),
                    "observation_offset": observation_offset.copy(),
                    "goal_membership": fuzzy_goal_membership,
                    "fuzzy_mismatch": fuzzy_mismatch,
                    "alpha_gain": terminal_target_alpha_gain,
                    "desired_goal": goal_pos.copy(),
                    "sensitivity_source": "finite_difference"
                    if sensitivity is not None
                    else (
                        "skipped_after_acados_failure"
                        if solver_failure is not None
                        else "unavailable"
                    ),
                    "solver_failure": solver_failure,
                }
            )
            if terminal_target_update["update_applied"]:
                self._set_terminal_target(terminal_target_key, z_target_after)
                self.mpc.update_parameters(
                    Q_diag, R_diag, obstacles, z_target=z_target_after
                )

            history = getattr(self, "terminal_target_update_history", None)
            if history is None:
                history = []
                self.terminal_target_update_history = history
            history.append(terminal_target_update)
            self._emit_episode_property_event(
                "terminal_target_update", **terminal_target_update
            )

        result = {
            "success":               True,
            "final_state_6d":        current_state.copy(),
            "trajectory":            traj_array,
            "total_distance":        total_distance,
            "straight_line_distance": straight_line,
            "path_efficiency":       straight_line / max(total_distance, 0.01),
            "final_error":           final_error,
            "true_final_error":      true_final_error,
            "observed_final_xy":     observed_final_xy.copy(),
            "observation_offset":    observation_offset.copy(),
            "fuzzy_location":        fuzzy_location,
            "fuzzy_goal_membership": fuzzy_goal_membership,
            "fuzzy_dominant_membership": float(fuzzy_dominant_membership),
            "fuzzy_mismatch":        fuzzy_mismatch,
            "steps":                 step,
            "execution_time":        step * 0.2,
            "avg_mpc_cost":          avg_cost,
            "dJ_dQ_avg":             dJ_dQ_avg,
            "dJ_dR_avg":             dJ_dR_avg,
            "dJ_dz_target_avg":      dJ_dz_target_avg,
            "num_sensitivities":     n_sens,
            "terminal_target_update": terminal_target_update,
            "solver_failure":        solver_failure,
            "nav_stack_used":        nav_used,
            "mpc_stats":             dict(self.mpc.stats),
        }

        if self.verbose:
            print(
                f"    [Leg] {'✓' if result['success'] else '✗'} "
                f"dist={total_distance:.1f}m, error={final_error:.2f}m, "
                f"fuzzy={fuzzy_location}, steps={step}, cost={avg_cost:.1f}, "
                f"sens={n_sens} (disabled)"
            )

        return result

    # -----------------------------------------------------------------
    # Full episode (5 phases)
    # -----------------------------------------------------------------

    def run_episode(
        self,
        total_available_actions: Optional[Dict] = None,
        start_location: str = "home",
        task_type: str = "medication",
    ) -> Dict:
        del total_available_actions  # Legacy runner argument; actions are PDDL-derived.
        self.episode_count += 1
        metrics = EpisodeMetrics()
        is_meal = task_type == "meal"

        if is_meal and not _HAS_MEAL_PREP:
            print("[FAIL] Meal prep not available")
            return {"success": False, "episode": self.episode_count, "reason": "no_meal_prep"}

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"EPISODE {self.episode_count} [{task_type.upper()}]")
            print(f"{'='*80}\n")

        # ── Perturb risk map (robustness experiments) ─────────────────
        self._perturb_risk_map()

                # ── Reset environment ─────────────────────────────────────────
        previous_battery = float(
            self.env.environment_state.get("battery_level", 1.0)
        )
        initial_pos      = self.env.locations[start_location]
        initial_state_6d = self.env.reset(initial_position=initial_pos)
        self.env.environment_state["battery_level"] = previous_battery

        if is_meal:
            task_state = self.meal_task_manager.get_initial_state(start_location)
        else:
            task_state = self.task_manager.get_initial_state(start_location)

        battery_start        = previous_battery
        
        task_state.battery_soc = battery_start
        self._emit_episode_property_event(
            "episode_start",
            task_type=task_type,
            start_location=start_location,
            start_state=self._state_copy(task_state),
            battery=float(battery_start),
        )

        if self.fuzzy_estimator is not None:
            fm_init = self.fuzzy_estimator.estimate(initial_pos, battery_start)
            task_state.location_memberships = dict(fm_init.location_memberships)
            if self.verbose:
                print(f"  {fm_init.summary()}")

        if hasattr(self.env, "get_all_stock_levels"):
            task_state.location_stock = self.env.get_all_stock_levels()
            if self.verbose:
                stock_str = ", ".join(
                    f"{k}={v}" for k, v in task_state.location_stock.items()
                )
                print(f"  [Stock] {stock_str}")

        # ==============================================================
        # PHASE 1: TASK PLANNING
        # ==============================================================
        if self.verbose:
            print("PHASE 1: HIGH-LEVEL TASK PLANNING\n")

        current_weights = self.preference_learner.get_current_weights()

        planning_weights, explore_info = self._perturb_weights_for_exploration(
            current_weights, self.episode_count,
        )
        if explore_info["explored"] and self.verbose:
            print(f"  [Explore] σ={explore_info['sigma']:.4f}")
            print(f"    Learned:  [{', '.join(f'{w:.3f}' for w in current_weights)}]")
            print(f"    Planning: [{', '.join(f'{w:.3f}' for w in planning_weights)}]")





        # ==============================================================
        # PHASE 2: REPLAN AND EXECUTE FIRST ACTION
        # ==============================================================
        if self.verbose:
            print(f"\nPHASE 2: REPLAN → EXECUTE FIRST ACTION\n")

        episode_features = {
            "total_time": 0.0, "total_distance": 0.0, "total_battery_used": 0.0,
            "proximity_min_dists": [], "approach_quality_scores": [],
        }
        execution_success = True
        current_6d_state  = initial_state_6d.copy()
        last_goal_location: Optional[str] = None
        final_position_error = None
        leg_count = 0
        mismatch_count = 0
        leg_mismatches: List[Dict] = []
        all_leg_trajectories: List = []
        terminal_target_updates: List[Dict] = []
        target_convergence_legs: List[Dict] = []
        actions: List = []
        states: List = [task_state.copy() if hasattr(task_state, "copy") else task_state]
        plan_info = {"success": True, "mode": "replan_first_action"}

        max_symbolic_steps = 50
        for replan_idx in range(max_symbolic_steps):
            if task_state.is_goal():
                break

            planned_actions, planned_states, plan_info = self._plan_with_pddl(
                task_state=task_state,
                planning_weights=planning_weights,
                is_meal=is_meal,
            )

            if not plan_info.get("success", False) or not planned_actions:
                print("[FAIL] Task planning failed!")
                return {
                    "success": False,
                    "episode": self.episode_count,
                    "reason": "task_planning_failed",
                    "plan_info": plan_info,
                }

            planned_action_names = [
                planned_action.value
                if hasattr(planned_action, "value")
                else str(planned_action)
                for planned_action in planned_actions
            ]
            self._emit_episode_property_event(
                "plan_ready",
                task_type=task_type,
                actions=planned_actions,
                pddl_action_names=planned_action_names,
                plan_length=len(planned_actions),
                plan_info=dict(plan_info),
            )

            action = planned_actions[0]
            actions.append(action)
            action_name = action.value if hasattr(action, "value") else str(action)
            self._emit_episode_property_event(
                "action_start",
                action=action,
                state=self._state_copy(task_state),
                battery=float(task_state.battery_soc),
            )
            print(f"First action: {action_name}")

            if self.verbose:
                print(f"\n--- Replan {replan_idx + 1}: execute {action_name} ---")

            if is_meal:
                is_nav_action = action in MEAL_NAV_ACTIONS
                goal_loc = ACTION_TARGET_LOCATIONS.get(action) if is_nav_action else None
            else:
                is_nav_action = action in self.task_manager.action_locations
                goal_loc = (
                    self.task_manager.action_locations.get(action)
                    if is_nav_action else None
                )

            if is_nav_action and goal_loc is not None:
                start_loc          = task_state.location
                last_goal_location = goal_loc
                near_patient       = "patient" in goal_loc.lower()
                leg_count         += 1

                if self.verbose:
                    print(f"   Moving: {start_loc} -> {goal_loc}")

                leg_result = self._execute_leg(
                    start_state_6d=current_6d_state,
                    goal_location=goal_loc,
                    start_location=start_loc,
                    near_patient=near_patient,
                    action=action,
                )

                if not leg_result["success"]:
                    print(f"   [FAIL] Leg failed (error={leg_result['final_error']:.2f}m)")
                    execution_success = False
                    break

                current_6d_state  = leg_result["final_state_6d"]
                distance_traveled = leg_result["total_distance"]
                time_elapsed      = leg_result["execution_time"]
                target_update = leg_result.get("terminal_target_update")
                if target_update is not None:
                    terminal_target_updates.append(target_update)
                target_convergence_legs.append(
                    {
                        "action": action_name,
                        "goal_location": goal_loc,
                        "fuzzy_location": leg_result.get("fuzzy_location"),
                        "fuzzy_mismatch": bool(
                            leg_result.get("fuzzy_mismatch", False)
                        ),
                        "target_update_applied": bool(
                            target_update.get("update_applied", False)
                        ) if isinstance(target_update, dict) else False,
                        "target_delta_norm": (
                            float(target_update.get("target_delta_norm"))
                            if isinstance(target_update, dict)
                            and target_update.get("target_delta_norm") is not None
                            else None
                        ),
                    }
                )

                episode_features["total_distance"] += distance_traveled
                episode_features["total_time"] += time_elapsed
                episode_features["total_battery_used"] += distance_traveled * 0.01

                if is_meal:
                    task_state = self.meal_task_manager.apply_action(task_state, action)
                    task_state.distance_traveled += distance_traveled
                    task_state.time_elapsed += time_elapsed
                else:
                    task_state = self.task_manager.apply_action(
                        task_state, action, distance_traveled, time_elapsed
                    )

                if leg_result.get("fuzzy_mismatch", False):
                    mismatch_count += 1
                    leg_mismatches.append(
                        {
                            "action": action_name,
                            "goal_location": goal_loc,
                            "fuzzy_location": leg_result.get("fuzzy_location"),
                            "final_error": float(leg_result.get("final_error", 0.0)),
                            "fuzzy_goal_membership": float(
                                leg_result.get("fuzzy_goal_membership", 0.0)
                            ),
                            "fuzzy_dominant_membership": float(
                                leg_result.get("fuzzy_dominant_membership", 0.0)
                            ),
                        }
                    )
                    align_pos = np.array(self.env.locations[goal_loc], dtype=float)
                    align_yaw = float(
                        getattr(self.translator, "location_orientations", {}).get(
                            goal_loc, current_6d_state[2]
                        )
                    )
                    if hasattr(self.env, "set_robot_pose"):
                        current_6d_state = self.env.set_robot_pose(align_pos, align_yaw)
                    else:
                        current_6d_state[:2] = align_pos
                        current_6d_state[2] = align_yaw
                        current_6d_state[3:] = 0.0
                        self.env.robot_state_6d = current_6d_state.copy()
                    task_state.location = goal_loc
                    task_state.location_memberships = {goal_loc: 1.0}
                    if goal_loc == "patient_bed_left":
                        task_state.approach_side = "left"
                    elif goal_loc == "patient_bed_right":
                        task_state.approach_side = "right"
                    print(
                        f"   [ALIGN] fuzzy={leg_result.get('fuzzy_location')} "
                        f"!= planner={goal_loc}; continuing from planner location"
                    )
                else:
                    task_state = self._update_task_state_with_fuzzy(
                        task_state, current_6d_state, goal_loc
                    )

                if leg_result.get("trajectory") is not None:
                    traj_xy = leg_result["trajectory"]
                    patient_pos = self.env.locations.get(
                        "patient_bed",
                        self.env.locations.get("patient_bed_left", np.zeros(2)),
                    )
                    patient_xy = np.array(patient_pos, dtype=float)
                    dists = np.linalg.norm(traj_xy - patient_xy[None, :], axis=1)
                    episode_features["proximity_min_dists"].append(float(np.min(dists)))
                    all_leg_trajectories.append(traj_xy)

                if self.verbose:
                    print(
                        f"   Reached {goal_loc} "
                        f"(dist={distance_traveled:.1f}m, time={time_elapsed:.1f}s)"
                    )

            elif is_meal:
                duration = MEAL_ACTION_DURATIONS.get(action, 5.0)
                episode_features["total_time"] += duration

                if action in MEAL_DELIVERY_ACTIONS:
                    candidate = last_goal_location or "patient_bed_left"
                    if "patient" in candidate.lower() and candidate in self.env.locations:
                        target_xy = np.array(self.env.locations[candidate], dtype=float)
                        desired_yaw = float(
                            getattr(self.translator, "location_orientations", {}).get(
                                candidate, 0.0
                            )
                        )
                    else:
                        default_patient = self.env.locations.get(
                            "patient_bed_left",
                            self.env.locations.get("patient_bed_right", np.zeros(2)),
                        )
                        target_xy = np.array(default_patient, dtype=float)
                        desired_yaw = 0.0

                    final_xy = current_6d_state[:2]
                    pos_err = float(np.linalg.norm(final_xy - target_xy))
                    final_position_error = pos_err
                    yaw_err = self._wrap_angle(float(current_6d_state[2]) - desired_yaw)
                    pos_score = self._pos_score_from_error(pos_err, max_ok=2.0)
                    approach_quality = 0.7 * pos_score + 0.3 * self._yaw_score(yaw_err)
                    episode_features["approach_quality_scores"].append(approach_quality)

                    metrics.delivery_position_error = pos_err
                    metrics.delivery_orientation_error = abs(yaw_err)
                    metrics.approach_quality = approach_quality

                task_state = self.meal_task_manager.apply_action(task_state, action)

            else:
                duration_before = getattr(task_state, "time_elapsed", 0.0)

                if action in MEDICATION_COLLECTION_ACTIONS | SUPPLEMENT_COLLECTION_ACTIONS:
                    if hasattr(self.env, "consume_stock"):
                        self.env.consume_stock(task_state.location)

                if action in MED_DELIVERY_ACTIONS:
                    candidate = last_goal_location or "patient_bed_left"
                    if "patient" in candidate.lower() and candidate in self.env.locations:
                        target_xy = np.array(self.env.locations[candidate], dtype=float)
                        desired_yaw = float(
                            getattr(self.translator, "location_orientations", {}).get(
                                candidate, 0.0
                            )
                        )
                    else:
                        default_patient = self.env.locations.get(
                            "patient_bed_left",
                            self.env.locations.get("patient_bed_right", np.zeros(2)),
                        )
                        target_xy = np.array(default_patient, dtype=float)
                        desired_yaw = 0.0

                    final_xy = current_6d_state[:2]
                    pos_err = float(np.linalg.norm(final_xy - target_xy))
                    final_position_error = pos_err
                    yaw_err = self._wrap_angle(float(current_6d_state[2]) - desired_yaw)
                    pos_score = self._pos_score_from_error(pos_err, max_ok=2.0)
                    approach_quality = 0.7 * pos_score + 0.3 * self._yaw_score(yaw_err)
                    episode_features["approach_quality_scores"].append(approach_quality)

                    metrics.delivery_position_error = pos_err
                    metrics.delivery_orientation_error = abs(yaw_err)
                    metrics.approach_quality = approach_quality

                task_state = self.task_manager.apply_action(task_state, action)
                episode_features["total_time"] += max(
                    0.0, getattr(task_state, "time_elapsed", 0.0) - duration_before
                )

            self._emit_episode_property_event(
                "action_end",
                action=action,
                state=self._state_copy(task_state),
                battery=float(task_state.battery_soc),
            )
            states.append(task_state.copy() if hasattr(task_state, "copy") else task_state)

        if not task_state.is_goal() and execution_success:
            execution_success = False
            print("[FAIL] Episode did not reach goal within replanning limit")

        if not execution_success:
            return {"success": False, "episode": self.episode_count, "reason": "execution_failed"}

        if is_meal:
            plan_structure = self._extract_meal_plan_structure(actions, task_state)
            plan_key = self._get_meal_plan_key(plan_structure)
        else:
            plan_structure = self._extract_plan_structure(actions)
            plan_key = self._get_med_plan_key(plan_structure)
        metrics.plan_length = len(actions)

        self.plan_history.append(plan_structure)
        if self.verbose and len(self.plan_history) > 1:
            unique_plans = set()
            for p in self.plan_history:
                if p.get("task_type") == "meal":
                    unique_plans.add(self._get_meal_plan_key(p))
                else:
                    unique_plans.add(self._get_med_plan_key(p))
            print(
                f"  [Diversity] Plan: {plan_key} | "
                f"Unique plans so far: {len(unique_plans)}/{len(self.plan_history)}"
            )

        # ==============================================================
        # PHASE 3: Translator φ/Q/R updates disabled
        # ==============================================================
        if self.verbose:
            print(f"\nPHASE 3: TRANSLATOR φ/Q/R UPDATES DISABLED\n")
            print("   Terminal-target z_target updates are applied per MPC leg")

        # ==============================================================
        # PHASE 4: OUTER LOOP — Preference learning
        # ==============================================================
        if self.verbose:
            print(f"\nPHASE 4: PREFERENCE LEARNING (patient feedback)\n")

        approach_quality_val = float(
            np.mean(episode_features["approach_quality_scores"])
        ) if episode_features["approach_quality_scores"] else 0.5
        raw_approach_badness = float(np.clip(1.0 - approach_quality_val, 0.0, 1.0))
        # Approach has a narrower natural range than time/proximity, so expand it
        # before the preference update to keep presentation-focused profiles identifiable.
        scaled_approach_badness = float(np.clip(2.0 * raw_approach_badness, 0.0, 1.0))
        approach_side = getattr(task_state, "approach_side", None)
        meal_type_str = getattr(task_state, "meal_type", None) if is_meal else None

        if is_meal and _HAS_MEAL_PREP:
            delivery_error  = final_position_error if final_position_error is not None else 1.5
            normalized_features = compute_meal_features(
                total_time=episode_features["total_time"],
                total_distance=episode_features["total_distance"],
                battery_start=battery_start,
                battery_end=task_state.battery_soc,
                delivery_error=delivery_error,
                approach_quality=approach_quality_val,
                meal_type=meal_type_str,
                max_time=150.0,
                max_distance=80.0,
            )
            if self.verbose:
                print(f"  Meal features ({meal_type_str}):")
                for k, v in normalized_features.items():
                    print(f"    {k:12s}: {v:.3f}")
        else:
            visited_risks  = [self._get_risk_value(s.location) for s in states[1:]]
            avg_risk       = float(np.mean(visited_risks)) if visited_risks else 0.0
            safety_badness = float(np.clip(avg_risk / 0.60, 0.0, 1.0))
            min_patient_dist = float(
                np.min(episode_features["proximity_min_dists"])
            ) if episode_features["proximity_min_dists"] else 3.0
            proximity_badness = float(np.clip((3.0 - min_patient_dist) / (3.0 - 0.8), 0.0, 1.0))

            normalized_features = {
                "time":      float(np.clip(episode_features["total_time"] / 120.0, 0.0, 1.0)),
                "safety":    float(np.clip(safety_badness, 0.0, 1.0)),
                "battery":   float(np.clip(episode_features["total_battery_used"] / 100.0, 0.0, 1.0)),
                "proximity": float(np.clip(proximity_badness, 0.0, 1.0)),
                "approach":  scaled_approach_badness,
            }

        ratings, update_info = self.preference_learner.process_episode(normalized_features)

        # ==============================================================
        # PHASE 5: EVALUATION METRICS
        # ==============================================================
        true_weights    = self.preference_learner.true_profile.weights
        learned_weights = update_info["new_weights"]

        metrics.preference_distance      = float(update_info["distance_to_true"])
        metrics.preference_weight_change = float(
            np.linalg.norm(update_info["new_weights"] - update_info["old_weights"])
        )
        metrics.preference_converged = bool(update_info["converged"])
        metrics.dominant_correct     = int(np.argmax(learned_weights)) == int(np.argmax(true_weights))

        last_mpc_stats = self.mpc.stats
        metrics.mpc_total_solves = last_mpc_stats.get("total_solves", 0)
        if metrics.mpc_total_solves > 0:
            successful = (
                last_mpc_stats.get("acados_solves", 0)
                + last_mpc_stats.get("casadi_solves", 0)
            )
            metrics.mpc_success_rate     = 100.0 * successful / metrics.mpc_total_solves
            metrics.mpc_avg_solve_time_ms = (
                1000.0 * last_mpc_stats.get("total_solve_time", 0.0) / metrics.mpc_total_solves
            )
        metrics.mpc_sensitivity_computes = last_mpc_stats.get("sensitivity_computes", 0)
        if metrics.mpc_sensitivity_computes > 0:
            metrics.mpc_avg_sens_time_ms = (
                1000.0 * last_mpc_stats.get("total_sens_time", 0.0)
                / metrics.mpc_sensitivity_computes
            )

        metrics.num_legs       = leg_count
        metrics.total_distance = episode_features["total_distance"]
        metrics.total_time     = episode_features["total_time"]
        metrics.nav_stack_used = False

        straight_total = 0.0
        prev_loc = start_location
        for action in actions:
            goal_loc = None
            if is_meal and action in MEAL_NAV_ACTIONS:
                goal_loc = ACTION_TARGET_LOCATIONS.get(action)
            elif not is_meal and action in self.task_manager.action_locations:
                goal_loc = self.task_manager.action_locations[action]
            if goal_loc is not None:
                if prev_loc in self.env.locations and goal_loc in self.env.locations:
                    straight_total += float(
                        np.linalg.norm(
                            self.env.locations[goal_loc] - self.env.locations[prev_loc]
                        )
                    )
                prev_loc = goal_loc
        metrics.path_efficiency      = straight_total / max(metrics.total_distance, 0.01)
        battery_end                  = task_state.battery_soc
        self.env.environment_state["battery_level"] = battery_end
        metrics.battery_used_pct     = (battery_start - battery_end) * 100
        metrics.battery_remaining_pct = battery_end * 100
        metrics.features             = normalized_features
        metrics.ratings              = ratings.copy()
        self.learning_tracker.record(metrics)

        # ==============================================================
        # BUILD EPISODE SUMMARY
        # ==============================================================
        full_trajectory_xy = (
            np.concatenate(all_leg_trajectories, axis=0).tolist()
            if all_leg_trajectories else []
        )

        episode_summary = {
            "success":    True,
            "episode":    self.episode_count,
            "task_type":  task_type,
            "task_state": task_state.to_dict(),
            "plan_info":       plan_info,
            "plan_structure":  plan_structure,
            "exploration": {
                "sigma":            explore_info["sigma"],
                "explored":         explore_info["explored"],
                "planning_weights": explore_info.get(
                    "perturbed_weights", current_weights.tolist()
                ),
            },
            "features":         normalized_features,
            "ratings":          ratings,
            "weights_before":   update_info["old_weights"],
            "weights_after":    update_info["new_weights"],
            "weight_change":    metrics.preference_weight_change,
            "distance_to_true": metrics.preference_distance,
            "converged":        metrics.preference_converged,
            "dominant_correct": metrics.dominant_correct,
            "total_time":       metrics.total_time,
            "total_distance":   metrics.total_distance,
            "total_steps":      metrics.plan_length,
            "executed_actions": [
                action.value if hasattr(action, "value") else str(action)
                for action in actions
            ],
            "mpc_stats": {
                "total_steps":       metrics.mpc_total_solves,
                "successful_steps":  int(
                    metrics.mpc_success_rate / 100 * metrics.mpc_total_solves
                ),
                "failed_steps":      0,
                "avg_solve_time_ms": metrics.mpc_avg_solve_time_ms,
                "sensitivity_computes": metrics.mpc_sensitivity_computes,
                "acados_solves":     last_mpc_stats.get("acados_solves", 0),
                "casadi_solves":     last_mpc_stats.get("casadi_solves", 0),
                "total_solve_time":  last_mpc_stats.get("total_solve_time", 0.0),
                "total_sens_time":   last_mpc_stats.get("total_sens_time", 0.0),
            },
            "mismatches": {
                "count": mismatch_count,
                "leg_count": leg_count,
                "rate": mismatch_count / max(leg_count, 1),
                "legs": leg_mismatches,
            },
            "translator_learning": {
                "phi_gradient_norm": metrics.phi_gradient_norm,
                "phi_param_change":  metrics.phi_param_change,
                "sensitivity_samples": metrics.num_sensitivity_samples,
                "avg_mpc_cost":      metrics.avg_mpc_cost,
            },
            "terminal_target_updates": terminal_target_updates,
            "target_convergence_legs": target_convergence_legs,
            "battery_start":         battery_start * 100,
            "battery_remaining":     metrics.battery_remaining_pct,
            "battery_used_pct":      metrics.battery_used_pct,
            "final_position_error":  final_position_error,
            "path_efficiency":       metrics.path_efficiency,
            "approach_quality":      metrics.approach_quality,
            "approach_diagnostics": {
                "approach_quality": approach_quality_val,
                "raw_approach_badness": raw_approach_badness,
                "scaled_approach_badness": normalized_features.get("approach"),
                "approach_side": approach_side,
                "meal_type": meal_type_str,
            },
            "trajectory_xy":         full_trajectory_xy,
            "states":                full_trajectory_xy,
            "metrics":               metrics.to_dict(),
        }

        self.episode_history.append(episode_summary)
        self._print_episode_summary(episode_summary)

        if self.save_summaries and self.summary_dir:
            ep_file = self.summary_dir / f"episode_{self.episode_count:03d}.json"
            self._save_json(episode_summary, ep_file)

        return episode_summary
