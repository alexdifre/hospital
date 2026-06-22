"""
integration/system.py — FullMedicationDeliverySystem

Complete integrated system for medication delivery with preference learning and
fixed MPC translator parameters:

    OUTER LOOP: Preference Learner  (patient feedback → w on probability simplex)
    TRANSLATOR: MPC parameters are held fixed during episode execution

Composed from:
    EpisodeRunnerMixin  — _execute_leg + run_episode
    ReportingMixin      — _print_*/save methods

Callers import via the thin shim:
    from integration.integrator2 import FullMedicationDeliverySystem
or directly:
    from integration.system import FullMedicationDeliverySystem
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Core framework ────────────────────────────────────────────────────
from core.environment.env import ExpandedHospitalMuJoCoEnv
from core.execution.hybrid import HybridMPC, filter_nearby_obstacles
from core.learning.preference_learner import PATIENT_PROFILES, PreferenceLearningEngine
from core.learning.learnable_translator import ObstacleAwareTranslator
from core.task_planning.pddl_engine import (
    DEFAULT_PDDL_PLANNING_ENGINE,
    normalize_pddl_planning_engine,
)

# ── Fuzzy state estimator ─────────────────────────────────────────────
try:
    from core.planning.fuzzy_state import FuzzyStateEstimator
    HAS_FUZZY = True
    print("✓ FuzzyStateEstimator imported")
except ImportError:
    HAS_FUZZY = False
    print("⚠ FuzzyStateEstimator not found — using crisp state transitions")

# ── Task-specific components ──────────────────────────────────────────
from tasks.medication_delivery.task_planner import HighLevelTaskPlanner
from tasks.medication_delivery.task_state_manager import TaskAction, TaskStateManager

# ── Meal preparation (optional) ───────────────────────────────────────
try:
    from tasks.meal_preparation.task_state_manager import MealTaskStateManager
    HAS_MEAL_PREP = True
    print("✓ Meal preparation task imported")
except ImportError as e:
    HAS_MEAL_PREP = False
    print(f"⚠ Meal preparation task not found — meal episodes disabled ({e})")

# ── Local modules ─────────────────────────────────────────────────────
from .metrics import LearningCurveTracker
from .episode_runner import EpisodeRunnerMixin
from .reporting import ReportingMixin


class FullMedicationDeliverySystem(EpisodeRunnerMixin, ReportingMixin):
    """
    Complete integrated system for medication delivery with preference learning
    and fixed MPC translator parameters.

    OUTER LOOP: Preference Learner (patient feedback → w on simplex)
    TRANSLATOR: MPC parameters are held fixed during episode execution

    Uses HybridMPC (Acados) for control.
    Each navigation leg uses a direct 21-point waypoint reference.
    FuzzyStateEstimator bridges continuous MPC ↔ discrete task planner.
    """

    BASE_RISK_MAP = {
        "pharmacy_north":   0.30,
        "supply_B":         0.30,
        "patient_bed_left": 0.15,
        "patient_bed_right": 0.15,
        "pharmacy_south":   0.05,
        "supply_A":         0.05,
        "charge_backup":    0.08,
        "charge_main":      0.05,
        "home":             0.02,
        "pantry":           0.15,
        "fridge":           0.17,
        "prep_station":     0.30,
        "stove":            0.70,
        "quality_check":    0.10,
    }

    def __init__(
        self,
        patient_profile_name: str = "speed_oriented",
        preference_learning_rate: float = 0.1,
        translator_learning_rate: float = 0.002,
        render: bool = False,
        verbose: bool = False,
        save_summaries: bool = True,
        summary_dir: Optional[str] = None,
        use_fuzzy: bool = True,
        sensitivity_interval: int = 3,
        explore_sigma: float = 0.15,
        explore_decay: float = 0.2,
        rating_noise: float = 0.3,
        fix_translator: bool = False,
        terminal_target_learning_rate: float = 0.05,
        terminal_target_alpha_gain: float = 10.0,
        terminal_target_fd_eps: float = 0.5,
        terminal_target_max_grad_norm: float = 5.0,
        terminal_observation_offsets: Optional[Dict[str, np.ndarray]] = None,
        terminal_observation_offset_scale: float = 3.0,
        dynamic_risk_perturbation: float = 0.0,
        lr_decay: float = 0.15,
        ema_alpha: float = 0.60,
        planning_engine: str = DEFAULT_PDDL_PLANNING_ENGINE,
    ):
        self.verbose              = verbose
        self.save_summaries       = save_summaries
        self.sensitivity_interval = sensitivity_interval
        self.fix_translator       = fix_translator
        self.terminal_target_learning_enabled = True
        self.terminal_target_learning_rate = terminal_target_learning_rate
        self.terminal_target_alpha_gain = terminal_target_alpha_gain
        self.terminal_target_fd_eps = terminal_target_fd_eps
        self.terminal_target_max_grad_norm = terminal_target_max_grad_norm
        self.terminal_targets: Dict[str, np.ndarray] = {}
        self.terminal_target_update_history: List[Dict] = []
        self.terminal_observation_offset_scale = terminal_observation_offset_scale
        self.dynamic_risk_perturbation = dynamic_risk_perturbation
        self._current_risk_map    = dict(self.BASE_RISK_MAP)
        self.explore_sigma        = explore_sigma
        self.explore_decay        = explore_decay
        self.plan_history: List[Dict] = []
        self.planning_engine = normalize_pddl_planning_engine(planning_engine)

        # Summary directory
        if self.save_summaries:
            if summary_dir is None:
                timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_dir = f"summaries_v4_{patient_profile_name}_{timestamp}"
            self.summary_dir = Path(summary_dir)
            self.summary_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving summaries to: {self.summary_dir}")
        else:
            self.summary_dir = None

        if self.verbose:
            print(f"\n{'='*80}")
            print("INITIALIZING MEDICATION DELIVERY SYSTEM (v4 - Hybrid MPC + Learning)")
            print(f"{'='*80}\n")

        # 1) Environment
        if self.verbose:
            print("1) Creating MuJoCo environment...")
        self.env = ExpandedHospitalMuJoCoEnv(render_mode="human" if render else None)
        if self.verbose:
            print("   Environment ready\n")
        self.terminal_observation_offsets = self._build_terminal_observation_offsets(
            terminal_observation_offsets
        )

        # 2a) Fuzzy state estimator
        self.use_fuzzy = use_fuzzy and HAS_FUZZY
        self.fuzzy_estimator: Optional["FuzzyStateEstimator"] = None
        if self.use_fuzzy:
            if self.verbose:
                print("2a) Creating fuzzy state estimator...")
            self.fuzzy_estimator = FuzzyStateEstimator(self.env)
            if self.verbose:
                print("    Fuzzy state estimator ready\n")
        elif self.verbose:
            print("2a) Fuzzy state estimation disabled — crisp transitions\n")

        # 2b) Task state manager
        if self.verbose:
            print("2b) Creating task state manager...")
        locations = list(self.env.locations.keys())
        self.task_manager = TaskStateManager(
            self.env, locations, fuzzy_estimator=self.fuzzy_estimator
        )
        if self.verbose:
            print("    Task manager ready\n")

        # 2c) Meal preparation state manager
        self.meal_task_manager: Optional[MealTaskStateManager] = None
        if HAS_MEAL_PREP:
            if self.verbose:
                print("2c) Creating meal preparation state manager...")
            self.meal_task_manager = MealTaskStateManager(
                env=self.env, fuzzy_estimator=self.fuzzy_estimator,
            )
            if self.verbose:
                print("    Meal task manager ready\n")
        elif self.verbose:
            print("2c) Meal preparation not available\n")

        # 3) Preference learner (OUTER LOOP)
        if self.verbose:
            print("3) Creating preference learning engine (outer loop)...")
        patient_profile = PATIENT_PROFILES[patient_profile_name]
        self.preference_learner = PreferenceLearningEngine(
            true_patient_profile=patient_profile,
            learning_rate=preference_learning_rate,
            rating_noise=rating_noise,
            lr_decay=lr_decay,
            ema_alpha=ema_alpha,
        )
        if self.verbose:
            print("   Preference learner ready\n")

        # 4) Translator (MPC parameters are kept fixed during episodes)
        if self.verbose:
            print("4) Creating translator (updates disabled)...")
        initial_weights = self.preference_learner.get_current_weights()
        self.translator = ObstacleAwareTranslator(
            environment=self.env,
            initial_preference_weights=initial_weights,
        )
        self.translator_lr = translator_learning_rate
        if self.verbose:
            print("   Translator ready\n")

        # 5) HybridMPC
        if self.verbose:
            print("5) Creating HybridMPC (Acados)...")
        self.mpc = HybridMPC(horizon=20, dt=0.2, n_obstacles=3)
        if self.verbose:
            print("   HybridMPC ready\n")

        # 6) Waypoint references are generated directly per leg.
        if self.verbose:
            print("6) Direct 21-point waypoint references ready\n")

        # 7) Task planner (initialised per episode)
        if self.verbose:
            print("7) Task planner ready (initialized per episode)\n")
            print(f"   PDDL engine: {self.planning_engine}")
        self.task_planner: Optional[HighLevelTaskPlanner] = None

        # State
        self.episode_count      = 0
        self.meal_episode_count = 0
        self.episode_history: List[Dict] = []
        self.learning_tracker = LearningCurveTracker()

        if self.verbose:
            print(f"{'='*80}")
            print("SYSTEM INITIALIZATION COMPLETE")
            print(f"  Outer loop: Preference learner (lr={preference_learning_rate})")
            print("  Translator φ/Q/R updates: disabled")
            print(
                "  Terminal target learning: "
                f"lr={terminal_target_learning_rate}, "
                f"alpha_gain={terminal_target_alpha_gain}, "
                f"fd_eps={terminal_target_fd_eps}"
            )
            print(
                "  Terminal observation offsets: "
                f"{len(self.terminal_observation_offsets)} locations"
            )
            print(f"  MPC: HybridMPC (Acados)")
            print("  Waypoint refs: direct 21-point")
            print(f"  Fuzzy state: {'enabled' if self.use_fuzzy else 'crisp'}")
            print("  Stage-cost sensitivity collection: disabled")
            print(f"  Rating noise: {rating_noise}")
            if self.fix_translator:
                print(f"  ⚠ Translator fixed flag set (updates already disabled)")
            if self.dynamic_risk_perturbation > 0:
                print(f"  ⚠ Dynamic risk: ±{self.dynamic_risk_perturbation:.0%} per episode")
            if self.explore_sigma > 0:
                print(
                    f"  Exploration: σ₀={self.explore_sigma}, "
                    f"decay={self.explore_decay} (Thompson sampling on weights)"
                )
            else:
                print(f"  Exploration: disabled")
            print(f"{'='*80}\n")

    # -----------------------------------------------------------------
    # Geometry helpers (static)
    # -----------------------------------------------------------------

    def _build_terminal_observation_offsets(
        self,
        explicit_offsets: Optional[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        if explicit_offsets is not None:
            return {
                str(name): np.array(offset, dtype=float).reshape(2)
                for name, offset in explicit_offsets.items()
            }

        locations = sorted(getattr(self.env, "locations", {}).keys())
        if not locations:
            return {}

        scale = float(self.terminal_observation_offset_scale)
        offsets: Dict[str, np.ndarray] = {}
        for idx, name in enumerate(locations):
            x_sign = -1.0 if idx % 2 else 1.0
            y_sign = -1.0 if (idx // 2) % 2 else 1.0
            x_mag = scale * (0.75 + 0.05 * (idx % 11))
            y_mag = scale * (0.75 + 0.05 * ((idx * 5 + 3) % 11))
            offsets[name] = np.array(
                [x_sign * x_mag, y_sign * y_mag],
                dtype=float,
            )
        return offsets

    @staticmethod
    def _wrap_angle(rad: float) -> float:
        return float(np.arctan2(np.sin(rad), np.cos(rad)))

    @staticmethod
    def _pos_score_from_error(pos_err: float, max_ok: float = 2.0) -> float:
        return float(1.0 - min(max(pos_err, 0.0) / max_ok, 1.0))

    @staticmethod
    def _yaw_score(yaw_err: float) -> float:
        return float(1.0 - min(abs(float(yaw_err)) / np.pi, 1.0))

    # -----------------------------------------------------------------
    # Risk map helpers
    # -----------------------------------------------------------------

    def _get_risk_value(self, location: str) -> float:
        return self._current_risk_map.get(location, 0.10)

    def _perturb_risk_map(self):
        """Perturb risk values ±dynamic_risk_perturbation each episode."""
        if self.dynamic_risk_perturbation <= 0:
            return
        for loc, base_risk in self.BASE_RISK_MAP.items():
            perturbation = np.random.uniform(
                -self.dynamic_risk_perturbation, self.dynamic_risk_perturbation
            )
            self._current_risk_map[loc] = float(
                np.clip(base_risk * (1.0 + perturbation), 0.0, 1.0)
            )
        if self.verbose:
            changed = {
                k: f"{self.BASE_RISK_MAP[k]:.2f}→{v:.2f}"
                for k, v in self._current_risk_map.items()
                if abs(v - self.BASE_RISK_MAP[k]) > 0.005
            }
            if changed:
                print(f"  [DynRisk] Perturbed: {changed}")

    # -----------------------------------------------------------------
    # Plan structure helpers
    # -----------------------------------------------------------------

    def _extract_plan_structure(self, actions: List[TaskAction]) -> Dict:
        structure = {
            "task_type": "medication", "pharmacy_choice": None,
            "supply_choice": None, "approach_choice": None,
            "recharge_added": False, "plan_length": len(actions),
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

    def _extract_meal_plan_structure(self, actions: List, final_state) -> Dict:
        return {
            "task_type":       "meal",
            "meal_type":       getattr(final_state, "meal_type", None),
            "approach_choice": getattr(final_state, "approach_side", None),
            "plan_length":     len(actions),
        }

    def _get_meal_plan_key(self, structure: Dict) -> Tuple:
        return ("meal", structure.get("meal_type"), structure.get("approach_choice"))

    def _get_med_plan_key(self, structure: Dict) -> Tuple:
        return (
            "med",
            structure.get("pharmacy_choice"),
            structure.get("supply_choice"),
            structure.get("approach_choice"),
        )

    # -----------------------------------------------------------------
    # Exploration helper
    # -----------------------------------------------------------------

    def _perturb_weights_for_exploration(
        self, weights: np.ndarray, episode: int,
    ) -> Tuple[np.ndarray, Dict]:
        sigma_eff = self.explore_sigma / (1.0 + self.explore_decay * episode)
        if sigma_eff < 1e-4:
            return weights.copy(), {"explored": False, "sigma": 0.0}
        noise   = np.random.normal(0.0, sigma_eff, size=weights.shape)
        w_noisy = np.maximum(weights + noise, 0.01)
        w_noisy = w_noisy / w_noisy.sum()
        return w_noisy, {
            "explored":         True,
            "sigma":            float(sigma_eff),
            "noise":            noise.tolist(),
            "original_weights": weights.tolist(),
            "perturbed_weights": w_noisy.tolist(),
        }

    # -----------------------------------------------------------------
    # Obstacle helper
    # -----------------------------------------------------------------

    def _get_obstacles_for_leg(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        exclude_locations: List[str],
        goal_clearance: float = 4.0,
        start_clearance: float = 3.0,
    ) -> List[Dict]:
        obstacles = []
        location_sizes = getattr(self.translator, "default_location_sizes", {})
        s2d = start_pos[:2] if len(start_pos) > 2 else start_pos
        g2d = goal_pos[:2]  if len(goal_pos)  > 2 else goal_pos

        for name, pos in self.env.locations.items():
            if name in exclude_locations:
                continue
            radius        = location_sizes.get(name, 0.8)
            dist_to_goal  = float(np.linalg.norm(pos[:2] - g2d))
            dist_to_start = float(np.linalg.norm(pos[:2] - s2d))
            if dist_to_goal < goal_clearance + radius:
                if self.verbose:
                    print(f"    [Obs] Excluding {name} (dist_to_goal={dist_to_goal:.1f}m)")
                continue
            if dist_to_start < start_clearance + radius:
                if self.verbose:
                    print(f"    [Obs] Excluding {name} (dist_to_start={dist_to_start:.1f}m)")
                continue
            obstacles.append({"x": float(pos[0]), "y": float(pos[1]),
                               "radius": float(radius), "name": name})

        return filter_nearby_obstacles(
            robot_pos=s2d, goal_pos=g2d, obstacles=obstacles, max_obstacles=3,
        )

    # -----------------------------------------------------------------
    # Fuzzy state update helper
    # -----------------------------------------------------------------

    def _pddl_location_names(self) -> Optional[set]:
        problem = getattr(self, "problem", None)
        if problem is None:
            return None
        try:
            location_type = problem.user_type("location")
            return {str(location) for location in problem.objects(location_type)}
        except Exception:
            return None

    def _update_task_state_with_fuzzy(self, task_state, current_6d_state, goal_location, battery_predicted=None):
        del battery_predicted
        if self.fuzzy_estimator is not None:
            fm = self.fuzzy_estimator.estimate(current_6d_state[:2], task_state.battery_soc)
            memberships = dict(fm.location_memberships)
            pddl_locations = self._pddl_location_names()
            if pddl_locations is not None:
                pddl_locations_lower = {location.lower() for location in pddl_locations}
                memberships = {
                    location: membership
                    for location, membership in memberships.items()
                    if location in pddl_locations or location.lower() in pddl_locations_lower
                }
            task_state.location_memberships = memberships

            if memberships:
                dominant_location, dominant_membership = max(
                    memberships.items(), key=lambda item: item[1]
                )
            else:
                dominant_location, dominant_membership = goal_location, 0.0

            if dominant_membership > 0.0:
                task_state.location = dominant_location
            else:
                task_state.location = "in_transit"
                task_state.location_memberships = None
            if self.verbose:
                print(f"    {fm.summary()}")
        else:
            task_state.location             = goal_location
            task_state.location_memberships = None

        return task_state

    # -----------------------------------------------------------------
    # Multi-episode orchestration
    # -----------------------------------------------------------------

    def run_multiple_episodes(
        self,
        num_episodes: int = 3,
        start_location: str = "home",
        add_variability: bool = True,
        task_type: str = "medication",
    ) -> List[Dict]:
        print(f"\n{'='*80}")
        print(
            f"RUNNING {num_episodes} LEARNING EPISODES [{task_type.upper()}]"
            f" (v4 - terminal-target learning)"
        )
        print(f"{'='*80}\n")

        results: List[Dict] = []
        for ep in range(num_episodes):
            try:
                if add_variability:
                    if task_type == "meal":
                        starts = ["home", "pantry", "charge_main", "home"]
                    else:
                        starts = ["home", "pharmacy_north", "supply_A", "charge_main"]
                    current_start = starts[ep % len(starts)]
                    battery_cycle = [0.4, 0.6, 0.8, 1.0, 0.9]
                    self.env.environment_state["battery_level"] = battery_cycle[
                        ep % len(battery_cycle)
                    ]
                else:
                    current_start = start_location

                summary = self.run_episode(start_location=current_start, task_type=task_type)
                results.append(summary)

                if not summary.get("success", False):
                    print(
                        f"[FAIL] Episode {ep+1}/{num_episodes}: {summary.get('reason')}"
                    )
            except Exception as e:
                print(f"[CRASH] Episode {ep+1}: {e}")
                import traceback; traceback.print_exc()

        self._print_final_summary(results)
        return results

    def run_mixed_episodes(
        self,
        num_episodes: int = 10,
        start_location: str = "home",
        task_roster: Optional[List[str]] = None,
        add_variability: bool = True,
    ) -> List[Dict]:
        if task_roster is None:
            task_roster = ["medication", "meal"] * (num_episodes // 2 + 1)
        task_roster = task_roster[:num_episodes]

        print(f"\n{'='*80}")
        print(f"RUNNING {num_episodes} MIXED EPISODES (medication + meal prep)")
        print(f"{'='*80}")
        print(f"Roster: {task_roster}\n")

        results: List[Dict] = []
        med_starts    = ["home", "pharmacy_north", "supply_A", "charge_main"]
        meal_starts   = ["home", "pantry", "charge_main", "home"]
        battery_cycle = [0.6, 0.8, 1.0, 0.9, 0.7]

        for ep in range(num_episodes):
            task_type = task_roster[ep]
            try:
                if add_variability:
                    current_start = (
                        meal_starts[ep % len(meal_starts)]
                        if task_type == "meal"
                        else med_starts[ep % len(med_starts)]
                    )
                    self.env.environment_state["battery_level"] = battery_cycle[
                        ep % len(battery_cycle)
                    ]
                else:
                    current_start = start_location

                summary = self.run_episode(start_location=current_start, task_type=task_type)
                results.append(summary)

                if not summary.get("success", False):
                    print(
                        f"[FAIL] Episode {ep+1}/{num_episodes} [{task_type}]: "
                        f"{summary.get('reason')}"
                    )
            except Exception as e:
                print(f"[CRASH] Episode {ep+1} [{task_type}]: {e}")
                import traceback; traceback.print_exc()

        self._print_final_summary(results)
        return results

    def close(self):
        self.env.close()


# ── Smoke test ────────────────────────────────────────────────────────

def test_full_system():
    print("=" * 80)
    print("FULL MEDICATION DELIVERY SYSTEM TEST (v4 - terminal-target learning)")
    print("=" * 80)

    system = FullMedicationDeliverySystem(
        patient_profile_name="safety_first",
        preference_learning_rate=0.12,
        translator_learning_rate=0.002,
        render=False,
        verbose=True,
        save_summaries=True,
        use_fuzzy=True,
        sensitivity_interval=3,
        explore_sigma=0.15,
        explore_decay=0.2,
    )

    results = system.run_multiple_episodes(
        num_episodes=15, start_location="home", add_variability=True,
    )

    system.visualize_learning(save_path="learning_v4.png")

    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")

    successful = [r for r in results if r.get("success", False)]
    print(f"Successful episodes: {len(successful)}/{len(results)}")

    if successful:
        final = successful[-1]
        print(f"Final distance to w*: {final['distance_to_true']:.4f}")
        print(f"Converged: {'YES ✓' if final['converged'] else 'NO'}")
        print(
            f"Dominant preference correct: "
            f"{'YES ✓' if final.get('dominant_correct') else 'NO'}"
        )
        tl = final.get("translator_learning", {})
        print(f"Final ||∂J/∂φ||: {tl.get('phi_gradient_norm', 0):.4f}")
        print(f"Final avg MPC cost: {tl.get('avg_mpc_cost', 0):.1f}")

    if system.plan_history:
        med_plans = [p for p in system.plan_history if p.get("task_type") != "meal"]
        if med_plans:
            unique = len(set(
                (p["pharmacy_choice"], p["supply_choice"], p["approach_choice"])
                for p in med_plans
            ))
            print(f"Plan diversity: {unique} unique plans / {len(med_plans)} med episodes")

    if system.summary_dir:
        print(f"\nSummaries saved to: {system.summary_dir}")

    system.close()
    print("\n✓ Full system test complete!")


if __name__ == "__main__":
    test_full_system()
