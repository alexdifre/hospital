"""
integration/episode_runner.py — EpisodeRunnerMixin for FullMedicationDeliverySystem.

Contains the two hot-path methods:
    _execute_leg   — one navigation leg: nav stack → waypoints → HybridMPC + sensitivities
    run_episode    — full 5-phase episode (plan → execute → inner loop → outer loop → metrics)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .metrics import EpisodeMetrics
from integration.integrator2 import FullMedicationDeliverySystem

# ── Meal-prep task imports (optional) ────────────────────────────────
try:
    from tasks.meal_preparation.task_actions import (
        MealAction,
        ACTION_TARGET_LOCATIONS,
        NAVIGATION_ACTIONS as MEAL_NAV_ACTIONS,
        IN_PLACE_ACTIONS as MEAL_IN_PLACE_ACTIONS,
    )
    from tasks.meal_preparation.task_planner import MealTaskPlanner
    from tasks.meal_preparation.meal_profiles import compute_meal_features

    _HAS_MEAL_PREP = True
except ImportError:
    _HAS_MEAL_PREP = False


class EpisodeRunnerMixin:
    """Core execution methods for FullMedicationDeliverySystem."""

    # -----------------------------------------------------------------
    # Leg execution: nav stack → waypoints → HybridMPC + sensitivities
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
            Q_diag, R_diag, horizon, _  = self.translator.get_mpc_params(near_patient)
            
            
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
            z_target = np.zeros(2)

        if self.verbose:
            print(
                f"    [Translator] Q_pos={Q_diag[0]:.1f}, Q_vel={Q_diag[3]:.1f}, "
                f"R={R_diag[0]:.2f}, horizon={horizon}"
            )

        # Step 2: Plan waypoints
        waypoints = [goal_pos]
        nav_used  = False

        if self.use_nav_stack and self.nav_stack is not None:
            success = self.nav_stack.plan_to(
                start=start_pos, goal=goal_pos, arrival_tolerance=1.0,
            )
            if success and self.nav_stack.waypoint_manager is not None:
                raw_wps = self.nav_stack.waypoint_manager.waypoints
                if len(raw_wps) > 0:
                    waypoints = [np.array(wp, dtype=float) for wp in raw_wps]
                    nav_used  = True
                    if self.verbose:
                        print(f"    [NavStack] {len(waypoints)} waypoints planned")

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
        fd_dJ_dQ_samples: List = []
        fd_dJ_dR_samples: List = []

        for wp_idx, wp_target in enumerate(waypoints):
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

                if step % self.sensitivity_interval == 0:
                    if self.use_finite_diff:
                        dJ_dQ_fd, dJ_dR_fd, J_fd = self._compute_finite_diff_sensitivities(
                            current_state, x_ref, Q_diag, R_diag, obstacles
                        )
                        if J_fd > 0:
                            fd_dJ_dQ_samples.append(dJ_dQ_fd)
                            fd_dJ_dR_samples.append(dJ_dR_fd)
                        sol = self.mpc.solve(current_state, x_ref)
                    else:
                        sol, _ = self.mpc.solve_with_sensitivities(current_state, x_ref)
                else:
                    sol = self.mpc.solve(current_state, x_ref)

                if not sol.success:
                    if self.verbose:
                        print(f"    [MPC] Solve failed at step {step}")
                    control = np.zeros(3)
                else:
                    control = sol.control
                    total_cost += sol.cost
                    cost_count += 1

                self.env.step(control)
                current_state = self.env.robot_state_6d.copy()
                trajectory.append(current_state[:2].copy())
                step += 1

        # Step 4: Aggregate sensitivities
        if self.use_finite_diff and fd_dJ_dQ_samples:
            dJ_dQ_avg = np.mean(fd_dJ_dQ_samples, axis=0)
            dJ_dR_avg = np.mean(fd_dJ_dR_samples, axis=0)
            dJ_dz_target_avg = np.zeros(2)
            n_sens    = len(fd_dJ_dQ_samples)
        else:
            dJ_dQ_avg, dJ_dR_avg, dJ_dz_target_avg = self.mpc.get_aggregated_sensitivities()
            n_sens = len([s for s in self.mpc.episode_sensitivities if s.success])

        traj_array = np.array(trajectory)
        if len(traj_array) > 1:
            diffs          = np.linalg.norm(traj_array[1:] - traj_array[:-1], axis=1)
            total_distance = float(np.sum(diffs))
        else:
            total_distance = 0.0

        straight_line = float(np.linalg.norm(goal_pos - start_pos))
        final_error   = float(np.linalg.norm(current_state[:2] - goal_pos))
        avg_cost      = total_cost / max(cost_count, 1)

        result = {
            "success":               final_error < 3.0,
            "final_state_6d":        current_state.copy(),
            "trajectory":            traj_array,
            "total_distance":        total_distance,
            "straight_line_distance": straight_line,
            "path_efficiency":       straight_line / max(total_distance, 0.01),
            "final_error":           final_error,
            "steps":                 step,
            "execution_time":        step * 0.2,
            "avg_mpc_cost":          avg_cost,
            "dJ_dQ_avg":             dJ_dQ_avg,
            "dJ_dR_avg":             dJ_dR_avg,
            "dJ_dz_target_avg":      dJ_dz_target_avg,
            "num_sensitivities":     n_sens,
            "nav_stack_used":        nav_used,
            "mpc_stats":             dict(self.mpc.stats),
        }

        if self.verbose:
            sens_method = "FD" if self.use_finite_diff else "IFT"
            print(
                f"    [Leg] {'✓' if result['success'] else '✗'} "
                f"dist={total_distance:.1f}m, error={final_error:.2f}m, "
                f"steps={step}, cost={avg_cost:.1f}, sens={n_sens} ({sens_method})"
            )

        return result

    # -----------------------------------------------------------------
    # Full episode (5 phases)
    # -----------------------------------------------------------------

    def run_episode(
        self,  total_available_actions: List[FullMedicationDeliverySystem], start_location: str = "home", task_type: str = "medication"
    ) -> Dict:
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
        initial_pos      = self.env.locations[start_location]
        initial_state_6d = self.env.reset(initial_position=initial_pos)

        if is_meal:
            task_state = self.meal_task_manager.get_initial_state(start_location)
        else:
            task_state = self.task_manager.get_initial_state(start_location)

        battery_start        = self.env.environment_state["battery_level"]
        
        task_state.battery_soc = battery_start

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
        self.translator.update_preference_weights(current_weights)

        planning_weights, explore_info = self._perturb_weights_for_exploration(
            current_weights, self.episode_count,
        )
        if explore_info["explored"] and self.verbose:
            print(f"  [Explore] σ={explore_info['sigma']:.4f}")
            print(f"    Learned:  [{', '.join(f'{w:.3f}' for w in current_weights)}]")
            print(f"    Planning: [{', '.join(f'{w:.3f}' for w in planning_weights)}]")





        
        '''  
        if is_meal:
            meal_planner = MealTaskPlanner(
                task_state_manager=self.meal_task_manager,
                preference_weights=planning_weights,
                fuzzy_estimator=self.fuzzy_estimator,
            )
            actions, states, plan_info = meal_planner.plan(
                initial_state=task_state, verbose=self.verbose,
            )
        else:
            from tasks.medication_delivery.task_planner import HighLevelTaskPlanner
            self.task_planner = HighLevelTaskPlanner(
                task_state_manager=self.task_manager,
                spatial_planner=None,
                preference_weights=planning_weights,
                fuzzy_estimator=self.fuzzy_estimator,
            )
            actions, states, plan_info = self.task_planner.plan(
                initial_state=task_state, verbose=self.verbose,
            )

        if not plan_info.get("success", False):
            print("[FAIL] Task planning failed!")
            return {"success": False, "episode": self.episode_count, "reason": "task_planning_failed"}

        if self.verbose:
            if is_meal:
                meal_planner.print_plan(actions, states)
            else:
                self.task_planner.print_plan(actions, states)

        if is_meal:
            final_meal_state = states[-1] if states else task_state
            plan_structure   = self._extract_meal_plan_structure(actions, final_meal_state)
            plan_key         = self._get_meal_plan_key(plan_structure)
        else:
            plan_structure = self._extract_plan_structure(actions)
            plan_key       = self._get_med_plan_key(plan_structure)
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
            '''  
        # ==============================================================
        # PHASE 2: EXECUTE PLAN
        # ==============================================================
        if self.verbose:
            print(f"\nPHASE 2: EXECUTE PLAN (HybridMPC + sensitivities)\n")

        episode_features = {
            "total_time": 0.0, "total_distance": 0.0, "total_battery_used": 0.0,
            "proximity_min_dists": [], "approach_quality_scores": [],
        }
        all_dJ_dQ: List = []
        all_dJ_dR: List = []
        all_dJ_dz_target: List = []
        all_mpc_costs: List = []
        total_sensitivity_samples = 0
        execution_success = True
        current_6d_state  = initial_state_6d.copy()
        last_goal_location: Optional[str] = None
        final_position_error = None
        leg_count = 0
        all_leg_trajectories: List = []

        phi_before = None
        if hasattr(self.translator, "params") and hasattr(
            self.translator.params, "to_vector"
        ):
            phi_before = self.translator.params.to_vector().copy()
            
            
            
            
            
            



        from unified_planning.io import PDDLReader
        from unified_planning.shortcuts import OneshotPlanner, get_environment

        get_environment().credits_stream = None

        if is_meal:
            domain_file = "domain_meal.pddl"
            problem_file = "problem_meal.pddl"
        else:
            domain_file = "domain_medication.pddl"
            problem_file = "problem_medication.pddl"

        reader = PDDLReader()
        problem = reader.parse_problem(domain_file, problem_file)

        battery_start = float(self.env.environment_state["battery_level"])
        battery_level = problem.fluent("battery_level")
        problem.set_initial_value(battery_level, battery_start)

        # 5) solve
        with OneshotPlanner(name="enhsp") as planner:
            result = planner.solve(problem)

        if result.plan is None:
            raise RuntimeError(f"No plan found. Status: {result.status}")

        first_action_name = result.plan.actions[0].action.name
        print(f"First action: {first_action_name}")

        if is_meal:
            action = MealAction(first_action_name)
        else:
            action = TaskAction(first_action_name)
            
            
            
            
            
            
            
    #for step_idx, (action, target_state) in enumerate(zip(actions, states[1:]), 1):
        #if self.verbose:
        #    action_name = action.value if hasattr(action, "value") else str(action)
        #    print(f"\n--- Step {step_idx}/{len(actions)}: {action_name} ---")

        if is_meal:
            is_nav_action = action in MEAL_NAV_ACTIONS
            goal_loc = ACTION_TARGET_LOCATIONS.get(action) if is_nav_action else None
        else:
            is_nav_action = action in self.task_manager.action_locations
            goal_loc = (
                self.task_manager.action_locations.get(action)
                if is_nav_action else None
            )

        # === MOVEMENT ACTIONS ===
        if is_nav_action and goal_loc is not None:
            start_loc          = task_state.location
            last_goal_location = goal_loc
            near_patient       = "patient" in goal_loc.lower()
            leg_count         += 1

            if self.verbose:
                print(f"   Moving: {start_loc} → {goal_loc}")

            leg_result = self._execute_leg(
                start_state_6d=current_6d_state,
                goal_location=goal_loc,
                start_location=start_loc,
                near_patient=near_patient,
                action = action
            )

            if not leg_result["success"]:
                print(f"   [FAIL] Leg failed (error={leg_result['final_error']:.2f}m)")
                execution_success = False
                break

            current_6d_state  = leg_result["final_state_6d"]
            distance_traveled = leg_result["total_distance"]
            time_elapsed      = leg_result["execution_time"]

            if leg_result["num_sensitivities"] > 0:
                all_dJ_dQ.append(leg_result["dJ_dQ_avg"])
                all_dJ_dR.append(leg_result["dJ_dR_avg"])
                all_dJ_dz_target.append(leg_result["dJ_dz_target_avg"])
                total_sensitivity_samples += leg_result["num_sensitivities"]
            all_mpc_costs.append(leg_result["avg_mpc_cost"])

            episode_features["total_distance"]    += distance_traveled
            episode_features["total_time"]        += time_elapsed
            episode_features["total_battery_used"] += distance_traveled * 0.01

            if is_meal:
                task_state = self.meal_task_manager.apply_action(task_state, action)
                task_state.distance_traveled += distance_traveled
                task_state.time_elapsed      += time_elapsed
                task_state.battery_soc        = max(
                    0.0, task_state.battery_soc - distance_traveled * 0.01
                )
            else:
                task_state = self.task_manager.apply_action(
                    task_state, action, distance_traveled, time_elapsed
                )

            task_state = self._update_task_state_with_fuzzy(
                task_state, current_6d_state, goal_loc
            )

            if leg_result.get("trajectory") is not None:
                traj_xy    = leg_result["trajectory"]
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

        # === NON-MOVEMENT ACTIONS (meal) ===
        elif is_meal:
            from tasks.meal_preparation.task_state_manager import (
                ACTION_DURATIONS as MEAL_DURATIONS,
            )
            duration = MEAL_DURATIONS.get(action, 5.0)
            episode_features["total_time"] += duration

            if action in (
                MealAction.COLLECT_SANDWICH_INGREDIENTS,
                MealAction.COLLECT_SOUP_INGREDIENTS,
                MealAction.COLLECT_MEAL_INGREDIENTS,
            ):
                meal_stock_map = {
                    MealAction.COLLECT_SANDWICH_INGREDIENTS: "pantry_sandwich",
                    MealAction.COLLECT_SOUP_INGREDIENTS:     "pantry_soup",
                    MealAction.COLLECT_MEAL_INGREDIENTS:     "pantry_full_meal",
                }
                stock_key = meal_stock_map[action]
                if hasattr(self.env, "consume_stock"):
                    self.env.consume_stock(stock_key)
                if self.verbose:
                    remaining = (
                        self.env.get_stock(stock_key)
                        if hasattr(self.env, "get_stock") else "?"
                    )
                    print(f"   Collected ingredients ({stock_key}, remaining: {remaining})")

            elif action == MealAction.DELIVER_MEAL:
                candidate = last_goal_location or "patient_bed"
                if "patient" in candidate.lower() and candidate in self.env.locations:
                    target_xy   = np.array(self.env.locations[candidate], dtype=float)
                    desired_yaw = float(
                        getattr(self.translator, "location_orientations", {}).get(
                            candidate, 0.0
                        )
                    )
                else:
                    default_patient = self.env.locations.get(
                        "patient_bed",
                        self.env.locations.get("patient_bed_left", np.zeros(2)),
                    )
                    target_xy   = np.array(default_patient, dtype=float)
                    desired_yaw = 0.0

                final_xy             = current_6d_state[:2]
                pos_err              = float(np.linalg.norm(final_xy - target_xy))
                final_position_error = pos_err
                yaw_err              = self._wrap_angle(float(current_6d_state[2]) - desired_yaw)
                pos_score            = self._pos_score_from_error(pos_err, max_ok=2.0)
                approach_quality     = 0.7 * pos_score + 0.3 * self._yaw_score(yaw_err)
                episode_features["approach_quality_scores"].append(approach_quality)

                metrics.delivery_position_error    = pos_err
                metrics.delivery_orientation_error = abs(yaw_err)
                metrics.approach_quality           = approach_quality

                if self.verbose:
                    meal_type = getattr(task_state, "meal_type", "?")
                    print(
                        f"   Delivered {meal_type}! pos_err={pos_err:.3f}m, "
                        f"approach={approach_quality:.2f}"
                    )

            elif action == MealAction.RECHARGE:
                task_state.battery_soc = min(1.0, task_state.battery_soc + 0.4)
            else:
                if self.verbose:
                    print(f"   {action.value} complete ({duration:.0f}s)")

            task_state = self.meal_task_manager.apply_action(task_state, action)

        # === NON-MOVEMENT ACTIONS (medication) ===
        else:
            from tasks.medication_delivery.task_state_manager import TaskAction
            if action == TaskAction.COLLECT_MEDICATION:
                task_state.has_medication = True
                episode_features["total_time"] += 5.0
                if hasattr(self.env, "consume_stock"):
                    loc = task_state.location
                    self.env.consume_stock(loc)
                    if self.verbose:
                        remaining = self.env.get_stock(loc)
                        print(f"   Collected medication at {loc} (stock: {remaining} remaining)")
            elif action == TaskAction.COLLECT_SUPPLEMENT:
                task_state.has_supplement = True
                episode_features["total_time"] += 5.0
                if hasattr(self.env, "consume_stock"):
                    loc = task_state.location
                    self.env.consume_stock(loc)
                    if self.verbose:
                        remaining = self.env.get_stock(loc)
                        print(f"   Collected supplement at {loc} (stock: {remaining} remaining)")
            elif action == TaskAction.RECHARGE:
                task_state.battery_soc = 1.0
                episode_features["total_time"] += 30.0
            elif action == TaskAction.DELIVER:
                task_state.delivered = True
                episode_features["total_time"] += 10.0

                candidate = last_goal_location or "patient_bed"
                if "patient" in candidate.lower() and candidate in self.env.locations:
                    target_xy   = np.array(self.env.locations[candidate], dtype=float)
                    desired_yaw = float(
                        getattr(self.translator, "location_orientations", {}).get(
                            candidate, 0.0
                        )
                    )
                else:
                    default_patient = self.env.locations.get(
                        "patient_bed",
                        self.env.locations.get("patient_bed_left", np.zeros(2)),
                    )
                    target_xy   = np.array(default_patient, dtype=float)
                    desired_yaw = 0.0

                final_xy             = current_6d_state[:2]
                pos_err              = float(np.linalg.norm(final_xy - target_xy))
                final_position_error = pos_err
                yaw_err              = self._wrap_angle(float(current_6d_state[2]) - desired_yaw)
                pos_score            = self._pos_score_from_error(pos_err, max_ok=2.0)
                approach_quality     = 0.7 * pos_score + 0.3 * self._yaw_score(yaw_err)
                episode_features["approach_quality_scores"].append(approach_quality)

                metrics.delivery_position_error    = pos_err
                metrics.delivery_orientation_error = abs(yaw_err)
                metrics.approach_quality           = approach_quality

                if self.verbose:
                    print(
                        f"   Delivered! pos_err={pos_err:.3f}m, "
                        f"approach={approach_quality:.2f}"
                    )

        if not execution_success:
            return {"success": False, "episode": self.episode_count, "reason": "execution_failed"}

        # ==============================================================
        # PHASE 3: INNER LOOP — Translator φ update
        # ==============================================================
        if self.verbose:
            print(f"\nPHASE 3: TRANSLATOR φ LEARNING\n")

        if self.fix_translator:
            if self.verbose:
                print("   [FIXED] Translator updates disabled (fix_translator=True)")
        elif all_dJ_dQ and hasattr(self.translator, "update_parameters"):
            dJ_dQ_episode = np.mean(all_dJ_dQ, axis=0)
            dJ_dR_episode = np.mean(all_dJ_dR, axis=0)
            dJ_dz_target_episode = np.mean(all_dJ_dz_target, axis=0) if all_dJ_dz_target else np.zeros(2)
            avg_cost      = float(np.mean(all_mpc_costs)) if all_mpc_costs else 0.0

            try:
                self.translator.update_parameters(
                    dJ_dQ=dJ_dQ_episode, dJ_dR=dJ_dR_episode, cost=avg_cost,
                    dJ_dz_target=dJ_dz_target_episode,
                )
            except TypeError:
                self.translator.update_parameters(
                    dJ_dQ=dJ_dQ_episode, dJ_dR=dJ_dR_episode,
                )

            metrics.num_sensitivity_samples = total_sensitivity_samples
            metrics.avg_mpc_cost            = avg_cost
            metrics.phi_gradient_norm       = float(
                np.linalg.norm(np.concatenate([dJ_dQ_episode, dJ_dR_episode, dJ_dz_target_episode]))
            )

            if self.verbose:
                sens_method = "FD" if self.use_finite_diff else "IFT"
                print(f"   [{sens_method}] dJ/dQ_avg: {dJ_dQ_episode}")
                print(f"   [{sens_method}] dJ/dR_avg: {dJ_dR_episode}")
                print(f"   ||gradient||: {metrics.phi_gradient_norm:.4f}")
                print(f"   Sensitivity samples: {total_sensitivity_samples}")
        else:
            if self.verbose:
                print("   No sensitivities collected — φ not updated")

        if phi_before is not None and hasattr(self.translator, "params"):
            phi_after             = self.translator.params.to_vector()
            metrics.phi_param_change = float(np.linalg.norm(phi_after - phi_before))

        # ==============================================================
        # PHASE 4: OUTER LOOP — Preference learning
        # ==============================================================
        if self.verbose:
            print(f"\nPHASE 4: PREFERENCE LEARNING (patient feedback)\n")

        if is_meal and _HAS_MEAL_PREP:
            approach_quality_val = float(
                np.mean(episode_features["approach_quality_scores"])
            ) if episode_features["approach_quality_scores"] else 0.5
            delivery_error  = final_position_error if final_position_error is not None else 1.5
            meal_type_str   = getattr(task_state, "meal_type", None)
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
            approach_badness  = 1.0 - float(
                np.mean(episode_features["approach_quality_scores"])
            ) if episode_features["approach_quality_scores"] else 0.5

            normalized_features = {
                "time":      float(np.clip(episode_features["total_time"] / 120.0, 0.0, 1.0)),
                "safety":    float(np.clip(safety_badness, 0.0, 1.0)),
                "battery":   float(np.clip(episode_features["total_battery_used"] / 100.0, 0.0, 1.0)),
                "proximity": float(np.clip(proximity_badness, 0.0, 1.0)),
                "approach":  float(np.clip(approach_badness, 0.0, 1.0)),
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
        metrics.nav_stack_used = self.use_nav_stack

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
            "mpc_stats": {
                "total_steps":       metrics.mpc_total_solves,
                "successful_steps":  int(
                    metrics.mpc_success_rate / 100 * metrics.mpc_total_solves
                ),
                "failed_steps":      0,
                "avg_solve_time_ms": metrics.mpc_avg_solve_time_ms,
                "sensitivity_computes": metrics.mpc_sensitivity_computes,
            },
            "translator_learning": {
                "phi_gradient_norm": metrics.phi_gradient_norm,
                "phi_param_change":  metrics.phi_param_change,
                "sensitivity_samples": metrics.num_sensitivity_samples,
                "avg_mpc_cost":      metrics.avg_mpc_cost,
            },
            "battery_start":         battery_start * 100,
            "battery_remaining":     metrics.battery_remaining_pct,
            "battery_used_pct":      metrics.battery_used_pct,
            "final_position_error":  final_position_error,
            "path_efficiency":       metrics.path_efficiency,
            "approach_quality":      metrics.approach_quality,
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
