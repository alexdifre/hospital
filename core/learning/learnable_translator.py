#!/usr/bin/env python3
"""
Learnable Translator
====================

Maps patient preference weights w ∈ Δ⁴ to MPC cost matrices (Q, R)
and horizon via a parametric formula whose 18 coefficients φ are
updated online by gradient descent (inner learning loop).

Learning signal:
    ∂J/∂φ = ∂J/∂Q · ∂Q/∂φ + ∂J/∂R · ∂R/∂φ   (chain rule)

where ∂J/∂Q and ∂J/∂R come from the MPC solver's IFT sensitivities,
and ∂Q/∂φ, ∂R/∂φ are computed analytically here.

Instrumentation (Section 8 figures):
    get_params()           → B7 per-episode φ snapshot
    param_history          → B7 full trajectory of all 18 φ
    cost_history           → B7 supplementary
    gradient_norms         → B7 supplementary
    computed_mpc_history   → B7 derived Q/R/H values over time
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Activation helpers ────────────────────────────────────────────────
# softplus guarantees R > 0 without a hard clip.
# sigmoid = softplus' (chain-rule derivative used in compute_parameter_gradients).

def _softplus(x: float) -> float:
    """log(1 + exp(x)) — numerically stable via log1p."""
    return float(np.log1p(np.exp(np.clip(x, -500.0, 500.0))))


def _sigmoid(x: float) -> float:
    """1 / (1 + exp(-x)) — derivative of softplus."""
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0))))


from core.learning.translator_params import (
    MPCParameterGradients,
    TranslatorParameters,
)


class LearnableTranslator:
    """
    Translator with learnable preference → MPC parameter mapping.

    Public interface:
        translate()                    — convert preferences + location to MPC config
        get_mpc_params()               — direct (Q_diag, R_diag, horizon) access
        get_params()                   — φ snapshot dict for JSON (B7 extraction)
        compute_parameter_gradients()  — ∂(Q,R)/∂φ for chain rule
        update_parameters()            — gradient descent step on φ
        update_preference_weights()    — called by preference learner each episode
    """

    def __init__(
        self,
        environment,
        initial_preference_weights: Optional[np.ndarray] = None,
        learning_rate: float = 0.001,
        parameters: Optional[TranslatorParameters] = None,
        max_grad_norm: float = 100.0,
    ):
        self.env = environment
        self.locations = environment.locations
        self.location_metadata = getattr(environment, "location_metadata", {})

        self.preference_weights = (
            np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            if initial_preference_weights is None
            else np.array(initial_preference_weights)
        )

        self.params = parameters if parameters is not None else TranslatorParameters()
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # ── History (B7 instrumentation) ─────────────────────────────
        self.param_history: List[np.ndarray] = [self.params.to_vector().copy()]
        self.gradient_history: List[np.ndarray] = []
        self.cost_history: List[float] = []
        self.gradient_norms: List[float] = []
        self.computed_mpc_history: List[Dict] = []
        self.param_change_history: List[float] = []

        self.last_update_info: Optional[Dict] = None
        self.update_count: int = 0

        # Fixed geometry (not learned)
        self.location_orientations = {
            "home": 0.0,
            "pharmacy_north": np.pi / 2,
            "pharmacy_south": np.pi / 2,
            "supply_A": 0.0,
            "supply_B": 0.0,
            "charge_main": -np.pi / 4,
            "charge_backup": -np.pi / 4,
            "nurse_station": 0.0,
            "equipment_storage": 0.0,
            "patient_bed": -np.pi / 4,
            "patient_bed_left": -np.pi / 6,
            "patient_bed_right": -np.pi / 3,
            "med_station": np.pi / 2,
        }
        self.default_location_sizes = {
            "home": 0.8, "pharmacy_north": 1.2, "pharmacy_south": 1.2,
            "supply_A": 1.0, "supply_B": 1.0,
            "charge_main": 0.8, "charge_backup": 0.8,
            "nurse_station": 1.5, "equipment_storage": 1.2,
            "patient_bed": 1.5, "patient_bed_left": 1.0,
            "patient_bed_right": 1.0, "med_station": 1.0,
        }
        self.Ts = 0.2
        self.max_obstacles = 3

        print("LearnableTranslator initialized")
        print(f"  Learnable parameters: {self.params.num_params}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Max gradient norm: {self.max_grad_norm}")
        print(f"  Initial params: {self.params.to_vector()[:6]}...")

    # ── B7 instrumentation ────────────────────────────────────────────

    def get_params(self) -> Dict:
        """
        Return current φ as a flat dict for JSON serialisation.

        Primary extraction method used by the experiment runner.
        Includes derived MPC values at current preference weights.
        """
        d = self.params.to_dict()
        try:
            Q_diag, R_diag, horizon, tol, z_target = self._compute_mpc_params(near_patient=False)
            d["_derived_Q_pos"]    = float(Q_diag[0])
            d["_derived_Q_vel"]    = float(Q_diag[3])
            d["_derived_Q_orient"] = float(Q_diag[2])
            d["_derived_R_ax"]     = float(R_diag[0])
            d["_derived_R_ay"]     = float(R_diag[1])
            d["_derived_R_alpha"]  = float(R_diag[2])
            d["_derived_R"]        = float(R_diag[0])  # backward-compat alias
            d["_derived_horizon"]  = int(horizon)
            d["_derived_tol"]      = float(tol)
            d["_derived_z_target"] = z_target.tolist()
        except Exception:
            pass
        d["_update_count"] = self.update_count
        return d

    # Property proxies — runner probes these as fallback attributes
    @property
    def q_base(self) -> float:      return self.params.q_base
    @property
    def q_time(self) -> float:      return self.params.q_time
    @property
    def q_safety(self) -> float:    return self.params.q_safety
    @property
    def q_proximity(self) -> float: return self.params.q_proximity
    @property
    def r_base(self) -> float:      return self.params.r_base
    @property
    def r_time(self) -> float:      return self.params.r_time
    @property
    def r_battery(self) -> float:   return self.params.r_battery
    @property
    def r_safety(self) -> float:    return self.params.qv_safety  # closest proxy
    @property
    def weights(self) -> np.ndarray: return self.preference_weights.copy()
    @property
    def bias(self) -> None:         return None

    # ── Preference weight interface ───────────────────────────────────

    def update_preference_weights(self, new_weights: np.ndarray) -> None:
        self.preference_weights = np.array(new_weights)

    # ── MPC parameter computation (learnable mapping) ─────────────────

    def _compute_mpc_params(
        self, near_patient: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, int, float, np.ndarray]:
        """
        Compute (Q_diag, R_diag, horizon, tol, z_target) from current φ and w.
        """
        w_time, w_safety, w_battery, w_proximity, w_approach = self.preference_weights
        φ = self.params
        near = 1.0 if near_patient else 0.0

        Q_pos = φ.q_base * (1.0 + φ.q_safety * w_safety + φ.q_time * w_time
                            + φ.q_proximity * near * w_proximity)
        Q_pos = np.clip(Q_pos, 5.0, 100.0)

        Q_vel = φ.qv_base * (1.0 + φ.qv_safety * w_safety + φ.qv_time * w_time)
        Q_vel = np.clip(Q_vel, 0.5, 20.0)

        Q_orient = φ.qo_base * (1.0 + φ.qo_approach * w_approach)
        Q_orient = np.clip(Q_orient, 0.5, 20.0)

        if Q_pos / Q_vel > 15.0:
            Q_vel = Q_pos / 15.0

        # R: per-axis raw values (affine in w), then softplus for positivity
        # R_raw_i = r_base_i * (1 + r_time*w_time + r_battery*w_battery + r_proximity*near*w_proximity)
        # R_i = softplus(R_raw_i)  — always positive, continuously differentiable
        f_r = (1.0 + φ.r_time * w_time + φ.r_battery * w_battery
               + φ.r_proximity * near * w_proximity)
        R_diag = np.array([
            _softplus(φ.r_base_ax    * f_r),
            _softplus(φ.r_base_ay    * f_r),
            _softplus(φ.r_base_alpha * f_r),
        ])

        horizon = int(np.clip(φ.h_base + φ.h_time * w_time + φ.h_safety * w_safety, 20, 60))
        tol     = np.clip(φ.tol_base + φ.tol_approach * w_approach, 0.3, 3.0)

        Q_diag = np.array([Q_pos, Q_pos, Q_orient, Q_vel, Q_vel, Q_vel])


        return Q_diag, R_diag, horizon, tol

    def get_mpc_params(
        self, near_patient: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """Return (Q_diag, R_diag, horizon, z_target) — convenience method for MPC calls."""
        Q_diag, R_diag, horizon, _,  = self._compute_mpc_params(near_patient)
        return Q_diag, R_diag, horizon

    # ── Gradient computation ──────────────────────────────────────────

    def compute_parameter_gradients(
        self, near_patient: bool = False
    ) -> MPCParameterGradients:
        """
        Compute ∂Q/∂φ, ∂R/∂φ, and ∂z_target/∂φ analytically for the chain rule.

        Parameter index order matches TranslatorParameters.to_vector() (length 32):
            0:q_base    1:q_safety   2:q_time     3:q_proximity
            4:qv_base   5:qv_safety  6:qv_time
            7:qo_base   8:qo_approach
            9:r_base_ax 10:r_base_ay 11:r_base_alpha
           12:r_time    13:r_battery 14:r_proximity
           15:h_base    16:h_time    17:h_safety
           18:tol_base  19:tol_approach
           20-24:z_A_00..z_A_04  (row 0 of z_target_A)
           25-29:z_A_10..z_A_14  (row 1 of z_target_A)
           30:z_b_0  31:z_b_1

        R gradients use the softplus chain rule:
            R_i = softplus(R_raw_i),  R_raw_i = r_base_i * f_r
            ∂R_i/∂φ_j = sigmoid(R_raw_i) * ∂R_raw_i/∂φ_j
        """
        w_time, w_safety, w_battery, w_proximity, w_approach = self.preference_weights
        w = self.preference_weights
        φ = self.params
        near = 1.0 if near_patient else 0.0
        n = φ.num_params  # 32

        dQ_dphi = np.zeros((6, n))
        dR_dphi = np.zeros((3, n))
        dH_dphi = np.zeros(n)
        dZ_dphi = np.zeros((2, n))

        # ∂Q_pos/∂φ (rows 0 and 1 — x and y share same weight)
        f_pos = 1.0 + φ.q_safety * w_safety + φ.q_time * w_time + φ.q_proximity * near * w_proximity
        dQ_dphi[0, 0] = f_pos
        dQ_dphi[0, 1] = φ.q_base * w_safety
        dQ_dphi[0, 2] = φ.q_base * w_time
        dQ_dphi[0, 3] = φ.q_base * near * w_proximity
        dQ_dphi[1, :] = dQ_dphi[0, :]

        # ∂Q_vel/∂φ (rows 3, 4, 5)
        f_vel = 1.0 + φ.qv_safety * w_safety + φ.qv_time * w_time
        dQ_dphi[3, 4] = f_vel
        dQ_dphi[3, 5] = φ.qv_base * w_safety
        dQ_dphi[3, 6] = φ.qv_base * w_time
        dQ_dphi[4, :] = dQ_dphi[3, :]
        dQ_dphi[5, :] = dQ_dphi[3, :]

        # ∂Q_orient/∂φ (row 2)
        f_ori = 1.0 + φ.qo_approach * w_approach
        dQ_dphi[2, 7] = f_ori
        dQ_dphi[2, 8] = φ.qo_base * w_approach

        # ∂R/∂φ — softplus chain rule: d/dφ softplus(R_raw) = sigmoid(R_raw) * dR_raw/dφ
        f_r = (1.0 + φ.r_time * w_time + φ.r_battery * w_battery
               + φ.r_proximity * near * w_proximity)
        r_bases = np.array([φ.r_base_ax, φ.r_base_ay, φ.r_base_alpha])
        R_raws  = r_bases * f_r
        sigs    = np.array([_sigmoid(R_raws[0]), _sigmoid(R_raws[1]), _sigmoid(R_raws[2])])

        # Per-axis base params (indices 9, 10, 11) — only affect their own R_i
        dR_dphi[0, 9]  = sigs[0] * f_r
        dR_dphi[1, 10] = sigs[1] * f_r
        dR_dphi[2, 11] = sigs[2] * f_r

        # Shared sensitivity params (indices 12, 13, 14) — affect all R_i
        for i in range(3):
            dR_dphi[i, 12] = sigs[i] * r_bases[i] * w_time
            dR_dphi[i, 13] = sigs[i] * r_bases[i] * w_battery
            dR_dphi[i, 14] = sigs[i] * r_bases[i] * near * w_proximity

        # ∂H/∂φ
        dH_dphi[15] = 1.0
        dH_dphi[16] = w_time
        dH_dphi[17] = w_safety

        # ∂z_target/∂φ
        # z_target_i = sum_j(z_target_A[i,j] * w[j]) + z_target_b[i]
        # ∂z_target[i]/∂z_target_A[i,j] = w[j]  → param index 20 + i*5 + j
        # ∂z_target[i]/∂z_target_b[i]   = 1.0   → param index 30 + i
        for i in range(2):
            for j in range(5):
                dZ_dphi[i, 20 + i * 5 + j] = w[j]
            dZ_dphi[i, 30 + i] = 1.0

        return MPCParameterGradients(
            dQ_dphi=dQ_dphi, dR_dphi=dR_dphi, dH_dphi=dH_dphi, dZ_dphi=dZ_dphi
        )

    # ── Parameter update (gradient descent) ──────────────────────────

    def update_parameters(
        self,
        dJ_dQ: np.ndarray,
        dJ_dR: np.ndarray,
        near_patient: bool = False,
        cost: Optional[float] = None,
        dJ_dz_target: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        One gradient descent step: φ ← φ - lr * ∂J/∂φ

        Args:
            dJ_dQ: (6,) sensitivity of MPC cost to Q diagonal
            dJ_dR: (3,) sensitivity of MPC cost to R diagonal
            near_patient: whether current segment is near the patient
            cost: optional MPC cost value for tracking
            dJ_dz_target: (2,) sensitivity of MPC cost to z_target (optional)
        """
        self.update_count += 1

        grads   = self.compute_parameter_gradients(near_patient)
        dJ_dphi = grads.dQ_dphi.T @ dJ_dQ + grads.dR_dphi.T @ dJ_dR
        if dJ_dz_target is not None:
            dJ_dphi += grads.dZ_dphi.T @ dJ_dz_target

        grad_norm = float(np.linalg.norm(dJ_dphi))
        if grad_norm > self.max_grad_norm:
            dJ_dphi = dJ_dphi * (self.max_grad_norm / grad_norm)
            grad_norm_clipped = self.max_grad_norm
        else:
            grad_norm_clipped = grad_norm

        old_params = self.params.to_vector().copy()
        new_params = self._apply_param_bounds(old_params - self.learning_rate * dJ_dphi)
        self.params.from_vector(new_params)

        param_change = float(np.linalg.norm(new_params - old_params))

        self.param_history.append(new_params.copy())
        self.gradient_history.append(dJ_dphi.copy())
        self.gradient_norms.append(grad_norm)
        self.param_change_history.append(param_change)
        if cost is not None:
            self.cost_history.append(cost)

        try:
            Q_diag, R_diag, horizon, _, _z = self._compute_mpc_params(near_patient=False)
            self.computed_mpc_history.append({
                "Q_pos": float(Q_diag[0]), "Q_vel": float(Q_diag[3]),
                "Q_orient": float(Q_diag[2]),
                "R_ax": float(R_diag[0]), "R_ay": float(R_diag[1]), "R_alpha": float(R_diag[2]),
                "horizon": int(horizon),
            })
        except Exception:
            pass

        self.last_update_info = {
            "gradient": dJ_dphi,
            "gradient_norm": grad_norm,
            "gradient_norm_clipped": grad_norm_clipped,
            "param_change": param_change,
            "old_params": old_params,
            "new_params": new_params,
            "update_count": self.update_count,
        }
        return self.last_update_info

    def _apply_param_bounds(self, params: np.ndarray) -> np.ndarray:
        bounds = [
            (5.0, 100.0), (0.0, 2.0), (-1.0, 1.0), (0.0, 1.0),      # q_*     0-3
            (2.0, 20.0),  (0.0, 1.0), (-0.5, 0.5),                    # qv_*    4-6
            (0.5, 10.0),  (0.0, 2.0),                                  # qo_*    7-8
            (0.1, 10.0),  (0.1, 10.0),  (0.1, 10.0),                  # r_base  9-11  (softplus guarantees positivity)
            (-1.0, 0.5), (0.0, 2.0), (0.0, 1.0),                      # r_sens  12-14
            (20.0, 60.0), (-20.0, 0.0), (0.0, 20.0),                  # h_*     15-17
            (0.3, 3.0),   (-1.0, 0.0),                                 # tol_*   18-19
            # z_target params (20-31): unconstrained within [-50, 50]
            (-50.0, 50.0), (-50.0, 50.0), (-50.0, 50.0), (-50.0, 50.0), (-50.0, 50.0),  # z_A row 0  20-24
            (-50.0, 50.0), (-50.0, 50.0), (-50.0, 50.0), (-50.0, 50.0), (-50.0, 50.0),  # z_A row 1  25-29
            (-50.0, 50.0), (-50.0, 50.0),                              # z_b     30-31
        ]
        bounded = params.copy()
        for i, (lo, hi) in enumerate(bounds):
            bounded[i] = np.clip(bounded[i], lo, hi)
        return bounded

    # ── Obstacle handling ─────────────────────────────────────────────

    def _get_location_size(self, loc_name: str) -> float:
        if loc_name in self.location_metadata:
            return float(self.location_metadata[loc_name].get("size", 1.0))
        return self.default_location_sizes.get(loc_name, 1.0)

    def _create_obstacle_list(
        self, start_name: str, goal_name: str, safety_margin: float = 0.3
    ) -> List[Dict]:
        obstacles = []
        for loc_name, loc_pos in self.locations.items():
            if loc_name in (start_name, goal_name):
                continue
            if (goal_name in ("patient_bed_left", "patient_bed_right")
                    and loc_name == "patient_bed"):
                continue
            obstacles.append({
                "name": loc_name,
                "x": float(loc_pos[0]),
                "y": float(loc_pos[1]),
                "radius": float(self._get_location_size(loc_name) + safety_margin),
            })
        return obstacles

    def _filter_obstacles_by_relevance(
        self,
        obstacles: List[Dict],
        robot_pos: np.ndarray,
        goal_pos: np.ndarray,
    ) -> List[Dict]:
        if len(obstacles) <= self.max_obstacles:
            return obstacles
        scored = sorted(
            obstacles,
            key=lambda o: min(
                np.linalg.norm(np.array([o["x"], o["y"]]) - robot_pos),
                np.linalg.norm(np.array([o["x"], o["y"]]) - goal_pos),
            ),
        )
        return scored[: self.max_obstacles]

    # ── Main translation method ───────────────────────────────────────

    def translate(
        self,
        start_location: str,
        goal_location: str,
        current_state: np.ndarray,
    ) -> Dict:
        """Convert (start, goal, preferences) to an MPC configuration dict."""
        if goal_location not in self.locations:
            return {"success": False, "reason": "unknown_goal_location"}

        target_pos = np.array(self.locations[goal_location])
        desired_ori = self.location_orientations.get(goal_location, 0.0)
        robot_pos   = current_state[:2]

        all_obs = self._create_obstacle_list(start_location, goal_location)
        obstacles = self._filter_obstacles_by_relevance(all_obs, robot_pos, target_pos)

        near_patient = "patient" in goal_location.lower()
        Q_diag, R_diag, horizon, conv_tol, _ = self._compute_mpc_params(near_patient)

        distance  = np.linalg.norm(target_pos - robot_pos)
        max_steps = int(distance * 1.5 / 0.8 / self.Ts) + 300

        return {
            "success": True,
            "obstacles": obstacles,
            "goal_state": np.array([target_pos[0], target_pos[1], desired_ori, 0.0, 0.0, 0.0]),
            "mpc_config": {"Q_diag": Q_diag, "R_diag": R_diag, "horizon": horizon},
            "max_steps": max_steps,
            "convergence_tolerance": conv_tol,
            "target_position": target_pos,
            "desired_orientation": desired_ori,
            "near_patient": near_patient,
            "path_info": {
                "straight_distance": float(distance),
                "num_obstacles": len(obstacles),
            },
        }

    # ── Diagnostics & persistence ─────────────────────────────────────

    def get_learning_snapshot(self) -> Dict:
        """Full learning state for post-hoc analysis."""
        return {
            "params": self.params.to_dict(),
            "update_count": self.update_count,
            "preference_weights": self.preference_weights.tolist(),
            "param_history_len": len(self.param_history),
            "cost_history": self.cost_history[-10:],
            "gradient_norms": self.gradient_norms[-10:],
            "param_changes": self.param_change_history[-10:],
            "computed_mpc_history": self.computed_mpc_history[-5:],
        }

    def print_learning_summary(self) -> None:
        if len(self.param_history) < 2:
            print("  No learning updates yet")
            return
        initial, final = self.param_history[0], self.param_history[-1]
        print(f"\n  {'Parameter':<12} {'Initial':>10} {'Final':>10} {'Change':>10}")
        print("  " + "-" * 44)
        for i, name in enumerate(self.params.PARAM_NAMES):
            change = final[i] - initial[i]
            if abs(change) > 0.001:
                print(f"  {name:<12} {initial[i]:>10.4f} {final[i]:>10.4f} {change:>+10.4f}")
        print(f"\n  Total parameter change: {np.linalg.norm(final - initial):.6f}")
        print(f"  Learning updates: {self.update_count}")
        if self.cost_history:
            print(f"  Cost: {self.cost_history[0]:.1f} → {self.cost_history[-1]:.1f}")
        if self.gradient_norms:
            print(f"  Avg gradient norm: {np.mean(self.gradient_norms):.2f}")

    def print_parameters(self) -> None:
        φ = self.params
        print("\nLearnable Translator Parameters:")
        print(f"  Q_pos:    base={φ.q_base:.2f}, safety={φ.q_safety:.3f}, time={φ.q_time:.3f}")
        print(f"  Q_vel:    base={φ.qv_base:.2f}, safety={φ.qv_safety:.3f}, time={φ.qv_time:.3f}")
        print(f"  Q_orient: base={φ.qo_base:.2f}, approach={φ.qo_approach:.3f}")
        print(f"  R:        base_ax={φ.r_base_ax:.2f}, base_ay={φ.r_base_ay:.2f}, base_alpha={φ.r_base_alpha:.2f}")
        print(f"            time={φ.r_time:.3f}, battery={φ.r_battery:.3f}, proximity={φ.r_proximity:.3f}")
        print(f"  Horizon:  base={φ.h_base:.1f}, time={φ.h_time:.2f}, safety={φ.h_safety:.2f}")

    def save_parameters(self, filepath: str) -> None:
        data = {
            "parameters": self.params.to_dict(),
            "param_history": [p.tolist() for p in self.param_history],
            "cost_history": self.cost_history,
            "gradient_norms": self.gradient_norms,
            "param_change_history": self.param_change_history,
            "computed_mpc_history": self.computed_mpc_history,
            "learning_rate": self.learning_rate,
            "preference_weights": self.preference_weights.tolist(),
            "update_count": self.update_count,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved translator parameters to {filepath}")

    def load_parameters(self, filepath: str) -> None:
        with open(filepath) as f:
            data = json.load(f)
        self.params = TranslatorParameters.from_dict(data["parameters"])
        self.learning_rate = data.get("learning_rate", self.learning_rate)
        self.update_count = data.get("update_count", 0)
        if "preference_weights" in data:
            self.preference_weights = np.array(data["preference_weights"])
        print(f"Loaded translator parameters from {filepath}")


# Alias preserved for existing import sites
ObstacleAwareTranslator = LearnableTranslator


# ── Self-test ─────────────────────────────────────────────────────────

def _test():
    """Quick smoke test — run with: python learnable_translator.py"""
    print("=" * 60)
    print("LEARNABLE TRANSLATOR — self test")
    print("=" * 60)

    class _MockEnv:
        locations = {
            "home": (0, 0), "pharmacy_north": (5, 18),
            "supply_A": (14, 10), "patient_bed_left": (20.5, 12),
        }

    t = LearnableTranslator(_MockEnv(), learning_rate=0.001)
    t.preference_weights = np.array([0.2, 0.4, 0.1, 0.2, 0.1])

    result = t.translate("home", "supply_A", np.zeros(6))
    print(f"Q_diag:  {result['mpc_config']['Q_diag']}")
    print(f"R_diag:  {result['mpc_config']['R_diag']}")
    print(f"Horizon: {result['mpc_config']['horizon']}")

    params = t.get_params()
    print(f"\nget_params() → {len(params)} keys")

    for _ in range(5):
        t.update_parameters(
            np.array([10., 10., 5., 2., 2., 2.]),
            np.array([5., 5., 5.]),
            cost=100.0,
        )
    print(f"\nAfter 5 updates: update_count={t.update_count}")
    t.print_learning_summary()
    print("\n✓ self test passed")


if __name__ == "__main__":
    _test()
