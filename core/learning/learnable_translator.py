#!/usr/bin/env python3
"""
Learnable Translator
====================

Maps fixed translator weights W = 1 to MPC cost matrices (Q, R)
via a parametric formula whose coefficients φ are
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
    computed_mpc_history   → B7 derived Q/R values over time
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
from core.execution.formulation import SharedMPCFormulation


class LearnableTranslator:
    """
    Translator with learnable MPC parameter mapping.

    Public interface:
        translate()                    — convert navigation context + location to MPC config
        get_mpc_params()               — direct (Q_diag, R_diag, Q_terminal_diag) access
        get_params()                   — φ snapshot dict for JSON (B7 extraction)
        compute_parameter_gradients()  — ∂(Q,R)/∂φ for chain rule
        update_parameters()            — gradient descent step on φ
        update_preference_weights()    — compatibility hook; ignored in simplified mode
    """

    def __init__(
        self,
        environment,
        initial_preference_weights: Optional[np.ndarray] = None,
        learning_rate: float = 0.001,
        parameters: Optional[TranslatorParameters] = None,
        max_grad_norm: float = 100.0,
    ):
        del initial_preference_weights  # Compatibility-only argument.
        self.env = environment
        self.locations = environment.locations
        self.location_metadata = getattr(environment, "location_metadata", {})

        # Simplified translator: keep a fixed all-ones modulation vector
        # internally. External learner weights are ignored, so Q/R do
        # not depend on patient-specific estimates anymore.
        self.preference_weights = np.ones(5, dtype=float)

        self.params = parameters if parameters is not None else TranslatorParameters()
        self.learning_rate = learning_rate
        self.terminal_learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.terminal_cost_diag = (
            SharedMPCFormulation.Q_default.copy()
            * SharedMPCFormulation.TERMINAL_COST_MULTIPLIER
        )
        self.action_terminal_costs: Dict[str, np.ndarray] = {}

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
            "patient_bed_left": -np.pi / 6,
            "patient_bed_right": -np.pi / 3,
            "pantry": 0.0,
            "fridge": 0.0,
            "prep_station": 0.0,
            "stove": 0.0,
            "quality_check": 0.0,
        }
        self.default_location_sizes = {
            "home": 0.8, "pharmacy_north": 1.2, "pharmacy_south": 1.2,
            "supply_A": 1.0, "supply_B": 1.0,
            "charge_main": 0.8, "charge_backup": 0.8,
            "patient_bed_left": 1.0, "patient_bed_right": 1.0,
            "pantry": 1.0, "fridge": 1.0, "prep_station": 0.8,
            "stove": 0.6, "quality_check": 0.8,
        }
        self.Ts = 0.2
        self.max_obstacles = 3

        print("LearnableTranslator initialized")
        print(f"  Learnable parameters: {self.params.num_params}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Terminal learning rate: {self.terminal_learning_rate}")
        print(f"  Max gradient norm: {self.max_grad_norm}")
        print(f"  Initial params: {self.params.to_vector()[:6]}...")

    # ── B7 instrumentation ────────────────────────────────────────────

    def get_params(self) -> Dict:
        """
        Return current φ as a flat dict for JSON serialisation.

        Primary extraction method used by the experiment runner.
        Includes derived MPC values at the current fixed internal modulation.
        """
        d = self.params.to_dict()
        try:
            Q_diag, R_diag, tol = self._compute_mpc_params(near_patient=False)
            d["_derived_Q_pos"]    = float(Q_diag[0])
            d["_derived_Q_vel"]    = float(Q_diag[3])
            d["_derived_Q_orient"] = float(Q_diag[2])
            d["_derived_R_ax"]     = float(R_diag[0])
            d["_derived_R_ay"]     = float(R_diag[1])
            d["_derived_R_alpha"]  = float(R_diag[2])
            d["_derived_R"]        = float(R_diag[0])  # backward-compat alias
            d["_derived_tol"]      = float(tol)
            d["_derived_Q_terminal"] = self.terminal_cost_diag.tolist()
            d["_action_terminal_costs"] = {
                key: value.tolist()
                for key, value in self.action_terminal_costs.items()
            }
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

    # ── Compatibility hook ────────────────────────────────────────────

    def update_preference_weights(self, new_weights: np.ndarray) -> None:
        del new_weights  # Compatibility-only argument.
        # Kept for compatibility with existing call sites.
        # The simplified translator ignores external weights and stays at W = 1.
        self.preference_weights = np.ones(5, dtype=float)

    # ── MPC parameter computation (learnable mapping) ─────────────────

    def _compute_mpc_params(
        self, near_patient: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute (Q_diag, R_diag, tol) from current φ and fixed W = 1.
        """
        w_time, w_safety, w_battery, w_proximity, w_approach = np.ones(5, dtype=float)
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

        # R: per-axis raw values (affine in the fixed internal modulation),
        # then softplus for positivity.
        # R_raw_i = r_base_i * (1 + r_time*w_time + r_battery*w_battery + r_proximity*near*w_proximity)
        # R_i = softplus(R_raw_i)  — always positive, continuously differentiable
        f_r = (1.0 + φ.r_time * w_time + φ.r_battery * w_battery
               + φ.r_proximity * near * w_proximity)
        R_diag = np.array([
            _softplus(φ.r_base_ax    * f_r),
            _softplus(φ.r_base_ay    * f_r),
            _softplus(φ.r_base_alpha * f_r),
        ])

        tol     = np.clip(φ.tol_base + φ.tol_approach * w_approach, 0.3, 3.0)

        Q_diag = np.array([Q_pos, Q_pos, Q_orient, Q_vel, Q_vel, Q_vel])


        return Q_diag, R_diag, tol

    @staticmethod
    def _action_key(action) -> Optional[str]:
        """Return a stable key for action-specific mismatch parameters."""
        if action is None:
            return None
        return str(getattr(action, "value", action))

    def _ensure_action_terminal_cost(self, action) -> np.ndarray:
        """Create/get p^w(a_t), represented as an action-specific terminal cost."""
        key = self._action_key(action)
        if key is None:
            return self.terminal_cost_diag
        if key not in self.action_terminal_costs:
            self.action_terminal_costs[key] = self.terminal_cost_diag.copy()
        return self.action_terminal_costs[key]

    def _compute_terminal_cost_diag(self, action=None) -> np.ndarray:
        """Return p^w(a_t) if an action is supplied, otherwise the base terminal cost."""
        key = self._action_key(action)
        if key is not None and key in self.action_terminal_costs:
            return np.clip(
                self.action_terminal_costs[key].copy(),
                0.0,
                SharedMPCFormulation.TERMINAL_COST_MAX,
            )
        return np.clip(
            self.terminal_cost_diag.copy(),
            0.0,
            SharedMPCFormulation.TERMINAL_COST_MAX,
        )

    @staticmethod
    def _align_terminal_error(E: np.ndarray) -> np.ndarray:
        """
        Align a user-provided terminal error vector to the MPC state dimension.

        For now we use a simple pad/truncate rule:
        - scalar → broadcast to all terminal dimensions
        - short vector → copy then zero-pad
        - long vector → truncate
        """
        flat = np.array(E, dtype=float).reshape(-1)
        nx = SharedMPCFormulation.nx
        if flat.size == 0:
            return np.zeros(nx, dtype=float)
        if flat.size == 1:
            return np.full(nx, float(flat[0]), dtype=float)

        aligned = np.zeros(nx, dtype=float)
        count = min(nx, flat.size)
        aligned[:count] = flat[:count]
        return aligned

    @staticmethod
    def _align_terminal_sensitivity(terminal_sensitivity: np.ndarray) -> np.ndarray:
        """
        Align dx_N*/dQ_terminal to shape (nx, nx).

        Rows correspond to terminal-state dimensions and columns to terminal
        weight-diagonal parameters.
        """
        nx = SharedMPCFormulation.nx
        arr = np.array(terminal_sensitivity, dtype=float)
        aligned = np.zeros((nx, nx), dtype=float)
        if arr.ndim != 2:
            return aligned
        rows = min(nx, arr.shape[0])
        cols = min(nx, arr.shape[1])
        aligned[:rows, :cols] = arr[:rows, :cols]
        return aligned

    @staticmethod
    def _align_terminal_error_jacobian(terminal_error_jacobian: np.ndarray) -> np.ndarray:
        """
        Align dE/dz_N to shape (nx, nx).

        Rows correspond to error dimensions; columns correspond to terminal-state
        dimensions.
        """
        nx = SharedMPCFormulation.nx
        arr = np.array(terminal_error_jacobian, dtype=float)
        aligned = np.zeros((nx, nx), dtype=float)
        if arr.ndim != 2:
            return aligned
        rows = min(nx, arr.shape[0])
        cols = min(nx, arr.shape[1])
        aligned[:rows, :cols] = arr[:rows, :cols]
        return aligned

    def get_mpc_params(
        self, near_patient: bool = False, action=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (Q_diag, R_diag, Q_terminal_diag) for MPC calls."""
        Q_diag, R_diag, _ = self._compute_mpc_params(near_patient)
        Q_terminal_diag = self._compute_terminal_cost_diag(action=action)
        return Q_diag, R_diag, Q_terminal_diag

    # ── Gradient computation ──────────────────────────────────────────

    def compute_parameter_gradients(
        self, near_patient: bool = False
    ) -> MPCParameterGradients:
        """
        Compute ∂Q/∂φ and ∂R/∂φ analytically for the chain rule.

        Parameter index order matches TranslatorParameters.to_vector() (length 20):
            0:q_base    1:q_safety   2:q_time     3:q_proximity
            4:qv_base   5:qv_safety  6:qv_time
            7:qo_base   8:qo_approach
            9:r_base_ax 10:r_base_ay 11:r_base_alpha
           12:r_time    13:r_battery 14:r_proximity
           15:h_base    16:h_time    17:h_safety
           18:tol_base  19:tol_approach

        R gradients use the softplus chain rule:
            R_i = softplus(R_raw_i),  R_raw_i = r_base_i * f_r
            ∂R_i/∂φ_j = sigmoid(R_raw_i) * ∂R_raw_i/∂φ_j
        """
        w = np.ones(5, dtype=float)
        w_time, w_safety, w_battery, w_proximity, w_approach = w
        φ = self.params
        near = 1.0 if near_patient else 0.0
        n = φ.num_params  # 32

        dQ_dphi = np.zeros((6, n))
        dR_dphi = np.zeros((3, n))
        dH_dphi = np.zeros(n)

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

        return MPCParameterGradients(
            dQ_dphi=dQ_dphi, dR_dphi=dR_dphi, dH_dphi=dH_dphi
        )

    # ── Parameter update (gradient descent) ──────────────────────────

    def update_parameters(
        self,
        dJ_dQ: np.ndarray,
        dJ_dR: np.ndarray,
        near_patient: bool = False,
        cost: Optional[float] = None,
        E: Optional[np.ndarray] = None,
        terminal_sensitivity: Optional[np.ndarray] = None,
        terminal_error_jacobian: Optional[np.ndarray] = None,
        action=None,
    ) -> Dict:
        """
        One gradient descent step: φ ← φ - lr * ∂J/∂φ

        Args:
            dJ_dQ: (6,) sensitivity of MPC cost to Q diagonal
            dJ_dR: (3,) sensitivity of MPC cost to R diagonal
            near_patient: whether current segment is near the patient
            cost: optional MPC cost value for tracking
            E: terminal membership-difference vector supplied externally.
               This is E_hat_t^psi in the mismatch update.
            terminal_sensitivity: (6,6) IFT sensitivity dx_N*/dQ_terminal_diag.
            terminal_error_jacobian: (6,6) local Jacobian dE_hat^psi/dz_N.
            action: action a_t whose action-specific mismatch parameter p^w(a_t)
               is updated.
        """
        self.update_count += 1

        grads   = self.compute_parameter_gradients(near_patient)
        dJ_dphi = grads.dQ_dphi.T @ dJ_dQ + grads.dR_dphi.T @ dJ_dR

        grad_norm = float(np.linalg.norm(dJ_dphi))
        if grad_norm > self.max_grad_norm:
            dJ_dphi = dJ_dphi * (self.max_grad_norm / grad_norm)
            grad_norm_clipped = self.max_grad_norm
        else:
            grad_norm_clipped = grad_norm

        old_params = self.params.to_vector().copy()
        terminal_target = self._ensure_action_terminal_cost(action)
        old_terminal = terminal_target.copy()
        new_params = self._apply_param_bounds(old_params - self.learning_rate * dJ_dphi)
        self.params.from_vector(new_params)

        terminal_change = 0.0
        terminal_gradient = None
        terminal_error_param_jacobian = None
        terminal_update_mode = "none"
        if E is not None:
            aligned_E = self._align_terminal_error(E)
            if terminal_sensitivity is not None and terminal_error_jacobian is not None:
                aligned_terminal_sensitivity = self._align_terminal_sensitivity(
                    terminal_sensitivity
                )
                aligned_error_jacobian = self._align_terminal_error_jacobian(
                    terminal_error_jacobian
                )
                terminal_error_param_jacobian = (
                    aligned_error_jacobian @ aligned_terminal_sensitivity
                )
                terminal_gradient = terminal_error_param_jacobian.T @ aligned_E
                terminal_target[:] = np.clip(
                    terminal_target
                    - self.terminal_learning_rate * terminal_gradient,
                    0.0,
                    SharedMPCFormulation.TERMINAL_COST_MAX,
                )
                terminal_change = float(
                    np.linalg.norm(terminal_target - old_terminal)
                )
                terminal_update_mode = "action_mismatch_chain_rule"
            else:
                terminal_gradient = aligned_E
                terminal_target[:] = np.clip(
                    terminal_target
                    - self.terminal_learning_rate * terminal_gradient,
                    0.0,
                    SharedMPCFormulation.TERMINAL_COST_MAX,
                )
                terminal_change = float(
                    np.linalg.norm(terminal_target - old_terminal)
                )
                terminal_update_mode = "direct_error"
        else:
            aligned_E = None

        param_change = float(np.linalg.norm(new_params - old_params))

        self.param_history.append(new_params.copy())
        self.gradient_history.append(dJ_dphi.copy())
        self.gradient_norms.append(grad_norm)
        self.param_change_history.append(param_change)
        if cost is not None:
            self.cost_history.append(cost)

        try:
            Q_diag, R_diag, _ = self._compute_mpc_params(near_patient=False)
            self.computed_mpc_history.append({
                "Q_pos": float(Q_diag[0]), "Q_vel": float(Q_diag[3]),
                "Q_orient": float(Q_diag[2]),
                "R_ax": float(R_diag[0]), "R_ay": float(R_diag[1]), "R_alpha": float(R_diag[2]),
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
            "old_terminal_cost": old_terminal,
            "new_terminal_cost": terminal_target.copy(),
            "terminal_action": self._action_key(action),
            "terminal_error": aligned_E,
            "terminal_gradient": terminal_gradient,
            "terminal_error_param_jacobian": terminal_error_param_jacobian,
            "terminal_update_mode": terminal_update_mode,
            "terminal_change": terminal_change,
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
        """Convert (start, goal, current_state) to an MPC configuration dict."""
        if goal_location not in self.locations:
            return {"success": False, "reason": "unknown_goal_location"}

        target_pos = np.array(self.locations[goal_location])
        desired_ori = self.location_orientations.get(goal_location, 0.0)
        robot_pos   = current_state[:2]

        all_obs = self._create_obstacle_list(start_location, goal_location)
        obstacles = self._filter_obstacles_by_relevance(all_obs, robot_pos, target_pos)

        near_patient = "patient" in goal_location.lower()
        Q_diag, R_diag, conv_tol = self._compute_mpc_params(near_patient)
        Q_terminal_diag = self._compute_terminal_cost_diag()

        distance  = np.linalg.norm(target_pos - robot_pos)
        max_steps = int(distance * 1.5 / 0.8 / self.Ts) + 300

        return {
            "success": True,
            "obstacles": obstacles,
            "goal_state": np.array([target_pos[0], target_pos[1], desired_ori, 0.0, 0.0, 0.0]),
            "mpc_config": {
                "Q_diag": Q_diag,
                "R_diag": R_diag,
                "Q_terminal_diag": Q_terminal_diag,
            },
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
            "terminal_cost_diag": self.terminal_cost_diag.tolist(),
            "action_terminal_costs": {
                key: value.tolist()
                for key, value in self.action_terminal_costs.items()
            },
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
        print(f"  Q_term:   {self.terminal_cost_diag}")
        print(f"  Q_term action-specific: {len(self.action_terminal_costs)} actions")

    def save_parameters(self, filepath: str) -> None:
        data = {
            "parameters": self.params.to_dict(),
            "param_history": [p.tolist() for p in self.param_history],
            "cost_history": self.cost_history,
            "gradient_norms": self.gradient_norms,
            "param_change_history": self.param_change_history,
            "computed_mpc_history": self.computed_mpc_history,
            "learning_rate": self.learning_rate,
            "terminal_learning_rate": self.terminal_learning_rate,
            "preference_weights": self.preference_weights.tolist(),
            "terminal_cost_diag": self.terminal_cost_diag.tolist(),
            "action_terminal_costs": {
                key: value.tolist()
                for key, value in self.action_terminal_costs.items()
            },
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
        self.terminal_learning_rate = data.get(
            "terminal_learning_rate", self.terminal_learning_rate
        )
        self.update_count = data.get("update_count", 0)
        self.preference_weights = np.ones(5, dtype=float)
        if "terminal_cost_diag" in data:
            self.terminal_cost_diag = np.array(data["terminal_cost_diag"], dtype=float)
        self.action_terminal_costs = {
            str(key): np.array(value, dtype=float)
            for key, value in data.get("action_terminal_costs", {}).items()
        }
        print(f"Loaded translator parameters from {filepath}")


# Alias preserved for existing import sites
ObstacleAwareTranslator = LearnableTranslator
