"""
CasADi IFT (Implicit Function Theorem) sensitivity engine.

Computes analytical sensitivities ∂J*/∂p and ∂u*_0/∂p by differentiating
the KKT conditions at the optimum — without re-solving the NLP.

Per paper equation (70):
    ∂κ_ν/∂ζ · dζ*/dp + ∂κ_ν/∂p = 0

Architecture (Section 6.7):
    Acados solve  ──►  (w*, λ*, p)  ──►  CasADi IFT  ──►  ∂J*/∂p
    (~1-5ms)                              (~5-10ms)
"""

from __future__ import annotations

import numpy as np
import casadi as ca
from typing import Dict, List, Optional, Tuple

from core.execution.formulation import (
    MPCSolution,
    MPCSensitivity,
    SharedMPCFormulation,
)


class CasADiSensitivityComputer:
    """
    Computes analytical sensitivities via IFT on KKT conditions.

    Given (w*, λ*, p) from Acados, computes:
        - ∂J*/∂p  (cost sensitivity)
        - ∂u*_0/∂p (policy sensitivity)

    Without re-solving the NLP!

    Per paper equation (70):
        ∂κ_ν/∂ζ · dζ*/dp + ∂κ_ν/∂p = 0

    Solve for dζ*/dp, then extract desired sensitivities.
    """

    def __init__(
        self,
        horizon: int = 40,
        dt: float = 0.2,
        n_obstacles: int = 3,
    ):
        self.N = horizon
        self.dt = dt
        self.nx = SharedMPCFormulation.nx
        self.nu = SharedMPCFormulation.nu
        self.n_obstacles = n_obstacles

        self._build_sensitivity_functions()

        # Also build a full NLP solver for fallback and warm-starting
        self._build_nlp_solver()

        self.w_warm = None
        self.lam_warm = None
        self._warm_anchor: Optional[np.ndarray] = None  # x_init when w_warm was computed

        # Fix 1: discard stale w_warm when robot moved more than this from anchor
        self._WARM_DIST_THRESHOLD = 1.0   # metres; validated in tests/mpc_coldstart/

        # z_target: learned terminal position target for the x/y dimensions.
        self.z_target = None
        self.active_constraint_tol = 1e-6
        self.active_multiplier_tol = 1e-7

    def _build_sensitivity_functions(self):
        """
        Build CasADi functions for sensitivity computation via IFT.

        These take the optimal (w*, λ*) and parameters p,
        and return ∂w*/∂p by solving the KKT sensitivity system.
        """

        # === Dimensions ===
        n_x_vars = self.nx * (self.N + 1)
        n_u_vars = self.nu * self.N
        n_s_vars = self.n_obstacles * (self.N + 1)  # slacks
        n_w = n_x_vars + n_u_vars + n_s_vars

        # === Symbolic variables ===

        # Primal variables
        X = ca.MX.sym("X", self.nx, self.N + 1)
        U = ca.MX.sym("U", self.nu, self.N)
        S = ca.MX.sym("S", self.n_obstacles, self.N + 1)

        # Parameters
        Q_diag = ca.MX.sym("Q_diag", self.nx)
        R_diag = ca.MX.sym("R_diag", self.nu)
        x_init = ca.MX.sym("x_init", self.nx)
        x_ref = ca.MX.sym("x_ref", self.nx)
        obs_params = ca.MX.sym("obs", self.n_obstacles * 3)
        slack_weight = ca.MX.sym("slack_w", 1)
        z_target = ca.MX.sym("z_target", 2)

        # === Cost ===
        cost = 0
        for k in range(self.N):
            x_err = X[:, k] - x_ref
            cost += ca.mtimes([x_err.T, ca.diag(Q_diag), x_err])
            cost += ca.mtimes([U[:, k].T, ca.diag(R_diag), U[:, k]])
            for i in range(self.n_obstacles):
                cost += slack_weight * S[i, k]
                cost += slack_weight * 0.1 * S[i, k] ** 2

        # Terminal: tracks the learned x/y target while preserving the rest of x_ref.
        x_ref_terminal = ca.vertcat(z_target, x_ref[2:])
        x_err_N = X[:, self.N] - x_ref_terminal
        cost += SharedMPCFormulation.TERMINAL_COST_MULTIPLIER * ca.mtimes(
            [x_err_N.T, ca.diag(Q_diag), x_err_N]
        )
        for i in range(self.n_obstacles):
            cost += slack_weight * S[i, self.N]

        # === Equality constraints ===
        g_eq = []

        # Initial condition
        g_eq.append(X[:, 0] - x_init)

        # Dynamics
        for k in range(self.N):
            x_next = SharedMPCFormulation.discrete_dynamics(X[:, k], U[:, k], self.dt)
            g_eq.append(X[:, k + 1] - x_next)

        g_eq = ca.vertcat(*g_eq)
        n_eq = g_eq.shape[0]

        # === Inequality constraints (obstacles, soft) ===
        g_ineq = []
        for k in range(self.N + 1):
            for i in range(self.n_obstacles):
                ox = obs_params[i * 3]
                oy = obs_params[i * 3 + 1]
                r = obs_params[i * 3 + 2]
                dist_sq = (X[0, k] - ox) ** 2 + (X[1, k] - oy) ** 2
                # Constraint: r² - dist² - slack <= 0
                g_ineq.append(r**2 - dist_sq - S[i, k])

        g_ineq = ca.vertcat(*g_ineq)
        n_ineq = g_ineq.shape[0]

        # === Pack variables ===
        w = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)), S.reshape((-1, 1)))
        # p structure: Q_diag(6), R_diag(3), x_init(6), x_ref(6), obs(n_obs*3), slack(1), z_target(2)
        p = ca.vertcat(Q_diag, R_diag, x_init, x_ref, obs_params, slack_weight, z_target)

        self.n_w = n_w
        self.n_p = p.shape[0]
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.n_x_vars = n_x_vars

        self.idx_Q = slice(0, 6)
        self.idx_R = slice(6, 9)

        # === Dual variables ===
        lam_eq = ca.MX.sym("lam_eq", n_eq)
        lam_ineq = ca.MX.sym("lam_ineq", n_ineq)

        # === Lagrangian ===
        L = cost + ca.dot(lam_eq, g_eq) + ca.dot(lam_ineq, g_ineq)

        # === KKT system ===

        # Gradient of Lagrangian w.r.t. primal
        grad_w_L = ca.gradient(L, w)

        # Hessian of Lagrangian w.r.t. primal
        hess_ww_L = ca.hessian(L, w)[0]

        # Jacobian of equality constraints
        jac_geq_w = ca.jacobian(g_eq, w)

        # Jacobian of inequality constraints
        jac_gineq_w = ca.jacobian(g_ineq, w)

        # === Sensitivity system (simplified for active set) ===
        # For equality constraints and active inequalities:
        # [H    A_eq^T   A_ineq^T] [dw/dp  ]   [∂²L/∂w∂p    ]
        # [A_eq   0        0     ] [dλ_eq/dp] = -[∂g_eq/∂p   ]
        # [A_ineq 0        0     ] [dλ_ineq/dp]  [∂g_ineq/∂p ]

        # Mixed partials
        grad_w_L_jac_p = ca.jacobian(grad_w_L, p)
        jac_geq_p = ca.jacobian(g_eq, p)
        jac_gineq_p = ca.jacobian(g_ineq, p)

        # === Build sensitivity functions ===

        # Full primal-dual variables
        lam = ca.vertcat(lam_eq, lam_ineq)

        # Cost sensitivity: ∂J*/∂p (via envelope theorem at optimum)
        # At optimum: ∂J*/∂p = ∂L/∂p
        grad_L_p = ca.gradient(L, p)

        self.cost_sensitivity_fn = ca.Function(
            "cost_sens",
            [w, lam, p],
            [grad_L_p],
            ["w", "lam", "p"],
            ["dJ_dp"],
        )

        self.kkt_terms_fn = ca.Function(
            "kkt_terms",
            [w, lam, p],
            [
                hess_ww_L,
                grad_w_L_jac_p,
                jac_geq_w,
                jac_geq_p,
                jac_gineq_w,
                jac_gineq_p,
                g_ineq,
            ],
            ["w", "lam", "p"],
            [
                "hess_ww_L",
                "grad_w_L_jac_p",
                "jac_geq_w",
                "jac_geq_p",
                "jac_gineq_w",
                "jac_gineq_p",
                "g_ineq",
            ],
        )
        self.u0_start = n_x_vars

        print("  ✓ CasADi sensitivity functions built (IFT with active-set KKT)")

        # z_target starts at index 22 + n_obs*3 in parameter vector
        # p: Q(6) + R(3) + x_init(6) + x_ref(6) + obs(n_obs*3) + slack(1) + z_target(2)
        self.idx_z_target = slice(22 + self.n_obstacles * 3, 24 + self.n_obstacles * 3)

    def _build_nlp_solver(self):
        """Build CasADi NLP solver for fallback and to get (w*, λ*)."""

        # Reuse the same formulation
        X = ca.MX.sym("X", self.nx, self.N + 1)
        U = ca.MX.sym("U", self.nu, self.N)
        S = ca.MX.sym("S", self.n_obstacles, self.N + 1)

        Q_diag = ca.MX.sym("Q_diag", self.nx)
        R_diag = ca.MX.sym("R_diag", self.nu)
        x_init = ca.MX.sym("x_init", self.nx)
        x_ref = ca.MX.sym("x_ref", self.nx)
        obs_params = ca.MX.sym("obs", self.n_obstacles * 3)
        slack_weight = ca.MX.sym("slack_w", 1)
        z_target = ca.MX.sym("z_target", 2)

        # Cost
        cost = 0
        for k in range(self.N):
            x_err = X[:, k] - x_ref
            cost += ca.mtimes([x_err.T, ca.diag(Q_diag), x_err])
            cost += ca.mtimes([U[:, k].T, ca.diag(R_diag), U[:, k]])
            for i in range(self.n_obstacles):
                cost += slack_weight * S[i, k]
                cost += slack_weight * 0.1 * S[i, k] ** 2

        # Terminal: tracks the learned x/y target while preserving the rest of x_ref.
        x_ref_terminal = ca.vertcat(z_target, x_ref[2:])
        x_err_N = X[:, self.N] - x_ref_terminal
        cost += SharedMPCFormulation.TERMINAL_COST_MULTIPLIER * ca.mtimes(
            [x_err_N.T, ca.diag(Q_diag), x_err_N]
        )
        for i in range(self.n_obstacles):
            cost += slack_weight * S[i, self.N]

        # Constraints
        g = []
        lbg, ubg = [], []

        # Initial condition
        g.append(X[:, 0] - x_init)
        lbg.extend([0.0] * self.nx)
        ubg.extend([0.0] * self.nx)

        # Dynamics
        for k in range(self.N):
            x_next = SharedMPCFormulation.discrete_dynamics(X[:, k], U[:, k], self.dt)
            g.append(X[:, k + 1] - x_next)
            lbg.extend([0.0] * self.nx)
            ubg.extend([0.0] * self.nx)

        # Obstacles
        for k in range(self.N + 1):
            for i in range(self.n_obstacles):
                ox = obs_params[i * 3]
                oy = obs_params[i * 3 + 1]
                r = obs_params[i * 3 + 2]
                dist_sq = (X[0, k] - ox) ** 2 + (X[1, k] - oy) ** 2
                g.append(r**2 - dist_sq - S[i, k])
                lbg.append(-ca.inf)
                ubg.append(0.0)

        g = ca.vertcat(*g)

        # Decision variables
        w = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)), S.reshape((-1, 1)))

        # Bounds
        lbw, ubw = [], []
        for _ in range(self.N + 1):
            lbw.extend(SharedMPCFormulation.x_min.tolist())
            ubw.extend(SharedMPCFormulation.x_max.tolist())
        for _ in range(self.N):
            lbw.extend(SharedMPCFormulation.u_min.tolist())
            ubw.extend(SharedMPCFormulation.u_max.tolist())
        for _ in range((self.N + 1) * self.n_obstacles):
            lbw.append(0.0)
            ubw.append(1e6)

        self.lbw = np.array(lbw)
        self.ubw = np.array(ubw)
        self.lbg = np.array(lbg)
        self.ubg = np.array(ubg)

        # Parameters: Q(6) + R(3) + x_init(6) + x_ref(6) + obs(n_obs*3) + slack(1) + z_target(2)
        p = ca.vertcat(Q_diag, R_diag, x_init, x_ref, obs_params, slack_weight, z_target)

        nlp = {"x": w, "f": cost, "g": g, "p": p}
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 200,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.tol": 1e-4,
        }

        self.nlp_solver = ca.nlpsol("casadi_mpc", "ipopt", nlp, opts)

    def _pack_params(
        self,
        Q_diag: np.ndarray,
        R_diag: np.ndarray,
        x_init: np.ndarray,
        x_ref: np.ndarray,
        obstacles: List[Dict],
        slack_weight: float = 50000.0,  # Very high penalty for obstacle violation
        z_target: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Pack parameters."""
        # Dummy positions at moderate distance for better numerical conditioning
        # (1000m causes huge Jacobians that destabilize QP solver)
        DUMMY_POSITIONS = [(50.0, 50.0), (60.0, 50.0), (50.0, 60.0)]

        obs_flat = []
        for i in range(self.n_obstacles):
            if i < len(obstacles):
                o = obstacles[i]
                obs_flat.extend([o["x"], o["y"], o["radius"]])
            else:
                dx, dy = DUMMY_POSITIONS[i % len(DUMMY_POSITIONS)]
                obs_flat.extend([dx, dy, 0.1])

        z = np.array(x_ref[:2], dtype=float) if z_target is None else z_target

        return np.concatenate(
            [Q_diag, R_diag, x_init, x_ref, np.array(obs_flat), [slack_weight], z]
        )

    def solve_and_get_sensitivities(
        self,
        x_init: np.ndarray,
        x_ref: np.ndarray,
        Q_diag: np.ndarray,
        R_diag: np.ndarray,
        obstacles: List[Dict],
        z_target: Optional[np.ndarray] = None,
    ) -> Tuple[MPCSolution, MPCSensitivity]:
        """
        Solve NLP and compute sensitivities.

        Used when Acados is not available, or for verification.
        """
        import time

        t_start = time.time()

        Q_diag = np.clip(Q_diag, 1.0, 200.0)
        R_diag = np.clip(R_diag, 0.1, 10.0)

        p = self._pack_params(Q_diag, R_diag, x_init, x_ref, obstacles, z_target=z_target)

        # Initial guess — Fix 1: residual-gated warm start.
        # Discard w_warm when robot has moved > threshold from the state at which
        # it was computed; stale trajectories cause 3-4× slowdowns in obstacle-dense
        # environments (validated in tests/mpc_coldstart/test_fixes.py).
        def _straight_line_w0():
            w0 = np.zeros(self.n_w)
            for k in range(self.N + 1):
                alpha = k / self.N
                w0[k * self.nx : (k + 1) * self.nx] = x_init * (1 - alpha) + x_ref * alpha
            return w0

        if self.w_warm is not None and self._warm_anchor is not None:
            dist = float(np.linalg.norm(self._warm_anchor[:2] - x_init[:2]))
            if dist > self._WARM_DIST_THRESHOLD:
                w0 = _straight_line_w0()   # discard stale warm-start
            else:
                w0 = self.w_warm
        elif self.w_warm is not None:
            w0 = self.w_warm
        else:
            w0 = _straight_line_w0()

        # Solve
        try:
            kwargs = dict(
                x0=w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=p
            )
            if self.lam_warm is not None:
                kwargs["lam_g0"] = self.lam_warm
            sol = self.nlp_solver(**kwargs)
            success = self.nlp_solver.stats()["success"]
        except Exception as e:
            print(f"CasADi solve failed: {e}")
            success = False
            sol = None

        solve_time = time.time() - t_start

        if not success or sol is None:
            return (
                MPCSolution(
                    success=False,
                    control=np.zeros(self.nu),
                    trajectory=None,
                    cost=np.inf,
                    solve_time=solve_time,
                    solver_used="casadi",
                ),
                MPCSensitivity(
                    success=False,
                    dJ_dQ=np.zeros(self.nx),
                    dJ_dR=np.zeros(self.nu),
                    du0_dQ=np.zeros((self.nu, self.nx)),
                    du0_dR=np.zeros((self.nu, self.nu)),
                    compute_time=0.0,
                ),
            )

        w_opt = np.array(sol["x"]).flatten()
        lam_opt = np.array(sol["lam_g"]).flatten()

        self.w_warm = w_opt
        self.lam_warm = lam_opt
        self._warm_anchor = x_init.copy()   # Fix 1: record where this solution was computed

        # Extract solution
        u0 = w_opt[self.n_x_vars : self.n_x_vars + self.nu]
        X_opt = w_opt[: self.n_x_vars].reshape((self.N + 1, self.nx))

        solution = MPCSolution(
            success=True,
            control=u0,
            trajectory=X_opt,
            cost=float(sol["f"]),
            solve_time=solve_time,
            solver_used="casadi",
            w_opt=w_opt,
            lam_opt=lam_opt,
        )

        # Compute sensitivities
        sens = self.compute_sensitivities(w_opt, lam_opt, p)

        return solution, sens

    def _active_bound_jacobian(self, w_opt: np.ndarray) -> np.ndarray:
        """Return Jacobian rows for active finite variable bounds."""
        if not hasattr(self, "lbw") or not hasattr(self, "ubw"):
            return np.zeros((0, self.n_w))

        lbw = np.asarray(self.lbw, dtype=float)
        ubw = np.asarray(self.ubw, dtype=float)
        w = np.asarray(w_opt, dtype=float)

        lower_active = np.isfinite(lbw) & (w <= lbw + self.active_constraint_tol)
        upper_active = np.isfinite(ubw) & (w >= ubw - self.active_constraint_tol)
        active_idx = np.flatnonzero(lower_active | upper_active)

        if active_idx.size == 0:
            return np.zeros((0, self.n_w))

        jac = np.zeros((active_idx.size, self.n_w))
        jac[np.arange(active_idx.size), active_idx] = 1.0
        return jac

    @staticmethod
    def _solve_kkt_linear_system(
        kkt_matrix: np.ndarray,
        kkt_rhs: np.ndarray,
    ) -> np.ndarray:
        """Solve KKT sensitivities, falling back to least squares if singular."""
        try:
            return np.linalg.solve(kkt_matrix, kkt_rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(kkt_matrix, kkt_rhs, rcond=1e-10)[0]

    def _compute_policy_sensitivity_active_set(
        self,
        w_opt: np.ndarray,
        lam_opt: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Compute ∂u*_0/∂p using only active inequalities in the KKT system."""
        (
            hess_ww_L,
            grad_w_L_jac_p,
            jac_geq_w,
            jac_geq_p,
            jac_gineq_w,
            jac_gineq_p,
            g_ineq,
        ) = self.kkt_terms_fn(w_opt, lam_opt, p)

        hess_ww_L = np.asarray(hess_ww_L, dtype=float)
        grad_w_L_jac_p = np.asarray(grad_w_L_jac_p, dtype=float)
        jac_geq_w = np.asarray(jac_geq_w, dtype=float)
        jac_geq_p = np.asarray(jac_geq_p, dtype=float)
        jac_gineq_w = np.asarray(jac_gineq_w, dtype=float)
        jac_gineq_p = np.asarray(jac_gineq_p, dtype=float)
        g_ineq = np.asarray(g_ineq, dtype=float).reshape(-1)

        lam_ineq = np.asarray(lam_opt[self.n_eq : self.n_eq + self.n_ineq], dtype=float)
        active_ineq = np.flatnonzero(
            (g_ineq >= -self.active_constraint_tol)
            | (np.abs(lam_ineq) > self.active_multiplier_tol)
        )

        constraint_jacobians = [jac_geq_w]
        constraint_param_jacobians = [jac_geq_p]

        if active_ineq.size > 0:
            constraint_jacobians.append(jac_gineq_w[active_ineq, :])
            constraint_param_jacobians.append(jac_gineq_p[active_ineq, :])

        active_bounds = self._active_bound_jacobian(w_opt)
        if active_bounds.shape[0] > 0:
            constraint_jacobians.append(active_bounds)
            constraint_param_jacobians.append(
                np.zeros((active_bounds.shape[0], self.n_p))
            )

        constraint_jacobian = np.vstack(constraint_jacobians)
        constraint_param_jacobian = np.vstack(constraint_param_jacobians)
        n_constraints = constraint_jacobian.shape[0]

        kkt_matrix = np.block(
            [
                [hess_ww_L, constraint_jacobian.T],
                [
                    constraint_jacobian,
                    np.zeros((n_constraints, n_constraints)),
                ],
            ]
        )
        kkt_rhs = -np.vstack([grad_w_L_jac_p, constraint_param_jacobian])

        dzeta_dp = self._solve_kkt_linear_system(kkt_matrix, kkt_rhs)
        dw_dp = dzeta_dp[: self.n_w, :]
        return dw_dp[self.u0_start : self.u0_start + self.nu, :]

    def compute_sensitivities(
        self,
        w_opt: np.ndarray,
        lam_opt: np.ndarray,
        p: np.ndarray,
    ) -> MPCSensitivity:
        """
        Compute sensitivities given optimal primal-dual solution.

        This is called after Acados solve to get gradients without re-solving.
        """
        import time

        t_start = time.time()

        try:
            # Cost sensitivity
            dJ_dp = np.array(self.cost_sensitivity_fn(w_opt, lam_opt, p)).flatten()

            # Policy sensitivity
            du0_dp = self._compute_policy_sensitivity_active_set(w_opt, lam_opt, p)

            # Extract Q, R, and z_target components
            dJ_dQ = dJ_dp[self.idx_Q]
            dJ_dR = dJ_dp[self.idx_R]
            dJ_dz_target = dJ_dp[self.idx_z_target]
            du0_dQ = du0_dp[:, self.idx_Q]
            du0_dR = du0_dp[:, self.idx_R]

            success = True
        except Exception as e:
            print(f"Sensitivity computation failed: {e}")
            dJ_dQ = np.zeros(self.nx)
            dJ_dR = np.zeros(self.nu)
            dJ_dz_target = np.zeros(2)
            du0_dQ = np.zeros((self.nu, self.nx))
            du0_dR = np.zeros((self.nu, self.nu))
            success = False

        return MPCSensitivity(
            success=success,
            dJ_dQ=dJ_dQ,
            dJ_dR=dJ_dR,
            du0_dQ=du0_dQ,
            du0_dR=du0_dR,
            compute_time=time.time() - t_start,
            dJ_dz_target=dJ_dz_target,
        )

    def reset(
        self,
        x_init: Optional[np.ndarray] = None,
        x_ref:  Optional[np.ndarray] = None,
    ):
        """Reset warm start.

        Fix 3: if x_init and x_ref are provided, pre-populate w_warm with a
        straight-line trajectory so the first post-reset solve starts from a
        reasonable guess instead of zero (7× speedup on large heading changes).
        """
        self.lam_warm     = None
        self._warm_anchor = None

        if x_init is not None and x_ref is not None:
            w0 = np.zeros(self.n_w)
            for k in range(self.N + 1):
                alpha = k / self.N
                w0[k * self.nx : (k + 1) * self.nx] = x_init * (1 - alpha) + x_ref * alpha
            self.w_warm       = w0
            self._warm_anchor = x_init.copy()
        else:
            self.w_warm = None
