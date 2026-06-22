"""
Patched solver implementations for cold-start fixes.

Three fixes from FINDINGS.md, implemented as subclasses/wrappers so the
production source in core/ is never touched on this branch.

Fix 1 - ResidualGatedNLP
    Discard w_warm when the robot has moved > threshold from the state
    where w_warm was computed (w_warm[0:nx]).  Prevents stale trajectories
    from dragging IPOPT into a bad basin in obstacle-dense environments.

Fix 2 - CostGatedWarmStart (wrapper around Fix1 NLP)
    Simulates the HybridMPC gate: only set w_warm from an Acados solution
    if its cost < sanity_threshold.  Prevents a failed/nonsense Acados
    solve from poisoning the CasADi fallback.

Fix 3 - StraightLineReset
    AcadosSolver.reset() currently zeros the trajectory.  This fix resets
    to a straight-line from x_current to x_ref, producing a better
    warm-start for the first solve after an episode reset or rebuild.
    Tested here purely in CasADi (Acados itself isn't invoked).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import casadi as ca
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs import DT, HORIZON, NU, NX, Q_DEFAULT, R_DEFAULT
from core.execution.formulation import SharedMPCFormulation


def _build_nlp(n_obstacles: int, N: int, dt: float):
    """
    Build the CasADi NLP that matches production CasADiSensitivityComputer.

    Returns (solver, lbw, ubw, lbg, ubg, n_w, n_x_vars).
    """
    nx, nu = NX, NU

    X = ca.MX.sym("X", nx, N + 1)
    U = ca.MX.sym("U", nu, N)
    S = ca.MX.sym("S", n_obstacles, N + 1)
    Q_d = ca.MX.sym("Q_d", nx)
    R_d = ca.MX.sym("R_d", nu)
    xi = ca.MX.sym("xi", nx)
    xr = ca.MX.sym("xr", nx)
    obs = ca.MX.sym("obs", n_obstacles * 3)
    sw = ca.MX.sym("sw", 1)

    cost = 0
    for k in range(N):
        xe = X[:, k] - xr
        cost += ca.mtimes([xe.T, ca.diag(Q_d), xe])
        cost += ca.mtimes([U[:, k].T, ca.diag(R_d), U[:, k]])
        for i in range(n_obstacles):
            cost += sw * S[i, k] + sw * 0.1 * S[i, k] ** 2
    xe_n = X[:, N] - xr
    cost += SharedMPCFormulation.TERMINAL_COST_MULTIPLIER * ca.mtimes(
        [xe_n.T, ca.diag(Q_d), xe_n]
    )
    for i in range(n_obstacles):
        cost += sw * S[i, N]

    g, lbg, ubg = [], [], []
    g.append(X[:, 0] - xi)
    lbg += [0.0] * nx
    ubg += [0.0] * nx
    for k in range(N):
        xn = SharedMPCFormulation.discrete_dynamics(X[:, k], U[:, k], dt)
        g.append(X[:, k + 1] - xn)
        lbg += [0.0] * nx
        ubg += [0.0] * nx
    for k in range(N + 1):
        for i in range(n_obstacles):
            ox = obs[i * 3]
            oy = obs[i * 3 + 1]
            radius = obs[i * 3 + 2]
            d2 = (X[0, k] - ox) ** 2 + (X[1, k] - oy) ** 2
            g.append(radius**2 - d2 - S[i, k])
            lbg.append(-ca.inf)
            ubg.append(0.0)

    w_s = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)), S.reshape((-1, 1)))
    p_s = ca.vertcat(Q_d, R_d, xi, xr, obs, sw)

    lbw, ubw = [], []
    for _ in range(N + 1):
        lbw += SharedMPCFormulation.x_min.tolist()
        ubw += SharedMPCFormulation.x_max.tolist()
    for _ in range(N):
        lbw += SharedMPCFormulation.u_min.tolist()
        ubw += SharedMPCFormulation.u_max.tolist()
    for _ in range((N + 1) * n_obstacles):
        lbw.append(0.0)
        ubw.append(1e6)

    nlp = {"x": w_s, "f": cost, "g": ca.vertcat(*g), "p": p_s}
    opts = {
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
        "ipopt.max_iter": 300,
        "ipopt.warm_start_init_point": "yes",
        "ipopt.tol": 1e-4,
    }
    solver = ca.nlpsol("nlp", "ipopt", nlp, opts)
    return (
        solver,
        np.array(lbw),
        np.array(ubw),
        np.array(lbg),
        np.array(ubg),
        w_s.shape[0],
        nx * (N + 1),
    )


def _pack_params(
    n_obstacles: int,
    xi: np.ndarray,
    xr: np.ndarray,
    obstacles: List[Dict],
    Q: np.ndarray = Q_DEFAULT,
    R: np.ndarray = R_DEFAULT,
    sw: float = 50000.0,
) -> np.ndarray:
    dummy = [(50.0, 50.0), (60.0, 50.0), (50.0, 60.0)]
    obs_flat = []
    for i in range(n_obstacles):
        if i < len(obstacles):
            obstacle = obstacles[i]
            obs_flat += [obstacle["x"], obstacle["y"], obstacle["radius"]]
        else:
            dx, dy = dummy[i % 3]
            obs_flat += [dx, dy, 0.1]
    return np.concatenate([Q, R, xi, xr, np.array(obs_flat), [sw]])


def _w0_straight(n_w: int, nx: int, N: int, xi: np.ndarray, xr: np.ndarray) -> np.ndarray:
    w0 = np.zeros(n_w)
    for k in range(N + 1):
        alpha = k / N
        w0[k * nx : (k + 1) * nx] = xi * (1 - alpha) + xr * alpha
    return w0


class ResidualGatedNLP:
    """
    Fix 1: discard w_warm when the robot has moved > dist_threshold metres.
    """

    def __init__(
        self,
        n_obstacles: int = 3,
        horizon: int = HORIZON,
        dt: float = DT,
        dist_threshold: float = 1.5,
    ):
        self.N = horizon
        self.dt = dt
        self.nx = NX
        self.nu = NU
        self.n_obs = n_obstacles
        self.dist_threshold = dist_threshold

        (
            self.solver,
            self.lbw,
            self.ubw,
            self.lbg,
            self.ubg,
            self.n_w,
            self.n_x_vars,
        ) = _build_nlp(n_obstacles, horizon, dt)

        self.w_warm: Optional[np.ndarray] = None
        self.lam_warm: Optional[np.ndarray] = None
        self._warm_anchor: Optional[np.ndarray] = None

    def reset(self):
        self.w_warm = None
        self.lam_warm = None
        self._warm_anchor = None

    def _pick_w0(self, xi: np.ndarray, xr: np.ndarray) -> Tuple[np.ndarray, str]:
        if self.w_warm is None:
            return _w0_straight(self.n_w, self.nx, self.N, xi, xr), "cold_straight"

        anchor = self._warm_anchor if self._warm_anchor is not None else self.w_warm[: self.nx]
        dist = float(np.linalg.norm(anchor[:2] - xi[:2]))

        if dist > self.dist_threshold:
            return _w0_straight(self.n_w, self.nx, self.N, xi, xr), f"discarded(dist={dist:.2f}m)"

        return self.w_warm, f"warm(dist={dist:.2f}m)"

    def solve(self, xi: np.ndarray, xr: np.ndarray, obstacles: List[Dict]) -> Dict:
        p = _pack_params(self.n_obs, xi, xr, obstacles)
        w0, reason = self._pick_w0(xi, xr)

        kwargs = dict(x0=w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=p)
        if self.lam_warm is not None and reason.startswith("warm"):
            kwargs["lam_g0"] = self.lam_warm

        t0 = time.perf_counter()
        try:
            sol = self.solver(**kwargs)
            success = self.solver.stats()["success"]
            iters = self.solver.stats().get("iter_count", -1)
        except Exception as exc:
            return {
                "success": False,
                "ms": 0,
                "iters": -1,
                "cost": np.inf,
                "w_opt": None,
                "reason": reason,
                "error": str(exc),
            }
        ms = (time.perf_counter() - t0) * 1000

        if not success:
            return {"success": False, "ms": ms, "iters": iters, "cost": np.inf, "w_opt": None, "reason": reason}

        w_opt = np.array(sol["x"]).flatten()
        self.w_warm = w_opt
        self.lam_warm = np.array(sol["lam_g"]).flatten()
        self._warm_anchor = xi.copy()

        return {"success": True, "ms": ms, "iters": iters, "cost": float(sol["f"]), "w_opt": w_opt, "reason": reason}


class CostGatedNLP:
    """
    Fix 2: only store w_opt as warm-start if cost < sanity threshold.
    """

    def __init__(
        self,
        n_obstacles: int = 3,
        horizon: int = HORIZON,
        dt: float = DT,
        dist_threshold: float = 1.5,
        cost_threshold: float = 1e6,
    ):
        self.N = horizon
        self.dt = dt
        self.nx = NX
        self.nu = NU
        self.n_obs = n_obstacles
        self.dist_threshold = dist_threshold
        self.cost_threshold = cost_threshold

        (
            self.solver,
            self.lbw,
            self.ubw,
            self.lbg,
            self.ubg,
            self.n_w,
            self.n_x_vars,
        ) = _build_nlp(n_obstacles, horizon, dt)

        self.w_warm: Optional[np.ndarray] = None
        self.lam_warm: Optional[np.ndarray] = None
        self._warm_anchor: Optional[np.ndarray] = None

    def reset(self):
        self.w_warm = None
        self.lam_warm = None
        self._warm_anchor = None

    def inject_bad_warmstart(self, xi: np.ndarray, xr: np.ndarray, cost: float = 1e8):
        self.w_warm = _w0_straight(self.n_w, self.nx, self.N, xi, xr) * 0.0
        self._cost_of_warm = cost
        self._warm_anchor = xi.copy()

    def _pick_w0(self, xi: np.ndarray, xr: np.ndarray) -> Tuple[np.ndarray, str]:
        if self.w_warm is None:
            return _w0_straight(self.n_w, self.nx, self.N, xi, xr), "cold_straight"

        anchor = self._warm_anchor if self._warm_anchor is not None else self.w_warm[: self.nx]
        dist = float(np.linalg.norm(anchor[:2] - xi[:2]))
        if dist > self.dist_threshold:
            return _w0_straight(self.n_w, self.nx, self.N, xi, xr), f"discarded_dist({dist:.2f}m)"

        cost_of_warm = getattr(self, "_cost_of_warm", 0.0)
        if cost_of_warm > self.cost_threshold:
            return _w0_straight(self.n_w, self.nx, self.N, xi, xr), f"discarded_cost({cost_of_warm:.0f})"

        return self.w_warm, f"warm(dist={dist:.2f}m,cost={cost_of_warm:.0f})"

    def solve(self, xi: np.ndarray, xr: np.ndarray, obstacles: List[Dict]) -> Dict:
        p = _pack_params(self.n_obs, xi, xr, obstacles)
        w0, reason = self._pick_w0(xi, xr)

        kwargs = dict(x0=w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=p)

        t0 = time.perf_counter()
        try:
            sol = self.solver(**kwargs)
            success = self.solver.stats()["success"]
            iters = self.solver.stats().get("iter_count", -1)
        except Exception as exc:
            return {
                "success": False,
                "ms": 0,
                "iters": -1,
                "cost": np.inf,
                "w_opt": None,
                "reason": reason,
                "error": str(exc),
            }
        ms = (time.perf_counter() - t0) * 1000

        if not success:
            return {"success": False, "ms": ms, "iters": iters, "cost": np.inf, "w_opt": None, "reason": reason}

        w_opt = np.array(sol["x"]).flatten()
        cost = float(sol["f"])

        if cost < self.cost_threshold:
            self.w_warm = w_opt
            self.lam_warm = np.array(sol["lam_g"]).flatten()
            self._warm_anchor = xi.copy()
            self._cost_of_warm = cost

        return {"success": True, "ms": ms, "iters": iters, "cost": cost, "w_opt": w_opt, "reason": reason}


class StraightLineResetNLP:
    """
    Fix 3: reset() initialises to a straight-line trajectory rather than zeros.
    """

    def __init__(
        self,
        n_obstacles: int = 3,
        horizon: int = HORIZON,
        dt: float = DT,
        dist_threshold: float = 1.5,
        cost_threshold: float = 1e6,
    ):
        self.N = horizon
        self.dt = dt
        self.nx = NX
        self.nu = NU
        self.n_obs = n_obstacles
        self.dist_threshold = dist_threshold
        self.cost_threshold = cost_threshold

        (
            self.solver,
            self.lbw,
            self.ubw,
            self.lbg,
            self.ubg,
            self.n_w,
            self.n_x_vars,
        ) = _build_nlp(n_obstacles, horizon, dt)

        self.w_warm: Optional[np.ndarray] = None
        self.lam_warm: Optional[np.ndarray] = None
        self._warm_anchor: Optional[np.ndarray] = None
        self._cost_of_warm: float = 0.0

    def reset(self, xi: Optional[np.ndarray] = None, xr: Optional[np.ndarray] = None):
        if xi is not None and xr is not None:
            self.w_warm = _w0_straight(self.n_w, self.nx, self.N, xi, xr)
            self._warm_anchor = xi.copy()
            self._cost_of_warm = 0.0
            self.lam_warm = None
        else:
            self.w_warm = None
            self.lam_warm = None
            self._warm_anchor = None
            self._cost_of_warm = 0.0

    def _pick_w0(self, xi: np.ndarray, xr: np.ndarray) -> Tuple[np.ndarray, str]:
        if self.w_warm is None:
            return _w0_straight(self.n_w, self.nx, self.N, xi, xr), "cold_straight"

        anchor = self._warm_anchor if self._warm_anchor is not None else self.w_warm[: self.nx]
        dist = float(np.linalg.norm(anchor[:2] - xi[:2]))
        if dist > self.dist_threshold:
            return _w0_straight(self.n_w, self.nx, self.N, xi, xr), f"discarded_dist({dist:.2f}m)"

        if self._cost_of_warm > self.cost_threshold:
            return _w0_straight(self.n_w, self.nx, self.N, xi, xr), "discarded_cost"

        return self.w_warm, f"warm(dist={dist:.2f}m)"

    def solve(self, xi: np.ndarray, xr: np.ndarray, obstacles: List[Dict]) -> Dict:
        p = _pack_params(self.n_obs, xi, xr, obstacles)
        w0, reason = self._pick_w0(xi, xr)

        kwargs = dict(x0=w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=p)
        if self.lam_warm is not None and "warm" in reason and "discarded" not in reason:
            kwargs["lam_g0"] = self.lam_warm

        t0 = time.perf_counter()
        try:
            sol = self.solver(**kwargs)
            success = self.solver.stats()["success"]
            iters = self.solver.stats().get("iter_count", -1)
        except Exception as exc:
            return {
                "success": False,
                "ms": 0,
                "iters": -1,
                "cost": np.inf,
                "w_opt": None,
                "reason": reason,
                "error": str(exc),
            }
        ms = (time.perf_counter() - t0) * 1000

        if not success:
            return {"success": False, "ms": ms, "iters": iters, "cost": np.inf, "w_opt": None, "reason": reason}

        w_opt = np.array(sol["x"]).flatten()
        cost = float(sol["f"])

        if cost < self.cost_threshold:
            self.w_warm = w_opt
            self.lam_warm = np.array(sol["lam_g"]).flatten()
            self._warm_anchor = xi.copy()
            self._cost_of_warm = cost

        return {"success": True, "ms": ms, "iters": iters, "cost": cost, "w_opt": w_opt, "reason": reason}
