"""
MPC solvers: Acados (fast real-time control) + HybridMPC orchestrator.

AcadosSolver   — SQP-RTI for ~1-5ms solve times; exports (w*, λ*) for IFT.
HybridMPC      — Routes solve calls to Acados or CasADi and can aggregate
                 sensitivities for standalone analyses.

Architecture (Section 6.7):
    CONTROL PATH:  Acados SQP-RTI  ──►  u* (~1-5ms)
    SENSITIVITY PATH: Acados solve ─► (w*, λ*, p) ─► CasADi IFT ─► ∂J*/∂p
"""

from __future__ import annotations

import numpy as np
import casadi as ca
import os
import shutil
import tempfile
import time
import uuid
import warnings
import importlib.util
from typing import Dict, List, Optional, Tuple

from core.execution.formulation import (
    MPCSolution,
    MPCSensitivity,
    SharedMPCFormulation,
)
from core.execution.ift_engine import CasADiSensitivityComputer


class AcadosRuntimeError(RuntimeError):
    """Raised when Acados is required but cannot provide the MPC control solve."""


# =============================================================================
# ACADOS SOLVER (Fast Real-Time Control)
# =============================================================================


class AcadosSolver:
    """
    Acados OCP solver for fast MPC.

    Uses SQP-RTI (Real-Time Iteration) for ~1-5ms solve times.
    Q_diag and R_diag are runtime parameters (no rebuild needed).

    Exports (w*, λ*) for CasADi sensitivity computation.
    """

    _dll_dir_handles = []
    _dll_dirs_added = set()

    def __init__(
        self,
        horizon: int = 40,
        dt: float = 0.2,
        n_obstacles: int = 3,
        build_dir: Optional[str] = None,
    ):
        self.N = horizon
        self.dt = dt
        self.nx = SharedMPCFormulation.nx
        self.nu = SharedMPCFormulation.nu
        self.n_obstacles = n_obstacles
        self._build_root = self._make_default_build_root(build_dir)
        self._build_index = 0
        self._model_name = "hybrid_mpc"
        self.build_dir: Optional[str] = None

        self.available = False
        self.ocp_solver = None
        self.ny = self.nx + self.nu
        self.last_error: Optional[str] = None
        self._current_obstacles = None
        self._acados_warm_anchor: Optional[np.ndarray] = None
        self._ACADOS_WARM_DIST_THRESHOLD = 1.0

        # Check if acados is available at all
        if importlib.util.find_spec("acados_template") is not None:
            self._acados_available = True
        else:
            self._acados_available = False
            self.last_error = "acados_template not installed"
            warnings.warn(
                "acados_template not installed. Run: pip install acados_template\n"
                "Also need Acados built: https://docs.acados.org/installation/"
            )

    @staticmethod
    def _make_default_build_root(build_dir: Optional[str] = None) -> str:
        root = build_dir or os.environ.get("MLC_ACADOS_BUILD_ROOT")
        if os.name == "nt":
            if root is None:
                root = os.path.join(os.getcwd(), ".acados_runtime")
            os.makedirs(root, exist_ok=True)
            return AcadosSolver._windows_short_path(root)
        if root is not None:
            os.makedirs(root, exist_ok=True)
            return root
        return tempfile.mkdtemp(prefix="acados_hybrid_mpc_")

    def _make_build_dir(self) -> str:
        self._build_index += 1
        suffix = f"{os.getpid()}_{self._build_index}_{uuid.uuid4().hex[:8]}"
        self._model_name = f"hybrid_mpc_{suffix}"
        build_dir = os.path.join(self._build_root, f"build_{suffix}")
        os.makedirs(build_dir, exist_ok=False)
        return build_dir

    @staticmethod
    def _windows_short_path(path: str) -> str:
        try:
            import ctypes

            size = ctypes.windll.kernel32.GetShortPathNameW(path, None, 0)
            if size == 0:
                return path
            buf = ctypes.create_unicode_buffer(size)
            ctypes.windll.kernel32.GetShortPathNameW(path, buf, size)
            return buf.value or path
        except Exception:
            return path

    @classmethod
    def _add_windows_dll_dir(cls, path: str):
        if not path or not os.path.isdir(path) or path in cls._dll_dirs_added:
            return
        cls._dll_dirs_added.add(path)
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            cls._dll_dir_handles.append(os.add_dll_directory(path))

    def _ensure_windows_dll_paths(self):
        if os.name != "nt":
            return

        acados_source = os.environ.get("ACADOS_SOURCE_DIR", "")
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        candidates = [
            self.build_dir,
            os.path.join(acados_source, "bin"),
            os.path.join(acados_source, "lib"),
            os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
            os.path.join(conda_prefix, "Library", "bin"),
            os.path.join(conda_prefix, "Scripts"),
        ]
        for path in candidates:
            self._add_windows_dll_dir(path)

    def _ensure_windows_solver_dll_alias(self, model_name: str):
        if os.name != "nt":
            return

        generated = os.path.join(
            self.build_dir, f"libacados_ocp_solver_{model_name}.dll"
        )
        expected = os.path.join(self.build_dir, f"acados_ocp_solver_{model_name}.dll")
        if os.path.exists(generated):
            shutil.copy2(generated, expected)

    @staticmethod
    def _obstacle_parameter_vector(obstacles: List[Dict]) -> np.ndarray:
        p_obs = np.zeros(3 * len(obstacles))
        for i, obs in enumerate(obstacles):
            p_obs[3 * i : 3 * i + 3] = [
                float(obs["x"]),
                float(obs["y"]),
                float(obs["radius"]),
            ]
        return p_obs

    def _build(self, obstacles: List[Dict] = None):
        """Build Acados OCP with baked obstacles."""
        try:
            from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
        except ImportError:
            warnings.warn(
                "acados_template not installed. Run: pip install acados_template\n"
                "Also need Acados built: https://docs.acados.org/installation/"
            )
            return

        try:
            obs_list = obstacles if obstacles else []
            self.last_error = None
            self._do_build(AcadosOcp, AcadosOcpSolver, AcadosModel, obs_list)
            self.available = True
            print(
                f"  ✓ Acados solver built (N={self.N}, dt={self.dt}, obs={len(obs_list)})"
            )
        except Exception as e:
            self.last_error = str(e)
            warnings.warn(f"Acados build failed: {e}")
            self.available = False

    def _do_build(self, AcadosOcp, AcadosOcpSolver, AcadosModel, obstacles: List[Dict]):
        """Internal build with obstacle constraints parameterized at runtime."""

        self.build_dir = self._make_build_dir()

        # === Model ===
        model = AcadosModel()
        model.name = self._model_name

        x = ca.MX.sym("x", self.nx)
        u = ca.MX.sym("u", self.nu)
        x_dot = ca.MX.sym("x_dot", self.nx)

        f_expl = SharedMPCFormulation.continuous_dynamics(x, u)

        model.x = x
        model.u = u
        model.xdot = x_dot
        model.f_expl_expr = f_expl
        model.f_impl_expr = x_dot - f_expl

        # Obstacle positions/radii are runtime parameters. The solver only needs
        # rebuilding when the number of obstacle constraints changes.
        n_obs = len(obstacles) if obstacles else 0
        if n_obs > 0:
            model.p = ca.MX.sym("p_obs", 3 * n_obs)

        # === Cost: LINEAR_LS (native Acados, works with soft constraints) ===
        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.dt

        ny = self.nx + self.nu
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        # Vx extracts state into output
        Vx = np.zeros((ny, self.nx))
        Vx[: self.nx, :] = np.eye(self.nx)
        ocp.cost.Vx = Vx

        # Vu extracts control into output
        Vu = np.zeros((ny, self.nu))
        Vu[self.nx :, :] = np.eye(self.nu)
        ocp.cost.Vu = Vu

        # Terminal: state only
        ocp.cost.Vx_e = np.eye(self.nx)

        # Default weight matrix (will be updated at solve time)
        Q_default = np.diag(SharedMPCFormulation.Q_default)
        R_default = np.diag(SharedMPCFormulation.R_default)
        W = np.zeros((ny, ny))
        W[: self.nx, : self.nx] = Q_default
        W[self.nx :, self.nx :] = R_default
        ocp.cost.W = W
        ocp.cost.W_e = (
            Q_default * SharedMPCFormulation.TERMINAL_COST_MULTIPLIER
        )

        # Reference (will be updated at solve time)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(self.nx)

        # === Constraints ===

        # Control bounds
        ocp.constraints.lbu = SharedMPCFormulation.u_min
        ocp.constraints.ubu = SharedMPCFormulation.u_max
        ocp.constraints.idxbu = np.arange(self.nu)

        # State bounds
        ocp.constraints.lbx = SharedMPCFormulation.x_min
        ocp.constraints.ubx = SharedMPCFormulation.x_max
        ocp.constraints.idxbx = np.arange(self.nx)

        # Initial state (set at runtime)
        ocp.constraints.lbx_0 = np.zeros(self.nx)
        ocp.constraints.ubx_0 = np.zeros(self.nx)
        ocp.constraints.idxbx_0 = np.arange(self.nx)

        # === Obstacle avoidance ===
        if n_obs > 0:
            h_expr = []
            for i in range(n_obs):
                ox = model.p[3 * i]
                oy = model.p[3 * i + 1]
                r = model.p[3 * i + 2]
                dist_sq = (x[0] - ox) ** 2 + (x[1] - oy) ** 2
                # h = r² - dist² ≤ 0 means dist ≥ r (outside obstacle)
                h_expr.append(r**2 - dist_sq)

            h = ca.vertcat(*h_expr)
            ns = n_obs

            # INITIAL stage (k=0)
            model.con_h_expr_0 = h
            ocp.constraints.lh_0 = np.full(ns, -1e9)
            ocp.constraints.uh_0 = np.zeros(ns)
            ocp.constraints.idxsh_0 = np.arange(ns)

            # PATH stages (k=1..N-1)
            model.con_h_expr = h
            ocp.constraints.lh = np.full(ns, -1e9)
            ocp.constraints.uh = np.zeros(ns)
            ocp.constraints.idxsh = np.arange(ns)

            # TERMINAL stage (k=N)
            model.con_h_expr_e = h
            ocp.constraints.lh_e = np.full(ns, -1e9)
            ocp.constraints.uh_e = np.zeros(ns)
            ocp.constraints.idxsh_e = np.arange(ns)

            # Slack penalties - match working ObstacleAwareMPC
            L1_penalty = 100000.0
            L2_penalty = 50000.0

            # Initial stage - set ALL slack penalties (both bounds)
            ocp.cost.zl_0 = L2_penalty * np.ones(ns)
            ocp.cost.Zl_0 = L1_penalty * np.ones(ns)
            ocp.cost.zu_0 = L2_penalty * np.ones(ns)
            ocp.cost.Zu_0 = L1_penalty * np.ones(ns)

            # Path stages
            ocp.cost.zl = L2_penalty * np.ones(ns)
            ocp.cost.Zl = L1_penalty * np.ones(ns)
            ocp.cost.zu = L2_penalty * np.ones(ns)
            ocp.cost.Zu = L1_penalty * np.ones(ns)

            # Terminal stage
            ocp.cost.zl_e = L2_penalty * np.ones(ns)
            ocp.cost.Zl_e = L1_penalty * np.ones(ns)
            ocp.cost.zu_e = L2_penalty * np.ones(ns)
            ocp.cost.Zu_e = L1_penalty * np.ones(ns)
            ocp.parameter_values = self._obstacle_parameter_vector(obstacles)

        # === Solver options ===
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP"  # Full SQP for obstacle avoidance
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.tol = 1e-4
        ocp.solver_options.print_level = 0  # suppress QP solver stdout noise

        # Build
        os.makedirs(self.build_dir, exist_ok=True)
        self._ensure_windows_dll_paths()
        json_file = os.path.join(self.build_dir, "acados_ocp.json")
        ocp.code_gen_opts.code_export_directory = self.build_dir
        ocp.code_gen_opts.json_file = json_file

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Call to deprecated function .*N\.",
                category=DeprecationWarning,
            )
            ocp.make_consistent(verbose=False)
        ocp.generate_external_functions()
        ocp.dump_to_json()
        ocp.render_templates()
        AcadosOcpSolver.build(self.build_dir, with_cython=False, verbose=False)
        self._ensure_windows_solver_dll_alias(model.name)
        self.ocp_solver = AcadosOcpSolver(
            None, json_file=json_file, generate=False, build=False, verbose=False
        )
        self.ny = ny
        self._current_obstacles = [obs.copy() for obs in obstacles] if obstacles else []

    def _needs_rebuild(self, obstacles: List[Dict]) -> bool:
        """Check if solver needs rebuilding due to obstacle constraint count change.

        Obstacle positions and radii are runtime Acados parameters. Rebuilds are
        only necessary when the compiled constraint dimension changes.
        """
        if not self._acados_available:
            return False  # Can't build anyway
        if self._current_obstacles is None:
            return True
        if len(obstacles) != len(self._current_obstacles):
            return True
        return False

    def solve(
        self,
        x_init: np.ndarray,
        x_ref: np.ndarray,
        Q_diag: np.ndarray,
        R_diag: np.ndarray,
        obstacles: List[Dict],
        z_target: Optional[np.ndarray] = None,
    ) -> MPCSolution:
        """
        Solve MPC with Acados (LINEAR_LS cost, runtime obstacle parameters).

        Rebuilds solver only if the obstacle constraint count changes.
        """
        if not self._acados_available:
            self.last_error = self.last_error or "acados_template unavailable"
            return MPCSolution(
                success=False,
                control=np.zeros(self.nu),
                trajectory=None,
                cost=np.inf,
                solve_time=0.0,
                solver_used="acados",
            )

        # Rebuild if obstacles changed
        if self._needs_rebuild(obstacles):
            self._build(obstacles)

        if not self.available:
            self.last_error = self.last_error or "Acados solver unavailable"
            return MPCSolution(
                success=False,
                control=np.zeros(self.nu),
                trajectory=None,
                cost=np.inf,
                solve_time=0.0,
                solver_used="acados",
            )

        t_start = time.time()

        Q_diag = np.clip(Q_diag, 1.0, 200.0)
        R_diag = np.clip(R_diag, 0.1, 10.0)

        # Build weight matrices for LINEAR_LS cost
        W = np.zeros((self.ny, self.ny))
        W[: self.nx, : self.nx] = np.diag(Q_diag)
        W[self.nx :, self.nx :] = np.diag(R_diag)
        W_e = (
            np.diag(Q_diag) * SharedMPCFormulation.TERMINAL_COST_MULTIPLIER
        )

        # Set initial state
        self.ocp_solver.set(0, "lbx", x_init)
        self.ocp_solver.set(0, "ubx", x_init)

        if obstacles:
            p_obs = self._obstacle_parameter_vector(obstacles)
            for k in range(self.N + 1):
                self.ocp_solver.set(k, "p", p_obs)

        # Set reference and weights at each stage
        yref = np.concatenate([x_ref, np.zeros(self.nu)])
        for k in range(self.N):
            self.ocp_solver.cost_set(k, "W", W)
            self.ocp_solver.cost_set(k, "yref", yref)

        # Terminal stage
        terminal_ref = x_ref.copy()
        if z_target is not None:
            terminal_ref[:2] = np.array(z_target, dtype=float)
        self.ocp_solver.cost_set(self.N, "W", W_e)
        self.ocp_solver.cost_set(self.N, "yref", terminal_ref)

        # Initialize trajectory — straight line from current state to reference.
        # This gives the SQP a consistent, bounded starting point every solve.
        for k in range(self.N + 1):
            alpha = k / self.N
            self.ocp_solver.set(k, "x", x_init * (1 - alpha) + x_ref * alpha)
        for k in range(self.N):
            self.ocp_solver.set(k, "u", np.zeros(self.nu))

        # Solve
        status = self.ocp_solver.solve()
        solve_time = time.time() - t_start

        if status == 0:
            u0 = self.ocp_solver.get(0, "u")

            # Extract full trajectory
            trajectory = np.zeros((self.N + 1, self.nx))
            for k in range(self.N + 1):
                trajectory[k] = self.ocp_solver.get(k, "x")

            # Extract primal solution for CasADi sensitivities
            w_opt = np.concatenate(
                [
                    trajectory.flatten(),
                    np.array(
                        [self.ocp_solver.get(k, "u") for k in range(self.N)]
                    ).flatten(),
                ]
            )

            cost = self.ocp_solver.get_cost()

            # Update anchor so next call can detect if robot has moved too far
            self._acados_warm_anchor = x_init.copy()

            return MPCSolution(
                success=True,
                control=u0,
                trajectory=trajectory,
                cost=cost,
                solve_time=solve_time,
                solver_used="acados",
                w_opt=w_opt,
                lam_opt=None,
            )
        else:
            self.last_error = f"Acados solver returned status {status}"
            return MPCSolution(
                success=False,
                control=np.zeros(self.nu),
                trajectory=None,
                cost=np.inf,
                solve_time=solve_time,
                solver_used="acados",
            )

    def reset(self, x_init=None, x_ref=None):
        """Reset solver state.

        If x_init and x_ref are provided, pre-populates the Acados trajectory
        with a straight-line interpolation so the first solve after reset avoids
        the zero-init cold-start penalty (Fix 3).
        """
        self._acados_warm_anchor = None
        if self.available:
            if x_init is not None and x_ref is not None:
                for k in range(self.N + 1):
                    alpha = k / self.N
                    x_k = x_init * (1 - alpha) + x_ref * alpha
                    self.ocp_solver.set(k, "x", x_k)
            else:
                for k in range(self.N + 1):
                    self.ocp_solver.set(k, "x", np.zeros(self.nx))
            for k in range(self.N):
                self.ocp_solver.set(k, "u", np.zeros(self.nu))


# =============================================================================
# HYBRID MPC CONTROLLER
# =============================================================================


class HybridMPC:
    """
    Hybrid MPC: Acados (fast solve) + CasADi (analytical sensitivities).

    Usage:
        mpc = HybridMPC(horizon=40, dt=0.2)
        mpc.update_parameters(Q_diag, R_diag, obstacles)

        # Fast control (Acados)
        solution = mpc.solve(x_init, x_ref)

        # Control + sensitivities (Acados + CasADi IFT)
        solution, sensitivities = mpc.solve_with_sensitivities(x_init, x_ref)
    """

    def __init__(
        self,
        horizon: int = 40,
        dt: float = 0.2,
        n_obstacles: int = 3,
        use_acados: bool = True,
    ):
        self.horizon = horizon
        self.dt = dt
        self.n_obstacles = n_obstacles
        self.nx = SharedMPCFormulation.nx
        self.nu = SharedMPCFormulation.nu

        # Current parameters
        self.Q_diag = SharedMPCFormulation.Q_default.copy()
        self.R_diag = SharedMPCFormulation.R_default.copy()
        self.obstacles: List[Dict] = []
        self.z_target: Optional[np.ndarray] = None

        # === Build solvers ===
        print("Building Hybrid MPC...")

        # CasADi (always available)
        print("  Building CasADi sensitivity computer...")
        self.casadi = CasADiSensitivityComputer(
            horizon=horizon, dt=dt, n_obstacles=n_obstacles
        )

        # Acados (optional, for speed - builds lazily on first solve)
        self.acados = None
        self.use_acados = use_acados

        if use_acados:
            print("  Building Acados solver...")
            self.acados = AcadosSolver(horizon=horizon, dt=dt, n_obstacles=n_obstacles)
            if self.acados._acados_available:
                print("  ✓ Acados available (will build on first solve with obstacles)")
            else:
                raise AcadosRuntimeError(
                    f"Acados requested but unavailable: {self.acados.last_error}"
                )

        if not self.use_acados:
            print("  ✓ CasADi-only mode")

        # Stats
        self.stats = {
            "total_solves": 0,
            "acados_solves": 0,
            "casadi_solves": 0,
            "sensitivity_computes": 0,
            "total_solve_time": 0.0,
            "total_sens_time": 0.0,
        }

        # Episode sensitivities
        self.episode_sensitivities: List[MPCSensitivity] = []

    def update_parameters(
        self,
        Q_diag: np.ndarray,
        R_diag: np.ndarray,
        obstacles: List[Dict],
        z_target: Optional[np.ndarray] = None,
    ):
        """Update MPC parameters from translator."""
        self.Q_diag = np.clip(Q_diag, 1.0, 200.0)
        self.R_diag = np.clip(R_diag, 0.1, 10.0)
        self.obstacles = obstacles
        self.z_target = None if z_target is None else z_target.copy()

    def solve(self, x_init: np.ndarray, x_ref: np.ndarray) -> MPCSolution:
        """
        Fast MPC solve (no sensitivities).

        Uses Acados when requested. CasADi control fallback is allowed only when
        the controller is explicitly constructed with use_acados=False.
        """
        self.stats["total_solves"] += 1

        if (
            self.use_acados
            and self.acados is not None
            and self.acados._acados_available
        ):
            self.stats["acados_solves"] += 1
            sol = self.acados.solve(
                x_init,
                x_ref,
                self.Q_diag,
                self.R_diag,
                self.obstacles,
                z_target=self.z_target,
            )
            self.stats["total_solve_time"] += sol.solve_time

            if sol.success:
                return sol
            raise AcadosRuntimeError(
                "Acados control solve failed; refusing CasADi fallback "
                f"(build_dir={self.acados.build_dir}, error={self.acados.last_error})"
            )

        if self.use_acados:
            raise AcadosRuntimeError("Acados requested but solver is not initialized")

        # CasADi solve
        self.stats["casadi_solves"] += 1
        sol, _ = self.casadi.solve_and_get_sensitivities(
            x_init, x_ref, self.Q_diag, self.R_diag, self.obstacles, z_target=self.z_target
        )
        self.stats["total_solve_time"] += sol.solve_time
        return sol

    def solve_with_sensitivities(
        self,
        x_init: np.ndarray,
        x_ref: np.ndarray,
    ) -> Tuple[MPCSolution, MPCSensitivity]:
        """
        MPC solve + analytical sensitivities.

        If Acados available:
            1. Acados solve → (u*, w*, trajectory)
            2. CasADi sensitivity → ∂J*/∂p, ∂u*/∂p

        If CasADi only:
            1. CasADi solve + sensitivities together
        """
        self.stats["total_solves"] += 1
        self.stats["sensitivity_computes"] += 1

        if (
            self.use_acados
            and self.acados is not None
            and self.acados._acados_available
        ):
            # Acados solve
            self.stats["acados_solves"] += 1
            sol = self.acados.solve(
                x_init,
                x_ref,
                self.Q_diag,
                self.R_diag,
                self.obstacles,
                z_target=self.z_target,
            )
            self.stats["total_solve_time"] += sol.solve_time

            if sol.success and sol.w_opt is not None:
                # Get dual variables from CasADi (Acados format is different)
                # We need to do a quick CasADi solve to get λ* in the right format
                # This is a limitation - ideally Acados would export λ* directly
                # Acados w_opt doesn't include slack variables, but CasADi does
                # Pad with zeros for slacks: w_casadi = [X, U, S]
                n_slack = self.n_obstacles * (self.horizon + 1)
                w_padded = np.concatenate([sol.w_opt, np.zeros(n_slack)])
                # Fix 2: only use Acados solution as warm-start if cost is sane.
                # A failed/nonsense Acados solve (cost >> 1e6) would otherwise
                # poison the CasADi warm-start and cause 500ms+ solves.
                if sol.cost < 1e6:
                    self.casadi.w_warm = w_padded

                casadi_sol, sens = self.casadi.solve_and_get_sensitivities(
                    x_init, x_ref, self.Q_diag, self.R_diag, self.obstacles, z_target=self.z_target
                )

                self.stats["total_sens_time"] += sens.compute_time

                if sens.success:
                    self.episode_sensitivities.append(sens)

                # Return Acados solution with CasADi sensitivities
                return sol, sens

            raise AcadosRuntimeError(
                "Acados sensitivity solve failed; refusing CasADi control fallback "
                f"(build_dir={self.acados.build_dir}, error={self.acados.last_error})"
            )

        if self.use_acados:
            raise AcadosRuntimeError("Acados requested but solver is not initialized")

        # CasADi-only path
        self.stats["casadi_solves"] += 1
        sol, sens = self.casadi.solve_and_get_sensitivities(
            x_init, x_ref, self.Q_diag, self.R_diag, self.obstacles, z_target=self.z_target
        )

        self.stats["total_solve_time"] += sol.solve_time
        self.stats["total_sens_time"] += sens.compute_time

        if sens.success:
            self.episode_sensitivities.append(sens)

        return sol, sens

    def get_aggregated_sensitivities(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate sensitivities for translator update.

        Returns mean ∂J/∂Q, ∂J/∂R, and ∂J/∂z_target over episode.
        """
        if not self.episode_sensitivities:
            return np.zeros(self.nx), np.zeros(self.nu), np.zeros(2)

        successful = [s for s in self.episode_sensitivities if s.success]
        if not successful:
            return np.zeros(self.nx), np.zeros(self.nu), np.zeros(2)

        dJ_dQ_mean = np.mean([s.dJ_dQ for s in successful], axis=0)
        dJ_dR_mean = np.mean([s.dJ_dR for s in successful], axis=0)
        dJ_dz_mean = np.mean([s.dJ_dz_target for s in successful], axis=0)

        return dJ_dQ_mean, dJ_dR_mean, dJ_dz_mean

    def reset_episode(self, x_init=None, x_ref=None):
        """Reset for new episode.

        Pass x_init and x_ref to pre-populate warm-starts with a straight-line
        trajectory (Fix 3), eliminating the zero-init cold-start penalty on the
        first solve of each episode.
        """
        self.episode_sensitivities = []
        self.casadi.reset(x_init=x_init, x_ref=x_ref)
        if self.acados is not None:
            self.acados.reset(x_init=x_init, x_ref=x_ref)
        self.stats = {
            k: 0 if isinstance(v, int) else 0.0 for k, v in self.stats.items()
        }

    def print_stats(self):
        """Print statistics."""
        total = self.stats["total_solves"]
        print(f"\n{'='*55}")
        print("HYBRID MPC STATS")
        print(f"{'='*55}")
        print(f"Total solves:         {total}")
        if total > 0:
            print(
                f"Acados solves:        {self.stats['acados_solves']} ({100*self.stats['acados_solves']/total:.1f}%)"
            )
            print(
                f"CasADi solves:        {self.stats['casadi_solves']} ({100*self.stats['casadi_solves']/total:.1f}%)"
            )
            print(
                f"Avg solve time:       {1000*self.stats['total_solve_time']/total:.2f}ms"
            )
        print(f"Sensitivity computes: {self.stats['sensitivity_computes']}")
        if self.stats["sensitivity_computes"] > 0:
            avg_sens = (
                self.stats["total_sens_time"] / self.stats["sensitivity_computes"]
            )
            print(f"Avg sensitivity time: {1000*avg_sens:.2f}ms")
        print(f"{'='*55}\n")
