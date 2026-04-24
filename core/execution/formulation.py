"""
Shared MPC problem formulation: data classes and dynamics.

Both AcadosSolver and CasADiSensitivityComputer use this as their
single source of truth for state/control dimensions, bounds, and dynamics.
"""

from __future__ import annotations

import numpy as np
import casadi as ca
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MPCSolution:
    """Result from MPC solve."""

    success: bool
    control: np.ndarray  # u*_0 (3,)
    trajectory: Optional[np.ndarray]  # X* (N+1, 6)
    cost: float
    solve_time: float
    solver_used: str  # "acados" or "casadi"

    # Primal-dual solution (needed for CasADi sensitivity computation)
    w_opt: Optional[np.ndarray] = None  # primal variables
    lam_opt: Optional[np.ndarray] = None  # dual variables (Lagrange multipliers)


@dataclass
class MPCSensitivity:
    """Sensitivities from IFT on KKT conditions."""

    success: bool
    dJ_dQ: np.ndarray  # (6,) ∂J*/∂Q_diag
    dJ_dR: np.ndarray  # (3,) ∂J*/∂R_diag
    du0_dQ: np.ndarray  # (3,6) ∂u*_0/∂Q_diag (policy sensitivity)
    du0_dR: np.ndarray  # (3,3) ∂u*_0/∂R_diag
    compute_time: float
    dJ_dz_target: np.ndarray = field(default_factory=lambda: np.zeros(2))  # (2,) ∂J*/∂z_target


# =============================================================================
# SHARED PROBLEM FORMULATION
# =============================================================================


class SharedMPCFormulation:
    """
    Single source of truth for the MPC problem.
    Both Acados and CasADi use this exact formulation.

    State:   x = [px, py, pz, vx, vy, vz]  (6D)
    Control: u = [ax, ay, az]              (3D)

    Dynamics: Double integrator (Euler discretization)
        x_{k+1} = A x_k + B u_k

    Cost: Quadratic tracking
        J = Σ (x_k - x_ref)^T Q (x_k - x_ref) + u_k^T R u_k
          + 10 * (x_N - x_ref)^T Q (x_N - x_ref)  (terminal)
          + slack penalties (soft obstacle constraints)

    Constraints:
        - Dynamics (equality)
        - Initial condition (equality)
        - Control bounds (inequality)
        - State bounds (inequality)
        - Obstacle avoidance (soft inequality with slack)
    """

    # Dimensions
    nx = 6
    nu = 3

    # Control limits
    u_min = np.array([-2.0, -2.0, -1.0])
    u_max = np.array([2.0, 2.0, 1.0])

    # State limits (wider velocity bounds to prevent infeasibility)
    x_min = np.array([-100, -100, -np.pi, -3.0, -3.0, -2.0])
    x_max = np.array([100, 100, np.pi, 3.0, 3.0, 2.0])

    # Default weights
    Q_default = np.array([50.0, 50.0, 5.0, 2.0, 2.0, 2.0])
    R_default = np.array([1.0, 1.0, 1.0])

    @staticmethod
    def continuous_dynamics(x: ca.MX, u: ca.MX) -> ca.MX:
        """Continuous dynamics: ẋ = f(x, u)"""
        return ca.vertcat(
            x[3],
            x[4],
            x[5],  # velocities
            u[0],
            u[1],
            u[2],  # accelerations
        )

    @staticmethod
    def discrete_dynamics(x: ca.MX, u: ca.MX, dt: float) -> ca.MX:
        """Discrete dynamics: x_{k+1} = f_d(x_k, u_k) via Euler"""
        return ca.vertcat(
            x[0] + x[3] * dt,
            x[1] + x[4] * dt,
            x[2] + x[5] * dt,
            x[3] + u[0] * dt,
            x[4] + u[1] * dt,
            x[5] + u[2] * dt,
        )

    @staticmethod
    def discrete_dynamics_numpy(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """NumPy version for simulation."""
        return np.array(
            [
                x[0] + x[3] * dt,
                x[1] + x[4] * dt,
                x[2] + x[5] * dt,
                x[3] + u[0] * dt,
                x[4] + u[1] * dt,
                x[5] + u[2] * dt,
            ]
        )
