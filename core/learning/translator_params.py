#!/usr/bin/env python3
"""
Learnable Translator — Parameter Definitions
=============================================

Contains the two dataclasses that define the learnable parameter space
for the preference → MPC mapping:

    TranslatorParameters  — the 18 learnable φ values with vector
                            serialisation and persistence helpers.
    MPCParameterGradients — container for the analytical gradients
                            ∂Q/∂φ, ∂R/∂φ, ∂H/∂φ used by the chain rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class TranslatorParameters:
    """
    Learnable parameters for the preference → MPC mapping.

    Mapping:
        Q_pos    = q_base  * (1 + q_safety*w_safety + q_time*w_time + q_proximity*near*w_proximity)
        Q_vel    = qv_base * (1 + qv_safety*w_safety + qv_time*w_time)
        Q_orient = qo_base * (1 + qo_approach*w_approach)
        R        = r_base  * (1 + r_time*w_time + r_battery*w_battery + r_proximity*near*w_proximity)
        horizon  = h_base  + h_time*w_time + h_safety*w_safety
        tol      = tol_base + tol_approach*w_approach

    All 18 coefficients are updated via gradient descent.
    """

    # Q_pos parameters
    q_base: float = 20.0
    q_safety: float = 0.5
    q_time: float = 0.2
    q_proximity: float = 0.1

    # Q_vel parameters
    qv_base: float = 8.0
    qv_safety: float = 0.2
    qv_time: float = 0.1

    # Q_orient parameters
    qo_base: float = 2.0
    qo_approach: float = 0.5

    # R parameters (per control dimension — ax, ay, α)
    r_base_ax: float = 2.0     # longitudinal acceleration
    r_base_ay: float = 2.0     # lateral acceleration
    r_base_alpha: float = 1.5  # angular acceleration
    r_time: float = -0.3
    r_battery: float = 0.8
    r_proximity: float = 0.3

    @property
    def r_base(self) -> float:
        """Backward-compat alias — returns r_base_ax."""
        return self.r_base_ax

    # Horizon parameters
    h_base: float = 40.0
    h_time: float = -8.0
    h_safety: float = 10.0

    # Convergence tolerance parameters
    tol_base: float = 1.0
    tol_approach: float = -0.3

    # z_target parameters: z_target(ŵ) = z_target_A @ ŵ + z_target_b
    # z_target_A: (2, 5) matrix — maps 5-dim preference weights to 2D position offset
    # z_target_b: (2,) bias vector
    # Initialized to zeros → z_target=[0,0] → identical to baseline behavior
    z_target_A: np.ndarray = field(default_factory=lambda: np.zeros((2, 5)), repr=False)
    z_target_b: np.ndarray = field(default_factory=lambda: np.zeros(2), repr=False)

    # Parameter name registry — order matches to_vector()
    PARAM_NAMES: list = field(
        default_factory=lambda: [
            "q_base", "q_safety", "q_time", "q_proximity",       # 0-3
            "qv_base", "qv_safety", "qv_time",                   # 4-6
            "qo_base", "qo_approach",                            # 7-8
            "r_base_ax", "r_base_ay", "r_base_alpha",           # 9-11
            "r_time", "r_battery", "r_proximity",                # 12-14
            "h_base", "h_time", "h_safety",                     # 15-17
            "tol_base", "tol_approach",                         # 18-19
            "z_A_00", "z_A_01", "z_A_02", "z_A_03", "z_A_04",  # 20-24
            "z_A_10", "z_A_11", "z_A_12", "z_A_13", "z_A_14",  # 25-29
            "z_b_0", "z_b_1",                                   # 30-31
        ],
        repr=False,
    )

    @property
    def num_params(self) -> int:
        return 32

    def to_vector(self) -> np.ndarray:
        """Convert to flat parameter vector for optimisation (length 32)."""
        return np.concatenate([
            np.array([
                self.q_base, self.q_safety, self.q_time, self.q_proximity,   # 0-3
                self.qv_base, self.qv_safety, self.qv_time,                  # 4-6
                self.qo_base, self.qo_approach,                              # 7-8
                self.r_base_ax, self.r_base_ay, self.r_base_alpha,          # 9-11
                self.r_time, self.r_battery, self.r_proximity,               # 12-14
                self.h_base, self.h_time, self.h_safety,                     # 15-17
                self.tol_base, self.tol_approach,                            # 18-19
            ]),
            self.z_target_A.flatten(),   # 20-29  (2×5 row-major)
            self.z_target_b,             # 30-31
        ])

    def from_vector(self, vec: np.ndarray) -> None:
        """Update all fields from a flat parameter vector (length 32, in-place)."""
        (
            self.q_base, self.q_safety, self.q_time, self.q_proximity,
            self.qv_base, self.qv_safety, self.qv_time,
            self.qo_base, self.qo_approach,
            self.r_base_ax, self.r_base_ay, self.r_base_alpha,
            self.r_time, self.r_battery, self.r_proximity,
            self.h_base, self.h_time, self.h_safety,
            self.tol_base, self.tol_approach,
        ) = vec[:20]
        self.z_target_A = vec[20:30].reshape(2, 5).copy()
        self.z_target_b = vec[30:32].copy()

    def to_dict(self) -> Dict:
        """Serialise to plain dict (JSON-safe)."""
        return {
            "q_base": self.q_base, "q_safety": self.q_safety,
            "q_time": self.q_time, "q_proximity": self.q_proximity,
            "qv_base": self.qv_base, "qv_safety": self.qv_safety,
            "qv_time": self.qv_time,
            "qo_base": self.qo_base, "qo_approach": self.qo_approach,
            "r_base_ax": self.r_base_ax, "r_base_ay": self.r_base_ay,
            "r_base_alpha": self.r_base_alpha,
            "r_time": self.r_time, "r_battery": self.r_battery,
            "r_proximity": self.r_proximity,
            "h_base": self.h_base, "h_time": self.h_time,
            "h_safety": self.h_safety,
            "tol_base": self.tol_base, "tol_approach": self.tol_approach,
            "z_target_A": self.z_target_A.tolist(),
            "z_target_b": self.z_target_b.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TranslatorParameters":
        """Create from dict, ignoring unrecognised keys (e.g. PARAM_NAMES).

        Handles legacy dicts that have a single 'r_base' key by mapping it
        to all three per-axis base values.
        """
        valid_keys = {
            "q_base", "q_safety", "q_time", "q_proximity",
            "qv_base", "qv_safety", "qv_time",
            "qo_base", "qo_approach",
            "r_base_ax", "r_base_ay", "r_base_alpha",
            "r_time", "r_battery", "r_proximity",
            "h_base", "h_time", "h_safety",
            "tol_base", "tol_approach",
        }
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        # Legacy compat: single r_base → all three per-axis bases
        if "r_base" in d and "r_base_ax" not in filtered:
            filtered["r_base_ax"] = d["r_base"]
            filtered["r_base_ay"] = d["r_base"]
            filtered["r_base_alpha"] = d["r_base"]
        obj = cls(**filtered)
        # z_target fields (absent in legacy dicts → default zeros are correct)
        if "z_target_A" in d:
            obj.z_target_A = np.array(d["z_target_A"], dtype=float).reshape(2, 5)
        if "z_target_b" in d:
            obj.z_target_b = np.array(d["z_target_b"], dtype=float)
        return obj


@dataclass
class MPCParameterGradients:
    """
    Analytical gradients of MPC cost matrices w.r.t. translator parameters.

    Computed by LearnableTranslator.compute_parameter_gradients() and
    consumed by update_parameters() for the chain rule:
        ∂J/∂φ = ∂J/∂Q · ∂Q/∂φ + ∂J/∂R · ∂R/∂φ + ∂J/∂z · ∂z/∂φ
    """

    dQ_dphi: np.ndarray  # (6, 32) — gradient of Q diagonal w.r.t. φ
    dR_dphi: np.ndarray  # (3, 32) — gradient of R diagonal w.r.t. φ
    dH_dphi: np.ndarray  # (32,)   — gradient of horizon w.r.t. φ
    dZ_dphi: np.ndarray  # (2, 32) — gradient of z_target w.r.t. φ
