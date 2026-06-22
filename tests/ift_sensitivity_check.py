#!/usr/bin/env python3
"""
Learning Loop Verification Test
================================

Tests that:
1. HybridMPC computes sensitivities (dJ/dQ, dJ/dR)
2. LearnableTranslator receives sensitivities and updates parameters
3. Parameters actually change over episodes

This is the core MLC loop:
    Translator(θ) → Q, R → MPC → J*, dJ/dQ, dJ/dR → Update θ

Run from project root:
    python tests/ift_sensitivity_check.py
"""

import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import HybridMPC from project
try:
    from core.execution.hybrid import HybridMPC

    print("✓ Imported HybridMPC from core.execution.hybrid")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from project root and hybrid.py is updated")
    sys.exit(1)


# ============================================================================
# SIMPLIFIED LEARNABLE TRANSLATOR (extracted from full version)
# ============================================================================


@dataclass
class TranslatorParams:
    """Learnable parameters for preference → MPC mapping."""

    # Q parameters
    q_base: float = 20.0
    q_safety: float = 0.5
    q_time: float = 0.2

    # R parameters
    r_base: float = 2.0
    r_time: float = -0.3
    r_battery: float = 0.5

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.q_base,
                self.q_safety,
                self.q_time,
                self.r_base,
                self.r_time,
                self.r_battery,
            ]
        )

    def from_vector(self, vec: np.ndarray):
        self.q_base, self.q_safety, self.q_time = vec[0], vec[1], vec[2]
        self.r_base, self.r_time, self.r_battery = vec[3], vec[4], vec[5]


class SimpleLearnableTranslator:
    """
    Simplified translator for testing learning loop.

    Maps: preference_weights → (Q_diag, R_diag)
    Learning: Uses dJ/dQ, dJ/dR from MPC to update parameters
    """

    def __init__(self, learning_rate: float = 0.01):
        self.params = TranslatorParams()
        self.learning_rate = learning_rate

        # Default preferences: [time, safety, battery, proximity, approach]
        self.preference_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        # History for analysis
        self.param_history: List[np.ndarray] = [self.params.to_vector().copy()]
        self.gradient_history: List[np.ndarray] = []
        self.cost_history: List[float] = []

    def compute_mpc_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Q, R from preferences using LEARNABLE mapping.

        Q_pos = q_base * (1 + q_safety * w_safety + q_time * w_time)
        R = r_base * (1 + r_time * w_time + r_battery * w_battery)
        """
        w_time, w_safety, w_battery, w_prox, w_approach = self.preference_weights
        φ = self.params

        # Position weight
        Q_pos = φ.q_base * (1 + φ.q_safety * w_safety + φ.q_time * w_time)
        Q_pos = np.clip(Q_pos, 5.0, 100.0)

        # Velocity weight (simpler)
        Q_vel = 2.0 * (1 + 0.2 * w_safety)

        # Control weight
        R_val = φ.r_base * (1 + φ.r_time * w_time + φ.r_battery * w_battery)
        R_val = np.clip(R_val, 0.5, 10.0)

        Q_diag = np.array([Q_pos, Q_pos, 2.0, Q_vel, Q_vel, Q_vel])
        R_diag = np.array([R_val, R_val, R_val])

        return Q_diag, R_diag

    def compute_param_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ∂Q/∂φ and ∂R/∂φ analytically.

        Returns:
            dQ_dphi: (6, 6) gradient of Q_diag w.r.t. params
            dR_dphi: (3, 6) gradient of R_diag w.r.t. params
        """
        w_time, w_safety, w_battery, _, _ = self.preference_weights
        φ = self.params
        n_params = 6

        dQ_dphi = np.zeros((6, n_params))
        dR_dphi = np.zeros((3, n_params))

        # ∂Q_pos/∂φ (indices 0, 1 for x and y)
        factor = 1 + φ.q_safety * w_safety + φ.q_time * w_time
        dQ_dphi[0, 0] = factor  # ∂Q_pos/∂q_base
        dQ_dphi[0, 1] = φ.q_base * w_safety  # ∂Q_pos/∂q_safety
        dQ_dphi[0, 2] = φ.q_base * w_time  # ∂Q_pos/∂q_time
        dQ_dphi[1, :] = dQ_dphi[0, :]  # Same for y

        # ∂R/∂φ
        factor_r = 1 + φ.r_time * w_time + φ.r_battery * w_battery
        for i in range(3):
            dR_dphi[i, 3] = factor_r  # ∂R/∂r_base
            dR_dphi[i, 4] = φ.r_base * w_time  # ∂R/∂r_time
            dR_dphi[i, 5] = φ.r_base * w_battery  # ∂R/∂r_battery

        return dQ_dphi, dR_dphi

    def update_from_sensitivities(
        self,
        dJ_dQ: np.ndarray,  # (6,) from MPC
        dJ_dR: np.ndarray,  # (3,) from MPC
        cost: float,
    ) -> Dict:
        """
        Update translator parameters using MPC sensitivities.

        Chain rule: ∂J/∂φ = ∂J/∂Q @ ∂Q/∂φ + ∂J/∂R @ ∂R/∂φ
        """
        # Get mapping gradients
        dQ_dphi, dR_dphi = self.compute_param_gradients()

        # Chain rule
        dJ_dphi = dQ_dphi.T @ dJ_dQ + dR_dphi.T @ dJ_dR

        # Gradient descent
        old_params = self.params.to_vector().copy()
        new_params = old_params - self.learning_rate * dJ_dphi

        # Bounds
        new_params = np.clip(
            new_params,
            [5.0, 0.0, -1.0, 0.5, -1.0, 0.0],
            [100.0, 2.0, 1.0, 10.0, 0.5, 2.0],
        )

        self.params.from_vector(new_params)

        # Track
        self.param_history.append(new_params.copy())
        self.gradient_history.append(dJ_dphi.copy())
        self.cost_history.append(cost)

        return {
            "gradient": dJ_dphi,
            "gradient_norm": np.linalg.norm(dJ_dphi),
            "param_change": np.linalg.norm(new_params - old_params),
            "old_params": old_params,
            "new_params": new_params,
        }

    def print_params(self):
        print(
            f"  Q: base={self.params.q_base:.2f}, safety={self.params.q_safety:.3f}, time={self.params.q_time:.3f}"
        )
        print(
            f"  R: base={self.params.r_base:.2f}, time={self.params.r_time:.3f}, battery={self.params.r_battery:.3f}"
        )


# ============================================================================
# LEARNING LOOP TEST
# ============================================================================


def run_learning_loop_check():
    """Test the full learning loop with real MPC sensitivities."""

    print("=" * 70)
    print("LEARNING LOOP VERIFICATION TEST")
    print("=" * 70)

    # Create components
    print("\n[1] Creating HybridMPC...")
    mpc = HybridMPC(
        horizon=15,
        dt=0.2,
        n_obstacles=3,
    )

    print("\n[2] Creating LearnableTranslator...")
    translator = SimpleLearnableTranslator(learning_rate=0.001)

    # Set a specific preference profile (safety-oriented)
    translator.preference_weights = np.array([0.15, 0.35, 0.2, 0.15, 0.15])
    print(
        f"  Preferences: time={translator.preference_weights[0]:.2f}, "
        f"safety={translator.preference_weights[1]:.2f}, "
        f"battery={translator.preference_weights[2]:.2f}"
    )

    print("\n[3] Initial translator parameters:")
    translator.print_params()

    # Run learning episodes
    print("\n[4] Running learning episodes...")
    print("-" * 70)

    n_episodes = 5
    n_steps_per_episode = 10

    for ep in range(n_episodes):
        print(f"\n--- Episode {ep+1}/{n_episodes} ---")

        # Get Q, R from translator
        Q_diag, R_diag = translator.compute_mpc_params()
        print(f"  Q_diag: [{Q_diag[0]:.1f}, {Q_diag[1]:.1f}, {Q_diag[2]:.1f}, ...]")
        print(f"  R_diag: [{R_diag[0]:.2f}, {R_diag[1]:.2f}, {R_diag[2]:.2f}]")

        # Update MPC with translator output
        mpc.update_parameters(Q_diag, R_diag, obstacles=[])

        # Simulate a trajectory segment
        x_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        x_ref = np.array([5.0, 3.0, 0.0, 0.0, 0.0, 0.0])

        # Accumulate sensitivities over steps
        total_dJ_dQ = np.zeros(6)
        total_dJ_dR = np.zeros(3)
        total_cost = 0.0
        n_sens = 0

        current_state = x_init.copy()

        for step in range(n_steps_per_episode):
            # Solve with sensitivities
            sol, sens = mpc.solve_with_sensitivities(current_state, x_ref)

            if sol.success and sens.success:
                total_dJ_dQ += sens.dJ_dQ
                total_dJ_dR += sens.dJ_dR
                total_cost += sol.cost
                n_sens += 1

                # Simulate state evolution (simple integration)
                dt = 0.2
                current_state[:2] += current_state[3:5] * dt
                current_state[3:5] += sol.control[:2] * dt

        if n_sens > 0:
            # Average sensitivities
            avg_dJ_dQ = total_dJ_dQ / n_sens
            avg_dJ_dR = total_dJ_dR / n_sens
            avg_cost = total_cost / n_sens

            print(f"  Avg cost: {avg_cost:.2f}")
            print(f"  dJ/dQ: [{avg_dJ_dQ[0]:.4f}, {avg_dJ_dQ[1]:.4f}, ...]")
            print(f"  dJ/dR: [{avg_dJ_dR[0]:.4f}, {avg_dJ_dR[1]:.4f}, ...]")

            # Update translator
            update_info = translator.update_from_sensitivities(
                avg_dJ_dQ, avg_dJ_dR, avg_cost
            )

            print(f"  Gradient norm: {update_info['gradient_norm']:.6f}")
            print(f"  Param change:  {update_info['param_change']:.6f}")
        else:
            print("  [!] No successful solves with sensitivities")

    # Final summary
    print("\n" + "=" * 70)
    print("LEARNING SUMMARY")
    print("=" * 70)

    print("\n[5] Final translator parameters:")
    translator.print_params()

    print("\n[6] Parameter evolution:")
    initial_params = translator.param_history[0]
    final_params = translator.param_history[-1]
    param_names = ["q_base", "q_safety", "q_time", "r_base", "r_time", "r_battery"]

    print(f"  {'Parameter':<12} {'Initial':>10} {'Final':>10} {'Change':>10}")
    print("  " + "-" * 44)
    for i, name in enumerate(param_names):
        change = final_params[i] - initial_params[i]
        print(
            f"  {name:<12} {initial_params[i]:>10.4f} {final_params[i]:>10.4f} {change:>+10.4f}"
        )

    total_change = np.linalg.norm(final_params - initial_params)
    print(f"\n  Total parameter change: {total_change:.6f}")

    if total_change > 0.001:
        print("\n✓ LEARNING VERIFIED - Parameters updated based on MPC sensitivities!")
    else:
        print("\n⚠ WARNING - Parameters barely changed. Check gradient flow.")

    # MPC stats
    print("\n[7] MPC Statistics:")
    mpc.print_stats()

    return translator


def run_gradient_flow_check():
    """Test that gradients flow correctly through chain rule."""

    print("\n" + "=" * 70)
    print("GRADIENT FLOW TEST")
    print("=" * 70)

    translator = SimpleLearnableTranslator()
    translator.preference_weights = np.array([0.2, 0.3, 0.2, 0.15, 0.15])

    # Mock MPC sensitivities (realistic values)
    dJ_dQ = np.array([0.05, 0.05, 0.01, 0.02, 0.02, 0.01])
    dJ_dR = np.array([0.03, 0.03, 0.02])

    print("\n[1] Input sensitivities from MPC:")
    print(f"  dJ/dQ: {dJ_dQ}")
    print(f"  dJ/dR: {dJ_dR}")

    print("\n[2] Translator mapping gradients:")
    dQ_dphi, dR_dphi = translator.compute_param_gradients()
    print(f"  dQ/dphi shape: {dQ_dphi.shape}")
    print(f"  dR/dphi shape: {dR_dphi.shape}")
    print(f"  dQ_pos/d(q_base): {dQ_dphi[0, 0]:.4f}")
    print(f"  dQ_pos/d(q_safety): {dQ_dphi[0, 1]:.4f}")
    print(f"  dR/d(r_base): {dR_dphi[0, 3]:.4f}")

    print("\n[3] Chain rule: dJ/dphi = dQ/dphi.T @ dJ/dQ + dR/dphi.T @ dJ/dR")
    dJ_dphi = dQ_dphi.T @ dJ_dQ + dR_dphi.T @ dJ_dR
    print(f"  dJ/dphi: {dJ_dphi}")
    print(f"  ||dJ/dphi||: {np.linalg.norm(dJ_dphi):.6f}")

    print("\n[4] Parameter update (lr=0.01):")
    old_params = translator.params.to_vector()
    translator.update_from_sensitivities(dJ_dQ, dJ_dR, cost=10.0)
    new_params = translator.params.to_vector()

    print(f"  Old: {old_params}")
    print(f"  New: {new_params}")
    print(f"  Change: {new_params - old_params}")

    if np.any(new_params != old_params):
        print("\n✓ Gradient flow verified - parameters changed!")
    else:
        print("\n✗ ERROR - parameters unchanged!")


if __name__ == "__main__":
    # Run gradient flow test first (quick sanity check)
    run_gradient_flow_check()

    # Then run full learning loop
    translator = run_learning_loop_check()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
