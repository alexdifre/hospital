"""
Hybrid MPC: Acados (Fast Solve) + CasADi (Analytical Sensitivities)
====================================================================

Architecture per Section 6.7 of the paper:

    ┌─────────────────────────────────────────────────────────────────┐
    │                      HYBRID MPC                                  │
    │                                                                  │
    │   CONTROL PATH (real-time, every timestep):                     │
    │   ──────────────────────────────────────────                    │
    │   Acados SQP-RTI  ───────────────────────────►  u* (control)    │
    │   (~1-5ms)                                                       │
    │                                                                  │
    │   LEARNING PATH (periodic, for translator training):            │
    │   ──────────────────────────────────────────                    │
    │   Acados solve  ──►  (w*, λ*, p)  ──►  CasADi IFT  ──►  ∂J*/∂p  │
    │   (~1-5ms)                              (~5-10ms)                │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

Canonical modules:
    core/execution/formulation.py   — MPCSolution, MPCSensitivity, SharedMPCFormulation
    core/execution/obstacle_utils.py — filter_nearby_obstacles
    core/execution/ift_engine.py    — CasADiSensitivityComputer
    core/execution/mpc_solver.py    — AcadosSolver, HybridMPC
"""

# Re-exports for backward compatibility
from core.execution.formulation import (  # noqa: F401
    MPCSolution,
    MPCSensitivity,
    SharedMPCFormulation,
)
from core.execution.obstacle_utils import filter_nearby_obstacles  # noqa: F401
from core.execution.ift_engine import CasADiSensitivityComputer  # noqa: F401
from core.execution.mpc_solver import AcadosSolver, HybridMPC  # noqa: F401

import numpy as np


def test_hybrid_mpc():
    """Test the hybrid MPC architecture."""
    print("=" * 60)
    print("TESTING HYBRID MPC: Acados (solve) + CasADi (sensitivities)")
    print("=" * 60)

    # Build with max 3 obstacles (filtered from larger set)
    mpc = HybridMPC(
        horizon=20,
        dt=0.1,
        n_obstacles=3,  # MPC handles up to 3 obstacles
        use_acados=True,
    )

    # Set parameters
    Q_diag = np.array([30.0, 30.0, 5.0, 2.0, 2.0, 2.0])
    R_diag = np.array([1.0, 1.0, 1.0])

    # Full environment has 10 obstacles
    all_obstacles = [
        {"x": 2.5, "y": 2.5, "radius": 0.8},  # On direct path!
        {"x": 4.0, "y": 1.0, "radius": 0.6},  # Near path
        {"x": 1.0, "y": 4.0, "radius": 0.5},  # Near path
        {"x": 3.5, "y": 3.5, "radius": 0.7},  # On direct path!
        {"x": 0.5, "y": 2.0, "radius": 0.4},  # Off to side
        {"x": 4.5, "y": 4.5, "radius": 0.5},  # Near goal
        {"x": -2.0, "y": -2.0, "radius": 1.0},  # Behind start (irrelevant)
        {"x": 8.0, "y": 8.0, "radius": 0.8},  # Past goal (irrelevant)
        {"x": 5.0, "y": 0.0, "radius": 0.6},  # Far from path
        {"x": 0.0, "y": 5.0, "radius": 0.6},  # Far from path
    ]

    x_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_ref = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0])

    # Filter to most relevant obstacles (with safety margin)
    filtered_obstacles = filter_nearby_obstacles(
        robot_pos=x_init[:2],
        goal_pos=x_ref[:2],
        obstacles=all_obstacles,
        max_distance=10.0,
        max_obstacles=3,
        safety_margin=0.15,  # Add 15cm buffer
    )

    print(f"\nObstacle filtering (with 0.15m safety margin):")
    print(f"  Total obstacles in env: {len(all_obstacles)}")
    print(f"  Filtered for MPC:       {len(filtered_obstacles)}")
    for i, obs in enumerate(filtered_obstacles):
        orig_r = obs["radius"] - 0.15  # Recover original for display
        print(
            f"    {i+1}. ({obs['x']}, {obs['y']}) r={orig_r:.1f} → {obs['radius']:.2f} (inflated)"
        )

    mpc.update_parameters(Q_diag, R_diag, filtered_obstacles)

    print(f"\nProblem setup:")
    print(f"  x_init: {x_init[:3]}")
    print(f"  x_ref:  {x_ref[:3]}")
    print(f"  Q_diag: {Q_diag}")
    print(f"  R_diag: {R_diag}")

    # === Test fast solve ===
    print("\n--- Fast Solve (no sensitivities) ---")
    sol = mpc.solve(x_init, x_ref)
    print(f"Success:    {sol.success}")
    print(f"Control:    {sol.control}")
    print(f"Cost:       {sol.cost:.2f}")
    print(f"Solve time: {sol.solve_time*1000:.2f}ms")
    print(f"Solver:     {sol.solver_used}")

    # === Test solve with sensitivities ===
    print("\n--- Solve with Sensitivities ---")
    sol, sens = mpc.solve_with_sensitivities(x_init, x_ref)

    print(f"Solution:")
    print(f"  Success:    {sol.success}")
    print(f"  Control:    {sol.control}")
    print(f"  Cost:       {sol.cost:.2f}")
    print(f"  Solve time: {sol.solve_time*1000:.2f}ms")

    print(f"\nSensitivities (IFT on KKT):")
    print(f"  Success:    {sens.success}")
    print(f"  Time:       {sens.compute_time*1000:.2f}ms")
    print(f"  ∂J/∂Q:      {sens.dJ_dQ}")
    print(f"  ∂J/∂R:      {sens.dJ_dR}")
    print(f"  ∂u₀/∂Q shape: {sens.du0_dQ.shape}")

    # === Simulate episode ===
    print("\n--- Episode Simulation ---")
    mpc.reset_episode(x_init=x_init, x_ref=x_ref)

    current_state = x_init.copy()
    for step in range(10):
        if step % 3 == 0:
            sol, sens = mpc.solve_with_sensitivities(current_state, x_ref)
        else:
            sol = mpc.solve(current_state, x_ref)

        if sol.success:
            current_state = SharedMPCFormulation.discrete_dynamics_numpy(
                current_state, sol.control, 0.1
            )

    print(f"Final position: {current_state[:2]}")
    print(f"Sensitivities collected: {len(mpc.episode_sensitivities)}")

    dJ_dQ_avg, dJ_dR_avg = mpc.get_aggregated_sensitivities()
    print(f"Aggregated ∂J/∂Q: {dJ_dQ_avg}")
    print(f"Aggregated ∂J/∂R: {dJ_dR_avg}")

    mpc.print_stats()

    # === Full trajectory with obstacle avoidance ===
    print("\n--- Full Trajectory (showing obstacle avoidance) ---")
    mpc.reset_episode(x_init=x_init, x_ref=x_ref)

    # Use SAME filtered obstacles throughout (no rebuilds during trajectory)
    mpc.update_parameters(Q_diag, R_diag, filtered_obstacles)

    current_state = x_init.copy()
    trajectory = [current_state.copy()]

    for step in range(50):  # Longer simulation
        sol = mpc.solve(current_state, x_ref)
        if not sol.success:
            print(f"  Step {step}: solve failed")
            break

        current_state = SharedMPCFormulation.discrete_dynamics_numpy(
            current_state, sol.control, 0.1
        )
        trajectory.append(current_state.copy())

        # Check if reached target
        dist_to_target = np.linalg.norm(current_state[:2] - x_ref[:2])
        if dist_to_target < 0.3:
            print(f"  Reached target at step {step+1}")
            break

    trajectory = np.array(trajectory)

    # Check obstacle clearance against ALL obstacles
    print(f"\n  Trajectory from {trajectory[0, :2]} to {trajectory[-1, :2]}")
    print(f"  Total steps: {len(trajectory)}")

    collisions = 0
    for i, obs in enumerate(all_obstacles):
        distances = np.sqrt(
            (trajectory[:, 0] - obs["x"]) ** 2 + (trajectory[:, 1] - obs["y"]) ** 2
        )
        min_dist = np.min(distances)
        clearance = min_dist - obs["radius"]
        if clearance < 0:
            collisions += 1
            print(
                f"  ✗ Obstacle {i+1} at ({obs['x']}, {obs['y']}) r={obs['radius']}: COLLISION (clearance={clearance:.2f})"
            )

    if collisions == 0:
        print(f"  ✓ All {len(all_obstacles)} obstacles avoided!")
    else:
        print(f"  {collisions}/{len(all_obstacles)} collisions")

    # Print key waypoints
    print(f"\n  Key positions:")
    for step in [0, 10, 20, 30, len(trajectory) - 1]:
        if step < len(trajectory):
            print(
                f"    Step {step:2d}: ({trajectory[step, 0]:.2f}, {trajectory[step, 1]:.2f})"
            )

    # === Verify against finite differences ===
    print("\n--- Verification: Finite Differences (using CasADi for consistency) ---")
    eps = 1e-4

    # Use CasADi directly for verification (not Acados)
    # This ensures we're comparing the same cost function
    mpc.casadi.reset()
    sol_base, sens = mpc.casadi.solve_and_get_sensitivities(
        x_init, x_ref, Q_diag, R_diag, filtered_obstacles
    )

    Q_pert = Q_diag.copy()
    Q_pert[0] += eps
    mpc.casadi.reset()
    sol_pert, _ = mpc.casadi.solve_and_get_sensitivities(
        x_init, x_ref, Q_pert, R_diag, filtered_obstacles
    )

    if (
        sol_base.success
        and sol_pert.success
        and np.isfinite(sol_base.cost)
        and np.isfinite(sol_pert.cost)
    ):
        dJ_dQ0_fd = (sol_pert.cost - sol_base.cost) / eps

        print(f"  ∂J/∂Q[0] analytical: {sens.dJ_dQ[0]:.4f}")
        print(f"  ∂J/∂Q[0] finite diff: {dJ_dQ0_fd:.4f}")
        rel_err = abs(sens.dJ_dQ[0] - dJ_dQ0_fd) / (abs(dJ_dQ0_fd) + 1e-8)
        print(f"  Relative error: {rel_err:.2%}")
    else:
        print(f"  Verification skipped (solver issue with base or perturbed solve)")
        print(f"  Base solve success: {sol_base.success}, cost: {sol_base.cost}")
        print(f"  Pert solve success: {sol_pert.success}, cost: {sol_pert.cost}")

    # Also show Acados vs CasADi cost difference
    mpc.update_parameters(Q_diag, R_diag, filtered_obstacles)
    sol_acados = mpc.solve(x_init, x_ref)
    print(
        f"\n  Note: Acados cost = {sol_acados.cost:.2f}, CasADi cost = {sol_base.cost:.2f}"
    )
    print(f"  (Difference due to slack handling - both are valid solutions)")

    print("\n✓ Hybrid MPC test complete!")


if __name__ == "__main__":
    test_hybrid_mpc()
