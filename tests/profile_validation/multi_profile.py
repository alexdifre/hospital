"""
Patient Profile Comparison: Safety-Oriented vs Approach-Oriented
================================================================
Runs the same hospital navigation scenario under two preference profiles
to demonstrate how the translator → MPC pipeline adapts robot behavior.

For presentation slides.
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.execution import (
    HybridMPC,
    SharedMPCFormulation,
    filter_nearby_obstacles,
)


# ── Translator mapping (simplified from LearnableTranslator) ──────────────
# Maps patient preference weights → MPC cost weights
# w = [w_time, w_safety, w_battery, w_proximity, w_approach]


def translator_map(w, near_patient=False):
    """
    Compute MPC parameters from patient preference weights.

    Key behavioral levers:
      - Q_pos: High → careful path tracking
      - Q_vel: High → penalizes speed, slower/smoother motion
      - Q_orient: High → robot orients toward goal (patient-facing)
      - R: High → smooth controls, Low → aggressive/fast controls
      - safety_margin: High for safety patients → wider obstacle clearance → different route
    """
    w_time, w_safety, w_battery, w_proximity, w_approach = w
    prox_flag = 1.0 if near_patient else 0.0

    # ── Position tracking ─────────────────────────────────────────────
    q_base = 15.0
    Q_pos = q_base * (
        1.0
        + 1.5 * w_safety
        + 0.3 * w_time
        + 0.5 * prox_flag * w_proximity
        - 0.4 * w_approach
    )

    # ── Velocity penalty ──────────────────────────────────────────────
    qv_base = 2.0
    Q_vel = qv_base * (1.0 + 2.0 * w_safety - 0.8 * w_time - 0.6 * w_approach)

    # ── Orientation tracking ──────────────────────────────────────────
    qo_base = 2.0
    Q_orient = qo_base * (1.0 + 3.0 * w_approach + 0.5 * w_proximity)

    # ── Control penalty ───────────────────────────────────────────────
    r_base = 0.8
    R_val = r_base * (
        1.0
        + 2.5 * w_safety
        + 1.0 * w_battery
        + 0.5 * prox_flag * w_proximity
        - 1.0 * w_time
        - 0.8 * w_approach
    )

    # ── Safety margin (THIS is what drives path differences) ──────────
    # Safety-oriented: 0.45m buffer → obstacles look much bigger → wider detour
    # Approach-oriented: 0.10m buffer → obstacles at true size → direct path
    margin_base = 0.10
    safety_margin = margin_base + 0.8 * w_safety - 0.1 * w_approach

    # Clamp
    Q_pos = max(Q_pos, 5.0)
    Q_vel = max(Q_vel, 0.5)
    Q_orient = max(Q_orient, 1.0)
    R_val = max(R_val, 0.15)
    safety_margin = max(safety_margin, 0.05)

    Q_diag = np.array([Q_pos, Q_pos, Q_orient, Q_vel, Q_vel, Q_vel])
    R_diag = np.array([R_val, R_val, R_val])

    return Q_diag, R_diag, safety_margin


# ── Patient profiles ──────────────────────────────────────────────────────

PROFILES = {
    "safety_oriented": {
        "label": "Safety-Oriented Patient",
        "desc": "Elderly/anxious patient: prioritizes caution, wide obstacle clearance, smooth motion",
        "weights": np.array([0.10, 0.40, 0.10, 0.25, 0.15]),
        # w_time=0.10, w_safety=0.40, w_battery=0.10, w_proximity=0.25, w_approach=0.15
    },
    "approach_oriented": {
        "label": "Approach-Oriented Patient",
        "desc": "Comfortable patient: prioritizes direct approach, quick service, orientation-aware",
        "weights": np.array([0.25, 0.08, 0.07, 0.15, 0.45]),
        # w_time=0.25, w_safety=0.08, w_battery=0.07, w_proximity=0.15, w_approach=0.45
    },
}


# ── Shared environment ────────────────────────────────────────────────────
# Layout designed so there's a DIRECT path (through a narrow gap) and a
# SAFE path (wider detour around the cluster). Safety-oriented robots
# should take the detour; approach-oriented robots should thread the gap.
#
#   Goal (6,4)
#     ↑
#   [obs cluster blocking direct path]
#     ↑
#   Start (0,0)

ALL_OBSTACLES = [
    # Main barrier — blocks the direct diagonal path
    {"x": 3.0, "y": 2.0, "radius": 0.8},  # Central blocker
    {"x": 3.0, "y": 3.5, "radius": 0.6},  # Upper blocker
    {"x": 1.8, "y": 2.8, "radius": 0.5},  # Left side
    # Right corridor
    {"x": 4.8, "y": 1.5, "radius": 0.4},  # Right corridor wall
    {"x": 5.2, "y": 3.0, "radius": 0.4},  # Near goal approach
    # Minor
    {"x": 1.0, "y": 0.8, "radius": 0.3},  # Near start
]

X_INIT = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
X_REF = np.array([6.0, 4.0, 0.0, 0.0, 0.0, 0.0])


def run_profile(profile_key: str, near_patient: bool = False):
    """Run a full trajectory under one patient profile. Returns stats dict."""
    profile = PROFILES[profile_key]
    w = profile["weights"]

    # Translator: preferences → MPC weights
    Q_diag, R_diag, safety_margin = translator_map(w, near_patient=near_patient)

    # Build MPC — use all obstacles since the layout is designed for this scenario
    n_obs = len(ALL_OBSTACLES)
    mpc = HybridMPC(horizon=20, dt=0.2, n_obstacles=n_obs)

    # Filter obstacles (take all since layout is curated)
    filtered = filter_nearby_obstacles(
        robot_pos=X_INIT[:2],
        goal_pos=X_REF[:2],
        obstacles=ALL_OBSTACLES,
        max_distance=15.0,
        max_obstacles=n_obs,
        safety_margin=safety_margin,
    )

    mpc.update_parameters(Q_diag, R_diag, filtered)
    obs_strs = [f"({o['x']},{o['y']}) r={o['radius']:.2f}" for o in filtered]
    print(f"    Safety margin: {safety_margin:.2f}m")
    print(f"    Filtered obstacles: {obs_strs}")

    # ── Run full trajectory ───────────────────────────────────────────────
    current = X_INIT.copy()
    trajectory = [current.copy()]
    controls = []
    solve_times = []
    sensitivities = []
    dt_sim = 0.1

    for step in range(80):
        # Collect sensitivities every 3rd step
        if step % 3 == 0:
            sol, sens = mpc.solve_with_sensitivities(current, X_REF)
            if sens.success:
                sensitivities.append(sens)
        else:
            sol = mpc.solve(current, X_REF)

        solve_times.append(sol.solve_time * 1000)

        if not sol.success:
            print(f"    [WARN] Solve failed at step {step}")
            break

        controls.append(sol.control.copy())
        current = SharedMPCFormulation.discrete_dynamics_numpy(
            current, sol.control, dt_sim
        )
        trajectory.append(current.copy())

        if np.linalg.norm(current[:2] - X_REF[:2]) < 0.3:
            break

    trajectory = np.array(trajectory)
    controls = np.array(controls) if controls else np.zeros((0, 3))

    # ── Compute metrics ───────────────────────────────────────────────────
    # Path length
    diffs = np.diff(trajectory[:, :2], axis=0)
    path_length = np.sum(np.linalg.norm(diffs, axis=1))

    # Min obstacle clearance (against ALL obstacles, not just filtered)
    min_clearance = float("inf")
    worst_obstacle = None
    collision_details = []
    for obs in ALL_OBSTACLES:
        dists = np.sqrt(
            (trajectory[:, 0] - obs["x"]) ** 2 + (trajectory[:, 1] - obs["y"]) ** 2
        )
        clearance = np.min(dists) - obs["radius"]
        if clearance < 0:
            collision_details.append((obs, clearance))
        if clearance < min_clearance:
            min_clearance = clearance
            worst_obstacle = obs
    if collision_details:
        for obs, cl in collision_details:
            print(
                f"    ⚠ Collision: obs ({obs['x']},{obs['y']}) r={obs['radius']}, clearance={cl:.3f}"
            )

    # Speed profile
    speeds = (
        np.linalg.norm(trajectory[1:, 3:5], axis=1)
        if len(trajectory) > 1
        else np.array([0])
    )

    # Control effort (sum of squared controls)
    ctrl_effort = np.sum(controls**2) if len(controls) > 0 else 0

    # Aggregate sensitivities
    if sensitivities:
        dJ_dQ_avg = np.mean([s.dJ_dQ for s in sensitivities], axis=0)
        dJ_dR_avg = np.mean([s.dJ_dR for s in sensitivities], axis=0)
    else:
        dJ_dQ_avg = np.zeros(6)
        dJ_dR_avg = np.zeros(3)

    return {
        "profile": profile,
        "weights": w,
        "Q_diag": Q_diag,
        "R_diag": R_diag,
        "safety_margin": safety_margin,
        "trajectory": trajectory,
        "controls": controls,
        "path_length": path_length,
        "steps": len(trajectory) - 1,
        "min_clearance": min_clearance,
        "worst_obstacle": worst_obstacle,
        "collisions": min_clearance < 0,
        "avg_speed": np.mean(speeds),
        "max_speed": np.max(speeds),
        "ctrl_effort": ctrl_effort,
        "solve_times": solve_times,
        "cold_start_ms": solve_times[0] if solve_times else 0,
        "warm_avg_ms": np.mean(solve_times[1:]) if len(solve_times) > 1 else 0,
        "dJ_dQ": dJ_dQ_avg,
        "dJ_dR": dJ_dR_avg,
        "n_sensitivities": len(sensitivities),
    }


def print_comparison(results: dict):
    """Print side-by-side comparison of the two profiles."""
    keys = list(results.keys())
    r1, r2 = results[keys[0]], results[keys[1]]

    print("\n" + "=" * 72)
    print("PATIENT PROFILE COMPARISON")
    print("=" * 72)

    print(f"\n{'':30s} {'SAFETY':>18s}  {'APPROACH':>18s}")
    print(f"{'':30s} {'─'*18}  {'─'*18}")

    # Preference weights
    w_labels = ["w_time", "w_safety", "w_battery", "w_proximity", "w_approach"]
    print(f"\n  Preference Weights:")
    for i, lbl in enumerate(w_labels):
        print(f"    {lbl:26s} {r1['weights'][i]:>18.2f}  {r2['weights'][i]:>18.2f}")

    # MPC parameters from translator
    print(f"\n  Translated MPC Params:")
    q_labels = ["Q_px", "Q_py", "Q_orient", "Q_vx", "Q_vy", "Q_vtheta"]
    for i, lbl in enumerate(q_labels):
        print(f"    {lbl:26s} {r1['Q_diag'][i]:>18.2f}  {r2['Q_diag'][i]:>18.2f}")
    print(
        f"    {'R (all axes)':26s} {r1['R_diag'][0]:>18.2f}  {r2['R_diag'][0]:>18.2f}"
    )
    print(
        f"    {'Safety margin (m)':26s} {r1['safety_margin']:>18.2f}  {r2['safety_margin']:>18.2f}"
    )

    # Trajectory metrics
    print(f"\n  Trajectory Metrics:")
    print(f"    {'Steps to goal':26s} {r1['steps']:>18d}  {r2['steps']:>18d}")
    print(
        f"    {'Path length (m)':26s} {r1['path_length']:>18.2f}  {r2['path_length']:>18.2f}"
    )
    print(
        f"    {'Min obstacle clearance (m)':26s} {r1['min_clearance']:>18.3f}  {r2['min_clearance']:>18.3f}"
    )
    print(
        f"    {'Collisions':26s} {'YES' if r1['collisions'] else 'NO':>18s}  {'YES' if r2['collisions'] else 'NO':>18s}"
    )
    print(
        f"    {'Avg speed (m/s)':26s} {r1['avg_speed']:>18.3f}  {r2['avg_speed']:>18.3f}"
    )
    print(
        f"    {'Max speed (m/s)':26s} {r1['max_speed']:>18.3f}  {r2['max_speed']:>18.3f}"
    )
    print(
        f"    {'Control effort':26s} {r1['ctrl_effort']:>18.2f}  {r2['ctrl_effort']:>18.2f}"
    )

    # Solver performance
    print(f"\n  Solver Performance:")
    print(
        f"    {'Cold start (ms)':26s} {r1['cold_start_ms']:>18.1f}  {r2['cold_start_ms']:>18.1f}"
    )
    print(
        f"    {'Warm-started avg (ms)':26s} {r1['warm_avg_ms']:>18.1f}  {r2['warm_avg_ms']:>18.1f}"
    )
    print(
        f"    {'Sensitivity computes':26s} {r1['n_sensitivities']:>18d}  {r2['n_sensitivities']:>18d}"
    )

    # Sensitivities (learning signal)
    print(f"\n  Gradient Signal (∂J/∂Q):")
    for i, lbl in enumerate(q_labels):
        print(f"    {lbl:26s} {r1['dJ_dQ'][i]:>18.2f}  {r2['dJ_dQ'][i]:>18.2f}")
    print(f"  Gradient Signal (∂J/∂R):")
    print(f"    {'R gradient':26s} {r1['dJ_dR'][0]:>18.2f}  {r2['dJ_dR'][0]:>18.2f}")

    # Behavioral summary
    print(f"\n  Behavioral Summary:")
    faster = keys[0] if r1["steps"] < r2["steps"] else keys[1]
    safer = keys[0] if r1["min_clearance"] > r2["min_clearance"] else keys[1]
    print(f"    Faster to goal:    {PROFILES[faster]['label']}")
    print(f"    Safer (clearance): {PROFILES[safer]['label']}")
    pct_path = (
        abs(r1["path_length"] - r2["path_length"])
        / max(r1["path_length"], r2["path_length"])
        * 100
    )
    pct_clear = (
        abs(r1["min_clearance"] - r2["min_clearance"])
        / max(r1["min_clearance"], r2["min_clearance"])
        * 100
    )
    print(f"    Path length diff:  {pct_path:.1f}%")
    print(f"    Clearance diff:    {pct_clear:.1f}%")

    # Key waypoints
    for key in keys:
        r = results[key]
        traj = r["trajectory"]
        print(f"\n  Trajectory [{PROFILES[key]['label']}]:")
        n = len(traj)
        for step_idx in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
            if step_idx < n:
                print(
                    f"    Step {step_idx:3d}: ({traj[step_idx, 0]:6.2f}, {traj[step_idx, 1]:6.2f})"
                )

    print("\n" + "=" * 72)


def main():
    print("=" * 72)
    print("HOSPITAL ROBOT: Safety vs Approach Patient Profiles")
    print("=" * 72)
    print(f"Start: {X_INIT[:2]}, Goal: {X_REF[:2]}")
    print(f"Obstacles: {len(ALL_OBSTACLES)}")

    results = {}

    for key in PROFILES:
        p = PROFILES[key]
        print(f"\n{'─'*72}")
        print(f"Running: {p['label']}")
        print(f"  {p['desc']}")
        print(f"  Weights: {p['weights']}")

        Q, R, margin = translator_map(p["weights"])
        print(f"  → Q_diag: [{', '.join(f'{v:.1f}' for v in Q)}]")
        print(f"  → R_diag: [{', '.join(f'{v:.2f}' for v in R)}]")
        print(f"  → Safety margin: {margin:.2f}m")

        t0 = time.time()
        results[key] = run_profile(key)
        elapsed = time.time() - t0

        r = results[key]
        print(
            f"  Done in {elapsed:.1f}s — {r['steps']} steps, "
            f"path={r['path_length']:.2f}m, clearance={r['min_clearance']:.3f}m"
        )

    print_comparison(results)
    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
