"""
MPC cold-start investigation — shared test configurations.

Defines canonical scenarios used across all test scripts so that each
script produces comparable results without duplicating setup.

Scenarios are picked to stress different aspects of the cold-start problem:
    CLEAR       — no obstacles between start and goal (baseline)
    BLOCKED     — straight-line path cuts through an obstacle radius
    SIDE        — obstacle sits just off the straight-line path (≈ 0.5m margin)
    MULTI_OBS   — 3 obstacles forcing a curved path
    LONG_RANGE  — 30m hop, high-curvature initial trajectory
    U_TURN      — goal is behind the robot (θ-flip needed)
"""

import numpy as np

# ---------------------------------------------------------------------------
# State / control dimensions
# ---------------------------------------------------------------------------
NX = 6   # [x, y, θ, vx, vy, ωz]
NU = 3   # [ax, ay, α]

# ---------------------------------------------------------------------------
# Default MPC weights (mid-range, not profile-specific)
# ---------------------------------------------------------------------------
Q_DEFAULT = np.array([20.0, 20.0, 2.0, 8.0, 8.0, 8.0])   # position x2, orient, vel x3
R_DEFAULT = np.array([1.5,  1.5,  1.0])                    # ax, ay, α

HORIZON  = 40
DT       = 0.2   # seconds

# ---------------------------------------------------------------------------
# Canonical scenarios
# ---------------------------------------------------------------------------

# CLEAR — straight corridor, nothing in between
CLEAR = dict(
    name="CLEAR",
    x_init=np.array([0.0, 0.0, 0.0,  0.0, 0.0, 0.0]),
    x_ref =np.array([14.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
    obstacles=[],
    note="No obstacles — pure cold-start NLP timing baseline",
)

# BLOCKED — pharmacy_north route where supply_A sits ~3m off the straight line
# In the hospital map supply_A is at (14,10); pharmacy_north at (5,18).
# The route home→supply_A has nurse_station at (12,0) within ~3m of mid-path.
BLOCKED = dict(
    name="BLOCKED",
    x_init=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    x_ref =np.array([14.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
    obstacles=[
        {"name": "nurse_station",      "x": 12.0, "y":  0.0, "radius": 2.5},  # in-path
        {"name": "equipment_storage",  "x": 22.0, "y":  6.0, "radius": 1.5},
        {"name": "charge_main",        "x":  3.0, "y":  5.0, "radius": 1.2},
    ],
    note="Nurse-station sits on straight-line path — straight-line init is INFEASIBLE",
)

# SIDE — obstacle just off path, straight-line init is feasible but tight
SIDE = dict(
    name="SIDE",
    x_init=np.array([0.0,  0.0, 0.0, 0.0, 0.0, 0.0]),
    x_ref =np.array([14.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
    obstacles=[
        {"name": "side_obs", "x": 7.0, "y": 4.5, "radius": 1.8},  # 0.6m margin
    ],
    note="Straight-line init is marginally feasible — tests constraint sensitivity",
)

# MULTI_OBS — three obstacles forcing a detour
MULTI_OBS = dict(
    name="MULTI_OBS",
    x_init=np.array([0.0,  0.0, 0.0, 0.0, 0.0, 0.0]),
    x_ref =np.array([14.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
    obstacles=[
        {"name": "obs_A", "x":  5.0, "y": 4.0, "radius": 2.0},
        {"name": "obs_B", "x":  9.0, "y": 6.5, "radius": 2.0},
        {"name": "obs_C", "x": 12.0, "y": 2.0, "radius": 1.8},
    ],
    note="Three obstacles — max-obstacle-count scenario, forces curved path",
)

# LONG_RANGE — 30 m hop (home → patient_bed scenario)
LONG_RANGE = dict(
    name="LONG_RANGE",
    x_init=np.array([ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0]),
    x_ref =np.array([20.5, 12.0, -np.pi/4, 0.0, 0.0, 0.0]),
    obstacles=[
        {"name": "nurse_station",     "x": 12.0, "y":  0.0, "radius": 2.5},
        {"name": "equipment_storage", "x": 22.0, "y":  6.0, "radius": 1.5},
        {"name": "supply_A",          "x": 14.0, "y": 10.0, "radius": 1.3},
    ],
    note="30m diagonal hop — long horizon where straight-line init diverges most",
)

# U_TURN — goal behind and to the right; large heading change needed
U_TURN = dict(
    name="U_TURN",
    x_init=np.array([5.0, 18.0, np.pi/2, 0.0, 0.0, 0.0]),   # facing north (pharmacy_north)
    x_ref =np.array([6.0, -15.0, np.pi/2, 0.0, 0.0, 0.0]),  # pharmacy_south — 33m south
    obstacles=[
        {"name": "home",          "x":  0.0, "y":  0.0, "radius": 1.2},
        {"name": "nurse_station", "x": 12.0, "y":  0.0, "radius": 2.5},
        {"name": "charge_main",   "x":  3.0, "y":  5.0, "radius": 1.2},
    ],
    note="180° reversal needed — worst case for straight-line trajectory init",
)

ALL_SCENARIOS = [CLEAR, BLOCKED, SIDE, MULTI_OBS, LONG_RANGE, U_TURN]

# ---------------------------------------------------------------------------
# Warm-start strategies under test
# ---------------------------------------------------------------------------

STRATEGIES = {
    "zero":          "All variables (x, u, s) initialised to zero",
    "straight_line": "States: linear interp x_init→x_ref; controls: zero; slacks: zero",
    "straight_slack": "States: linear interp; controls: zero; slacks: violation-aware (max(0, r²-dist²))",
    "prev_sol":      "Previous NLP solution reused (episode warm-start — normal operation)",
}
