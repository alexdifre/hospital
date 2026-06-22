# core/execution/

MPC controllers for real-time robot trajectory tracking.

## Module Structure

`hybrid.py` was split into focused modules. All public names are still importable from `hybrid.py` for backward compatibility.

### `formulation.py`
Shared problem definition for Acados.

- `SharedMPCFormulation` — dimensions, bounds, default weights, continuous/discrete dynamics
- `MPCSolution` — dataclass: control, trajectory, cost, solve time, primal solution
- `MPCSensitivity` — dataclass: `dJ_dQ`, `dJ_dR`, `du0_dQ`, `du0_dR`

### `obstacle_utils.py`
Obstacle geometry helpers.

- `filter_nearby_obstacles()` — trims the full environment obstacle list to the 3–5 most relevant obstacles for the MPC horizon, scoring by proximity to the robot-to-goal path segment and inflating radii by a configurable safety margin

### `mpc_solver.py`
Acados solver and hybrid orchestrator.

- `AcadosSolver` — SQP-RTI solver; bakes obstacle constraints into C code at build time; rebuilds only when obstacle count changes
- `HybridMPC` — routes all `solve()` calls to Acados and computes Q/R cost sensitivities from the Acados trajectory

### `hybrid.py`
Backward-compatibility shim. Re-exports all public names from the four modules above. Also contains `test_hybrid_mpc()` and the `__main__` entry point.

---

## Architecture (Section 6.7)

```
CONTROL PATH (every timestep, ~1-5ms):
  x_t  ──►  AcadosSolver.solve()  ──►  u*

LEARNING PATH (periodic, for translator training):
  x_t  ──►  AcadosSolver.solve()  ──►  optimal trajectory ──► ∂J*/∂Q, ∂J*/∂R
```

Key insight: Acados is the only controller solve path. No fallback MPC solve
or separate symbolic sensitivity engine is used.

## Robot Model

| Property | Value |
|----------|-------|
| State | 6D: `[px, py, pz, vx, vy, vz]` |
| Control | 3D: `[ax, ay, az]` |
| Dynamics | Double integrator (Euler) |
| Control limits | `ax, ay ∈ [−2, 2]` m/s², `az ∈ [−1, 1]` m/s² |
| Velocity limits | `vx, vy ∈ [−3, 3]` m/s, `vz ∈ [−2, 2]` m/s |
| Default horizon | N = 40, dt = 0.2 s |

## MPC Cost

```
Stage cost (k = 0..N-1):
  ℓ(xk, uk) = (xk − x_ref)ᵀ Q (xk − x_ref) + ukᵀ R uk

Terminal cost (k = N):
  ℓ_N(xN) = (xN − x_terminal_ref)ᵀ Q_terminal (xN − x_terminal_ref)

  where x_terminal_ref[:2] = z_target

  + soft obstacle penalties (quadratic slack)
```

`Q` and `R` are set by the translator and kept fixed by the integrated runner.
The learnable terminal parameter is `z_target`. After each MPC leg, the runner
applies a fixed per-location observation offset, evaluates
`μ(goal_location)`, and updates `z_target` from `E = 1 - μ(goal_location)`
through `J_E_z_target.T @ E`; if no analytical membership sensitivity is
available, it uses a finite-difference rollout fallback.

Obstacle constraints are soft (always feasible, penalised quadratically).

## Dependency Graph

```
formulation ◄── obstacle_utils
     ▲
mpc_solver ◄── hybrid (shim + test)
```
