# core/execution/

MPC controllers for real-time robot trajectory tracking.

## Module Structure

`hybrid.py` was split into four focused modules. All public names are still importable from `hybrid.py` for backward compatibility.

### `formulation.py`
Shared problem definition — single source of truth for both Acados and CasADi.

- `SharedMPCFormulation` — dimensions, bounds, default weights, continuous/discrete dynamics (CasADi and NumPy variants)
- `MPCSolution` — dataclass: control, trajectory, cost, solve time, primal/dual solution
- `MPCSensitivity` — dataclass: `dJ_dQ`, `dJ_dR`, `dJ_dz_target`, `du0_dQ`, `du0_dR`

### `obstacle_utils.py`
Obstacle geometry helpers.

- `filter_nearby_obstacles()` — trims the full environment obstacle list to the 3–5 most relevant obstacles for the MPC horizon, scoring by proximity to the robot-to-goal path segment and inflating radii by a configurable safety margin

### `ift_engine.py`
CasADi IFT sensitivity engine (learning path, ~5–10 ms).

- `CasADiSensitivityComputer` — builds symbolic KKT sensitivity functions at construction time; at runtime accepts `(w*, λ*, p)` from Acados and returns `∂J*/∂p`, `∂u*₀/∂p` without re-solving the NLP (per paper eq. 70)
- Falls back to a full IPOPT solve when Acados is unavailable

### `mpc_solver.py`
Acados solver and hybrid orchestrator (control path, ~1–5 ms).

- `AcadosSolver` — SQP-RTI solver; bakes obstacle constraints into C code at build time; rebuilds only when obstacle count changes; exports `w*` for IFT
- `HybridMPC` — routes `solve()` calls to Acados (or CasADi fallback), routes `solve_with_sensitivities()` through both paths, aggregates episode sensitivities for the translator update loop

### `hybrid.py`
Backward-compatibility shim. Re-exports all public names from the four modules above. Also contains `test_hybrid_mpc()` and the `__main__` entry point.

---

## Architecture (Section 6.7)

```
CONTROL PATH (every timestep, ~1-5ms):
  x_t  ──►  AcadosSolver.solve()  ──►  u*

LEARNING PATH (periodic, for translator training, ~5-10ms):
  x_t  ──►  AcadosSolver.solve()  ──►  (w*, λ*)
                                           │
                                     CasADiSensitivityComputer
                                           │
                                       ∂J*/∂p, ∂u*₀/∂p
```

Key insight: Acados and CasADi share the exact same `SharedMPCFormulation`.
Acados solves fast; CasADi differentiates the KKT conditions at the Acados solution.

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
  ℓ(xk, uk) = (xk − x_ref_stage)ᵀ Q (xk − x_ref_stage) + ukᵀ R uk

  where x_ref_stage = [x_ref[:2] + z_target(ŵ), x_ref[2:]]
        z_target(ŵ) = A @ ŵ + b   (A: 2×5, b: 2D — learnable)

Terminal cost (k = N):
  ℓ_N(xN) = 10 · (xN − x_ref)ᵀ Q (xN − x_ref)   ← tracks waypoint exactly

  + soft obstacle penalties (quadratic slack)
```

`Q`, `R`, and `z_target` are all set by the learnable translator from preference
weights `ŵ`. The z_target offset allows the system to learn preference-conditioned
position biases along the trajectory (e.g. a safety-conscious patient causes the
robot to bias its path away from obstacles). Initialised to zero — identical to
fixed x_ref tracking until learned.

The IFT engine automatically computes `∂J*/∂z_target` alongside `∂J*/∂Q` and
`∂J*/∂R`, so A and b are updated via the same gradient pipeline with no
architectural change.

Obstacle constraints are soft (always feasible, penalised quadratically).

## Dependency Graph

```
formulation ◄── obstacle_utils
     ▲
ift_engine  ◄── mpc_solver ◄── hybrid (shim + test)
```
