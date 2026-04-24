# Hospital Robot System — Architecture

## Design Philosophy

Three layers, each independently interpretable:

1. **Symbolic task planning** — A* over discrete task states produces a high-level action sequence
2. **Continuous MPC execution** — Hybrid Acados/CasADi controller tracks waypoints in 6-DOF state space
3. **Dual learning loops** — outer loop learns patient preference weights; inner loop learns translator parameters via IFT chain rule

The translation layer between planning and execution is what makes the system adaptive without sacrificing determinism: same preference weights always produce the same control matrices.

---

## Module Structure

### `core/` — Task-agnostic framework

#### `core/execution/`

Hybrid MPC controller split into four focused modules:

| Module | Contents |
|--------|----------|
| `formulation.py` | `MPCSolution`, `MPCSensitivity` dataclasses; `SharedMPCFormulation` (dynamics, bounds, default Q/R) |
| `ift_engine.py` | `CasADiSensitivityComputer` — builds KKT sensitivity functions; `solve_and_get_sensitivities()` |
| `mpc_solver.py` | `AcadosSolver` (SQP-RTI real-time control); `HybridMPC` orchestrator |
| `obstacle_utils.py` | `filter_nearby_obstacles()` with point-to-segment distance |
| `hybrid.py` | Backward-compat shim re-exporting all public names |

`__init__.py` re-exports everything so `from core.execution import HybridMPC` works directly.

The two solvers serve different purposes:
- **Acados** runs every timestep for real-time control (1–5 ms)
- **CasADi** runs periodically to compute `∂J*/∂φ` for translator learning (5–10 ms)

#### `core/planning/`

A* grid planner over the occupancy map. Produces waypoint sequences for the MPC to track. `FuzzyStateEstimator` bridges continuous robot position to discrete location memberships used by the task planner.

#### `core/learning/`

| Module | Contents |
|--------|----------|
| `preference_learning_engine.py` | Projected gradient descent on the 5-simplex; updates `w_hat` after each episode |
| `learnable_translator.py` | Maps `w → (Q, R, safety_margin)` via learned affine parameters `φ` |
| `translator_params.py` | Parameter initialisation and bounds for `φ` |

#### `core/task_planning/`

Shared base classes extracted to avoid duplication across task packages:

| Module | Contents |
|--------|----------|
| `base_state.py` | `TaskStateMixin` — `get_discrete_battery_level()`, `needs_recharge()`, `_shared_copy_kwargs()`, `_shared_to_dict()` |
| `base_planner.py` | `BaseTaskPlanner` — full A* loop with `_reconstruct_path()`; abstract `_expand()` + `_heuristic()` |

#### `core/environment/`

MuJoCo 6-DOF hospital simulation (`env.py`). State space `[x, y, θ, vx, vy, ωz]`, control space `[ax, ay, α]`. 15 named locations with fuzzy membership, stock levels, risk values, and congestion zones.

---

### `tasks/` — Task-specific implementations

Each task package follows the same structure:

| File | Purpose |
|------|---------|
| `task_state.py` | Dataclass inheriting `TaskStateMixin`; task-specific progression flags |
| `task_actions.py` | Action enum, durations, valid locations |
| `task_planner.py` | Inherits `BaseTaskPlanner`; implements `_expand()` + `_heuristic()` |
| `task_state_manager.py` | Precondition checking and state transitions |

#### `tasks/medication_delivery/`

Robot navigates home → pharmacy → (optional supply depot) → patient bed.

- Two pharmacy options with different risk/distance tradeoffs
- `reward_engine.py` normalises execution metrics to 5D feature vector `f ∈ [0,1]⁵`
- Preference-weighted A* chooses pharmacy and approach side

#### `tasks/meal_preparation/`

Robot prepares and delivers one of three meal types from the kitchen (env3 locations).

| Path | Steps | Safety cost | Approach cost |
|------|-------|-------------|---------------|
| Sandwich | collect → assemble → deliver | 0.00 | +0.15 |
| Soup | collect → chop → cook → deliver | 0.15 | +0.05 |
| Full Meal | collect → chop → cook → plate → deliver | 0.25 | 0.00 |

`meal_profiles.py` generates meal-type-specific feature adjustments that create structural diversity in the preference learning signal.

---

### `integration/`

Full system orchestration:

| Module | Contents |
|--------|----------|
| `system.py` | `FullMedicationDeliverySystem` — builds all components, owns the episode loop |
| `episode_runner.py` | `EpisodeRunnerMixin` — per-episode execution logic |
| `reporting.py` | `ReportingMixin` — convergence summaries, plan printing |
| `metrics.py` | Feature extraction and normalisation helpers |
| `integrator2.py` | 2-line backward-compat shim → `system.py` |

---

### `tests/`

| File/Directory | Purpose |
|----------------|---------|
| `run_section8_experiments.py` | Full experiment runner (all conditions × profiles × seeds) |
| `generate_section7_figures.py` | Section 7 setup figures (floor plan, state diagrams, profiles) |
| `generate_section8_figures.py` | Section 8 result figures (convergence, ablations, robustness) |
| `profile_validation/run_profile_tests.py` | Unified profile validation runner (5 profiles, config-table driven) |
| `profile_validation/harness.py` | Shared test harness (`ProfileConfig`, `run_suite`, `test_2_route_choice`) |
| `profile_validation/med_delivery_energy.py` | Battery-specific validation (custom logic, kept separate) |
| `ift_sensitivity_check.py` | Standalone IFT gradient sanity check |

---

## Data Flow (Single Episode)

```
w_hat (current estimate)
    │
    ▼
TaskPlanner._expand() × A*          ← preference-weighted cost function
    │  action sequence
    ▼
LearnableTranslator(w_hat, φ)       ← learned affine map
    │  Q, R, safety_margin
    ▼
NavigationStack.plan()              ← A* over occupancy grid
    │  waypoints
    ▼  (for each waypoint)
HybridMPC.solve()                   ← Acados SQP-RTI
    │  u* → MuJoCo.step()
    │  ∂J*/∂φ ← CasADiSensitivityComputer (periodic)
    ▼
RewardEngine.compute_features()     ← normalise to f ∈ [0,1]⁵
    │
    ▼
PreferenceLearner.update(f, ratings)   ← projected gradient → w_hat
LearnableTranslator.update(∂J*/∂φ)    ← IFT chain rule → φ
```

---

## Learning Loop Details

### Outer Loop — Preference Learning

After each episode the patient rates each feature dimension (simulated from hidden `w*`). The learner performs projected gradient descent on the 5-simplex:

```
w_hat ← Π_Δ( w_hat - η · ∇_w L(w_hat, ratings) )
```

with learning rate decay `η_t = η₀ / (1 + decay · t)` and EMA smoothing to dampen oscillation.

### Inner Loop — Translator Learning (IFT)

The translator parameters `φ` are updated using the Implicit Function Theorem to propagate MPC optimality conditions:

```
∂J*/∂φ = −(∂²H/∂u²)⁻¹ · (∂²H/∂u∂φ)   [from KKT sensitivity]
φ ← φ − α · (∂J*/∂φ)ᵀ
```

`CasADiSensitivityComputer` builds the KKT Jacobians symbolically at construction time; `solve_and_get_sensitivities()` evaluates them at each MPC solution point.

---

## Adding a New Task

1. Create `tasks/<new_task>/` with the four standard files
2. Inherit `TaskStateMixin` in `task_state.py`
3. Inherit `BaseTaskPlanner` in `task_planner.py`; implement `_expand()` and `_heuristic()`
4. Add a feature extractor (like `reward_engine.py` or `meal_profiles.py`)
5. Register the task type in `integration/system.py`

No changes to `core/` are needed.
