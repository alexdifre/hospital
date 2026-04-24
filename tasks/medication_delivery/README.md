# tasks/medication_delivery/

Complete medication delivery task implementation. The robot navigates from home through the pharmacy (and optionally a supply depot) to the patient's bedside, adapting its route and speed profile to the patient's learned preferences.

## Files

### `task_state.py` — Discrete Task State

`TaskState(TaskStateMixin)` — inherits battery helpers and shared copy/dict utilities from `core/task_planning/base_state.py`.

Task-specific state variables:
- `location: str` — current named location (dominant fuzzy membership)
- `has_medication: bool` — collected from pharmacy
- `has_supplement: bool` — collected from supply depot
- `delivered: bool` — medication handed to patient

Shared via mixin:
- `battery_soc: float` — continuous [0,1], discretised to 8 levels for A* hashing
- `approach_side: str | None` — `'left'`, `'right'`, or `None`
- `location_memberships: Dict[str, float]` — fuzzy position estimates
- `location_stock: Dict[str, int]` — remaining stock per stocked location

---

### `task_actions.py` — Action Set

Navigation and in-place actions:

| Category | Actions |
|----------|---------|
| Navigation | `GO_TO_PHARMACY_{NORTH,SOUTH}`, `GO_TO_SUPPLY_{A,B}`, `GO_TO_CHARGE_{MAIN,BACKUP}`, `GO_TO_PATIENT_{LEFT,RIGHT}` |
| In-place | `COLLECT_MEDICATION`, `COLLECT_SUPPLEMENT`, `RECHARGE`, `DELIVER` |

Action durations: `RECHARGE = 30 s`, navigation actions `5–10 s` depending on distance.

---

### `task_planner.py` — A* Over Task Space

`HighLevelTaskPlanner(BaseTaskPlanner)` — inherits the A* loop from `core/task_planning/base_planner.py`. Implements:
- `_expand(state)` — calls `estimate_action_cost` + `apply_action` on the state manager, then `_calculate_action_cost`
- `_heuristic(state)` — admissible distance-to-goal estimate over remaining pharmacy/supply/patient legs

**Cost function (preference-weighted):**
```
cost = w_time    × time_estimate(action)
     + w_safety  × risk_estimate(target_location)
     + w_battery × distance_estimate(action)
     + w_proximity × delivery_error_estimate
     + w_approach × approach_quality_estimate
```

**Planning decisions shaped by preferences:**
- **Speed-oriented** patient → picks pharmacy_north (shorter distance, higher risk accepted)
- **Safety-first** patient → picks pharmacy_south (longer route, lower risk 0.05 vs 0.30)
- **Energy-conscious** patient → minimises total distance; adds a recharge step if battery is low enough that not charging would cost more energy overall

Plans are re-computed after each action based on actual outcomes (new location, battery level, stock).

---

### `task_state_manager.py` — State Transitions

Validates and applies action outcomes:
- **Preconditions**: `DELIVER` requires `has_medication`; `COLLECT_SUPPLEMENT` requires being at a supply location
- **Dynamic stock**: Pharmacy stock decrements during planning to avoid planning around a depleted location
- **Goal check**: `has_medication ∧ has_supplement ∧ delivered`

---

### `reward_engine.py` — Feature Extraction

Accumulates per-step measurements during execution and normalises them to [0, 1] at episode end for the preference learner.

| Feature | What is measured | Normalisation |
|---------|-----------------|---------------|
| time | Episode duration (s) | `t / max_time` |
| safety | Min distance to patient/obstacles | Safety score → [0,1] |
| battery | Total energy consumed | `Δbattery / 1.0` |
| proximity | Movement comfort near patient (2 m zone) | Comfort scores averaged |
| approach | Final positioning precision | Approach quality score |

---

### `learnable_translator.py` — Shim

Backward-compat re-export. Canonical code lives in `core/learning/learnable_translator.py`.

### `translator_params.py` — Shim

Backward-compat re-export. Canonical code lives in `core/learning/translator_params.py`.

---

## Task Flow

```
Episode start
    │
    ▼
TaskPlanner.plan(state, w_hat)    → action sequence
    │
    ▼  (for each action)
TaskStateManager.apply(action)    → new task state
FuzzyStateEstimator.estimate()    → location memberships
SpatialPlanner.plan(current, target) → waypoints
HybridMPC.solve(waypoints, Q, R)  → u*, sensitivities
MuJoCo.step(u*)                   → physics
RewardEngine.record(step_data)
    │
    ▼  (episode end)
features = RewardEngine.compute_features()
w_hat = PreferenceLearner.update(features, ratings)
φ = Translator.update(sensitivities)
```
