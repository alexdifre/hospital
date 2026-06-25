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
- `battery_soc: float` — continuous [0,1], discretised to 8 levels for state hashing
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

### PDDL/ENHSP Task Planning

Medication planning is defined in `unified_planning/domain_med.pddl` and `unified_planning/problem_med.pddl`. The episode runner syncs the current Python state into the PDDL initial state and calls ENHSP-opt through Unified Planning.
- `_expand(state)` — calls `estimate_action_cost` + `apply_action` on the state manager, then `_calculate_action_cost`
- `_heuristic(state)` — returns zero, making the search uniform-cost until a proven admissible task heuristic is introduced

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
direct waypoint reference         → 21 start-to-goal points
HybridMPC.solve(Q, R)             → u*, sensitivities
MuJoCo.step(u*)                   → physics
    │
    ▼  (episode end)
features = EpisodeRunner normalised metrics
w_hat = PreferenceLearner.update(features, ratings)
φ = Translator.update(sensitivities)
```
