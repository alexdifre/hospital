# tasks/

Task-specific implementations. Each sub-package is self-contained and follows the same internal structure so new tasks can be added without modifying `core/`.

## Sub-packages

| Directory | Task | Status |
|-----------|------|--------|
| [medication_delivery/](medication_delivery/) | Robot delivers prescribed medication from pharmacy to patient | Complete |
| [meal_preparation/](meal_preparation/) | Robot prepares and delivers a patient meal | Complete |

## Common Structure

Every task package contains:

| File | Purpose |
|------|---------|
| `task_state.py` | Discrete task state — inherits `TaskStateMixin` from `core/task_planning/` |
| `task_actions.py` | Action enum and constants (durations, valid locations) |
| `task_state_manager.py` | Precondition checking and state transitions |

Task-specific extras (reward computation, translator parameters, meal profiles) live alongside these core files.

## Shared Base Classes (`core/task_planning/`)

| Module | Provides |
|--------|---------|
| `base_state.py` — `TaskStateMixin` | `get_discrete_battery_level()`, `needs_recharge()`, `_shared_copy_kwargs()`, `_shared_to_dict()` |
| `pddl_engine.py` | ENHSP-opt engine selection for Unified Planning/PDDL |

## How Tasks Plug Into the Framework

```
PDDL Task Planner (ENHSP-opt)
    ↓ action sequence
Task State Manager (validates transitions)
    ↓ next action + target location
FuzzyStateEstimator → direct waypoint reference → HybridMPC
    ↓ episode features [time, safety, battery, proximity, approach]
Episode Runner / Meal Profiles (normalises features)
    ↓
Preference Learner (updates w_hat)
```

The task layer is responsible for defining *what* constitutes good performance; the `core/` layer is responsible for *how* to execute and learn.
