# tasks/meal_preparation/

Meal preparation and delivery task. The robot selects a meal complexity path, prepares the food in the kitchen area of env3, and delivers it to the patient — with the path choice itself adapting to the patient's learned preferences.

## Files

### `task_state.py` — Meal Task State

`MealTaskState(TaskStateMixin)` — inherits battery helpers and shared copy/dict utilities from `core/task_planning/base_state.py`.

Task-specific progression flags:
```
has_ingredients → is_chopped → is_cooked → is_plated → meal_ready → delivered
```

- `meal_type: str` — `'sandwich'`, `'soup'`, or `'full_meal'` (set at episode start)
- Shared fields (location, battery_soc, approach_side, memberships, tracking) via mixin

---

### `task_actions.py` — Meal Action Set

| Category | Actions |
|----------|---------|
| Navigation | `GO_TO_PANTRY`, `GO_TO_PREP_STATION`, `GO_TO_STOVE`, `GO_TO_PATIENT_{LEFT,RIGHT}`, `GO_TO_CHARGE_{MAIN,BACKUP}` |
| Preparation | `COLLECT_INGREDIENTS`, `ASSEMBLE` (sandwich only), `CHOP` (soup/full), `COOK` (soup/full), `PLATE` (full only) |
| Delivery | `DELIVER_MEAL` |

---

### PDDL/ENHSP Task Planning

Meal planning is defined in `unified_planning/domain_meal.pddl` and `unified_planning/problem_meal.pddl`. The episode runner syncs the current Python state into the PDDL initial state and calls ENHSP-opt through Unified Planning.
- `_expand(state)` — calls `manager.apply_action` for each available action, then `_calculate_action_cost`
- `_heuristic(state)` — counts remaining prep steps and estimates distance to patient

The planner simultaneously evaluates all three meal paths and selects the one with lowest preference-weighted cost.

**Meal paths:**

| Path | Steps | Duration | Safety cost | Approach cost |
|------|-------|----------|-------------|---------------|
| Sandwich | collect → assemble → deliver | Short | 0.00 (cold) | +0.15 (no plating) |
| Soup | collect → chop → cook → deliver | Medium | 0.15 (hot) | +0.05 |
| Full Meal | collect → chop → cook → plate → deliver | Long | 0.25 (hot, complex) | 0.00 (plated) |

**Preference-driven path selection:**

- **Speed-oriented** patient → sandwich (fastest; poor approach quality is tolerated)
- **Safety-first** patient → sandwich or soup (avoids stove risk 0.70)
- **Presentation-focused** patient → full meal (plating bonus dominates; stove risk accepted)
- **Energy-conscious** patient → sandwich (fewest steps, minimal movement)

---

### `task_state_manager.py` — Ordered Preconditions

Enforces meal progression ordering:
- `CHOP` requires `has_ingredients`
- `COOK` requires `is_chopped`
- `PLATE` requires `is_cooked`
- `DELIVER_MEAL` requires `meal_ready` (or at minimum `has_ingredients` for sandwich path)

Goal: `meal_ready ∧ delivered`

---

### `meal_profiles.py` — Feature Generation and Quality Bonuses

Converts raw execution metrics into the 5D feature vector `f ∈ [0,1]⁵` with meal-type-specific quality adjustments:

| Meal Type | Time | Safety | Battery | Proximity | Approach |
|-----------|------|--------|---------|-----------|----------|
| Sandwich | +0.05 | — | — | +0.45 | +0.15 |
| Soup | — | +0.10 | — | +0.05 | — |
| Full Meal | −0.05 | +0.12 | — | — | −0.20 |

These shifts reflect real-world meal quality differences:
- Sandwiches are quick but presentation suffers (`approach +0.15`)
- Full meals are slow but beautifully plated (`approach −0.20`, i.e. better)
- Soup carries freshness timing constraints (`safety +0.10`)

The meal type feature distribution is structurally different from medication delivery, providing richer learning signal for the approach and safety dimensions.

---

## Kitchen Layout (env3 locations)

```
pantry (−3, 15)         → ingredient collection
prep_station (0, 20)    → assembly (sandwich) and plating (full meal)
stove (−2, 22)          → cooking (soup, full meal) — highest risk location (0.70)
patient_bed (22, 12)    → delivery
```

---

## Why Meal Preparation Matters for Learning

Medication delivery produces similar feature patterns every episode (the task structure is fixed). Mixing in meal preparation episodes creates **structural diversity** in the feature space:

- Medication: moderate time, moderate safety, moderate approach
- Full meal: slow, riskier (stove), excellent approach
- Sandwich: fast, safe, poor approach

This diversity helps the preference learner disambiguate, for example, patients who want good approach quality (presentation-focused) from patients who just want fast delivery (speed-oriented) — two patterns that might look similar in medication-only data.
