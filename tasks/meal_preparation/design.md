# Meal Preparation Task — Architecture Design

## 1. Core Idea (from Sebastien's email)

> "The plan for the tasks should be complex — only a set sequence of steps
> of a cooking preparation yields a meal, and each different valid meal
> preparation carries a different distribution of enjoyment from the human."

The planner must **choose which meal to make**, then execute the correct
step ordering. Different meals produce different feature vectors, so the
preference learner gets genuinely diverse training signal — fixing the
convergence bottleneck from medication delivery.

---

## 2. Meal Types (3 options)

### A) Cold Sandwich

- **Route**: pantry → prep_station → patient
- **Steps**: collect_ingredients → assemble → deliver_meal
- **Character**: Fast, low battery use, but lower patient satisfaction on quality
- **Prep time cost**: 5s (assemble only)

### B) Warm Soup

- **Route**: pantry → prep_station → stove → patient
- **Steps**: collect_ingredients → chop → cook → deliver_meal
- **Character**: Medium speed, requires stove visit, warming/comforting
- **Prep time cost**: 5s (chop) + 10s (cook) = 15s

### C) Full Cooked Meal

- **Route**: pantry → prep_station → stove → prep_station → patient
- **Steps**: collect_ingredients → chop → cook → plate → deliver_meal
- **Character**: Slowest, must return to prep_station for plating, highest quality
- **Prep time cost**: 5s (chop) + 10s (cook) + 5s (plate) = 20s

Key structural difference: **the full meal requires revisiting prep_station
after the stove**, creating a genuinely longer route. This is what makes
the planner's meal choice a real tradeoff.

---

## 3. Action Enum

```python
class MealAction(Enum):
    # Navigation
    GO_TO_PANTRY = "go_to_pantry"
    GO_TO_PREP_STATION = "go_to_prep_station"
    GO_TO_STOVE = "go_to_stove"
    GO_TO_PATIENT_LEFT = "go_to_patient_left"
    GO_TO_PATIENT_RIGHT = "go_to_patient_right"
    GO_TO_CHARGE_MAIN = "go_to_charge_main"

    # Meal-specific collection (determines which meal path)
    COLLECT_SANDWICH_INGREDIENTS = "collect_sandwich_ingredients"
    COLLECT_SOUP_INGREDIENTS = "collect_soup_ingredients"
    COLLECT_MEAL_INGREDIENTS = "collect_meal_ingredients"

    # Preparation (location-gated)
    ASSEMBLE = "assemble"           # sandwich only, at prep_station
    CHOP = "chop"                   # soup/full meal, at prep_station
    COOK = "cook"                   # soup/full meal, at stove
    PLATE = "plate"                 # full meal only, at prep_station (after cooking)

    # Delivery
    DELIVER_MEAL = "deliver_meal"

    # Utility
    RECHARGE = "recharge"
```

---

## 4. Task State

```python
@dataclass
class MealTaskState:
    location: str

    # What's been collected/done (progression flags)
    meal_type: Optional[str] = None      # 'sandwich', 'soup', 'full_meal', None
    has_ingredients: bool = False
    is_chopped: bool = False
    is_cooked: bool = False
    is_assembled: bool = False            # sandwich only
    is_plated: bool = False               # full meal only
    meal_ready: bool = False              # ready for delivery
    delivered: bool = False

    # Shared state
    battery_soc: float = 1.0
    approach_side: Optional[str] = None
    location_memberships: Optional[Dict[str, float]] = None
    location_stock: Optional[Dict[str, int]] = None

    step_count: int = 0
    time_elapsed: float = 0.0
    distance_traveled: float = 0.0
```

---

## 5. Precondition Graph (the ordering constraints)

```
COLLECT_SANDWICH_INGREDIENTS:
  requires: location ∈ {pantry}, meal_type is None
  effects:  meal_type = 'sandwich', has_ingredients = True

COLLECT_SOUP_INGREDIENTS:
  requires: location ∈ {pantry}, meal_type is None
  effects:  meal_type = 'soup', has_ingredients = True

COLLECT_MEAL_INGREDIENTS:
  requires: location ∈ {pantry}, meal_type is None
  effects:  meal_type = 'full_meal', has_ingredients = True

ASSEMBLE:
  requires: location ∈ {prep_station}, meal_type == 'sandwich', has_ingredients
  effects:  is_assembled = True, meal_ready = True

CHOP:
  requires: location ∈ {prep_station}, meal_type ∈ {'soup', 'full_meal'}, has_ingredients, NOT is_chopped
  effects:  is_chopped = True

COOK:
  requires: location ∈ {stove}, is_chopped, NOT is_cooked
  effects:  is_cooked = True
            if meal_type == 'soup': meal_ready = True  (soup serves from pot)

PLATE:
  requires: location ∈ {prep_station}, meal_type == 'full_meal', is_cooked, NOT is_plated
  effects:  is_plated = True, meal_ready = True

DELIVER_MEAL:
  requires: location ∈ {patient_bed_left, patient_bed_right}, meal_ready
  effects:  delivered = True  → GOAL
```

This gives the valid sequences:

```
Sandwich:  pantry → collect_sandwich → prep → assemble → patient → deliver
Soup:      pantry → collect_soup → prep → chop → stove → cook → patient → deliver
Full meal: pantry → collect_meal → prep → chop → stove → cook → prep → plate → patient → deliver
```

---

## 6. New Environment Locations

```python
# Kitchen area (north-west quadrant, near pharmacy_north)
"pantry":       np.array([-3.0, 15.0]),    # ingredient storage
"prep_station": np.array([0.0, 20.0]),     # chopping/assembly/plating
"stove":        np.array([-2.0, 22.0]),    # cooking

# Stock for ingredient availability
"pantry": {
    "type": "food_storage",
    "initial_stock": {
        "sandwich": 10,
        "soup": 5,
        "full_meal": 3,
    },
    "size": 1.0,
}
```

Placed in the north-west to create interesting distance tradeoffs:

- Pantry is close to home → sandwich ingredients nearby
- Stove requires going further north → cooking meals costs more distance
- Patient is far east → long delivery leg regardless of meal type
- But prep_station is between pantry and stove → chop doesn't add much

---

## 7. Feature Mapping (5 dimensions, same as medication)

| Feature   | What it measures for meal prep                           |
| --------- | -------------------------------------------------------- |
| time      | Total time from start to delivery (normalized)           |
| safety    | Hygiene + careful handling (penalize rushing, hot items) |
| battery   | Energy consumption (proportional to distance + cooking)  |
| proximity | Freshness — time between food-ready and delivery         |
| approach  | Presentation — plating quality + approach angle          |

### Feature generation per meal type:

**Sandwich**: low time (good), low safety concern, low battery, medium proximity, no plating bonus
**Soup**: medium time, higher safety (hot liquid), medium battery, medium proximity, no plating bonus  
**Full meal**: high time (bad), highest safety concern (hot, complex), high battery, lower proximity (travels further), plating bonus

This means the **same patient profile** rates meals differently:

- Safety-first patient: prefers soup/full meal (careful prep) but penalizes rushing
- Speed-first patient: prefers sandwich strongly
- Quality/approach patient: prefers full meal (plating bonus)

---

## 8. Enjoyment Distributions (how patients rate each meal)

The patient rating model extends naturally. The existing model:

```
r_i = 5 - 4 * f_i * w_i + noise
```

For meal prep, we add a **meal quality bonus** that modulates the base rating:

```python
MEAL_QUALITY = {
    'sandwich':  {'time': 0.0, 'safety': 0.0, 'battery': 0.0, 'proximity': 0.0, 'approach': -0.15},
    'soup':      {'time': 0.0, 'safety': 0.1, 'battery': 0.0, 'proximity': 0.05, 'approach': 0.0},
    'full_meal': {'time': 0.0, 'safety': 0.1, 'battery': 0.0, 'proximity': 0.1,  'approach': 0.2},
}
```

So a full meal gets +0.2 on approach (plating bonus), +0.1 on proximity (freshness from
careful prep), +0.1 on safety (more careful handling). But it also has worse time features
(it takes longer), which the speed dimension penalizes.

---

## 9. Why This Fixes Convergence

With medication delivery: 2 plan variants, nearly identical features every episode.

With meal prep added:

- **3 meal types** × 2 approach sides × variable start locations = many more plan variants
- Each meal type produces **structurally different** feature vectors
- The planner must choose the right meal for the patient's preferences
- The preference learner sees diverse {feature, rating} pairs → faster convergence
- Running both tasks (medication + meal) in alternating episodes doubles the signal

---

## 10. File Structure

```
tasks/
  meal_preparation/
    __init__.py
    task_actions.py          # MealAction enum
    task_state.py            # MealTaskState dataclass
    task_state_manager.py    # Preconditions, transitions, available actions
    meal_profiles.py         # MEAL_QUALITY bonuses, feature generation
```

## 11. Integration Plan

1. Add pantry/prep_station/stove to env2.py (coordinates + metadata)
2. Build task_state_manager.py (preconditions + transitions)
3. Define meal branches in `unified_planning/domain_meal.pddl`
4. Build meal_profiles.py (feature generation + quality bonuses)
5. Wire into full_system_integrator_v4.py as second task type
6. Test: ENHSP-opt solves valid meal plans under different weight profiles
