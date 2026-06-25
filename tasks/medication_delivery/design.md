# Medication Delivery Task — Architecture Design

## 1. Core Idea

The robot must collect a primary medication from a pharmacy, a supplementary
item from a supply room, and deliver both to the patient's bedside. The
planner chooses which pharmacy, which supply room, which approach side, and
whether to insert a recharge stop — producing different plan variants whose
cost profiles depend on the patient's learned preference weights.

This is the primary task in the MLC stack. Unlike meal preparation (which
generates diverse features through meal-type branching), medication delivery's
diversity comes from **route selection**: different source locations carry
different risk, distance, and battery tradeoffs.

---

## 2. Plan Structure (branching choices)

Every valid plan follows one template with 4 decision points:

```
[start] → pharmacy_{north|south}   → collect_medication
        → supply_{A|B}             → collect_supplement
        → (optional) charge_{main|backup} → recharge
        → patient_bed_{left|right} → deliver
```

### Decision points:

**A) Pharmacy** — North vs South

- pharmacy_north: closer to home and nurse_station, but higher congestion
  risk (0.30). Faster route, riskier environment.
- pharmacy_south: further away, but very low risk (0.05). Slower route,
  safer environment.

**B) Supply room** — A vs B

- supply_A: low risk (0.05), good stock levels
- supply_B: moderate risk (0.30), may have different stock availability

**C) Recharge** — Skip vs charge_main vs charge_backup

- Inserted by the planner when battery_soc is below threshold
- charge_main (risk 0.05) vs charge_backup (risk 0.08)
- Adds 30s but prevents mission failure from battery depletion

**D) Approach side** — Left vs Right

- patient_bed_left vs patient_bed_right
- Affects final position + orientation alignment (approach quality score)

Total skeleton space: 2 × 2 × 3 × 2 = 24, though A\* typically finds
4-6 dominant variants depending on weights and start location.

---

## 3. Action Enum

```python
class TaskAction(Enum):
    # Navigation actions (8)
    GO_TO_PHARMACY_NORTH = "go_to_pharmacy_north"
    GO_TO_PHARMACY_SOUTH = "go_to_pharmacy_south"
    GO_TO_SUPPLY_A = "go_to_supply_a"
    GO_TO_SUPPLY_B = "go_to_supply_b"
    GO_TO_CHARGE_MAIN = "go_to_charge_main"
    GO_TO_CHARGE_BACKUP = "go_to_charge_backup"
    GO_TO_PATIENT_LEFT = "go_to_patient_left"
    GO_TO_PATIENT_RIGHT = "go_to_patient_right"

    # In-place actions (4)
    COLLECT_MEDICATION = "collect_medication"
    COLLECT_SUPPLEMENT = "collect_supplement"
    RECHARGE = "recharge"
    DELIVER = "deliver"
```

Module-level constants (matching meal prep pattern):

```python
ACTION_TARGET_LOCATIONS = {TaskAction.GO_TO_PHARMACY_NORTH: "pharmacy_north", ...}
NAVIGATION_ACTIONS = set(ACTION_TARGET_LOCATIONS.keys())
IN_PLACE_ACTIONS = {COLLECT_MEDICATION, COLLECT_SUPPLEMENT, RECHARGE, DELIVER}
ACTION_DURATIONS = {COLLECT_MEDICATION: 5.0, COLLECT_SUPPLEMENT: 5.0, RECHARGE: 30.0, DELIVER: 10.0}
```

---

## 4. Task State

```python
@dataclass
class TaskState:
    location: str

    # Task progression flags
    has_medication: bool = False
    has_supplement: bool = False
    delivered: bool = False

    # Shared state
    battery_soc: float = 1.0
    approach_side: Optional[str] = None      # 'left', 'right', or None
    location_memberships: Optional[Dict[str, float]] = None
    location_stock: Optional[Dict[str, int]] = None

    # Episode tracking
    step_count: int = 0
    time_elapsed: float = 0.0
    distance_traveled: float = 0.0
    num_replans: int = 0
```

Hashing discretises battery into 8 levels for A\* closed-set membership.
Goal check: `self.delivered == True`.

---

## 5. Precondition Graph (ordering constraints)

```
COLLECT_MEDICATION:
  requires: location ∈ {pharmacy_north, pharmacy_south}, NOT has_medication,
            stock(location) > 0
  effects:  has_medication = True, stock(location) -= 1

COLLECT_SUPPLEMENT:
  requires: location ∈ {supply_A, supply_B}, NOT has_supplement,
            stock(location) > 0
  effects:  has_supplement = True, stock(location) -= 1

RECHARGE:
  requires: location ∈ {charge_main, charge_backup}, battery_soc < 1.0
  effects:  battery_soc = 1.0

DELIVER:
  requires: location ∈ {patient_bed_left, patient_bed_right},
            has_medication AND has_supplement
  effects:  delivered = True  → GOAL
```

Fuzzy mode: location checks use membership threshold (μ ≥ 0.7) instead
of crisp equality. FuzzyStateEstimator bridges continuous robot position
from MPC execution into discrete precondition checks.

Stock tracking: `state.location_stock` maintains a planning-time copy of
inventory levels. The planner decrements stock on collection actions so
A\* can reason about stockout scenarios and route around depleted sources.

---

## 6. Environment Locations

```python
# Pre-existing hospital locations
"home":               np.array([0.0, 0.0]),
"pharmacy_north":     np.array([5.0, 15.0]),
"pharmacy_south":     np.array([-5.0, -10.0]),
"supply_A":           np.array([10.0, 5.0]),
"supply_B":           np.array([-3.0, 8.0]),
"nurse_station":      np.array([8.0, 12.0]),
"equipment_storage":  np.array([12.0, -5.0]),
"charge_main":        np.array([15.0, 0.0]),
"charge_backup":      np.array([-2.0, -15.0]),
"patient_bed_left":   np.array([20.0, 10.0]),
"patient_bed_right":  np.array([22.0, 8.0]),
```

Key spatial relationships:

- pharmacy_north is near nurse_station (high congestion zone)
- pharmacy_south is isolated (low risk, but far from home/patient)
- Patient beds are in the east — every route has a long delivery leg
- Charging stations are positioned so that detours vary by start location

### Location risk map

```
nurse_station:      0.60    (highest — congestion + equipment)
equipment_storage:  0.40
pharmacy_north:     0.30
supply_B:           0.30
patient_bed_left:   0.15
patient_bed_right:  0.15
pharmacy_south:     0.05
supply_A:           0.05
charge_backup:      0.08
charge_main:        0.05
home:               0.02
```

The risk map feeds into the safety cost component. Visiting pharmacy_north
(risk 0.30, near nurse_station 0.60) vs pharmacy_south (risk 0.05) is the
primary safety-vs-speed tradeoff in the task.

---

## 7. Feature Mapping (5 dimensions, shared with meal prep)

| Feature   | What it measures                                   | Normalisation           |
| --------- | -------------------------------------------------- | ----------------------- |
| Time      | Total start-to-delivery elapsed time               | ÷ 120s                  |
| Safety    | Avg risk of all visited locations (from risk map)  | ÷ 0.60 (max risk)       |
| Battery   | Total energy consumed during navigation            | distance × 0.01         |
| Proximity | Min distance to patient across all navigation legs | (3.0 - d) / (3.0 - 0.8) |
| Approach  | 0.7 × position_score + 0.3 × yaw_score at delivery | [0, 1]                  |

No quality bonuses (unlike meal prep). The learning signal comes entirely
from route choice: different source locations produce different feature
vectors along the safety, time, and battery dimensions.

The rating model is identical to meal prep:

```
r_i = 5 - 4 · f_i · w_i + noise
```

---

## 8. A\* Cost Function (multi-objective)

The planner uses a weighted sum of 5 cost components:

```python
total_cost = w · [c_time, c_safety, c_battery, c_proximity, c_approach]
```

### Per-component costs:

**Time cost**: `time / 60.0` — normalised by typical mission duration.

**Safety cost** (fuzzy mode):

```python
battery_risk = fuzzy_battery_penalty(soc)     # high if soc < 0.25
congestion   = fuzzy_congestion_penalty(pos)  # high near nurse_station
scarcity     = 0.3 / (1 + stock)             # high when stock is low
c_safety = battery_risk + congestion + scarcity
```

**Safety cost** (crisp fallback):

```python
if location == "nurse_station":  += 0.3
if location == "equipment_storage":  += 0.2
if battery_soc < 0.15:  += 0.5
if battery_soc < 0.25:  += 0.2
```

**Battery cost**: `state.battery_soc - next_state.battery_soc`

**Proximity cost**: 0.0 at patient_bed_left, 0.1 at patient_bed_right.

**Approach cost**: 0.0 for left approach, 0.05 for right approach.

---

## 9. Why Different Profiles Choose Different Routes

**Speed-oriented (w = [0.50, 0.12, 0.14, 0.14, 0.10])** — Time dominates
at 50%. The planner picks whichever pharmacy and supply room minimises total
distance. Safety risk (12%) is noise. pharmacy_north is usually faster from
home despite higher risk. Shortest plan wins.

**Safety-first (w = [0.10, 0.50, 0.15, 0.15, 0.10])** — Safety dominates
at 50%. The planner avoids pharmacy_north (risk 0.30) and routes through
pharmacy_south (risk 0.05). supply_A preferred over supply_B. This adds
distance but slashes the safety cost. The 0.25 risk difference × 50% weight
= 0.125 cost savings — far outweighing the time penalty at 10% weight.

**Energy-conscious (w = [0.15, 0.15, 0.45, 0.15, 0.10])** — Battery at
45%. Minimises total distance (proportional to energy drain). Similar to
speed in route choice, but more willing to add recharge when battery is low.
The 30s recharge time only costs 15% × (30/120) = 0.0375 in the objective
— cheap insurance against mission failure.

**Comfort-focused (w = [0.15, 0.15, 0.10, 0.40, 0.20])** — Proximity at
40%, approach at 20%. Route to pharmacy/supply is less constrained; the
delivery leg matters most. Approach side choice driven by which gives better
position + yaw alignment scores at the bedside.

---

## 10. Comparison with Meal Preparation Task

| Aspect                | Medication Delivery                      | Meal Preparation                                     |
| --------------------- | ---------------------------------------- | ---------------------------------------------------- |
| Plan diversity source | Route selection (which pharmacy, supply) | Meal type selection (sandwich, soup, full)           |
| Decision count        | 4 (pharmacy, supply, recharge, side)     | 1 major (meal type) + 1 minor (side)                 |
| Feature diversity     | Moderate (route changes safety/time)     | High (meal type changes all 5 dimensions)            |
| Quality bonuses       | None                                     | Per-meal-type modifiers on approach/proximity/safety |
| Ordering constraints  | Linear (collect → collect → deliver)     | Branching (different prep sequences per meal)        |
| Navigation complexity | 3-4 legs, all different locations        | 4-9 steps, kitchen cluster + delivery                |
| Structural insight    | Route tradeoffs under preference weights | Meal selection as implicit preference expression     |

The two tasks complement each other for learning: medication provides
stable baseline signal from route optimization, while meal prep provides
high-variance signal from meal-type branching. Running both in alternating
episodes doubles the feature diversity the preference learner sees.

---

## 11. File Structure (refactored)

```
tasks/medication_delivery/
  __init__.py              # Package exports
  task_actions.py          # TaskAction enum + module-level constants
  task_state.py            # TaskState dataclass
  task_state_manager.py    # Preconditions, transitions, stock tracking
  lhmoaro.py               # LearnableTranslator (shared infrastructure)
```

Mirrors `tasks/meal_preparation/` layout. Module-level constants
(`NAVIGATION_ACTIONS`, `ACTION_TARGET_LOCATIONS`, `IN_PLACE_ACTIONS`,
`ACTION_DURATIONS`) match the meal prep pattern for symmetric dispatch
in the integrator.

Backwards compatible: `task_state_manager.py` re-exports `TaskAction` and
`TaskState`, so existing imports continue to work.

---

## 12. Integration

Both tasks share:

- Same environment (env3.py)
- Same 5-dimension preference weight vector w\*
- Same preference learner (outer loop)
- Same translator with learnable φ (inner loop)
- Same HybridMPC controller (Acados + trajectory/terminal sensitivities)
- Same direct waypoint reference strategy as medication delivery
- Same FuzzyStateEstimator (continuous position → discrete state)

The integrator dispatches to the correct task planner and state manager
based on `task_type` ("medication" or "meal"). Both produce the same 5D
feature vector for the shared preference learner.

---

## 13. Integration Test Results

_Run `test_med_integration_{safety,speed,energy}.py` to populate._

| Metric                | Safety-First                       | Speed-Oriented                     | Energy-Conscious                   |
| --------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- |
| True weights          | [0.10, **0.50**, 0.15, 0.15, 0.10] | [**0.50**, 0.12, 0.14, 0.14, 0.10] | [0.15, 0.15, **0.45**, 0.15, 0.10] |
| Learned dominant      | —                                  | —                                  | —                                  |
| Final distance to w\* | —                                  | —                                  | —                                  |
| Best distance to w\*  | —                                  | —                                  | —                                  |
| First convergence     | —                                  | —                                  | —                                  |
| Unique plans          | —                                  | —                                  | —                                  |
| Pharmacy choice       | —                                  | —                                  | —                                  |
| Recharge inserted     | —                                  | —                                  | —                                  |
| Success rate          | —                                  | —                                  | —                                  |
| Feature separation    | —                                  | —                                  | —                                  |

### Expected behaviours to validate

- Safety profile → pharmacy_south (avoids risk 0.30)
- Speed profile → closest pharmacy (regardless of risk)
- Energy profile → shortest route; recharge when battery starts low
- All 3 correctly identify dominant weight after 10 episodes
- Feature separation between medication and meal episodes > 0.05
