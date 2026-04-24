# core/task_planning/

Shared base classes for discrete task planning. Both task packages inherit from here instead of duplicating the A* loop and battery utilities.

## Files

### `base_state.py` — `TaskStateMixin`

A plain mixin (not a dataclass) that contributes only methods. Task state dataclasses inherit it alongside their own fields.

**Methods:**

| Method | Description |
|--------|-------------|
| `get_discrete_battery_level()` | Discretizes `battery_soc` to 8 levels (0–7) for A* hashing |
| `needs_recharge(threshold=0.2)` | Returns `True` if battery is critically low |
| `_shared_copy_kwargs()` | Returns shared fields as kwargs for use inside `copy()` |
| `_shared_to_dict()` | Returns shared fields as a dict for use inside `to_dict()` |

**Shared fields expected on the subclass:**

```
battery_soc: float
approach_side: Optional[str]
location_memberships: Optional[Dict[str, float]]
location_stock: Optional[Dict[str, int]]
step_count: int
time_elapsed: float
distance_traveled: float
num_replans: int
```

**Usage pattern:**

```python
@dataclass
class MyTaskState(TaskStateMixin):
    location: str
    has_item: bool = False
    # ... task-specific fields

    def copy(self):
        return MyTaskState(
            location=self.location,
            has_item=self.has_item,
            **self._shared_copy_kwargs(),   # battery, tracking, fuzzy fields
        )

    def to_dict(self):
        return {"location": self.location, **self._shared_to_dict()}
```

---

### `base_planner.py` — `PlanNode` + `BaseTaskPlanner`

#### `PlanNode`

Ordered dataclass used in the A* priority queue:

| Field | Type | Description |
|-------|------|-------------|
| `f_cost` | float | Total estimated cost (`g + h`) — used for heap ordering |
| `g_cost` | float | Actual cost from root |
| `h_cost` | float | Heuristic estimate to goal |
| `state` | Any | Task state at this node |
| `parent` | PlanNode | Parent pointer for path reconstruction |
| `action` | Any | Action that produced this state |

#### `BaseTaskPlanner(ABC)`

Provides the full A* loop. Subclasses implement two abstract methods and optionally override `print_plan`.

**Public interface:**

```python
planner = MyTaskPlanner(preference_weights=w_hat, fuzzy_estimator=fuzzy)
actions, states, info = planner.plan(initial_state, max_nodes=2000)
```

`info` dict: `{"success", "nodes_expanded", "nodes_generated", "total_cost", "plan_length"}`

**Abstract methods to implement:**

```python
def _expand(self, state) -> List[Tuple[action, next_state, edge_cost]]:
    """Generate successors. Call your task state manager here."""
    ...

def _heuristic(self, state) -> float:
    """Admissible cost-to-go estimate."""
    ...
```

**Concrete implementations:**

| Subclass | Location |
|----------|----------|
| `HighLevelTaskPlanner` | `tasks/medication_delivery/task_planner.py` |
| `MealTaskPlanner` | `tasks/meal_preparation/task_planner.py` |
