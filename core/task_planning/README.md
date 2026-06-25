# core/task_planning/

Shared task-state utilities and PDDL engine selection. Runtime symbolic planning is done through Unified Planning with ENHSP-opt and the files in `unified_planning/`.

## Files

### `base_state.py` — `TaskStateMixin`

A plain mixin (not a dataclass) that contributes only methods. Task state dataclasses inherit it alongside their own fields.

**Methods:**

| Method | Description |
|--------|-------------|
| `get_discrete_battery_level()` | Discretizes `battery_soc` to 8 levels (0–7) for state hashing |
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

### `pddl_engine.py`

Provides the canonical PDDL planner selection used by the episode runner:

```python
make_pddl_oneshot_planner("enhsp-opt")
```

The runner syncs the current Python task state into the PDDL initial state at each replan, solves with ENHSP-opt, converts the returned PDDL action names into task action enums, executes only the first action, and replans.
