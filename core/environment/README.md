# core/environment/

MuJoCo hospital ward simulation environment.

## Files

### `env.py` — Unified 14-Location Hospital

Single environment covering both medication delivery and meal preparation tasks.

**Physical setup:**

| Property | Value |
|----------|-------|
| Ward size | 50 × 50 m |
| Physics timestep | dt = 0.01 s |
| Control timestep | Ts = 0.2 s (20 MuJoCo substeps per control step) |
| Robot DOF | 6 — state `[x, y, θ, vx, vy, ωz]` |
| Control inputs | 3 — `[ax, ay, αz]` |

**Locations (14 total, matching the PDDL problem files):**

| Location | Coordinates | Risk | Congestion | Notes |
|----------|-------------|------|------------|-------|
| home | (0, 0) | — | — | Episode start |
| pharmacy_north | (5, 18) | 0.30 | 0.2 | High-risk, stocked (5) |
| pharmacy_south | (6, −15) | 0.05 | 0.1 | Low-risk, stocked (2) |
| supply_A | (14, 10) | — | — | Stock prob 0.7, initial 7 |
| supply_B | (15, −12) | — | — | Stock prob 0.9, initial 1 |
| charge_main | (3, 5) | — | — | Charging station |
| charge_backup | (17, −18) | — | — | Charging station |
| patient_bed_left | (20.5, 12) | — | — | Left approach variant |
| patient_bed_right | (23.5, 10) | — | — | Right approach variant |
| pantry | (−3, 15) | — | 0.15 | Kitchen: ingredient source (sandwich/soup/full_meal stock) |
| fridge | (−6, 17.5) | — | 0.17 | Kitchen: refrigerated ingredient source |
| prep_station | (0, 20) | — | 0.2 | Kitchen: assembly and plating |
| stove | (−3, 20.5) | 0.70 | 0.25 | Kitchen: cooking — highest risk in ward |
| quality_check | (3, 21) | — | 0.1 | Kitchen: final quality check |

Stock is tracked per-ingredient at the pantry: `pantry_sandwich` (10), `pantry_soup` (5), `pantry_full_meal` (3).

---

## Environment API

```python
env = ExpandedHospitalMuJoCoEnv(render_mode="human")  # or "rgb_array" / None
obs = env.reset()
obs, reward, done, info = env.step(control)  # control: np.array([ax, ay, αz])
env.close()
```

The `info` dict includes:
- `robot_pos`: `(x, y, θ)`
- `battery_soc`: State of charge `[0, 1]`
- `min_obstacle_distance`: Current clearance in metres
- `location_stocks`: `Dict[str, int]` — remaining stock per location

The occupancy grid is exposed via `env.get_occupancy_grid()` for use by the navigation stack.
