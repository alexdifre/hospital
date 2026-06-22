# integration/

Full system integrator that wires all framework components into runnable end-to-end experiments.

## Module Structure

`integrator2.py` (2079 lines) was split into focused modules. All existing imports from `integration.integrator2` and `integration` continue to work.

### `system.py` — `FullMedicationDeliverySystem` *(canonical)*
Top-level class. Wires all 7 subsystems in `__init__`, owns all helper methods, and exposes the public API.

Subsystems initialised:
- `ExpandedHospitalMuJoCoEnv` — 14-location MuJoCo physics environment
- `HybridMPC` — Acados control
- Direct waypoint references — 21 start-to-goal points per executed leg
- `PreferenceLearner` — outer loop: weight vector `w` update on probability simplex
- `LearnableTranslator` — MPC parameter map; `φ/Q/R` stay fixed in the runner
- Terminal target learner — applies a fixed per-location observation offset, computes `μ(goal_location)`, then updates `z_target` from `1 - μ(goal_location)` after each MPC leg
- Episode feature extraction — normalised from execution metrics in `episode_runner.py`
- `FuzzyStateEstimator` — fuzzy inference for task state tracking

Key helpers on `FullMedicationDeliverySystem`:
- Risk map: `_get_risk_value`, `_perturb_risk_map`
- Plan structure: `_extract_plan_structure`, `_extract_meal_plan_structure`
- Geometry: `_wrap_angle`, `_pos_score_from_error`, `_yaw_score`
- Exploration: `_perturb_weights_for_exploration`
- Multi-episode runners: `run_multiple_episodes`, `run_mixed_episodes`

**Ablation flags** (pass to `__init__`):

| Flag | Effect |
|------|--------|
| `fix_translator` | Kept for compatibility; `φ/Q/R` updates are already disabled |
| `dynamic_risk_perturbation` | Randomises location risk mid-episode — robustness test |
| `rating_noise` | Adds Gaussian noise to patient ratings — noise sweep experiments |

### `episode_runner.py` — `EpisodeRunnerMixin`
Hot-path execution logic mixed into `FullMedicationDeliverySystem`.

- `_execute_leg()` — single navigation leg: MPC solve loop → physics step → feature accumulation
- `run_episode()` — five phases: plan → execute legs + terminal-target update → outer loop (w) → metrics

### `reporting.py` — `ReportingMixin`
Output and persistence methods mixed into `FullMedicationDeliverySystem`.

- `_print_episode_summary()`, `_print_final_summary()`
- `_save_json()`, `_save_final_summary()`
- `visualize_learning()`

### `metrics.py` — `EpisodeMetrics`, `LearningCurveTracker`
Per-episode and cross-episode metric tracking.

- `EpisodeMetrics` — scalar fields per episode with `to_dict()`
- `LearningCurveTracker` — aggregates across episodes; `record()`, `print_summary()`, `export_csv()`

### `integrator2.py` — shim
```python
from integration.system import FullMedicationDeliverySystem  # noqa: F401
```
Preserved so existing test imports (`from integration.integrator2 import FullMedicationDeliverySystem`) continue to work.

### `__init__.py`
Re-exports `FullMedicationDeliverySystem` so `from integration import FullMedicationDeliverySystem` also works.

---

## Full Architecture

```
Task Planner (A*)
    → FuzzyStateEstimator
    → Direct 21-point waypoint reference
    → HybridMPC (Acados control)
    → MuJoCo (physics)
    → Episode feature extraction
    → PreferenceLearner (outer loop: w update)
    → Terminal target learner (z_target update; φ/Q/R fixed)
```

---

## Episode Result Schema

All episodes emit a JSON result dict:

```json
{
  "episode": 3,
  "task_type": "medication_delivery",
  "success": true,
  "features": [0.42, 0.18, 0.31, 0.24, 0.15],
  "weights_before": [0.20, 0.20, 0.20, 0.20, 0.20],
  "weights_after":  [0.22, 0.18, 0.21, 0.20, 0.19],
  "learner_mse": 0.031,
  "translator_params": [...],
  "trajectory_xy": [[0.0, 0.0], [1.2, 0.3], ...],
  "battery_used_pct": 12.4,
  "path_efficiency": 0.87
}
```

---

## Running Experiments

```bash
# Single episode (quick smoke test)
python integration/system.py

# Full experiment suite (all profiles, all ablations)
python tests/run_section8_experiments.py
```
