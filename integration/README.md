# integration/

Full system integrator that wires all framework components into runnable end-to-end experiments.

## Module Structure

`integrator2.py` (2079 lines) was split into focused modules. All existing imports from `integration.integrator2` and `integration` continue to work.

### `system.py` — `FullMedicationDeliverySystem` *(canonical)*
Top-level class. Wires all 7 subsystems in `__init__`, owns all helper methods, and exposes the public API.

Subsystems initialised:
- `ExpandedHospitalMuJoCoEnv` — 14-location MuJoCo physics environment
- `HybridMPC` — Acados control + CasADi IFT sensitivities
- `NavigationStack` — A* grid planner → waypoint sequence
- `PreferenceLearner` — outer loop: weight vector `w` update on probability simplex
- `LearnableTranslator` — inner loop: MPC parameter map `φ` update via IFT chain rule
- `RewardEngine` — feature extraction from episode state
- `FuzzyStateEstimator` — fuzzy inference for task state tracking

Key helpers on `FullMedicationDeliverySystem`:
- Risk map: `_get_risk_value`, `_perturb_risk_map`
- Plan structure: `_extract_plan_structure`, `_extract_meal_plan_structure`
- Geometry: `_wrap_angle`, `_pos_score_from_error`, `_yaw_score`
- Exploration: `_perturb_weights_for_exploration`
- Ablation: `_compute_finite_diff_sensitivities`
- Multi-episode runners: `run_multiple_episodes`, `run_mixed_episodes`

**Ablation flags** (pass to `__init__`):

| Flag | Effect |
|------|--------|
| `fix_translator` | Disables inner loop (φ frozen) — outer-only baseline |
| `use_finite_diff` | Replaces IFT sensitivities with numerical gradients |
| `dynamic_risk_perturbation` | Randomises location risk mid-episode — robustness test |
| `rating_noise` | Adds Gaussian noise to patient ratings — noise sweep experiments |

### `episode_runner.py` — `EpisodeRunnerMixin`
Hot-path execution logic mixed into `FullMedicationDeliverySystem`.

- `_execute_leg()` — single navigation leg: MPC solve loop → physics step → feature accumulation
- `run_episode()` — five phases: plan → execute legs → inner loop (φ) → outer loop (w) → metrics

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
    → NavigationStack (A* grid → waypoints)
    → HybridMPC (Acados control + CasADi sensitivities)
    → MuJoCo (physics)
    → RewardEngine (feature extraction)
    → PreferenceLearner (outer loop: w update)
    → LearnableTranslator (inner loop: φ update via IFT chain rule)
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
