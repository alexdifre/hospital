# tests/

Experimental validation for the paper. Organised into three layers matching the development timeline.

## Structure

```
tests/
  profile_validation/     Per-profile correctness checks (run before full experiments)
  results/                Saved experiment outputs
  run_section8_experiments.py   Full experiment runner (Section 8)
  run_section8_figures.py       Convenience wrapper: run + plot
  generate_section8_figures.py  Figure generator (reads saved results)
  rerun_outer_only.py           Reruns outer-loop-only ablation
```

---

## [profile_validation/](profile_validation/)

Per-patient-profile correctness tests. Run a small number of episodes against a single known profile to verify correct route selection, meal path choice, and preference-driven behaviour — **before** running the full experiment suite.

| Script | Task | Profile |
|--------|------|---------|
| `med_delivery_speed.py` | Medication | speed_oriented |
| `med_delivery_safety.py` | Medication | safety_first |
| `med_delivery_energy.py` | Medication | energy_conscious |
| `meal_integration_presentation.py` | Meal | presentation_focused |
| `meal_integration_safety.py` | Meal | safety_first |
| `meal_integration_approach.py` | Meal | approach-focused |
| `meal_prep.py` | Meal | Unit test (all paths) |
| `multi_profile.py` | Both | Cross-profile demo |
| `diagnose_profile.py` | Any | Debug utility |

---

## Section 8 Experiment Scripts

These live at the top of `tests/` and are **not moved** — `rerun_outer_only.py` imports directly from `run_section8_experiments.py` and both must stay co-located.

### `run_section8_experiments.py`

Main experiment runner. Conditions:

| Category | Variants |
|----------|---------|
| Full | 5 profiles × N seeds |
| Baselines | uniform, random, outer-only, bandit |
| Ablations | crisp state, no decay, med-only, meal-only, finite-diff |
| Robustness | noise sweep [0.05–0.40], random init, dynamic risk, ambiguous profiles |

Output goes to `results/section8/` as JSON, one file per episode.

### `generate_section8_figures.py`

Reads saved JSON results and generates publication plots:

| Figure | Content |
|--------|---------|
| B1–B5 | Preference weight convergence per profile |
| B6 | Per-dimension MSE over episodes |
| B7 | Translator φ parameter drift |
| B8 | Robot trajectory (x, y) visualisation |
| B9 | Energy efficiency and path quality |

### `run_section8_figures.py`

Wrapper that runs experiments then immediately generates figures.

### `rerun_outer_only.py`

Reruns the outer-loop-only ablation (`fix_translator=True`) across all profiles and 5 seeds. Imports `run_section8_experiments` directly — must stay in the same directory.

---

## Logs

`*.log` files are gitignored. They are terminal output from long experiment runs and can be large (10+ MB). If you need to inspect a run, redirect stdout when launching:

```bash
python tests/run_section8_experiments.py --condition full 2>&1 | tee tests/my_run.log
```
