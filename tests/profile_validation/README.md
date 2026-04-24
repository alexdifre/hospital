# tests/profile_validation/

Per-patient-profile correctness tests. Each script runs a small number of episodes (typically 3–10) against a single known patient profile and verifies the system behaves as expected — correct route selection, meal path choice, and preference-driven trade-offs.

These are run **before** the full Section 8 experiment suite to confirm each task and profile works in isolation.

## Medication Delivery

| File | Profile (w*) | What it verifies |
|------|-------------|-----------------|
| `med_delivery_speed.py` | speed_oriented [0.50, 0.12, 0.14, 0.14, 0.10] | Picks pharmacy_north (shorter, riskier) over pharmacy_south |
| `med_delivery_safety.py` | safety_first [0.10, 0.50, 0.15, 0.15, 0.10] | Routes via pharmacy_south (risk 0.05 vs 0.30) despite longer distance |
| `med_delivery_energy.py` | energy_conscious [0.15, 0.15, 0.45, 0.15, 0.10] | Minimises total distance; inserts recharge step when beneficial |

## Meal Preparation

| File | Profile | What it verifies |
|------|---------|-----------------|
| `meal_integration_presentation.py` | presentation_focused [0.05, 0.10, 0.05, 0.20, 0.60] | Converges to full_meal path (plating approach bonus dominates) |
| `meal_integration_safety.py` | safety_first | Avoids stove (risk 0.70); prefers sandwich path |
| `meal_integration_approach.py` | approach-focused | Selects full_meal or soup based on approach/proximity weights |

## Unit & Cross-Profile

| File | Purpose |
|------|---------|
| `meal_prep.py` | Unit test for meal task logic: precondition ordering, all 3 paths reach goal, stock depletion, planner path selection under different weights |
| `multi_profile.py` | Runs safety-oriented vs approach-oriented side-by-side — used for slides/demos to visually show preference-conditioned behaviour |
| `diagnose_profile.py` | Debug utility: prints per-step features, weights, and gradients for a specified profile. Useful when a profile isn't converging as expected |

## Usage

```bash
# Run a single profile test
python tests/profile_validation/med_delivery_speed.py

# Diagnose a specific profile
python tests/profile_validation/diagnose_profile.py --profile safety_first --episodes 20
```
