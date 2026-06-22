# core/learning/

Patient preference learning via projected gradient descent on the probability simplex.

## Files

### `preference_learner.py`

The central learning engine. Maintains and updates the robot's estimate of a patient's hidden preference profile.

### `translator_params.py`

Dataclasses for the learnable MPC parameter space.

- `TranslatorParams` — 20-parameter vector `φ`:
  - indices 0–5: `Q_diag` (position/velocity tracking weights)
  - indices 6–8: `r_base_ax, r_base_ay, r_base_alpha` (pre-softplus control cost params)
  - indices 9–19: remaining Q/R/horizon/tolerance coefficients
- `MPCParameterGradients` — holds `dQ_dphi (6×20)`, `dR_dphi (3×20)` for the chain-rule update

### `learnable_translator.py`

Maps the fixed internal modulation vector to MPC parameters `(Q, R, Q_terminal)`.

#### Learned Diagonal R (softplus activation)

Each axis has an independent pre-activation parameter:
```
R_i = log(1 + exp(r_base_i)) + ε      # softplus, always positive
```
where `r_base_ax`, `r_base_ay`, `r_base_alpha` are learned. This guarantees `R > 0` without clamping and gives a smooth gradient through zero.

#### Terminal Target Learning

The integrated runner keeps the translator's `Q/R` coefficients fixed and learns
the MPC terminal target `z_target` instead. After applying the per-location
observation offset, `E = 1 - μ(goal_location)` updates `z_target` through
`J_E_z_target.T @ E`; it does not directly update `Q_terminal_diag`.

---

## Problem Formulation

A patient has a hidden preference profile `w* ∈ Δ⁴` (a point on the 5-simplex):
```
w* = [w_time, w_safety, w_battery, w_proximity, w_approach]
w* ≥ 0,  Σ w*_i = 1
```

The robot maintains an estimate `w_hat` and updates it after each episode. Learning is **multi-dimensional** — each dimension carries an independent signal; ratings are never collapsed to a scalar.

---

## Learning Loop (per episode)

```
1. Execute episode → extract features f ∈ [0,1]⁵
       f = [time/max_time, safety_score, battery_used, proximity_error, approach_quality]
       (0 = best, 1 = worst for each dimension)

2. Patient provides ratings r ∈ [1,5]⁵  (one per dimension)

3. Compute loss:
       L(w) = ‖w ⊙ f − r‖²   (per-dimension MSE)

4. Gradient step:
       w ← w − η ∇_w L
       ∇_w L = (w ⊙ f − r) ⊙ f

5. Project back onto simplex:
       w_hat ← Π_Δ(w)

6. Decay learning rate:
       η ← η₀ / (1 + decay × episode)
```

---

## Key Classes

### `PatientProfile`
Dataclass holding a named ground-truth preference vector `w*`. Used only in simulation experiments to generate synthetic ratings; the real robot never has access to it.

### `PreferenceLearningEngine`
Maintains `w_hat` across episodes. Exposes:
- `update(features, ratings) → w_hat` — runs one gradient + projection step
- `loss_history`, `per_dim_loss_history` — convergence data for Section 8 figures
- `gradient_norm_history` — learning diagnostics

---

## Predefined Patient Profiles

Used as ground-truth `w*` in experiments:

| Name | Time | Safety | Battery | Proximity | Approach |
|------|------|--------|---------|-----------|----------|
| speed_oriented | 0.50 | 0.12 | 0.14 | 0.14 | 0.10 |
| safety_first | 0.10 | 0.50 | 0.15 | 0.15 | 0.10 |
| comfort_focused | 0.15 | 0.15 | 0.10 | 0.40 | 0.20 |
| energy_conscious | 0.15 | 0.15 | 0.45 | 0.15 | 0.10 |
| presentation_focused | 0.05 | 0.10 | 0.05 | 0.20 | 0.60 |

---

## Convergence

Target: `‖w_hat − w*‖ < 0.05` (typically reached in 15–20 episodes).

The `per_dim_loss_history` tracks per-dimension MSE for Section 8 figure B6 — showing which preference dimensions converge fastest and whether any dimensions are structurally harder to learn (e.g., approach quality requires the right task path to generate informative features).
